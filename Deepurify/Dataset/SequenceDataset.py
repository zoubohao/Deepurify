import os
import pickle
import random
from copy import deepcopy
from math import floor
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from Deepurify.Utils.IOUtils import loadTaxonomyTree, readFile
from Deepurify.Utils.SequenceUtils import (
    ConvertSeqToImageTensorMoreFeatures, ConvertTextToIndexTensor,
    RandomlyReturnNegTaxoDiffPhy, RandomReturnNegTaxoSamePhy,
    SeqCutToModelLengthIntervalAndAddNoisy, returnTaxoTextsInSameLevel,
    sampleSeqFromFasta)


class SequenceTrainingDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        min_model_len: int,
        max_model_len: int,
        taxo_vocabulary: Dict,
        vocab_3Mer: Dict,
        vocab_4Mer: Dict,
        taxomonyTreePath: str,
        sampleName2weightPath: str,
        misMatchNum: int,
        gmmModelPath: str,
        finetune=False,
    ):
        self.finetune = finetune
        self.max_model_len = max_model_len
        self.min_model_len = min_model_len
        self.taxo_vocabulary = taxo_vocabulary
        self.vocab_3Mer = vocab_3Mer
        self.vocab_4Mer = vocab_4Mer
        self.file_path = file_path
        self.files = os.listdir(file_path)
        self.length = len(self.files)
        self.num_mismatch = misMatchNum
        self.spName2weight = {}
        self.phylums = set()
        self.tree = loadTaxonomyTree(taxomonyTreePath)
        self.phy2Index = {}  # 0 is for out of distribution data
        k = 0
        for child in self.tree["Children"]:
            self.phylums.add(child["Name"])
            self.phy2Index[child["Name"]] = k
            k += 1
        self.num_phylum = len(self.phylums)
        with open(sampleName2weightPath, "r") as rh:
            for line in rh:
                split_info = line.strip("\n").split("\t")
                self.spName2weight[split_info[0]] = float(split_info[1])
        rb = open(gmmModelPath, "rb")
        self.gmmModel = pickle.load(rb)
        rb.close()

        self.spe2index = {}
        index = 0
        for name, _ in self.taxo_vocabulary.items():
            if "s__" == name[0:3]:
                self.spe2index[name] = index
                index += 1


    def _get_random_num(self, len_val: int = 6):
        if len_val == 6:
            return np.random.choice(6, None, replace=False, p=[0.1, 0.11, 0.12, 0.13, 0.14, 0.40]) + 1
        return np.random.choice(5, None, replace=False, p=[0.11, 0.13, 0.15, 0.17, 0.44]) + 2


    def __getitem__(self, index):
        # data variable is a tuple, the first position is the sequence and the second position is the taxonomy label of this sequence
        seq, if_noisy = sampleSeqFromFasta(
            os.path.join(self.file_path, self.files[index]),
            self.min_model_len,
            self.max_model_len,
            not self.finetune)
        # data = readFile(os.path.join(self.file_path, str(index) + ".txt"))
        # add noisy
        # seq, if_noisy = SeqCutToModelLengthIntervalAndAddNoisy(
        #     seq,
        #     min_model_len=self.min_model_len,
        #     max_model_len=self.max_model_len,
        #     gmmModel=self.gmmModel,
        #     if_noisy=not self.finetune
        # )
        # convert to tensor
        ori_rev_tensor, feature_3Mer, feature_3Mer_rev_com, feature_4Mer, feature_4Mer_rev_com = \
            ConvertSeqToImageTensorMoreFeatures(
                self.max_model_len, seq, self.vocab_3Mer, self.vocab_4Mer
            )
        # generateing negative labelsf.vocab_4Mer)
        matchTaxoLevel = self._get_random_num()
        texts = "_".join(self.files[index].split("_")[0:-1])
        spec = texts.split("@")[-1]
        matchText = texts.split("@")[0: matchTaxoLevel]
        misMatchTensorList = []
        sameLevelMisMatches = returnTaxoTextsInSameLevel(matchText, self.num_mismatch // 2, self.tree)
        for misMatchText in sameLevelMisMatches:
            misMatchTensorList.append(ConvertTextToIndexTensor(self.taxo_vocabulary, misMatchText))
        oriPhy = matchText[0]
        copyPhys = deepcopy(self.phylums)
        copyPhys.remove(oriPhy)
        mismatchPhylums = list(copyPhys)
        curNum = len(misMatchTensorList)
        left = self.num_mismatch - curNum
        num_diff_phy = floor(left * 0.3) + 1
        num_sam_phy = self.num_mismatch - curNum - num_diff_phy
        for _ in range(num_diff_phy):
            random.shuffle(mismatchPhylums)
            startPhylum = mismatchPhylums[np.random.randint(self.num_phylum - 1)]
            misMatchTaxoLevel = self._get_random_num()
            misMatchText = RandomlyReturnNegTaxoDiffPhy(self.tree, startPhylum, misMatchTaxoLevel, texts)
            misMatchTensorList.append(ConvertTextToIndexTensor(self.taxo_vocabulary, misMatchText))
        for _ in range(num_sam_phy):
            misMatchTaxoLevel = self._get_random_num(5)
            misMatchText = RandomReturnNegTaxoSamePhy(self.tree, oriPhy, misMatchTaxoLevel, texts)
            if misMatchText is not None:
                misMatchTensorList.append(ConvertTextToIndexTensor(self.taxo_vocabulary, misMatchText))
            else:
                random.shuffle(mismatchPhylums)
                startPhylum = mismatchPhylums[np.random.randint(self.num_phylum - 1)]
                misMatchTaxoLevel = self._get_random_num()
                misMatchText = RandomlyReturnNegTaxoDiffPhy(self.tree, startPhylum, misMatchTaxoLevel, texts)
                misMatchTensorList.append(ConvertTextToIndexTensor(self.taxo_vocabulary, misMatchText))
        matchTextTensor = ConvertTextToIndexTensor(self.taxo_vocabulary, matchText)
        textList = misMatchTensorList
        insertIndex = np.random.choice(self.num_mismatch + 1, 1, replace=False)
        textList.insert(insertIndex[0], matchTextTensor)
        weight = self.spName2weight[texts]
        # label
        pairLabel = torch.tensor(insertIndex[0], dtype=torch.int64)
        weight = torch.tensor(weight, dtype=torch.float32)
        phyLabel = torch.tensor(self.phy2Index[oriPhy], dtype=torch.int64)
        speLabel = torch.tensor(self.spe2index[spec], dtype=torch.int64)
        # Constrain for label, the phylum tensor will be set as a anchor.
        oriPhyTensor = ConvertTextToIndexTensor(self.taxo_vocabulary, [oriPhy])
        lowMatchTaxoLevel = self._get_random_num(5)
        lowMatchText = texts.split("@")[0: lowMatchTaxoLevel]
        lowMatchTextTensor = ConvertTextToIndexTensor(self.taxo_vocabulary, lowMatchText)
        # outer text tensor.
        random.shuffle(mismatchPhylums)
        startPhylum = mismatchPhylums[np.random.randint(self.num_phylum - 1)]
        misMatchTaxoLevel = self._get_random_num()
        misMatchText = RandomlyReturnNegTaxoDiffPhy(self.tree, startPhylum, misMatchTaxoLevel, texts)
        outerMisMatchTextTensor = ConvertTextToIndexTensor(self.taxo_vocabulary, misMatchText)
        if_noisy_tensor = torch.tensor(data=if_noisy, dtype=torch.float32)
        return (
            ori_rev_tensor,
            feature_3Mer,
            feature_3Mer_rev_com,
            feature_4Mer,
            feature_4Mer_rev_com,
            torch.stack(textList, dim=0),
            pairLabel,
            weight,
            oriPhyTensor,
            lowMatchTextTensor,
            outerMisMatchTextTensor,
            if_noisy_tensor,
            phyLabel,
            speLabel
        )


    def __len__(self):
        return self.length


class SequenceSampledTestDataset(Dataset):
    def __init__(
        self,
        file_path,
        min_model_len,
        max_model_len,
        taxo_vocabulary: Dict,
        vocab_3Mer: Dict,
        vocab_4Mer: Dict,
        taxomonyTreePath: str,
        sampleName2weightPath: str,
        misMatchNum: int,
        gmmModelPath: str,
    ):
        self.max_model_len = max_model_len
        self.min_model_len = min_model_len
        self.taxo_vocabulary = taxo_vocabulary
        self.vocab_3Mer = vocab_3Mer
        self.vocab_4Mer = vocab_4Mer
        self.file_path = file_path
        k = 0
        for file in os.listdir(file_path):
            try:
                int(os.path.splitext(file)[0])
                k += 1
            except:
                pass
        self.length = k
        self.num_mismatch = misMatchNum
        self.phylums = set()
        self.tree = loadTaxonomyTree(taxomonyTreePath)
        self.phy2Index = {}
        k = 0
        for child in self.tree["Children"]:
            self.phylums.add(child["Name"])
            self.phy2Index[child["Name"]] = k
            k += 1
        self.num_phylum = len(self.phylums)
        rb = open(gmmModelPath, "rb")
        self.gmmModel = pickle.load(rb)
        rb.close()


    def __getitem__(self, index):
        with torch.no_grad():
            data = readFile(os.path.join(self.file_path, str(index) + ".txt"))
            seq = data[0]
            seq, if_noisy = SeqCutToModelLengthIntervalAndAddNoisy(
                seq, min_model_len=self.min_model_len, max_model_len=self.max_model_len, gmmModel=self.gmmModel, if_noisy=False
            )
            assert if_noisy == 0, "error"
            ori_rev_tensor, feature_3Mer, feature_3Mer_rev_com, feature_4Mer, feature_4Mer_rev_com = ConvertSeqToImageTensorMoreFeatures(
                self.max_model_len, seq, self.vocab_3Mer, self.vocab_4Mer
            )
            matchTaxoLevel = np.random.choice(6, 1, replace=False, p=[0.13, 0.14, 0.15, 0.16, 0.17, 0.25]) + 1
            texts = data[1]
            matchText = texts.split("@")[0: matchTaxoLevel[0]]
            misMatchTensorList = []
            sameLevelMisMatches = returnTaxoTextsInSameLevel(matchText, self.num_mismatch // 2, self.tree)
            for misMatchText in sameLevelMisMatches:
                misMatchTensorList.append(ConvertTextToIndexTensor(self.taxo_vocabulary, misMatchText))
            oriPhy = matchText[0]
            copyPhys = deepcopy(self.phylums)
            copyPhys.remove(oriPhy)
            mismatchPhylums = list(copyPhys)
            curNum = len(misMatchTensorList)
            left = self.num_mismatch - curNum
            num_diff_phy = floor(left * 0.3) + 1
            num_sam_phy = self.num_mismatch - curNum - num_diff_phy
            for _ in range(num_diff_phy):
                random.shuffle(mismatchPhylums)
                startPhylum = mismatchPhylums[np.random.randint(self.num_phylum - 1)]
                misMatchTaxoLevel = np.random.choice(6, 1, replace=False, p=[0.13, 0.14, 0.15, 0.16, 0.17, 0.25]) + 1
                misMatchText = RandomlyReturnNegTaxoDiffPhy(self.tree, startPhylum, misMatchTaxoLevel[0], texts)
                misMatchTensorList.append(ConvertTextToIndexTensor(self.taxo_vocabulary, misMatchText))
            for _ in range(num_sam_phy):
                misMatchTaxoLevel = np.random.choice(5, 1, replace=False, p=[0.14, 0.16, 0.18, 0.20, 0.32]) + 2
                misMatchText = RandomReturnNegTaxoSamePhy(self.tree, oriPhy, misMatchTaxoLevel[0], texts)
                if misMatchText is not None:
                    misMatchTensorList.append(ConvertTextToIndexTensor(self.taxo_vocabulary, misMatchText))
                else:
                    random.shuffle(mismatchPhylums)
                    startPhylum = mismatchPhylums[np.random.randint(self.num_phylum - 1)]
                    misMatchTaxoLevel = np.random.choice(6, 1, replace=False, p=[0.13, 0.14, 0.15, 0.16, 0.17, 0.25]) + 1
                    misMatchText = RandomlyReturnNegTaxoDiffPhy(self.tree, startPhylum, misMatchTaxoLevel[0], texts)
                    misMatchTensorList.append(ConvertTextToIndexTensor(self.taxo_vocabulary, misMatchText))
            matchTextTensor = ConvertTextToIndexTensor(self.taxo_vocabulary, matchText)
            textList = misMatchTensorList
            insertIndex = np.random.choice(self.num_mismatch + 1, 1, replace=False)
            textList.insert(insertIndex[0], matchTextTensor)
            # label
            pairLabel = torch.tensor(insertIndex[0], dtype=torch.int64)
        return ori_rev_tensor, feature_3Mer, feature_3Mer_rev_com, feature_4Mer, feature_4Mer_rev_com, torch.stack(textList, dim=0), pairLabel


    def __len__(self):
        return self.length
