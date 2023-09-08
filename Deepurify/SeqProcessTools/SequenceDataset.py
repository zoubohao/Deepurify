import contextlib
import os
import pickle
import random
from copy import deepcopy
from math import floor
from typing import Dict

import numpy as np
import torch
from Deepurify.IOUtils import loadTaxonomyTree, readFile
from Deepurify.SeqProcessTools.SequenceUtils import (ConvertSeqToImageTensorMoreFeatures, ConvertTextToIndexTensor,
    RandomlyReturnNegTaxoDiffPhy, RandomReturnNegTaxoSamePhy,
    SeqCutToModelLengthIntervalAndAddNoisy, maskAndPredict, maskSeq,
    returnTaxoTextsInSameLevel)
from torch.utils.data import Dataset


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
        maskedLength=100,
    ):
        self.finetune = finetune
        self.max_model_len = max_model_len
        self.min_model_len = min_model_len
        self.taxo_vocabulary = taxo_vocabulary
        self.vocab_3Mer = vocab_3Mer
        self.vocab_4Mer = vocab_4Mer
        self.file_path = file_path
        self.geneLen = maskedLength
        k = 0
        for file in os.listdir(file_path):
            try:
                int(os.path.splitext(file)[0])
                k += 1
            except:
                pass
        self.length = k
        self.num_mismatch = misMatchNum
        self.spName2weight = {}
        self.phylums = set()
        self.tree = loadTaxonomyTree(taxomonyTreePath)
        self.phy2Index = {}  # 0 is for out of distribution data
        for k, child in enumerate(self.tree["Children"]):
            self.phylums.add(child["Name"])
            self.phy2Index[child["Name"]] = k
        self.num_phylum = len(self.phylums)
        with open(sampleName2weightPath, "r") as rh:
            for line in rh:
                split_info = line.strip("\n").split("\t")
                self.spName2weight[split_info[0]] = float(split_info[1])
        with open(gmmModelPath, "rb") as rb:
            self.gmmModel = pickle.load(rb)

    def __getitem__(self, index):
        with torch.no_grad():
            # data variable is a tuple, the first position is the sequence and the second position is the taxonomy label of this sequence
            data = readFile(os.path.join(self.file_path, f"{str(index)}.txt"))
            seq = data[0]
            # add noisy
            seq, if_noisy = SeqCutToModelLengthIntervalAndAddNoisy(
                seq, min_model_len=self.min_model_len, max_model_len=self.max_model_len, gmmModel=self.gmmModel, if_noisy=not self.finetune
            )
            # convert to tensor
            ori_rev_tensor, feature_3Mer, feature_3Mer_rev_com, feature_4Mer, feature_4Mer_rev_com = ConvertSeqToImageTensorMoreFeatures(
                self.max_model_len, seq, self.vocab_3Mer, self.vocab_4Mer
            )
            # mask tokens
            ori_rev_tensor, feature_3Mer, feature_3Mer_rev_com, feature_4Mer, feature_4Mer_rev_com, generateLabel, selectMask = maskAndPredict(
                seq, ori_rev_tensor, feature_3Mer, feature_3Mer_rev_com, feature_4Mer, feature_4Mer_rev_com, len(seq), self.geneLen
            )
            if self.finetune is False:
                ori_rev_tensor, feature_3Mer, feature_3Mer_rev_com, feature_4Mer, feature_4Mer_rev_com = maskSeq(
                    ori_rev_tensor, feature_3Mer, feature_3Mer_rev_com, feature_4Mer, feature_4Mer_rev_com, len(seq)
                )
            # generateing negative labels
            matchTaxoLevel = np.random.choice(6, 1, replace=False, p=[0.14, 0.15, 0.16, 0.17, 0.18, 0.2]) + 1
            texts = data[1]
            matchText = texts.split("@")[: matchTaxoLevel[0]]
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
                misMatchTaxoLevel = np.random.choice(6, 1, replace=False, p=[0.14, 0.15, 0.16, 0.17, 0.18, 0.2]) + 1
                misMatchText = RandomlyReturnNegTaxoDiffPhy(self.tree, startPhylum, misMatchTaxoLevel[0], texts)
                misMatchTensorList.append(ConvertTextToIndexTensor(self.taxo_vocabulary, misMatchText))
            for _ in range(num_sam_phy):
                misMatchTaxoLevel = np.random.choice(5, 1, replace=False, p=[0.16, 0.18, 0.20, 0.22, 0.24]) + 2
                misMatchText = RandomReturnNegTaxoSamePhy(self.tree, oriPhy, misMatchTaxoLevel[0], texts)
                if misMatchText is None:
                    random.shuffle(mismatchPhylums)
                    startPhylum = mismatchPhylums[np.random.randint(self.num_phylum - 1)]
                    misMatchTaxoLevel = np.random.choice(6, 1, replace=False, p=[0.14, 0.15, 0.16, 0.17, 0.18, 0.2]) + 1
                    misMatchText = RandomlyReturnNegTaxoDiffPhy(self.tree, startPhylum, misMatchTaxoLevel[0], texts)
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
            # Constrain for label, the phylum tensor will be set as a anchor.
            oriPhyTensor = ConvertTextToIndexTensor(self.taxo_vocabulary, [oriPhy])
            lowMatchTaxoLevel = np.random.choice(5, 1, replace=False, p=[0.16, 0.18, 0.20, 0.22, 0.24]) + 2
            lowMatchText = texts.split("@")[: lowMatchTaxoLevel[0]]
            lowMatchTextTensor = ConvertTextToIndexTensor(self.taxo_vocabulary, lowMatchText)
            # outer text tensor.
            random.shuffle(mismatchPhylums)
            startPhylum = mismatchPhylums[np.random.randint(self.num_phylum - 1)]
            misMatchTaxoLevel = np.random.choice(6, 1, replace=False, p=[0.14, 0.15, 0.16, 0.17, 0.18, 0.2]) + 1
            misMatchText = RandomlyReturnNegTaxoDiffPhy(self.tree, startPhylum, misMatchTaxoLevel[0], texts)
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
                generateLabel,
                selectMask,
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
            with contextlib.suppress(Exception):
                int(os.path.splitext(file)[0])
                k += 1
        self.length = k
        self.num_mismatch = misMatchNum
        self.phylums = set()
        self.tree = loadTaxonomyTree(taxomonyTreePath)
        self.phy2Index = {}
        for k, child in enumerate(self.tree["Children"]):
            self.phylums.add(child["Name"])
            self.phy2Index[child["Name"]] = k
        self.num_phylum = len(self.phylums)
        with open(gmmModelPath, "rb") as rb:
            self.gmmModel = pickle.load(rb)

    def __getitem__(self, index):
        with torch.no_grad():
            data = readFile(os.path.join(self.file_path, f"{str(index)}.txt"))
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
            matchText = texts.split("@")[:matchTaxoLevel[0]]
            sameLevelMisMatches = returnTaxoTextsInSameLevel(matchText, self.num_mismatch // 2, self.tree)
            misMatchTensorList = [
                ConvertTextToIndexTensor(self.taxo_vocabulary, misMatchText)
                for misMatchText in sameLevelMisMatches
            ]
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
                if misMatchText is None:
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
