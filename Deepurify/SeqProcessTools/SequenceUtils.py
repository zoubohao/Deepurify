import random
from typing import Dict, List, Tuple, TypeVar, Union

import numpy as np
import torch

Tensor = TypeVar("Tensor")  # The tensor of pytorch


def ConvertTextToIndexTensor(vocabulary: Dict, labelText: List) -> Tensor:
    labelTextLength = len(labelText)
    if labelTextLength > 6 or labelTextLength < 0:
        raise ValueError("The length of label text must smaller or equal with 6, since there are only 6 taxonomy level.")
    seq = []
    for word in labelText:
        if word in vocabulary:
            seq.append(vocabulary[word])
        else:
            raise ValueError("Word does not in the vocabulary.")
    if labelTextLength < 6:
        seq += [0 for _ in range(6 - labelTextLength)]
    seq = torch.from_numpy(np.array(seq, dtype=np.int64))
    return seq


index2Taxo = {1: "phylum", 2: "class", 3: "order", 4: "family", 5: "genus", 6: "species"}


def RandomlyReturnNegTaxoDiffPhy(taxoTree: Dict, startPhylum: str, stopLevel: int, truthText: str) -> List[str]:
    truthInfo = truthText.split("@")
    assert startPhylum != truthInfo[0], ValueError("Must with different phylum name.")
    res = []
    phys = taxoTree["Children"]
    signal = True
    startPhyObj = None
    for child in phys:
        if child["Name"] == startPhylum:
            startPhyObj = child
            signal = False
    if signal:
        raise ValueError("This phylum name is not in taxonomy tree.")
    if stopLevel < 1 or stopLevel > 6:
        raise ValueError("stop level error.")

    def inner(curTaxoTree):
        if isinstance(curTaxoTree, Dict):
            curLevel = curTaxoTree["TaxoLevel"]
            curChildren = curTaxoTree["Children"]
            nextIndex = np.random.randint(len(curChildren))
            res.append(curTaxoTree["Name"])
            if curLevel != index2Taxo[stopLevel]:
                inner(curChildren[nextIndex])
        else:
            res.append(curTaxoTree)

    inner(startPhyObj)
    return res


def RandomReturnNegTaxoSamePhy(taxoTree: Dict, startPhylum: str, stopLevel: int, truthText: str) -> Union[List[str], None]:
    truthInfo = truthText.split("@")
    assert startPhylum == truthInfo[0], ValueError("Must with same phylum name.")
    res = []
    phys = taxoTree["Children"]
    signal = True
    startPhyTree = None
    for child in phys:
        if child["Name"] == startPhylum:
            startPhyTree = child
            signal = False
    if signal:
        raise ValueError("This phylum name is not in taxonomy tree.")
    # This means you must select other phylum as neg sample since the current match text is just at phy level.
    if stopLevel <= 1 or stopLevel > 6:
        raise ValueError("stop level error.")

    def inner(curTaxoTree):
        if isinstance(curTaxoTree["Children"][0], Dict):
            nextLevel = curTaxoTree["Children"][0]["TaxoLevel"]
        else:
            nextLevel = "species"
        if nextLevel != index2Taxo[stopLevel]:
            curChildren = curTaxoTree["Children"]
            nextIndex = np.random.randint(len(curChildren))
            res.append(curTaxoTree["Name"])
            inner(curChildren[nextIndex])
        else:
            curChildren = curTaxoTree["Children"]
            newChildren = []
            for child in curChildren:
                name = child
                if isinstance(child, Dict):
                    name = child["Name"]
                if name != truthInfo[stopLevel - 1]:
                    newChildren.append(child)
            if not newChildren:
                return res.append(None)
            nextIndex = np.random.randint(len(newChildren))
            res.append(curTaxoTree["Name"])
            if isinstance(newChildren[nextIndex], str):
                res.append(newChildren[nextIndex])
            else:
                res.append(newChildren[nextIndex]["Name"])

    inner(startPhyTree)
    return None if res[-1] is None else res


def returnTaxoTextsInSameLevel(matchTextOuter: List, maxNum: int, taxoTree: Dict):
    results = []

    def inner(matchTextInner, taxoTree):
        children = taxoTree["Children"]
        if len(matchTextInner) == 1:
            for child in children:
                if isinstance(child, Dict):
                    if child["Name"] != matchTextInner[-1]:
                        results.append(matchTextOuter[:-1] + [child["Name"]])
                else:
                    if child != matchTextInner[-1]:
                        results.append(matchTextOuter[:-1] + [child])
        else:
            for child in children:
                if child["Name"] == matchTextInner[0]:
                    inner(matchTextInner[1:], child)

    inner(matchTextOuter, taxoTree)
    random.shuffle(results)
    return results[:maxNum]


# Padding char "X"
nt2index = {"X": 0, "N": 1, "A": 2, "T": 3, "C": 4, "G": 5, "R": 1, "Y": 1, "M": 1, "K": 1, "W": 1, "H": 1, "B": 1, "V": 1, "S": 1, "D": 1}
nt2nt = {"A": "T", "T": "A", "C": "G", "G": "C"}


def buildSeqFeatures(seq: str, vocab_3Mer: Dict, vocab_4Mer: Dict) -> Tuple[List[str], List[int], List[int], List[int], List[int]]:
    reverse_complement = []
    feature_3Mer = []
    feature_4Mer = []
    feature_3Mer_rev_com = []
    feature_4Mer_rev_com = []
    seqLen = len(seq)
    for i in range(seqLen):
        revIndex = seqLen - 1 - i
        if i + 3 <= seqLen:
            mer3 = seq[i: i + 3]
            if mer3 in vocab_3Mer:
                feature_3Mer.append(vocab_3Mer[mer3])
            else:
                feature_3Mer.append(vocab_3Mer["[UNK]"])
        if i + 4 <= seqLen:
            mer4 = seq[i: i + 4]
            if mer4 in vocab_4Mer:
                feature_4Mer.append(vocab_4Mer[mer4])
            else:
                feature_4Mer.append(vocab_4Mer["[UNK]"])
        if seq[revIndex] in nt2nt:
            reverse_complement.append(nt2nt[seq[revIndex]])
        else:
            reverse_complement.append("N")
        if i >= 3:
            rev_mer3 = "".join(reverse_complement[i - 3: i])
            if rev_mer3 in vocab_3Mer:
                feature_3Mer_rev_com.append(vocab_3Mer[rev_mer3])
            else:
                feature_3Mer_rev_com.append(vocab_3Mer["[UNK]"])
        if i >= 4:
            rev_mer4 = "".join(reverse_complement[i - 4: i])
            if rev_mer4 in vocab_4Mer:
                feature_4Mer_rev_com.append(vocab_4Mer[rev_mer4])
            else:
                feature_4Mer_rev_com.append(vocab_4Mer["[UNK]"])
    rev_mer3 = "".join(reverse_complement[i - 2: i + 1])
    if rev_mer3 in vocab_3Mer:
        feature_3Mer_rev_com.append(vocab_3Mer[rev_mer3])
    else:
        feature_3Mer_rev_com.append(vocab_3Mer["[UNK]"])
    rev_mer4 = "".join(reverse_complement[i - 3: i + 1])
    if rev_mer4 in vocab_4Mer:
        feature_4Mer_rev_com.append(vocab_4Mer[rev_mer4])
    else:
        feature_4Mer_rev_com.append(vocab_4Mer["[UNK]"])
    return reverse_complement, feature_3Mer, feature_3Mer_rev_com, feature_4Mer, feature_4Mer_rev_com


def ConvertSeqToImageTensorMoreFeatures(max_model_len: int, seq: str, vocab_3Mer: Dict, vocab_4Mer: Dict) -> Tensor:
    """
    This function requires the seq does not have padding char 'X'. The seq is the original seq.
    """
    # assert "X" not in seq, ValueError("'X' in the seq. ")
    seqLength = len(seq)
    assert seqLength <= max_model_len, "Your seq length is bigger than max_model_len."
    oriSeq = seq + "".join(["X" for _ in range(max_model_len - seqLength)])
    oriSeqIndex = torch.from_numpy(np.array(list(map(lambda x: nt2index[x], oriSeq)), dtype=np.int64)).view([max_model_len, 1])
    oriSeqTensor = torch.zeros([max_model_len, 6]).scatter_(dim=-1, index=oriSeqIndex, value=1.0).permute(1, 0).float()  # [6, max_model_len]
    # Other features
    reverse_complement, feature_3Mer, feature_3Mer_rev_com, feature_4Mer, feature_4Mer_rev_com = buildSeqFeatures(seq, vocab_3Mer, vocab_4Mer)
    reverse_complement = reverse_complement + ["X" for _ in range(max_model_len - seqLength)]
    rev_comp_index = torch.from_numpy(np.array(list(map(lambda x: nt2index[x], reverse_complement)), dtype=np.int64)).view([max_model_len, 1])
    rev_compTensor = torch.zeros([max_model_len, 6]).scatter_(dim=-1, index=rev_comp_index, value=1.0).permute(1, 0).float()  # [6, max_model_len]
    ###
    feature_3Mer += [0 for _ in range(max_model_len - len(feature_3Mer))]
    feature_3Mer = torch.from_numpy(np.array(feature_3Mer, dtype=np.int64))
    feature_4Mer += [0 for _ in range(max_model_len - len(feature_4Mer))]
    feature_4Mer = torch.from_numpy(np.array(feature_4Mer, dtype=np.int64))
    ###
    feature_3Mer_rev_com += [0 for _ in range(max_model_len - len(feature_3Mer_rev_com))]
    feature_3Mer_rev_com = torch.from_numpy(np.array(feature_3Mer_rev_com, dtype=np.int64))
    feature_4Mer_rev_com += [0 for _ in range(max_model_len - len(feature_4Mer_rev_com))]
    feature_4Mer_rev_com = torch.from_numpy(np.array(feature_4Mer_rev_com, dtype=np.int64))
    return torch.cat([oriSeqTensor, rev_compTensor], dim=0), feature_3Mer, feature_3Mer_rev_com, feature_4Mer, feature_4Mer_rev_com


nt2ntList = {"A": ["T", "C", "G"], "T": ["A", "C", "G"], "C": ["T", "A", "G"], "G": ["T", "C", "A"]}
nt = ["T", "C", "G", "A"]


def SeqSimulateSNV(seq: str, vRatio=0.05) -> str:
    newSeq = []
    for c in seq:
        if random.random() >= vRatio:
            newSeq.append(c)
        else:
            index = np.random.randint(0, 3, dtype=np.int64)
            if c in nt2ntList:
                newSeq.append(nt2ntList[c][index])
            else:
                index = np.random.randint(0, 4, dtype=np.int64)
                newSeq.append(nt[index])
    return "".join(newSeq)


def GenerateNoisySeq(g_len: int) -> str:
    index2nt = {0: "A", 1: "T", 2: "C", 3: "G"}
    intSeq = np.random.randint(0, 4, size=[g_len], dtype=np.int64)
    return "".join(map(lambda x: index2nt[x], intSeq))


def SeqCutToModelLengthIntervalAndAddNoisy(seq: str, min_model_len: int, max_model_len: int, gmmModel, if_noisy=True):
    assert min_model_len * 1.5 <= max_model_len, "The max length must bigger than min length 1.5 times."
    oriSeqLen = len(seq)
    # For seq length smaller than minimium length, zero probability
    if oriSeqLen < min_model_len:
        if if_noisy is False:
            return seq, 0
        noisySeq = GenerateNoisySeq(min_model_len - oriSeqLen)
        if np.random.rand() <= 0.5:
            return cutSeq + noisySeq, 1
        return noisySeq + cutSeq, 1
    # For seq length smaller than maximum length, zero probability
    if oriSeqLen <= max_model_len:
        curLength = oriSeqLen
        randN = np.random.rand()
        cutSeq = seq
        if if_noisy is False:
            return cutSeq, 0
        return _extracted_from_SeqCutToModelLengthIntervalAndAddNoisy_19(
            randN, cutSeq, curLength, max_model_len
        )
    # For seq length bigger than maximum length
    if np.random.rand() <= 0.4:  # 0.4 probability to have max_model_len
        startIndex = np.random.randint(0, oriSeqLen - max_model_len)
        cutSeq = seq[startIndex: startIndex + max_model_len]
        return cutSeq, 0
    if np.random.rand() <= 0.16666:  # 0.1 probability to have min_model_len
        return _extracted_from_SeqCutToModelLengthIntervalAndAddNoisy_34(
            oriSeqLen, min_model_len, seq, if_noisy
        )
    # 0.5 probability to have sampled length
    startIndex = np.random.randint(0, oriSeqLen - min_model_len)
    curLength = int(np.clip(gmmModel.sample(1)[0][0], min_model_len, max_model_len)[0])
    cutSeq = seq[startIndex: startIndex + curLength]
    if if_noisy is False:
        return cutSeq, 0
    randN = np.random.rand()
    return _extracted_from_SeqCutToModelLengthIntervalAndAddNoisy_19(
        randN, cutSeq, curLength, max_model_len
    )


# TODO Rename this here and in `SeqCutToModelLengthIntervalAndAddNoisy`
def _extracted_from_SeqCutToModelLengthIntervalAndAddNoisy_34(oriSeqLen, min_model_len, seq, if_noisy):
    startIndex = np.random.randint(0, oriSeqLen - min_model_len)
    cutSeq = seq[startIndex: startIndex + min_model_len]
    randN = np.random.rand()
    if if_noisy is False:
        return cutSeq, 0
    if randN <= 0.5:
        return cutSeq, 0
    noisySeqLen = int(min_model_len * (np.random.rand() * 0.25 + 0.25))
    return _extracted_from_SeqCutToModelLengthIntervalAndAddNoisy_42(
        noisySeqLen, cutSeq
    )


# TODO Rename this here and in `SeqCutToModelLengthIntervalAndAddNoisy`
def _extracted_from_SeqCutToModelLengthIntervalAndAddNoisy_42(noisySeqLen, cutSeq):
    noisySeq = GenerateNoisySeq(noisySeqLen)
    if np.random.rand() <= 0.5:
        return cutSeq + noisySeq, 1
    return noisySeq + cutSeq, 1


# TODO Rename this here and in `SeqCutToModelLengthIntervalAndAddNoisy`
def _extracted_from_SeqCutToModelLengthIntervalAndAddNoisy_19(randN, cutSeq, curLength, max_model_len):
    if randN <= 0.5:
        return cutSeq, 0
    noisySeqLen = int(curLength * (np.random.rand() * 0.2 + 0.2))
    if noisySeqLen + curLength > max_model_len:
        noisySeqLen = max_model_len - curLength
    return _extracted_from_SeqCutToModelLengthIntervalAndAddNoisy_42(
        noisySeqLen, cutSeq
    )


def maskSeq(
    ori_rev_tensor: Tensor, feature_3Mer: Tensor, feature_3Mer_rev_com: Tensor, feature_4Mer: Tensor, feature_4Mer_rev_com: Tensor, seq_len: int
):
    ### 50% mask, 50% unchange
    randN = np.random.rand()
    if randN <= 0.5:
        return ori_rev_tensor, feature_3Mer, feature_3Mer_rev_com, feature_4Mer, feature_4Mer_rev_com
    elif 0.5 < randN <= 0.75:
        maskRatio = np.random.rand() * 0.1
        maskLength = int(seq_len * maskRatio)
        unMaskLength = seq_len - maskLength
        preLen = np.random.randint(0, unMaskLength)
        afterLen = unMaskLength - preLen
        preTensor = torch.ones([preLen])
        maskTensor = torch.zeros([maskLength])
        afterTensor = torch.ones([afterLen])
        paddingTensor = torch.ones([feature_3Mer.shape[0] - seq_len])
        mask = torch.cat([preTensor, maskTensor, afterTensor, paddingTensor], dim=0).long()
        return ori_rev_tensor * mask.unsqueeze(0), feature_3Mer * mask, feature_3Mer_rev_com * mask, feature_4Mer * mask, feature_4Mer_rev_com * mask
    else:
        keepRatio = np.random.rand() * 0.1 + 0.9
        mask = torch.ones(size=[seq_len]).bernoulli_(p=keepRatio)
        paddingTensor = torch.ones([feature_3Mer.shape[0] - seq_len])
        mask = torch.cat([mask, paddingTensor], dim=0).long()
        return ori_rev_tensor * mask.unsqueeze(0), feature_3Mer * mask, feature_3Mer_rev_com * mask, feature_4Mer * mask, feature_4Mer_rev_com * mask


def maskAndPredict(
    seq: str,
    ori_rev_tensor: Tensor,
    feature_3Mer: Tensor,
    feature_3Mer_rev_com: Tensor,
    feature_4Mer: Tensor,
    feature_4Mer_rev_com: Tensor,
    seq_len: int,
    maskedWordsNum=40,
):
    generateArry = np.arange(start=0, stop=seq_len - 10, step=4)
    maskedWordIndices = torch.from_numpy(np.random.choice(generateArry, size=maskedWordsNum, replace=False)).long()
    maskList = [maskedWordIndices]
    maskList.extend(maskedWordIndices + i + 1 for i in range(3))
    maskedWordIndices = torch.stack(maskList, dim=-1).flatten()
    # print("mask words legnth", maskedWordIndices)

    orimask = torch.ones(size=[seq_len])
    orimask = torch.scatter(orimask, dim=0, index=maskedWordIndices, value=0)
    paddingTensor = torch.ones([feature_3Mer.shape[0] - seq_len])
    mask = torch.cat([orimask, paddingTensor], dim=0).long()

    oriSeqIndex = torch.from_numpy(np.array(list(map(lambda x: nt2index[x], seq)), dtype=np.int64))
    selectMask = 1.0 - orimask
    labels = torch.masked_select(oriSeqIndex, selectMask.bool())
    assert labels.shape[0] == maskedWordIndices.shape[0]
    selectMask = torch.cat([selectMask, torch.zeros([feature_3Mer.shape[0] - seq_len])], dim=0).long()
    return (
        ori_rev_tensor * mask.unsqueeze(0),
        feature_3Mer * mask,
        feature_3Mer_rev_com * mask,
        feature_4Mer * mask,
        feature_4Mer_rev_com * mask,
        labels,
        selectMask,
    )
