import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Callable, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from Deepurify.IOUtils import (getNumberOfPhylum, loadTaxonomyTree, readFasta,
                               readPickle, readVocabulary, writeAnnotResult,
                               writePickle)
from Deepurify.Model.EncoderModels import DeepurifyModel
from Deepurify.SeqProcessTools.SequenceUtils import (
    ConvertSeqToImageTensorMoreFeatures, ConvertTextToIndexTensor)


def gatherValues(v1, t2, num_labels):
    """
    t1 = torch.randn([3, 4])
    t2 = torch.randn([3, 5, 4])
    print(gatherValues(t1, t2, 5))
    t2v = t2.view(15, 4)
    print(t1 @ t2v.T)
    """
    b1 = v1.size(0)
    b2 = t2.size(0)
    assert b1 == b2, ValueError("Batch size is not equal.")
    dotTensor = torch.tensordot(v1, t2, dims=([1], [2])).permute([0, 2, 1])  # [b1, num_labels, b2]
    index = torch.arange(b1).expand([num_labels, b1]).transpose(1, 0).unsqueeze(-1).to(dotTensor.device)
    return torch.gather(dotTensor, dim=-1, index=index).squeeze(-1)


def buildTextsRepNormVector(taxo_tree: Dict, model: nn.Module, vocabulary: Dict[str, int], device: str, outputPath: str) -> Dict:
    text2repV = {}

    def inner(cur_taxo_tree: Dict, cur_text: str):
        if cur_taxo_tree["TaxoLevel"] != "genus":
            for child in cur_taxo_tree["Children"]:
                this_name = child["Name"]
                if not cur_text:
                    textTensor = ConvertTextToIndexTensor(vocabulary, [this_name]).unsqueeze(0).unsqueeze(0).to(device)
                    text2repV[this_name] = model.textRepNorm(textTensor).detach().cpu().squeeze()
                    inner(child, f"{this_name}@")
                else:
                    preNames = cur_text.split("@")[:-1]
                    textTensor = ConvertTextToIndexTensor(vocabulary, preNames + [this_name]).unsqueeze(0).unsqueeze(0).to(device)
                    text2repV[cur_text + this_name] = model.textRepNorm(textTensor).detach().cpu().squeeze()
                    inner(child, cur_text + this_name + "@")
        else:
            for this_name in cur_taxo_tree["Children"]:
                preNames = cur_text.split("@")[:-1]
                textTensor = ConvertTextToIndexTensor(vocabulary, preNames + [this_name]).unsqueeze(0).unsqueeze(0).to(device)
                text2repV[cur_text + this_name] = model.textRepNorm(textTensor).detach().cpu().squeeze()

    inner(taxo_tree, "")
    if os.path.exists(outputPath) is False:
        writePickle(outputPath, text2repV)
    return text2repV


def taxonomyLabelGreedySearch(
    visRepVector: torch.Tensor,
    cur_anntotated: str,
    taxo_tree: Dict,
    annotated_level: int,
    text2repNormVector: Dict,
    logitNum: torch.Tensor,
    curMaxList: List,
    phy_fc: Union[None, nn.Module],
):
    """
    visRepVector: one dimension sequence reprentation, shape: [dimension]
    cur_annotated: current annotated string. Taxonomy levels are split by '@'
    taxo_tree: The taxonomy tree, which is a map that likes a json structure.
    annotated_level: What level you want to stop. The range of this is 1 ~ 6. 1 indicates phylum and 6 indicates species.
    """
    index2Taxo = {1: "phylum", 2: "class", 3: "order", 4: "family", 5: "genus", 6: "species"}
    visRepVector = visRepVector.unsqueeze(0)
    visRepNorm = visRepVector / visRepVector.norm(dim=-1, keepdim=True)
    curstackedTextsTensorList = []
    curTextsNames = []
    curNextChild = []
    if not cur_anntotated:
        for child in taxo_tree["Children"]:
            curstackedTextsTensorList.append(text2repNormVector[child["Name"]])
            curTextsNames.append(child["Name"])
            curNextChild.append(child)
        textNorm = torch.stack(curstackedTextsTensorList, dim=0).unsqueeze(0)
        innerSFT = torch.softmax(gatherValues(visRepNorm, textNorm, len(curstackedTextsTensorList)).squeeze(0) * logitNum, dim=-1)
        if len(innerSFT.shape) == 2:
            innerSFT = innerSFT.squeeze(0)
        innerMaxIndex = innerSFT.argmax()
        phySFT = torch.softmax(phy_fc(visRepNorm), dim=-1)
        if len(phySFT.shape) == 2:
            phySFT = phySFT.squeeze(0)
        phyMaxIndex = phySFT.argmax()
        if (
            innerMaxIndex != phyMaxIndex
            and innerSFT[innerMaxIndex] > phySFT[phyMaxIndex]
            or innerMaxIndex == phyMaxIndex
        ):
            maxIndex = innerMaxIndex
            curMaxList.append(innerSFT[maxIndex].item())
        else:
            maxIndex = phyMaxIndex
            curMaxList.append(phySFT[maxIndex].item())
        annotatedRes = curTextsNames[maxIndex]
    else:
        for child in taxo_tree["Children"]:
            if isinstance(child, str):
                curstackedTextsTensorList.append(text2repNormVector[cur_anntotated + child])
                curTextsNames.append(cur_anntotated + child)
            else:
                curstackedTextsTensorList.append(text2repNormVector[cur_anntotated + child["Name"]])
                curTextsNames.append(cur_anntotated + child["Name"])
                curNextChild.append(child)
        textNorm = torch.stack(curstackedTextsTensorList, dim=0).unsqueeze(0)
        innerSFT = torch.softmax(gatherValues(visRepNorm, textNorm, len(curstackedTextsTensorList)).squeeze(0) * logitNum, dim=-1)
        if len(innerSFT.shape) == 2:
            innerSFT = innerSFT.squeeze(0)
        maxIndex = innerSFT.argmax()
        curMaxList.append(innerSFT[maxIndex].item())
        annotatedRes = curTextsNames[maxIndex]
        if not curNextChild:
            return annotatedRes
    next_taxo_tree = curNextChild[maxIndex]
    if next_taxo_tree["TaxoLevel"] == index2Taxo[annotated_level]:
        return annotatedRes
    visRepVector = visRepVector.squeeze(0)
    return taxonomyLabelGreedySearch(
        visRepVector,
        f"{annotatedRes}@",
        next_taxo_tree,
        annotated_level,
        text2repNormVector,
        logitNum,
        curMaxList,
        None,
    )


def taxonomyLabelTopkSearch(
    result: List,
    visRepVector: torch.Tensor,
    cur_anntotated: str,
    taxo_tree: Dict,
    text2repNormVector: Dict,
    logitNum: torch.Tensor,
    phy_fc: Union[None, nn.Module],
    topK: int,
    probs: List[float],
    value=0.0,
    curLevel=0,
):
    """
    visRepVector: one dimension sequence reprentation, shape: [dimension]
    cur_annotated: current annotated string. Taxonomy levels are split by '@'
    taxo_tree: The taxonomy tree, which is a map that likes a json structure.
    """
    assert topK >= 2, "topK must bigger than 2."
    bouns = [1.6, 1.5, 1.4, 1.3, 1.2, 1.0]
    visRepVector = visRepVector.unsqueeze(0)
    visRepNorm = visRepVector / visRepVector.norm(dim=-1, keepdim=True)
    curstackedTextsTensorList = []
    curTextsNames = []
    curNextChild = []
    nextPairs = []
    if not cur_anntotated:
        for child in taxo_tree["Children"]:
            curstackedTextsTensorList.append(text2repNormVector[child["Name"]])
            curTextsNames.append(child["Name"])
            curNextChild.append(child)
        textNorm = torch.stack(curstackedTextsTensorList, dim=0).unsqueeze(0)
        innerSFT = torch.softmax(gatherValues(visRepNorm, textNorm, len(curstackedTextsTensorList)).squeeze(0) * logitNum, dim=-1)
        if len(innerSFT.shape) == 2:
            innerSFT = innerSFT.squeeze(0)
        innerMaxIndex = innerSFT.argmax()
        phySFT = torch.softmax(phy_fc(visRepNorm), dim=-1)
        if len(phySFT.shape) == 2:
            phySFT = phySFT.squeeze(0)
        phyMaxIndex = phySFT.argmax()
        if innerMaxIndex != phyMaxIndex:
            curProbI = innerSFT[innerMaxIndex].item()
            nextPairs.append((innerMaxIndex, curTextsNames[innerMaxIndex], curProbI, curProbI))
            curProbP = phySFT[phyMaxIndex].item()
            nextPairs.append((phyMaxIndex, curTextsNames[phyMaxIndex], curProbP, curProbP))
        else:
            curProb = innerSFT[innerMaxIndex].item()
            nextPairs.append((innerMaxIndex, curTextsNames[innerMaxIndex], curProb, curProb))
        for pair in nextPairs:
            next_taxo_tree = curNextChild[pair[0]]
            annotatedRes = pair[1]
            curProbs = [pair[2]]
            v = pair[3] * bouns[curLevel]
            visRepVector = visRepVector.squeeze(0)
            taxonomyLabelTopkSearch(
                result,
                visRepVector,
                f"{annotatedRes}@",
                next_taxo_tree,
                text2repNormVector,
                logitNum,
                None,
                topK,
                curProbs,
                v,
                curLevel + 1,
            )
    else:
        for child in taxo_tree["Children"]:
            if isinstance(child, str):
                curstackedTextsTensorList.append(text2repNormVector[cur_anntotated + child])
                curTextsNames.append(cur_anntotated + child)
            else:
                curstackedTextsTensorList.append(text2repNormVector[cur_anntotated + child["Name"]])
                curTextsNames.append(cur_anntotated + child["Name"])
                curNextChild.append(child)
        textNorm = torch.stack(curstackedTextsTensorList, dim=0).unsqueeze(0)
        innerSFT = torch.softmax(gatherValues(visRepNorm, textNorm, len(curstackedTextsTensorList)).squeeze(0) * logitNum, dim=-1)
        thisTopK = topK - 1 if curLevel <= 2 else topK
        if len(innerSFT.shape) == 2:
            innerSFT = innerSFT.squeeze(0)
        if innerSFT.shape[-1] >= thisTopK:
            topValues, topIndices = torch.topk(innerSFT, thisTopK, dim=-1)
        else:
            topValues, topIndices = torch.topk(innerSFT, innerSFT.shape[-1], dim=-1)
        bestProb = topValues[0].item()
        addScore = [0.9, 0.4, 0.2, 0.1, 0.05, 0.025]
        for i, values in enumerate(topValues):
            curIndex = topIndices[i].item()
            curProb = values.item()
            curAnnotatedRes = curTextsNames[curIndex]
            curProbs = deepcopy(probs) + [curProb]
            if curProb >= 1.0:
                curProb = 0.5
            if i == 0:
                nextPairs.append((curIndex, curAnnotatedRes, curProbs, value * curProb * 1.3))
            elif abs(bestProb - curProb) <= 0.2 and curProb > addScore[curLevel]:
                nextPairs.append((curIndex, curAnnotatedRes, curProbs, value * curProb))
        for pair in nextPairs:
            if curNextChild:
                next_taxo_tree = curNextChild[pair[0]]
                annotatedRes = pair[1]
                curProbs = pair[2]
                v = pair[3] * bouns[curLevel]
                visRepVector = visRepVector.squeeze(0)
                taxonomyLabelTopkSearch(
                    result,
                    visRepVector,
                    f"{annotatedRes}@",
                    next_taxo_tree,
                    text2repNormVector,
                    logitNum,
                    None,
                    topK,
                    curProbs,
                    v,
                    curLevel + 1,
                )
            else:
                result.append((pair[1], pair[2], pair[3], text2repNormVector[pair[1]]))


def splitLongContig(name2seq: Dict[str, str], max_model_len: int, min_model_len: int, overlappingRatio=0.5):
    newName2seq = {}
    for name, seq in name2seq.items():
        seqLen = len(seq)
        if seqLen > max_model_len:
            start = 0
            k = 0
            while start + max_model_len <= seqLen:
                newName2seq[f"{name}___{str(k)}"] = seq[start: start + max_model_len]
                start += int(max_model_len * (1.0 - overlappingRatio))
                k += 1
            newName2seq[f"{name}___{str(k)}"] = seq[start:]
        else:
            newName2seq[name] = seq
    return newName2seq


def getBestLabelAndProbs(
    annoteList: List[List[str]], maxProbList: List[List[float]], length: List[int], results: List[str], probs: List[float], level=0
):
    assert len(annoteList) == len(maxProbList) == len(length), "The length of those parameters are not equal with each other."
    countsProbWeightList = [1.0, 1.04, 1.14, 1.2475, 1.37, 1.72]
    lengthProbWeightList = [1.0, 1.03, 1.1, 1.23, 1.34, 1.62]
    curLevelStrs = []
    curMaxProbs = []
    curLength = []
    for i, curAnnote in enumerate(annoteList):
        curLevelStrs.append(curAnnote[0])
        curMaxProbs.append(maxProbList[i][0])
        curLength.append(length[i])
    values, counts = np.unique(curLevelStrs, return_counts=True)
    nd = np.argmax(counts)
    # This means there is only 1 element in the list.
    if len(counts) == 1:
        results.append(values[0])
        probs.append(sum(curMaxProbs) / len(curMaxProbs) + 0.0)
    elif counts[nd] == 1:  # This means the max count of those elements is 1.
        lnd = np.argmax(curLength)
        secondLongest = sorted(curLength)[-2]
        longest = curLength[lnd]
        snd = None
        for i, ele in enumerate(curLength):
            if ele == secondLongest:
                snd = i
        lProb = curMaxProbs[lnd]
        sProb = curMaxProbs[snd]
        r1 = longest / (longest + secondLongest) + 0.0
        r2 = secondLongest / (longest + secondLongest) + 0.0
        if (lProb + r1 * lengthProbWeightList[level]) >= (sProb + r2 * lengthProbWeightList[level]):
            results.append(curLevelStrs[lnd])
            probs.append(curMaxProbs[lnd])
        else:
            results.append(curLevelStrs[snd])
            probs.append(curMaxProbs[snd])
    else:  # For multi elements
        countsProb = counts / sum(counts)
        secondLagestProb = sorted(countsProb)[-2]
        snd = None
        for i, ele in enumerate(countsProb):
            if ele == secondLagestProb:
                snd = i
        sumV1 = 0
        k1 = 0
        sumLen1 = 0
        sumV2 = 0
        k2 = 0
        sumLen2 = 0
        for i, st in enumerate(curLevelStrs):
            if st == values[nd]:
                sumV1 += curMaxProbs[i]
                sumLen1 += curLength[i]
                k1 += 1
            if st == values[snd]:
                sumV2 += curMaxProbs[i]
                sumLen2 += curLength[i]
                k2 += 1
        meanV1 = sumV1 / k1 + 0.0
        meanV2 = sumV2 / k2 + 0.0
        summLen = sumLen1 + sumLen2 + 0.0
        score1 = meanV1 + countsProb[nd] * countsProbWeightList[level] + (sumLen1 / summLen + 0.0) * lengthProbWeightList[level]
        score2 = meanV2 + countsProb[snd] * countsProbWeightList[level] + (sumLen2 / summLen + 0.0) * lengthProbWeightList[level]
        if score1 >= score2:
            results.append(values[nd])
            probs.append(meanV1)
        else:
            results.append(values[snd])
            probs.append(meanV2)
    newNextAnnot = []
    newNextMaxProb = []
    newLength = []
    for i, curAnnot in enumerate(annoteList):
        if curAnnot[0] == results[-1] and len(curAnnot[1:]) != 0:
            newNextAnnot.append(curAnnot[1:])
            newNextMaxProb.append(maxProbList[i][1:])
            newLength.append(length[i])
    if len(newNextAnnot) != 0:
        return getBestLabelAndProbs(newNextAnnot, newNextMaxProb, newLength, results, probs, level + 1)
    else:
        return results, probs


class BTree:
    def __init__(self, value) -> None:
        self.value = value
        self.left = None
        self.right = None

    def insertLeft(self, value):
        self.left = BTree(value)
        return self.left

    def insertRight(self, value):
        self.right = BTree(value)
        return self.right


def getBestMultiLabelsAndProbs(annoteList: List[List[str]], maxProbList: List[List[float]], length: List[int], results: BTree, probs: BTree, level=0):
    assert len(annoteList) == len(maxProbList) == len(length), "The length of those parameters are not equal with each other."
    countsProbWeightList = [1.0, 1.04, 1.14, 1.2475, 1.37, 1.72]
    lengthProbWeightList = [1.0, 1.196, 1.302, 1.3655, 1.4832, 1.88]
    curLevelStrs = []
    curMaxProbs = []
    curLength = []
    curInfoPair = []
    for i, curAnnote in enumerate(annoteList):
        curLevelStrs.append(curAnnote[0])
        curMaxProbs.append(maxProbList[i][0])
        curLength.append(length[i])
        curInfoPair.append((curAnnote[0], maxProbList[i][0], length[i]))
    values, counts = np.unique(curLevelStrs, return_counts=True)
    nd = np.argmax(counts)
    leftNode = None
    leftProb = None
    rightNode = None
    rightProb = None
    # This means there is only 1 element in the list.
    if len(counts) == 1:
        leftNode = results.insertLeft(values[0])
        leftProb = probs.insertLeft(sum(curMaxProbs) / len(curMaxProbs) + 0.0)
    elif counts[nd] == 1:  # This means the max count of those elements is 1.
        sortedCurInfoPair = list(sorted(curInfoPair, key=lambda x: x[-1], reverse=True))
        for i, pair in enumerate(sortedCurInfoPair):
            if i == 0:
                leftNode = results.insertLeft(pair[0])
                leftProb = probs.insertLeft(pair[1])
            elif i == 1:
                rightNode = results.insertRight(pair[0])
                rightProb = probs.insertRight(pair[1])
            else:
                break
    else:  # For multi elements
        countsProb = counts / sum(counts)
        secondLagestProb = sorted(countsProb)[-2]
        snd = None
        for i, ele in enumerate(countsProb):
            if ele == secondLagestProb:
                snd = i
        sumV1 = 0
        k1 = 0
        sumLen1 = 0
        sumV2 = 0
        k2 = 0
        sumLen2 = 0
        for i, st in enumerate(curLevelStrs):
            if st == values[nd]:
                sumV1 += curMaxProbs[i]
                sumLen1 += curLength[i]
                k1 += 1
            if st == values[snd]:
                sumV2 += curMaxProbs[i]
                sumLen2 += curLength[i]
                k2 += 1
        meanV1 = sumV1 / k1 + 0.0
        meanV2 = sumV2 / k2 + 0.0
        summLen = sumLen1 + sumLen2 + 0.0
        score1 = meanV1 + countsProb[nd] * countsProbWeightList[level] + (sumLen1 / summLen + 0.0) * lengthProbWeightList[level]
        score2 = meanV2 + countsProb[snd] * countsProbWeightList[level] + (sumLen2 / summLen + 0.0) * lengthProbWeightList[level]
        if score1 >= score2:
            leftNode = results.insertLeft(values[nd])
            leftProb = probs.insertLeft(meanV1)
        else:
            leftNode = results.insertLeft(values[snd])
            leftProb = probs.insertLeft(meanV2)
        level2abs = {0: 0.112, 1: 0.225, 2: 0.365, 3: 0.485, 4: 0.575, 5: 0.685}
        if abs(score1 - score2) < level2abs[level]:
            leftNode = results.insertLeft(values[nd])
            leftProb = probs.insertLeft(meanV1)
            rightNode = results.insertRight(values[snd])
            rightProb = probs.insertRight(meanV2)
    newNextAnnotLeft = []
    newNextMaxProbLeft = []
    newLengthLeft = []
    newNextAnnotRight = []
    newNextMaxProbRight = []
    newLengthRight = []
    for i, curAnnot in enumerate(annoteList):
        if len(curAnnot[1:]) != 0:
            if leftNode is not None and leftNode.value == curAnnot[0]:
                newNextAnnotLeft.append(curAnnot[1:])
                newNextMaxProbLeft.append(maxProbList[i][1:])
                newLengthLeft.append(length[i])
            if rightNode is not None and rightNode.value == curAnnot[0]:
                newNextAnnotRight.append(curAnnot[1:])
                newNextMaxProbRight.append(maxProbList[i][1:])
                newLengthRight.append(length[i])
    if len(newNextAnnotLeft) != 0 and leftNode is not None:
        getBestMultiLabelsAndProbs(newNextAnnotLeft, newNextMaxProbLeft, newLengthLeft, leftNode, leftProb, level + 1)
    if len(newNextAnnotRight) != 0 and rightNode is not None:
        getBestMultiLabelsAndProbs(newNextAnnotRight, newNextMaxProbRight, newLengthRight, rightNode, rightProb, level + 1)


def getBestMultiLabelsForFiltering(annoteList: List[List[str]], maxProbList: List[List[float]], length: List[int]) -> List[str]:
    def traverseTree(tree: BTree, previousValue: List, result: List):
        curStr = tree.value
        previousValue.append(curStr)
        if tree.left is not None:
            traverseTree(tree.left, deepcopy(previousValue), result)
        if tree.right is not None:
            traverseTree(tree.right, deepcopy(previousValue), result)
        if tree.left is None and tree.right is None:
            result.append(deepcopy(previousValue))

    results = BTree("Root")
    probs = BTree("Root")
    getBestMultiLabelsAndProbs(annoteList, maxProbList, length, results, probs)

    coresList = []
    traverseTree(results, [], coresList)

    return coresList


def reverseLabeledResult(name2annotatRes: Dict[str, str], name2maxList: Dict[str, List[float]], name2contigLen: Dict[str, int]):
    newRes = {}
    newMaxList = {}
    newLength = {}
    for name, res in name2annotatRes.items():
        if "___" not in name:
            newRes[name] = res
            newMaxList[name] = name2maxList[name]
            newLength[name] = name2contigLen[name]
        else:
            contigName, _ = name.split("___")
            if contigName not in newRes:
                newLength[contigName] = [name2contigLen[name]]
                newRes[contigName] = [res.split("@")]
                newMaxList[contigName] = [name2maxList[name]]
            else:
                newLength[contigName].append(name2contigLen[name])
                newRes[contigName].append(res.split("@"))
                newMaxList[contigName].append(name2maxList[name])
    for contigName, annoRes in newRes.items():
        if isinstance(annoRes, list):
            core, probs = getBestLabelAndProbs(annoRes, newMaxList[contigName], newLength[contigName], [], [])
            annot = "@".join(core)
            newRes[contigName] = annot
            newMaxList[contigName] = probs
    return newRes, newMaxList


def subProcessLabelGreedySearch(
    inputVectorList: List[torch.Tensor],
    names: List[str],
    taxo_tree: Dict,
    annotated_level: int,
    text2repNormVector: Dict[str, torch.Tensor],
    logitNum: torch.Tensor,
    phy_fc: nn.Module,
):
    name2res = {}
    for i, inputVector in enumerate(inputVectorList):
        curMaxList = []
        with torch.no_grad():
            anntotated_res = taxonomyLabelGreedySearch(
                inputVector, "", taxo_tree, annotated_level, text2repNormVector, logitNum, curMaxList, phy_fc
            )
        name2res[names[i]] = (anntotated_res, curMaxList)
    return name2res


def subProcessLabelTopkSearch(
    inputVectorList: List[torch.Tensor],
    names: List[str],
    taxo_tree: Dict,
    text2repNormVector: Dict[str, torch.Tensor],
    logitNum: torch.Tensor,
    phy_fc: nn.Module,
    topK: int,
):
    name2res = {}
    with torch.no_grad():
        for i, inputVector in enumerate(inputVectorList):
            result = []
            taxonomyLabelTopkSearch(result, inputVector, "", taxo_tree, text2repNormVector, logitNum, phy_fc, topK, None, 1.0)
            sortedRes = list(sorted(result, key=lambda x: x[2], reverse=True))
            n = len(sortedRes)
            k = n // 4 * 3
            if k == 0:
                k = n
            result = sortedRes[0:k]
            annotNames = []
            annotTextNormTensors = []
            annotProbs = []
            annotScore = []
            for pair in result:
                annotNames.append(pair[0])
                annotProbs.append(pair[1])
                annotScore.append(pair[2])
                annotTextNormTensors.append(pair[3])
            inputVector = inputVector.unsqueeze(0)
            visRepNorm = inputVector / inputVector.norm(dim=-1, keepdim=True)
            textNorm = torch.stack(annotTextNormTensors, dim=0).unsqueeze(0)
            innerSFT = torch.softmax(gatherValues(visRepNorm, textNorm, len(annotTextNormTensors)).squeeze(0) * logitNum, dim=-1)
            if len(innerSFT.shape) >= 2:
                innerSFT = innerSFT.squeeze(0)
            annotScore = np.array(annotScore, dtype=np.float32)
            annotScore = torch.softmax(torch.from_numpy(annotScore), dim=-1)
            innerSFT = innerSFT + annotScore
            innerMaxIndex = innerSFT.argmax()
            name2res[names[i]] = (annotNames[innerMaxIndex], annotProbs[innerMaxIndex])
    return name2res


def labelBinFastaFile(
    binFasta: Union[str, Dict],
    model,
    mer3_vocabulary,
    mer4_vocabulary,
    taxo_tree,
    text2repNormVector,
    logitNum,
    device,
    model_config=None,
    batch_size=1,
    annotated_level=6,
    num_cpu=6,
    overlapping_ratio=0.5,
    cutSeqLength=8192,
    th="",
    n="",
    binName="",
    topkORgreedy="topk",
    topK=3,
):
    pid = str(os.getpid())
    if isinstance(binFasta, str):
        name2seq = readFasta(binFasta)
    elif isinstance(binFasta, dict):
        name2seq = binFasta
    else:
        raise ValueError("binFasta is not a fasta file path and not is a dict that key is contig name value is seq.")
    # Split contig if longer than max_model_len,
    # Since we split the long contigs, than we need to reverse to the original
    name2seq = splitLongContig(name2seq, max_model_len=cutSeqLength, min_model_len=model_config["min_model_len"], overlappingRatio=overlapping_ratio)
    names = []
    visRepVectorList = []
    batchList = []
    nsL = len(name2seq)
    k = 0
    for i, (name, seq) in enumerate(name2seq.items()):
        with torch.no_grad():
            ori_rev_tensor, feature_3Mer, feature_3Mer_rev_com, feature_4Mer, feature_4Mer_rev_com = ConvertSeqToImageTensorMoreFeatures(
                model_config["max_model_len"], seq, mer3_vocabulary, mer4_vocabulary
            )
            ori_rev_tensor = ori_rev_tensor.to(device)  # [C, L]
            feature_3Mer = feature_3Mer.to(device)  # [L]
            feature_3Mer_rev_com = feature_3Mer_rev_com.to(device)  # [L]
            feature_4Mer = feature_4Mer.to(device)  # [L]
            feature_4Mer_rev_com = feature_4Mer_rev_com.to(device)  # [L]
            catedTensror = model.annotatedConcatTensors(ori_rev_tensor, feature_3Mer, feature_3Mer_rev_com, feature_4Mer, feature_4Mer_rev_com)
        names.append(name)
        batchList.append(catedTensror)
        if (i + 1) % batch_size == 0:
            if k % 5 == 0:
                statusStr = "    " + "PROCESSER {}, {:.4}% complete for {}. (Current / Total) --> ({} / {})".format(
                    pid, (i + 1.0) * 100.0 / nsL + 0.0, binName, th + 1, n
                )
                cn = len(statusStr)
                if cn < 150:
                    statusStr = statusStr + "".join([" " for _ in range(150 - cn)])
                statusStr += "\r"
                sys.stderr.write("%s\r" % statusStr)
                sys.stderr.flush()
            k += 1
            stacked = torch.stack(batchList, dim=0).to(device)
            with torch.no_grad():
                brepVectors = model.visionRep(stacked)
                for repVector in brepVectors.detach().cpu():
                    visRepVectorList.append(repVector)
            batchList = []
    if len(batchList) != 0:
        stacked = torch.stack(batchList, dim=0).to(device)
        with torch.no_grad():
            brepVectors = model.visionRep(stacked)
            for repVector in brepVectors.detach().cpu():
                visRepVectorList.append(repVector)
    assert len(names) == len(visRepVectorList), "The length is not equal with each other."
    name2Labeled = {}
    name2maxList = {}
    name2contigLen = {}
    for i, name in enumerate(names):
        name2contigLen[name] = len(name2seq[name])
    phy_fc = model.phy_fc.to("cpu")
    processList = []
    step = len(names) // num_cpu + 1
    with ThreadPoolExecutor(max_workers=num_cpu) as t:
        for i in range(num_cpu):
            if topkORgreedy.lower() == "greedy":
                p = t.submit(
                    subProcessLabelGreedySearch,
                    visRepVectorList[step * i: step * (i + 1)],
                    names[step * i: step * (i + 1)],
                    taxo_tree,
                    annotated_level,
                    text2repNormVector,
                    logitNum,
                    phy_fc,
                )
            elif topkORgreedy.lower() == "topk":
                p = t.submit(
                    subProcessLabelTopkSearch,
                    visRepVectorList[step * i: step * (i + 1)],
                    names[step * i: step * (i + 1)],
                    taxo_tree,
                    text2repNormVector,
                    logitNum,
                    phy_fc,
                    topK,
                )
            else:
                raise ValueError("No Implement Other Searching Algorithms Besides Top-K or Greedy Search.")
            processList.append(p)
        for async_res in as_completed(processList):
            name2res = async_res.result()
            for name, data in name2res.items():
                name2Labeled[name] = data[0]
                name2maxList[name] = data[1]
    # Reverse to original
    return reverseLabeledResult(name2Labeled, name2maxList, name2contigLen)


def labelONEBinAndWrite(
    inputPath: str,
    outputFolder: str,
    model,
    mer3_vocabulary,
    mer4_vocabulary,
    taxo_tree,
    text2repNormVector,
    logitNum,
    device,
    model_config=None,
    batch_size=6,
    annotated_level=6,
    num_cpu=6,
    overlapping_ratio=0.5,
    cutSeqLength=8192,
    th="",
    n="",
    binName="",
    topkORgreedy="topk",
    topK=3,
):
    name2annotated, name2maxList = labelBinFastaFile(
        inputPath,
        model,
        mer3_vocabulary,
        mer4_vocabulary,
        taxo_tree,
        text2repNormVector,
        logitNum,
        device,
        model_config,
        batch_size=batch_size,
        annotated_level=annotated_level,
        num_cpu=num_cpu,
        overlapping_ratio=overlapping_ratio,
        cutSeqLength=cutSeqLength,
        th=th,
        n=n,
        binName=binName,
        topkORgreedy=topkORgreedy,
        topK=topK,
    )
    outputPath = os.path.join(outputFolder, binName + ".txt")
    writeAnnotResult(outputPath, name2annotated, name2maxList)


def labelBinsFolder(
    inputBinFolder: str,
    outputFolder: str,
    device: str,
    modelWeightPath: str,
    mer3Path: str,
    mer4Path: str,
    taxoVocabPath: str,
    taxoTreePath: str,
    taxoName2RepNormVecPath: str,
    batch_size=6,
    annotated_level=6,
    bin_suffix="fasta",
    filesList=None,
    num_cpu=6,
    overlapping_ratio=0.5,
    cutSeqLength=8192,
    topkORgreedy="topk",
    topK=3,
    error_queue=None,
    model_config=None,
):
    try:
        assert bin_suffix.lower() != "txt" or bin_suffix.lower() != ".txt"
        files = os.listdir(inputBinFolder)
        if filesList is not None:
            files = filesList
        num_files = len(files)
        taxo_tree = loadTaxonomyTree(taxoTreePath)
        taxo_vocabulary = readVocabulary(taxoVocabPath)
        mer3_vocabulary = readVocabulary(mer3Path)
        mer4_vocabulary = readVocabulary(mer4Path)
        if model_config is None:
            model_config = {
                "min_model_len": 1000,
                "max_model_len": 1024 * 8,
                "inChannel": 108,
                "expand": 1.5,
                "IRB_num": 3,
                "head_num": 6,
                "d_model": 738,
                "num_GeqEncoder": 7,
                "num_lstm_layers": 5,
                "feature_dim": 1024,
            }
        model = DeepurifyModel(
            max_model_len=model_config["max_model_len"],
            in_channels=model_config["inChannel"],
            taxo_dict_size=len(taxo_vocabulary),
            vocab_3Mer_size=len(mer3_vocabulary),
            vocab_4Mer_size=len(mer4_vocabulary),
            phylum_num=getNumberOfPhylum(taxo_tree),
            head_num=model_config["head_num"],
            d_model=model_config["d_model"],
            num_GeqEncoder=model_config["num_GeqEncoder"],
            num_lstm_layer=model_config["num_lstm_layers"],
            IRB_layers=model_config["IRB_num"],
            expand=model_config["expand"],
            feature_dim=model_config["feature_dim"],
            drop_connect_ratio=0.0,
            dropout=0.0,
        )
        model.to(device)
        ########### IMPORT ##########
        state = torch.load(modelWeightPath, map_location=torch.device(device))
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.no_grad():
            text2repNormVector = readPickle(taxoName2RepNormVecPath)
            logitNum = model.logit_scale.exp().cpu()
        c = 0
        for i, file in enumerate(files):
            if os.path.splitext(file)[-1][1:] == bin_suffix:
                c += 1
                binName = os.path.splitext(file)[0]
                if os.path.exists(os.path.join(outputFolder, binName + ".txt")):
                    continue
                labelONEBinAndWrite(
                    os.path.join(inputBinFolder, file),
                    outputFolder,
                    model,
                    mer3_vocabulary,
                    mer4_vocabulary,
                    taxo_tree,
                    text2repNormVector,
                    logitNum,
                    device,
                    model_config,
                    batch_size,
                    annotated_level,
                    num_cpu,
                    overlapping_ratio,
                    cutSeqLength,
                    i,
                    num_files,
                    binName,
                    topkORgreedy,
                    topK,
                )
        error_queue.put(None)
        if c == 0:
            raise ValueError("Can not find any MAGs in this folder. Please check your bin_suffix or MAGs' PATH !!!!")
    except:
        traceback.print_exc()
        error_queue.put(1)
        sys.exit(1)
