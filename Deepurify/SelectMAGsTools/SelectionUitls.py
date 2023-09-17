# selection no duplicated and best bins
import os
from copy import deepcopy
from shutil import copy
from typing import Dict, List, Tuple

from Deepurify.IOUtils import readBinName2Annot, readCheckMResultAndStat, readPickle


index2Taxo = {1: "T1_filter", 2: "T2_filter", 3: "T3_filter", 4: "T4_filter", 5: "T5_filter", 6: "T6_filter"}

class Tree:
    def __init__(self, binName: str, annotName: str, qualityValues: Tuple[float, float, str], core: List[bool]) -> None:
        self.binName = binName
        self.annotName = annotName
        self.qualityValues = qualityValues
        self.core = core
        self.canSelect = True
        self.children = []
        self.pathBests = []

    def insert(self, binName: str, annotName: str, qualityValues: Tuple[float, float, str], core: List[bool]):
        node = Tree(binName, annotName, qualityValues, core)
        self.children.append(node)
        return node


def getScore(qualityValues: Tuple[float, float, str]) -> float:
    score = qualityValues[0] - 5. * qualityValues[1]
    if qualityValues[-1] == "HighQuality":
        score += 400.0
    elif qualityValues[-1] == "MediumQuality":
        score += 150.0
    return score


def getScoreForLowquality(qualityValues: Tuple[float, float, str]) -> float:
    return qualityValues[0] - qualityValues[1]


def dfsBuildTree(
        tree: Tree,
        level: int,
        sevenFilteredChekcMList: List[Dict[str, Tuple[float, float, str]]],
        filterOutputFolder: str,
        binName: str):
    if level > 6:
        return
    assert 1 <= level <= 6, ValueError("Error with a wrong level information.")
    curFolder = os.path.join(filterOutputFolder, index2Taxo[level])
    curCheckMRes = sevenFilteredChekcMList[level]
    curBinCoreName2lineage = readBinName2Annot(
        os.path.join(curFolder, f"{binName}_BinNameToLineage.ann")
    )
    preAnnotName = tree.annotName
    preCores = tree.core
    annot2_binCoreNameChekmValue = {}
    for binCoreName, annot in curBinCoreName2lineage.items():
        thisBinCheckmVal = curCheckMRes[binCoreName]
        score = getScore(thisBinCheckmVal)
        if annot not in annot2_binCoreNameChekmValue:
            annot2_binCoreNameChekmValue[annot] = [(binCoreName, thisBinCheckmVal, score)]
        else:
            annot2_binCoreNameChekmValue[annot].append((binCoreName, thisBinCheckmVal, score))
    for annot, binCoreNameCheckmValue in annot2_binCoreNameChekmValue.items():
        if preAnnotName in annot:
            sortedValues = list(sorted(binCoreNameCheckmValue, key=lambda x: x[-1], reverse=True))
            bestBinInfoTuple = sortedValues[0]
            if annot == curBinCoreName2lineage[binName]:
                preCores.append(True)
            else:
                preCores.append(False)
            node = tree.insert(bestBinInfoTuple[0], annot, bestBinInfoTuple[1], deepcopy(preCores))
            dfsBuildTree(node, level + 1, sevenFilteredChekcMList, filterOutputFolder, binName)
            preCores.pop(-1)


def buildTreeForBin(
        binFileName: str,
        deepurifyFolder: str,
        sevenFilteredChekcMList: List[Dict[str, Tuple[float, float, str]]]) -> Tree:
    binName, _ = os.path.splitext(binFileName)
    filterOutputFolder = os.path.join(deepurifyFolder, "FilterOutput")
    root = Tree(binName, "", sevenFilteredChekcMList[0][binName], [True])
    dfsBuildTree(root, 1, sevenFilteredChekcMList, filterOutputFolder, binName)
    return root


def dfsFindBestBins(
        tree: Tree):
    curChildren = tree.children
    if len(curChildren) == 0:
        tree.pathBests.append((tree.qualityValues, tree))
    else:
        for child in curChildren:
            dfsFindBestBins(child)
        for child in curChildren:
            for qualityValuesC, child in child.pathBests:
                if qualityValuesC[-1] in ["MediumQuality", "HighQuality"]:
                    tree.pathBests.append((qualityValuesC, child))
        highQNum = 0
        mediumQNum = 0
        for qulaityValuesChild, _ in tree.pathBests:
            if qulaityValuesChild[-1] == "HighQuality":
                highQNum += 1
            elif qulaityValuesChild[-1] == "MediumQuality":
                mediumQNum += 1
        if highQNum == 0:
            if tree.qualityValues[-1] == "HighQuality":
                if mediumQNum < 2:
                    tree.pathBests = [(tree.qualityValues, tree)]
            elif tree.qualityValues[-1] == "MediumQuality":
                if mediumQNum == 0:
                    tree.pathBests.append((tree.qualityValues, tree))
                elif mediumQNum == 1:
                    if getScore(tree.qualityValues) > getScore(tree.pathBests[0][0]):
                        tree.pathBests = [(tree.qualityValues, tree)]
        elif highQNum == 1 and mediumQNum == 0 and tree.qualityValues[-1] == "HighQuality":
            if getScore(tree.qualityValues) > getScore(tree.pathBests[0][0]):
                tree.pathBests = [(tree.qualityValues, tree)]


def findCoreOutput(
        node: Tree,
        res: List):
    res.append((getScoreForLowquality(node.qualityValues), node))
    if len(node.children) != 0:
        for child in node.children:
            if child.core[-1] is True:
                findCoreOutput(child, res)


def findBestBinsAfterFiltering(
    binFileName: str,
    inputFileFolder: str,
    deepurifyFolderPath: str,
    originalCheckMPath: str,
    outputPath: str
):
    sevenFilteredChekcMList = [readCheckMResultAndStat(originalCheckMPath)[0]]
    for i in range(1, 7):
        ch = readPickle(os.path.join(os.path.join(deepurifyFolderPath, "FilterOutput"), index2Taxo[i].split("_")[0] + "_checkm.pkl"))
        sevenFilteredChekcMList.append(ch)
    # build tree
    root = buildTreeForBin(binFileName, deepurifyFolderPath, sevenFilteredChekcMList)
    # find best bins for this bin
    dfsFindBestBins(root)
    # assign the taxonomy lineage for the root node
    for child in root.children:
        if child.core[-1] is True:
            root.annotName = child.annotName
            break
    outInfo = []
    n = len(root.pathBests)
    _, suffix = os.path.splitext(binFileName)
    # It has medium or high quality bins
    if n > 0:
        for k, (qualityValues, treeNode) in enumerate(root.pathBests):
            level = len(treeNode.core) - 1
            curBinName = treeNode.binName
            perfix = curBinName.split("___")[0]
            outName = f"{perfix}_{str(k)}{suffix}"
            outInfo.append((outName, qualityValues, treeNode.annotName))
            if level == 0:
                copy(os.path.join(inputFileFolder, binFileName), os.path.join(outputPath, outName))
            else:
                copy(os.path.join(deepurifyFolderPath, "FilterOutput", index2Taxo[level], treeNode.binName + suffix),
                     os.path.join(outputPath, outName))
    else:
        res = []
        findCoreOutput(root, res)
        res = list(sorted(res, key=lambda x: x[0], reverse=True))
        coreLeaf = res[0][1]
        curBinName = coreLeaf.binName
        perfix = curBinName.split("___")[0]
        l = len(coreLeaf.core) - 1
        outName = f"{perfix}_0{suffix}"
        outInfo.append((outName, coreLeaf.qualityValues, coreLeaf.annotName))
        if l == 0:
            copy(os.path.join(inputFileFolder, binFileName), os.path.join(outputPath, outName))
        else:
            copy(os.path.join(deepurifyFolderPath, "FilterOutput", index2Taxo[l], coreLeaf.binName + suffix), os.path.join(outputPath, outName))
    return outInfo
