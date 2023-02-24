# selection no duplicated and best bins
import os
from copy import deepcopy
from shutil import copy
from typing import Dict, List, Tuple

from ..IOUtils import readBinName2Annot, readCheckMResultAndStat

index2Taxo = {1: "phylum_filter", 2: "class_filter", 3: "order_filter", 4: "family_filter", 5: "genus_filter", 6: "species_filter"}


class Tree:
    def __init__(self, binName: str, annotName: str, qualityValues: Tuple[float, float, str], core: List[bool]) -> None:
        self.binName = binName
        self.annotName = annotName
        self.qualityValues = qualityValues
        self.core = core
        self.canSelect = True
        self.childern = []
        self.pathBests = []

    def insert(self, binName: str, annotName: str, qualityValues: Tuple[float, float, str], core: List[bool]):
        node = Tree(binName, annotName, qualityValues, core)
        self.childern.append(node)
        return node


def getScore(qualityValues: Tuple[float, float, str]) -> float:
    score = qualityValues[0] - 5.0 * qualityValues[1]
    if qualityValues[-1] == "HighQuality":
        score += 200.0
    elif qualityValues[-1] == "MediumQuality":
        score += 100.0
    return score


def dfsBuildTree(tree: Tree, level: int, sevenFilteredChekcMList: List[Dict[str, Tuple[float, float, str]]], filterOutputFolder: str, binName: str):
    if level <= 6:
        assert 1 <= level <= 6
        curFolder = os.path.join(filterOutputFolder, index2Taxo[level])
        curCheckMRes = sevenFilteredChekcMList[level]
        curBinCoreName2lineage = readBinName2Annot(os.path.join(curFolder, binName + "_BinNameToLineage.ann"))
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


def buildTreeForBin(binFileName: str, deepurifyFolder: str, sevenFilteredChekcMList: List[Dict[str, Tuple[float, float, str]]]) -> Tree:
    binName, _ = os.path.splitext(binFileName)
    filterOutputFolder = os.path.join(deepurifyFolder, "FilterOutput")
    root = Tree(binName, "", sevenFilteredChekcMList[0][binName], [True])
    dfsBuildTree(root, 1, sevenFilteredChekcMList, filterOutputFolder, binName)
    return root


def dfsFindBestBins(tree: Tree):
    curChildren = tree.childern
    if len(curChildren) == 0:
        tree.pathBests.append((tree.qualityValues, tree))
    else:
        for child in curChildren:
            dfsFindBestBins(child)
        for child in curChildren:
            for qualityValuesC, child in child.pathBests:
                if qualityValuesC[-1] == "MediumQuality" or qualityValuesC[-1] == "HighQuality":
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


def findLeafNode(node: Tree) -> Tree:
    if len(node.childern) == 0:
        return node
    else:
        for child in node.childern:
            if child.core[-1] is True:
                return findLeafNode(child)


def traverse(tree: Tree, close: bool):
    if close:
        tree.canSelect = False
    else:
        print(tree.binName, tree.annotName, len(tree.childern), tree.qualityValues, tree.core, tree.canSelect)
    if len(tree.childern) != 0:
        for child in tree.childern:
            traverse(child, close)


def findBestBinsAfterFiltering(
    binFileName: str, inputFileFolder: str, deepurifyFolderPath: str, originalCheckMPath: str, outputPath: str
):
    sevenFilteredChekcMList = [readCheckMResultAndStat(originalCheckMPath)[0]]
    for i in range(1, 7):
        ch = readCheckMResultAndStat(os.path.join(os.path.join(deepurifyFolderPath, "FilterOutput"), index2Taxo[i].split("_")[0] + "_checkm.txt"))
        sevenFilteredChekcMList.append(ch[0])
    # build tree
    root = buildTreeForBin(binFileName, deepurifyFolderPath, sevenFilteredChekcMList)
    # find best bins for this bin
    dfsFindBestBins(root)
    n = len(root.pathBests)
    outInfo = []
    _, suffix = os.path.splitext(binFileName)
    if n > 0:
        for qualityValues, treeNode in root.pathBests:
            level = len(treeNode.core) - 1
            curBinName = treeNode.binName
            outName = curBinName + "_" + str(level) + suffix
            if treeNode.annotName == "":  # assign the taxonomy lineage for the root node
                for child in treeNode.childern:
                    if child.core[-1] is True:
                        treeNode.annotName = child.annotName
                        break
            outInfo.append((outName, qualityValues, treeNode.annotName))
            if level == 0:
                copy(os.path.join(inputFileFolder, binFileName), os.path.join(outputPath, outName))
            else:
                copy(os.path.join(deepurifyFolderPath, "FilterOutput", index2Taxo[level], treeNode.binName + suffix),
                     os.path.join(outputPath, outName))
    else:
        coreLeaf = findLeafNode(root)
        if coreLeaf is None:
            traverse(root, close=False)
            raise ValueError("The leaf node is None.")
        curBinName = coreLeaf.binName
        outName = curBinName + "_" + str(6) + suffix
        outInfo.append((outName, coreLeaf.qualityValues, coreLeaf.annotName))
        copy(os.path.join(deepurifyFolderPath, "FilterOutput", index2Taxo[6], coreLeaf.binName + suffix), os.path.join(outputPath, outName))
    return outInfo

    #     coreLeaf = None
    #     for child in root.childern:
    #         if child.core[-1] is True:
    #             coreLeaf = child
    #     curBinName = coreLeaf.binName
    #     outName = curBinName + "_" + str(1) + suffix
    #     outInfo.append((outName, coreLeaf.qualityValues, coreLeaf.annotName))
    #     copy(os.path.join(deepurifyFolderPath, "FilterOutput", index2Taxo[1], coreLeaf.binName + suffix), os.path.join(outputPath, outName))
    # return outInfo
