# selection no duplicated and best bins
import os
from copy import deepcopy
from typing import Dict, List, Tuple

from Deepurify.Utils.IOUtils import readBinName2Annot, readCheckm2Res

index2Taxo = {
    0: "T0_filter",
    1: "T1_filter",
    2: "T2_filter",
    3: "T3_filter",
    4: "T4_filter",
    5: "T5_filter",
    6: "T6_filter"}


class Tree:
    def __init__(self, annotName: str,
                qualityValues_dict: Dict[str, Tuple[float, float, str]]) -> None:
        self.annotName = annotName
        self.qualityValues_dict = qualityValues_dict
        self.children = []
        self.pathBests = []
        self.last = False

    def insert(self,annotName: str,
                qualityValues_dict: Dict[str, Tuple[float, float, str]]):
        node = Tree(annotName, qualityValues_dict)
        self.children.append(node)
        return node


def getScore(
    qualityValues: Tuple[float, float, str]
) -> float:
    if qualityValues[-1] == "HighQuality":
        score = qualityValues[0] - 5. * qualityValues[1] + 100.
    elif qualityValues[-1] == "MediumQuality":
        score = qualityValues[0] - 5. * qualityValues[1] + 50.
    else:
        score = qualityValues[0] - 5. * qualityValues[1]
    return score


def compare(q_dict1, q_dict2):
    highQNum1 = 0
    mediumQNum1 = 0
    sum_score1 = 0.
    for _, val1 in q_dict1.items():
        if val1[-1] == "HighQuality":
            sum_score1 += getScore(val1)
            highQNum1 += 1
        elif val1[-1] == "MediumQuality":
            sum_score1 += getScore(val1)
            mediumQNum1 += 1

    highQNum2 = 0
    mediumQNum2 = 0
    sum_score2 = 0.
    for _, val2 in q_dict2.items():
        if val2[-1] == "HighQuality":
            sum_score2 += getScore(val2)
            highQNum2 += 1
        elif val2[-1] == "MediumQuality":
            sum_score2 += getScore(val2)
            mediumQNum2 += 1

    # selection
    if highQNum1 > highQNum2:
        return q_dict1
    elif highQNum1 == highQNum2:
        if mediumQNum1 > mediumQNum2:
            return q_dict1
        elif mediumQNum1 == mediumQNum2:
            if sum_score1 >= sum_score2:
                return q_dict1
            else:
                return q_dict2
        else:
            return q_dict2
    else:
        return q_dict2


def dfsFindBestBins(tree: Tree):
    curChildren = tree.children
    if len(curChildren) == 0:
        tree.pathBests.append(deepcopy(tree.qualityValues_dict))
    else:
        for child in curChildren:
            dfsFindBestBins(child)

        for child in curChildren:
            for qualityValues_dict in child.pathBests:
                tree.pathBests.append(deepcopy(qualityValues_dict))

        child_highQNum = 0
        child_mediumQNum = 0
        child_sum_score = 0.
        for qulaityValuesChild_dict in tree.pathBests:
            for _, val in qulaityValuesChild_dict.items():
                if val[-1] == "HighQuality":
                    child_sum_score += getScore(val)
                    child_highQNum += 1
                elif val[-1] == "MediumQuality":
                    child_sum_score += getScore(val)
                    child_mediumQNum += 1

        cur_highQNum = 0
        cur_mediumQNum = 0
        cur_sum_score = 0.
        for _, val in tree.qualityValues_dict.items():
            if val[-1] == "HighQuality":
                cur_sum_score += getScore(val)
                cur_highQNum += 1
            elif val[-1] == "MediumQuality":
                cur_sum_score += getScore(val)
                cur_mediumQNum += 1

        # selection
        if child_highQNum > cur_highQNum:
            pass
        elif child_highQNum == cur_highQNum:
            if child_mediumQNum > cur_mediumQNum:
                pass
            elif child_mediumQNum == cur_mediumQNum:
                if child_sum_score >= cur_sum_score:
                    pass
                else:
                    tree.pathBests = [deepcopy(tree.qualityValues_dict)]
            else:
                tree.pathBests = [deepcopy(tree.qualityValues_dict)]
        else:
            tree.pathBests = [deepcopy(tree.qualityValues_dict)]


def dfsBuildTree(
        tree: Tree,
        level: int,
        sevenFilteredChekcMList: List[Dict[str, Tuple[float, float, str]]],
        filterOutputFolder: str,
        binName: str):
    if level > 5:
        return
    assert 1 <= level <= 5, ValueError("Error with a wrong level information.")
    curFolder = os.path.join(filterOutputFolder, index2Taxo[level])
    curCheckMRes = sevenFilteredChekcMList[level]
    curBinCoreName2lineage = readBinName2Annot(
        os.path.join(curFolder, f"{binName}_BinNameToLineage.ann")
    )
    preAnnotName = tree.annotName
    annot2_binCoreName2ChekmValue = {}
    for binCoreName, annot in curBinCoreName2lineage.items():
        if binCoreName in curCheckMRes:
            thisBinCheckmVal = curCheckMRes[binCoreName]
        else:
            thisBinCheckmVal = (0., 0., "LowQuality")

        if annot not in annot2_binCoreName2ChekmValue:
            new_dict = {}
            if thisBinCheckmVal[-1] == "HighQuality" or thisBinCheckmVal[-1] == "MediumQuality":
                new_dict[f"{level}||{binCoreName}"] = thisBinCheckmVal
            annot2_binCoreName2ChekmValue[annot] = new_dict
        else:
            cur_dict = annot2_binCoreName2ChekmValue[annot]
            if thisBinCheckmVal[-1] == "HighQuality" or thisBinCheckmVal[-1] == "MediumQuality":
                cur_dict[f"{level}||{binCoreName}"] = thisBinCheckmVal

    for annot, binCoreName2CheckmValue in annot2_binCoreName2ChekmValue.items():
        o_val = {}
        s_val = {}
        for level_curBinName, val in binCoreName2CheckmValue.items():
            if "___o___" in level_curBinName:
                o_val[level_curBinName] = val
            elif "___s___" in level_curBinName:
                s_val[level_curBinName] = val
            else:
                raise ValueError("Error in the name.")
        if preAnnotName in annot:
            node = tree.insert(annot, compare(o_val, s_val))
            dfsBuildTree(node, level + 1, sevenFilteredChekcMList, filterOutputFolder, binName)


def last(
        root: Tree,
        checkMRes: Dict[str, Tuple[float, float, str]],
        filterOutputFolder: str,
        binName: str):

    curFolder = os.path.join(filterOutputFolder, index2Taxo[6])
    curBinCoreName2lineage = readBinName2Annot(
        os.path.join(curFolder, f"{binName}_BinNameToLineage.ann")
    )

    def inner(
            tree: Tree,
            annot_list: List[str],
            level: int,
            annot: str,
            val):
        if_insert = True
        for child in tree.children:
            child_annot_name_list = child.annotName.split("@")
            if child_annot_name_list[level] == annot_list[level] and child.last is False:
                inner(child, annot_list, level + 1, annot, val)
                if_insert = False
                break
        if if_insert:
            node = tree.insert(annot, val)
            node.last = True

    annot2_binCoreName2ChekmValue = {}
    for binCoreName, annot in curBinCoreName2lineage.items():
        if binCoreName in checkMRes:
            thisBinCheckmVal = checkMRes[binCoreName]
        else:
            thisBinCheckmVal = (0., 0., "LowQuality")

        if annot not in annot2_binCoreName2ChekmValue:
            new_dict = {}
            if thisBinCheckmVal[-1] == "HighQuality" or thisBinCheckmVal[-1] == "MediumQuality":
                new_dict[f"{6}||{binCoreName}"] = thisBinCheckmVal
            annot2_binCoreName2ChekmValue[annot] = new_dict
        else:
            cur_dict = annot2_binCoreName2ChekmValue[annot]
            if thisBinCheckmVal[-1] == "HighQuality" or thisBinCheckmVal[-1] == "MediumQuality":
                cur_dict[f"{6}||{binCoreName}"] = thisBinCheckmVal

    for annot, binCoreName2CheckmValue in annot2_binCoreName2ChekmValue.items():
        o_val = {}
        s_val = {}
        for level_curBinName, val in binCoreName2CheckmValue.items():
            if "___o___" in level_curBinName:
                o_val[level_curBinName] = val
            elif "___s___" in level_curBinName:
                s_val[level_curBinName] = val
            else:
                raise ValueError("Error in the name.")
        c_val = compare(o_val, s_val)
        annot_list = annot.split("@")
        inner(root, annot_list, 0, annot, c_val)


def buildTreeForBin(
        binFileName: str,
        deepurifyTmpFolder: str,
        sevenFilteredChekcMList: List[Dict[str, Tuple[float, float, str]]]) -> Tree:
    binNamePro, bin_suffix = os.path.splitext(binFileName)
    filterOutputFolder = os.path.join(deepurifyTmpFolder, "FilterOutput")
    
    if binNamePro in sevenFilteredChekcMList[0]:
        ori_quality = sevenFilteredChekcMList[0][binNamePro]
        real_name = f"-1||{binFileName}"
    else:
        real_name = f"-1||{binFileName}"
        ori_quality = (0., 0., "LowQuality")
    ori_dict = {real_name : ori_quality}
    
    sub_dict = {}
    for real_name, qu_val in sevenFilteredChekcMList[0].items():
        real_name_pro = real_name.split("___")[0]
        if "___s___" in real_name and real_name_pro == binNamePro:
            if qu_val[-1] == "HighQuality" or qu_val[-1] == "MediumQuality":
                sub_dict[f"0||{real_name}"] = qu_val
    root = Tree("", compare(ori_dict, sub_dict))
    dfsBuildTree(root, 1, sevenFilteredChekcMList, filterOutputFolder, binNamePro)
    last(root, sevenFilteredChekcMList[-1], filterOutputFolder, binNamePro)
    return root


def findBestBinsAfterFiltering(
    binFileName: str,
    inputFileFolder: str,
    deepurifyTmpFolderPath: str,
    originalCheckMPath: str
):
    cur_bin_prefix, suffix = os.path.splitext(binFileName)
    ori_checkm_res = readCheckm2Res(originalCheckMPath)[0]
    ori_checkm_res.update(readCheckm2Res(os.path.join(deepurifyTmpFolderPath,
                    "FilterOutput", index2Taxo[0] + "_checkm2_res",
                    "quality_report.tsv"))[0])
    sevenFilteredChekcMList = [ori_checkm_res]

    for i in range(1, 7):
        ch = readCheckm2Res(
            os.path.join(deepurifyTmpFolderPath,
            "FilterOutput",
            index2Taxo[i] + "_checkm2_res",
            "quality_report.tsv"))[0]
        sevenFilteredChekcMList.append(ch)

    root = buildTreeForBin(
        binFileName,
        deepurifyTmpFolderPath,
        sevenFilteredChekcMList)

    dfsFindBestBins(root)

    outInfo = []
    collect_list = []
    n = 0
    for qulaityValuesChild_dict in root.pathBests:
        for oneBinNameInNode, val in qulaityValuesChild_dict.items():
            if val[-1] == "HighQuality":
                collect_list.append((oneBinNameInNode, val))
                n += 1
            elif val[-1] == "MediumQuality":
                collect_list.append((oneBinNameInNode, val))
                n += 1

    if n > 0:
        for oneBinNameInNode, val in collect_list:
            level, curBinName = oneBinNameInNode.split("||")
            level = int(level)
            if level == -1:
                outInfo.append((val, os.path.join(
                    inputFileFolder, binFileName)))
            else:
                outInfo.append((val, os.path.join(
                    deepurifyTmpFolderPath, "FilterOutput",
                    index2Taxo[level], curBinName + suffix)))
    else:
        curBinName = f"{cur_bin_prefix}"
        if curBinName in ori_checkm_res:
            quality = ori_checkm_res[curBinName]
        else:
            quality = (0., 0., "LowQuality")
        outInfo.append((quality,  os.path.join(inputFileFolder, binFileName)))
    return outInfo
