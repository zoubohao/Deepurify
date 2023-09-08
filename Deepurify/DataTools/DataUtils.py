import math
import os
from copy import deepcopy
from shutil import copyfile
from typing import Callable, Dict, List, Tuple

import numpy as np
from Deepurify.IOUtils import readVocabulary


def backTrace(curSelected: List, selection: List, res: set, k: int) -> None:
    if len(curSelected) == k:
        res.add(tuple(curSelected))
    else:
        for element in selection:
            curSelected.append(element)
            backTrace(curSelected, selection, res, k)
            curSelected.pop(-1)


def filterSpeciesNumInPhylumSmallerThanThreFiles(folder: str, outputFolder: str, threshold: int) -> None:
    files = os.listdir(folder)
    phy2num = {}
    for file in files:
        info = file.split("@")
        if info[0] not in phy2num:
            phy2num[info[0]] = 1
        else:
            phy2num[info[0]] += 1
    includePhySet = {"Unclassified"}
    for phyName, val in phy2num.items():
        if val >= threshold:
            includePhySet.add(phyName)
    try:
        includePhySet.remove("Unclassified")
    except:
        pass
    for file in files:
        info = file.split("@")
        if info[0] in includePhySet:
            copyfile(os.path.join(folder, file), os.path.join(outputFolder, file))
    files = os.listdir(outputFolder)
    phy2num = {}
    for file in files:
        info = file.split("@")
        if info[0] not in phy2num:
            phy2num[info[0]] = 1
        else:
            phy2num[info[0]] += 1
    for key, val in phy2num.items():
        print(key, val)
    print("The number of phylum in dataset is: ", len(phy2num))


index2Taxo = {6: "phylum", 5: "class", 4: "order", 3: "family", 2: "genus", 1: "species"}


def insert(taxoList: List[str], curDict: Dict) -> None:
    length = len(taxoList)
    if length == 1:
        if taxoList[0] not in curDict["Children"]:
            curDict["Children"].append(taxoList[0])
    else:
        signal = True
        for child in curDict["Children"]:
            if child["Name"] == taxoList[0]:
                copyTaxo = deepcopy(taxoList)
                copyTaxo.pop(0)
                insert(copyTaxo, child)
                signal = False
        if signal:
            newDict = {"TaxoLevel": index2Taxo[length], "Name": taxoList[0], "Children": []}
            copyTaxo = deepcopy(taxoList)
            copyTaxo.pop(0)
            insert(copyTaxo, newDict)
            curDict["Children"].append(newDict)


def taxonomyTreeBuild(split_func: Callable, file_path=None) -> Dict:
    """
    This function is used for buliding a taxonomy tree with the map data structure. Like the json structure.
    The biggest taxonomy level is the superkingdom of bacteria.
    There are 6 sub-level, its are phylum, class, order, family, genus, and species.

    1. For the levels of phylum, class, order, family, and genus, each objects in those level will be represented
    as a map with following attributes:
    "TaxoLevel" -> Depicts the taxonomy level of this object,
    "Name" -> The  name of the object,
    "Children" -> The list of next level objects.
    2. For the species level, since there is no next level for species, therefore, the objects in "Children" attribute of genus
    are just strings, which are the name of the species that belong to corresponding genus.

    split_func: The split function must return a tuple that contains the name of "phylum, class, order, family, genus, and species" in
    this order.
    file_path: the path of taxonomy txt file. Each line must contain the taxonomy of one species. Each line will be pharsed by split_func.
    """
    taxonomyTree = {"TaxoLevel": "superkingdom", "Name": "bacteria", "Children": []}
    with open(file_path, mode="r") as rh:
        for line in rh:
            oneLine = line.strip("\n")
            insert(split_func(oneLine), taxonomyTree)
    return taxonomyTree


def split_Pro_function(oneLine: str) -> List:
    """
    This function is used to split one line of file "TaxonomyProGenomes.txt".
    It returns the taxonomy level from phylum to species in order.
    If the species has no taxonomy in one level (class, order, ...), it will be marked "Unclassified" at that level.
    All of blank char will be replaced with "-". All of "/" char will be replaced with "".
    """
    levelsInfor = oneLine.split("\t")
    res = []
    for i in range(2, len(levelsInfor)):
        if levelsInfor[i] != " ":
            infor_split = levelsInfor[i].split(" ")
            curStr = ""
            for vocab in infor_split[1:]:
                curStr = curStr + vocab.replace(" ", "-").replace("/", "-").replace(":", "-").replace("_", "-").replace(".", "-") + "-"
            res.append(curStr[:-1])
        else:
            res.append("Unclassified")
    return res


def split_file_function(oneLine: str) -> List:
    return oneLine.split(".txt")[0].split("@")


def checkTaxoExistInTree(taxoLevels: List, tree: Dict) -> bool:
    """
    :param taxoLevels: a list with the name of taxonomy levels: phylum, class, order, family, genus, and species in this order. List
    :param tree: the initial tree that is constructed with map structure. Dict
    """
    children = tree["Children"]
    curTaxo = taxoLevels[0]
    signal = True
    if len(taxoLevels) != 1:
        nextChild = None
        for child in children:
            if child["Name"] == curTaxo:
                signal = False
                nextChild = child
        return False if signal else checkTaxoExistInTree(taxoLevels[1:], nextChild)
    else:
        for child in children:
            if child == curTaxo:
                signal = False
        return not signal


def buildVocabularyAndAssignWeightByPhylum(folder: str, vocab_path: str, samples_weight_path: str, base=math.e, if_weight=True) -> None:
    """
    phylum, class, order, family, genus, and species
    """

    def count(taxoCountMap, name):
        if name not in taxoCountMap:
            taxoCountMap[name] = 1
        else:
            taxoCountMap[name] += 1

    def ratioCal(curNum, ratioNum):
        return math.log(ratioNum / curNum, base) + 1.0

    files = os.listdir(folder)
    vocab_dict = {"[PAD]": 0}
    vocab_set = set()
    samples_weight = {}
    phylum2Num = {}
    k = 0
    for file in files:
        split_info = file.split(".txt")[0].split("@")
        k += 1
        phylumName = "".join(split_info[:1])
        count(phylum2Num, phylumName)
        for word in split_info:
            vocab_set.add(word)
    k = 1
    for word in vocab_set:
        vocab_dict[word] = k
        k += 1
    ratioList = []
    ratioNumPhylum = np.mean(np.array(list(sorted(list(phylum2Num.values()))))[2:-2])
    for file in files:
        split_info = file.split(".txt")[0].split("@")
        phylumNum = phylum2Num["".join(split_info[:1])]
        if if_weight:
            samples_weight[file.split(".txt")[0]] = np.clip(ratioCal(phylumNum, ratioNumPhylum), 1.0, 3.0)
        else:
            samples_weight[file.split(".txt")[0]] = 1.0
        ratioList.append(samples_weight[file.split(".txt")[0]])
    with open(vocab_path, mode="w") as wh:
        for key, val in vocab_dict.items():
            wh.write(key + "\t" + str(val) + "\n")
    with open(samples_weight_path, mode="w") as wh:
        for key, val in samples_weight.items():
            wh.write(key + "\t" + str(val) + "\n")
    print("Max value in those ratios is :", max(ratioList))
    print("Min value in those ratios is :", min(ratioList))


def sampleOneTimeFromContigSeqForTraining(
    contigSeq: str, max_data_length: int, cur_sample_times: int, max_sample_times: int, min_data_length=1000
) -> Tuple[List[str], int]:
    """
    This function do guarantee the length of sequences all exactly equal with max_data_length,
    !!!! BUT you must notice the max length of DATA sampling can be longer than the max length DURING training process !!!!
    OVERLAPPING SAMPLEING.
    """
    sampledList = []
    if cur_sample_times >= max_sample_times:
        return sampledList, cur_sample_times
    seqLen = len(contigSeq)
    # If the length of this seq smaller than min_data_length, then it would not be added into training dataset.
    if seqLen < min_data_length:
        return sampledList, cur_sample_times
    # If the length of this seq smaller than max_data_length, then it would be added into training dataset derictly.
    elif seqLen <= max_data_length:
        sampledList.append(contigSeq)
        return sampledList, cur_sample_times + 1
    # If the length of this seq bigger than max_data_length, then it would be added into training dataset by splitting it
    # into smaller pieces with random overlapping length. The length for each pieces is max_data_length
    else:
        startIndex = 0
        while startIndex + max_data_length < seqLen:
            seq = contigSeq[startIndex: startIndex + max_data_length]
            sampledList.append(seq)
            startIndex += int(max_data_length / (np.random.rand(1)[0] * 8.0 + 2.0))
            cur_sample_times += 1
            if cur_sample_times >= max_sample_times:
                return sampledList, cur_sample_times
        finalSampled = contigSeq[startIndex: startIndex + max_data_length]
        sampledList.append(finalSampled)
        cur_sample_times += 1
        return sampledList, cur_sample_times


def sampleOneTimeFromContigSeqForTesting(
    contigSeq: str, max_data_length: int, cur_sample_times: int, max_sample_times: int, min_data_length=1000
) -> Tuple[List[str], int]:
    """
    This function do guarantee the length of sequences all exactly equal with max_data_length
    NOT OVERLAPPING SAMPLEING. BUT RANDOM SAMPLING
    """
    sampledList = []
    if cur_sample_times >= max_sample_times:
        return sampledList, cur_sample_times
    seqLen = len(contigSeq)
    if seqLen < min_data_length:
        return sampledList, cur_sample_times
    elif seqLen <= max_data_length:
        sampledList.append(contigSeq)
        return sampledList, cur_sample_times + 1
    else:
        startIndex = np.random.randint(0, seqLen - min_data_length)
        curLength = np.random.randint(min_data_length, max_data_length + 1)
        cutSeq = contigSeq[startIndex: startIndex + curLength]
        sampledList.append(cutSeq)
        return sampledList, cur_sample_times + 1


def concatContigs(txtFilePath: str) -> List[str]:
    contigs = []
    with open(txtFilePath, "r") as rh:
        contigs.extend(line.strip("\n") for line in rh)
    new_seq = "".join(contigs)
    return [new_seq]


def approx(phyCount: int, baseNum: int, sampledNum: int) -> int:
    times = sampledNum // phyCount // baseNum
    return times + 1 if times * phyCount * baseNum <= sampledNum else times


def sampleFromContigsAndWriteFiles(
    genome_folder: str,
    out_path: str,
    phy2countPath: str,
    min_data_length=1000,
    max_data_length=256 * 256,
    base_sample_times=5,
    if_training=True,
    smallPhyUpperBound=500,
) -> None:
    files = os.listdir(genome_folder)
    phy2count = readVocabulary(phy2countPath)
    totalNum = sum(phy2count.values())
    minVal = min(phy2count.values())
    index = 0
    for i, file in enumerate(files):
        phyName = file.split("@")[0]
        times = approx(phy2count[phyName], base_sample_times, smallPhyUpperBound)
        if times <= 0:
            times = 1
        if times == 1 and phy2count[phyName] < smallPhyUpperBound:
            times += 0.5
        max_sample_times = int(base_sample_times * times)
        print(i, file, max_sample_times, phy2count[phyName], totalNum)
        longestContigs = concatContigs(os.path.join(genome_folder, file))[
            :max_sample_times
        ]
        cur_samp_times = 0
        while cur_samp_times < max_sample_times:
            for i, seq in enumerate(longestContigs):
                if if_training:
                    samples, cur_samp_times = sampleOneTimeFromContigSeqForTraining(
                        seq, max_data_length, cur_samp_times, max_sample_times, min_data_length
                    )
                else:
                    samples, cur_samp_times = sampleOneTimeFromContigSeqForTesting(
                        seq, max_data_length, cur_samp_times, max_sample_times, min_data_length
                    )
                for s in samples:
                    with open(os.path.join(out_path, f"{str(index)}.txt"), "w") as wh:
                        wh.write(s + "\t" + file.split(".txt")[0] + "\n")
                    index += 1
                if cur_samp_times >= max_sample_times:
                    break
