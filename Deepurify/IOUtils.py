import os
import pickle
from typing import Dict, List, Tuple


def readFile(file_path: str) -> Tuple[str, str]:
    data = None
    with open(file_path, "r") as rh:
        for line in rh:
            data = line.strip("\n").split("\t")
    return data[0], data[1]


def readVocabulary(path: str) -> Dict:
    vocabulary = {}
    with open(path, mode="r") as rh:
        for line in rh:
            oneLine = line.strip("\n").split("\t")
            vocabulary[oneLine[0]] = int(oneLine[1])
    return vocabulary


def loadTaxonomyTree(pkl_path: str) -> Dict:
    with open(pkl_path, mode="rb") as rb:
        tree = pickle.load(rb)
    return tree


def readFasta(path: str) -> Dict[str, str]:
    """This function is used to read fasta file and
    it will return a dict, which key is the name of seq and the value is the sequence.

    Args:
        path (str): _description_

    Returns:
        Dict[str, str]: _description_
    """
    contig2Seq = {}
    curContig = ""
    curSeq = ""
    with open(path, mode="r") as rh:
        for line in rh:
            curLine = line.strip("\n")
            if curLine[0] == ">":
                if "plasmid" not in curContig.lower():
                    contig2Seq[curContig] = curSeq
                    curContig = curLine
                curSeq = ""
            else:
                curSeq += curLine
    if "plasmid" not in curContig.lower():
        contig2Seq[curContig] = curSeq
    contig2Seq.pop("")
    return contig2Seq


def writeFasta(name2seq: Dict, writePath: str):
    with open(writePath, "w") as wh:
        for key, val in name2seq.items():
            if key[0] != ">":
                wh.write(f">{key}" + "\n")
            else:
                wh.write(key + "\n")
            wh.write(val + "\n")


def convertFastaToTXT(fastPath: str, txtOutputPath: str):
    contig2seq = readFasta(fastPath)
    with open(txtOutputPath, "w") as wh:
        for _, val in contig2seq.items():
            wh.write(val + "\n")


def readPickle(readPath: str) -> object:
    with open(readPath, "rb") as rh:
        obj = pickle.load(rh)
    return obj


def writePickle(writePath: str, obj: object) -> None:
    with open(writePath, "wb") as wh:
        pickle.dump(obj, wh, pickle.HIGHEST_PROTOCOL)
        wh.flush()


def readTXT(path: str) -> Dict:
    name2seq = {}
    k = 0
    with open(path, "r") as rh:
        for line in rh:
            oneLine = line.strip("\n")
            name2seq[k] = oneLine
            k += 1
    return name2seq


def getNumberOfPhylum(taxoTree: Dict) -> int:
    return len(taxoTree["Children"])


def readAnnotResult(testBinPath: str) -> Tuple[Dict[str, str], Dict[str, List[float]]]:
    name2annote = {}
    name2probsList = {}
    with open(testBinPath, "r") as rh:
        for line in rh:
            info = line.strip("\n").split("\t")
            name2annote[info[0]] = info[1]
            probs = [float(prob) for prob in info[2:] if prob != ""]
            name2probsList[info[0]] = probs
    return name2annote, name2probsList


def writeAnnotResult(outputPath: str, name2annotated: Dict, name2maxList: Dict):
    with open(outputPath, "w") as wh:
        for key, val in name2annotated.items():
            wh.write(key + "\t" + val + "\t")
            for prob in name2maxList[key]:
                wh.write(str(prob)[:10] + "\t")
            wh.write("\n")


def readCheckMResultAndStat(
    checkMPath: str,
) -> Tuple[Dict[str, Tuple[float, float, str],], int, int, int]:
    name2res = {}
    highQuality = 0
    mediumQuality = 0
    lowQuality = 0
    if os.path.exists(checkMPath) is False:
        print("##################################################")
        print("### Error Occured During Reading CheckM Result ###")
        print("##################################################")
        raise ValueError(f"CheckM result file {checkMPath} not found...")
    with open(checkMPath, "r") as rh:
        for line in rh:
            if line[0] != "-" and "Marker lineage" not in line:
                info = line.strip("\n").split(" ")
                newInfo = []
                for ele in info:
                    if ele != "":
                        if "\t" in ele:
                            newInfo.extend(iter(ele.split("\t")))
                        else:
                            newInfo.append(ele)
                state = None
                comp = float(newInfo[-3])
                conta = float(newInfo[-2])
                if comp >= 90 and conta <= 5:
                    state = "HighQuality"
                    highQuality += 1
                elif comp >= 50 and conta <= 10:
                    state = "MediumQuality"
                    mediumQuality += 1
                else:
                    state = "LowQuality"
                    lowQuality += 1
                name2res[newInfo[0]] = (comp, conta, state)
    return name2res, highQuality, mediumQuality, lowQuality


def readHMMFile(file_path: str, ratio_cutoff, acc_cutoff) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, int]]]:
    gene2contigNames = {}
    contigName2_gene2num = {}
    if os.path.exists(file_path) is False:
        raise ValueError("HMM file does not exist.")
    with open(file_path, "r") as rh:
        for line in rh:
            if line[0] != "#":
                info = line.strip("\n").split(" ")
                newInfo = [ele for ele in info if ele != ""]
                aligFrom = float(newInfo[17])
                aligTo = float(newInfo[18])
                seqLen = float(newInfo[2])

                geneName = newInfo[4]
                acc = float(newInfo[21])
                contigName = ">" + "_".join(newInfo[0].split("_")[:-1])
                if (aligTo - aligFrom) / seqLen >= ratio_cutoff and acc >= acc_cutoff:
                    if geneName not in gene2contigNames:
                        gene2contigNames[geneName] = [contigName]
                    else:
                        gene2contigNames[geneName].append(contigName)
                    if contigName not in contigName2_gene2num:
                        newDict = {geneName: 1}
                        contigName2_gene2num[contigName] = newDict
                    else:
                        curDict = contigName2_gene2num[contigName]
                        if geneName not in curDict:
                            curDict[geneName] = 1
                        else:
                            curDict[geneName] += 1
    return gene2contigNames, contigName2_gene2num


def writeAnnot2BinNames(annot2binNames: Dict[str, List[str]], outputPath: str):
    with open(outputPath, "w") as wh:
        for annot, binList in annot2binNames.items():
            for binName in binList:
                wh.write(binName + "\t" + annot + "\n")


def readBinName2Annot(binName2LineagePath: str) -> Dict[str, str]:
    res = {}
    with open(binName2LineagePath, "r") as rh:
        for line in rh:
            info = line.strip("\n").split("\t")
            name, suffix = os.path.splitext(info[0])
            res[name] = info[1]
    return res
