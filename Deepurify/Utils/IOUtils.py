import os
import pickle
import sys
from typing import Dict, List, Tuple

import numpy as np

from Deepurify.Utils.HmmUtils import (HmmerHitDOM, addHit,
                                      identifyAdjacentMarkerGenes)


def readFile(file_path: str) -> Tuple[str, str]:
    data = None
    with open(file_path, "r", encoding="utf-8") as rh:
        for line in rh:
            data = line.strip("\n").split("\t")
    return data[0], data[1]


def readVocabulary(path: str) -> Dict:
    vocabulary = {}
    with open(path, mode="r", encoding="utf-8") as rh:
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
    with open(path, mode="r", encoding="utf-8") as rh:
        for line in rh:
            curLine = line.strip("\n")
            if curLine[0] == ">":
                if "plasmid" not in curContig.lower():
                    contig2Seq[curContig] = curSeq.upper()
                    curContig = curLine
                curSeq = ""
            else:
                curSeq += curLine
    if "plasmid" not in curContig.lower():
        contig2Seq[curContig] = curSeq
    contig2Seq.pop("")
    return contig2Seq


def readBinName2Annot(binName2LineagePath: str) -> Dict[str, str]:
    res = {}
    with open(binName2LineagePath, "r", encoding="utf-8") as rh:
        for line in rh:
            info = line.strip("\n").split("\t")
            name, suffix = os.path.splitext(info[0])
            res[name] = info[1]
    return res


def readCheckm2Res(file_path: str):
    res = {}
    h = 0
    m = 0
    l = 0
    with open(file_path, "r", encoding="utf-8") as rh:
        for line in rh:
            if "Name" not in line:
                info = line.strip("\n").split("\t")
                comp = float(info[1])
                conta = float(info[2])
                if comp >= 90 and conta <= 5:
                    state = "HighQuality"
                    h += 1
                elif comp >= 50 and conta <= 10:
                    state = "MediumQuality"
                    m += 1
                else:
                    state = "LowQuality"
                    l += 1
                res[info[0]] = (comp, conta, state)
    return res, h, m, l


def readDiamond(file_path: str, res: Dict):
    with open(file_path, "r", encoding="utf-8") as rh:
        for line in rh:
            thisline = line.strip("\n").split("\t")
            bin_name, contig_name = thisline[0].split("Ω")
            if bin_name not in res:
                cur_dict = {}
                cur_dict[contig_name] = thisline[1:]
                res[bin_name] = cur_dict
            else:
                cur_dict = res[bin_name]
                cur_dict[contig_name] = thisline[1:]
                res[bin_name] = cur_dict


def readPickle(readPath: str) -> object:
    with open(readPath, "rb") as rh:
        obj = pickle.load(rh)
    return obj


def readTXT(path: str) -> Dict:
    name2seq = {}
    k = 0
    with open(path, "r", encoding="utf-8") as rh:
        for line in rh:
            oneLine = line.strip("\n")
            name2seq[k] = oneLine
            k += 1
    return name2seq


def readAnnotResult(testBinPath: str) -> Tuple[Dict[str, str], Dict[str, List[float]]]:
    name2annote = {}
    name2probsList = {}
    with open(testBinPath, "r", encoding="utf-8") as rh:
        for line in rh:
            info = line.strip("\n").split("\t")
            name2annote[info[0]] = info[1]
            probs = [float(prob) for prob in info[2:] if prob != ""]
            name2probsList[info[0]] = probs
    return name2annote, name2probsList


def readCheckMResultAndStat(
    checkMPath: str,
) -> Tuple[Dict[str, Tuple[float, float, str], ], int, int, int]:
    name2res = {}
    highQuality = 0
    mediumQuality = 0
    lowQuality = 0
    if os.path.exists(checkMPath) is False:
        print("##################################################")
        print("### Error Occured During Reading CheckM Result ###")
        print("##################################################")
        raise ValueError(f"CheckM result file {checkMPath} not found...")
    with open(checkMPath, "r", encoding="utf-8") as rh:
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


def readHMMFile(file_path: str, hmmAcc2model, accs_set: set, phy_name=None) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, int]]]:
    gene2contigNames = {}
    contigName2_gene2num = {}

    if os.path.exists(file_path) is False:
        raise ValueError("HMM file does not exist.")

    markerHits = {}
    with open(file_path, "r", encoding="utf-8") as rh:
        for line in rh:
            if line[0] != "#":
                info = line.strip("\n").split(" ")
                newInfo = [ele for ele in info if ele != ""]
                pre = newInfo[0: 22]
                aft = "_".join(newInfo[22:])
                try:
                    hit = HmmerHitDOM(pre + [aft])
                except:
                    hit = None
                if hit is not None and hit.query_accession in accs_set:
                    addHit(hit, markerHits, hmmAcc2model)
    identifyAdjacentMarkerGenes(markerHits)

    for query_accession, hitDoms in markerHits.items():
        geneName = query_accession
        for hit in hitDoms:
            contigName = ">" + "_".join(hit.target_name.split("_")[0:-1])
            assert hit.query_accession == geneName, ValueError("The hit query accession is not equal with gene name.")
            assert hit.contig_name == contigName, ValueError(f"hit contig name: {hit.contig_name}, cur contigName: {contigName}")

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


def readHMMFileReturnDict(
    file_path: str
):
    contigName2hits = {}
    with open(file_path, "r", encoding="utf-8") as rh:
        for line in rh:
            if line[0] != "#":
                info = line.strip("\n").split(" ")
                newInfo = [ele for ele in info if ele != ""]
                pre = newInfo[0: 22]
                aft = "_".join(newInfo[22:])
                hit = HmmerHitDOM(pre + [aft])
                cur_contigName = ">" + "_".join(hit.target_name.split("_")[0:-1])
                assert hit.contig_name == cur_contigName, ValueError(f"hit contig name: {hit.contig_name}, cur contigName: {cur_contigName}")
                if cur_contigName in contigName2hits:
                    contigName2hits[cur_contigName].append(hit)
                else:
                    contigName2hits[cur_contigName] = [hit]
    return contigName2hits


def progressBar(j, N):
    statusStr = "          " + "{} / {}".format(j + 1, N)
    cn = len(statusStr)
    if cn < 50:
        statusStr += "".join([" " for _ in range(50 - cn)])
    statusStr += "\r"
    sys.stderr.write("%s\r" % statusStr)
    sys.stderr.flush()


def readMetaInfo(file_path: str):
    res = {}
    h = 0
    m = 0
    l = 0
    with open(file_path, "r", encoding="utf-8") as rh:
        for line in rh:
            info = line.strip("\n").split("\t")
            comp = float(info[1])
            conta = float(info[2])
            if comp >= 90 and conta <= 5:
                state = "HighQuality"
                h += 1
            elif comp >= 50 and conta <= 10:
                state = "MediumQuality"
                m += 1
            else:
                state = "LowQuality"
                l += 1
            res[info[0]] = (comp, conta, state)
    return res, h, m, l


def readCSV(file_path):
    csv = []
    with open(file_path, "r", encoding="utf-8") as rh:
        for line in rh:
            oneline = line.strip("\n").split(",")
            csv.append(oneline)
    return csv


def convertFastaToTXT(fastPath: str, txtOutputPath: str):
    contig2seq = readFasta(fastPath)
    with open(txtOutputPath, "w", encoding="utf-8") as wh:
        for _, val in contig2seq.items():
            wh.write(val + "\n")


def getNumberOfPhylum(taxoTree: Dict) -> int:
    return len(taxoTree["Children"])


def writePickle(writePath: str, obj: object) -> None:
    with open(writePath, "wb") as wh:
        pickle.dump(obj, wh, pickle.HIGHEST_PROTOCOL)
        wh.flush()


def write_result(
        outputBinFolder,
        collected_list,
        wh):
    for i, (qualityValues, cor_path) in enumerate(collected_list):
        outName = f"Deepurify_Bin_{i}.fasta"
        wh.write(
            outName
            + "\t"
            + str(qualityValues[0])
            + "\t"
            + str(qualityValues[1])
            + "\t"
            + str(qualityValues[2])
            + "\n"
        )
        writeFasta(readFasta(cor_path), os.path.join(outputBinFolder, outName))


def writeAnnot2BinNames(annot2binNames: Dict[str, List[str]], outputPath: str):
    with open(outputPath, "w", encoding="utf-8") as wh:
        for annot, binList in annot2binNames.items():
            for binName in binList:
                wh.write(binName + "\t" + annot + "\n")


def writeFasta(name2seq: Dict, writePath: str, change_name=False):
    index = 0
    with open(writePath, "w", encoding="utf-8") as wh:
        for key, val in name2seq.items():
            if change_name:
                wh.write(f">Contig_{index}_{len(val)}\n")
            else:
                if key[0] != ">":
                    wh.write(f">{key}\n")
                else:
                    wh.write(key + "\n")
            index += 1
            for i in range(0, len(val), 60):
                wh.write(val[i: i + 60] + "\n")


def writeAnnotResult(outputPath: str, name2annotated: Dict, name2maxList: Dict):
    with open(outputPath, "w", encoding="utf-8") as wh:
        for key, val in name2annotated.items():
            wh.write(key + "\t" + val + "\t")
            for prob in name2maxList[key]:
                wh.write(str(prob)[:10] + "\t")
            wh.write("\n")
