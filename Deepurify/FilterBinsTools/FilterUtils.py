import math
import os
import sys
from copy import deepcopy
from multiprocessing import Process
from typing import Dict, List, Set, Tuple, Union

from Deepurify.IOUtils import (readAnnotResult, readCheckMResultAndStat,
                               readFasta, readHMMFile, writeAnnot2BinNames,
                               writeFasta)
from Deepurify.LabelContigTools.LabelBinUtils import \
    getBestMultiLabelsForFiltering

index2Taxo = {1: "T1_filter", 2: "T2_filter", 3: "T3_filter", 4: "T4_filter", 5: "T5_filter", 6: "T6_filter"}


def summedLengthCal(name2seq: Dict[str, str]) -> int:
    return sum(len(seq) for seq in name2seq.values())


def allocate(
    splitContigSetList: List[Set[str]],
    splitRecordGenes: List[Dict[str, int]],
    info: Tuple[str, Dict[str, int], int],
    replication_times_threashold: int,
) -> None:
    if not splitContigSetList:
        allocate_new_set(info, splitContigSetList, splitRecordGenes)
    else:
        insertIndex = None
        for i, record in enumerate(splitRecordGenes):
            if_insert = True
            for gene, num in info[1].items():
                if gene in record:
                    recordNum = record[gene]
                    if (recordNum + num) > replication_times_threashold:
                        if_insert = False
                        break
            if if_insert is True:
                insertIndex = i
                break
        if insertIndex is not None:
            splitContigSetList[insertIndex].add(info[0])
            curRecord = splitRecordGenes[insertIndex]
            for gene, num in info[1].items():
                if gene not in curRecord:
                    curRecord[gene] = num
                else:
                    curRecord[gene] += num
        else:
            allocate_new_set(info, splitContigSetList, splitRecordGenes)


def allocate_new_set(info, splitContigSetList, splitRecordGenes):
    curSet = {info[0]}
    splitContigSetList.append(curSet)
    curDict = {}
    curDict |= info[1]
    splitRecordGenes.append(curDict)


def summedRecord(recordList):
    summedValue = 0.0
    for num, _ in recordList:
        summedValue += num
    return summedValue


def splitContigs(
    contigName2seq: Dict[str, str],
    gene2contigNames: Dict[str, List[str]],
    contigName2_gene2num: Dict[str, Dict[str, int]],
    replication_times_threashold: int,
    estimate_completeness_threshold: float,
    core: bool
) -> List[Dict[str, str]]:
    c1 = deepcopy(contigName2seq)
    c2 = deepcopy(gene2contigNames)
    c3 = deepcopy(contigName2_gene2num)
    contigSeqPair = [
        (contigName, len(seq)) for contigName, seq in contigName2seq.items()
    ]
    exist_contigs = [
        contig
        for contig, _ in sorted(
            contigSeqPair, key=lambda x: x[1], reverse=False
        )
    ]
    existGene2contigNames = {}  # subset of gene2contigNames
    existcontig2_gene2num = {}
    notExistGeneContig = set()
    notExistGeneContig2seq = {}
    # find the exist genes in those input contigs
    for contig in exist_contigs:
        if contig in contigName2_gene2num:
            curExistGenes2num = contigName2_gene2num[contig]
            existcontig2_gene2num[contig] = curExistGenes2num
            for gene, _ in curExistGenes2num.items():
                if gene not in existGene2contigNames:
                    existGene2contigNames[gene] = gene2contigNames[gene]
        else:
            notExistGeneContig.add(contig)
            notExistGeneContig2seq[contig] = deepcopy(contigName2seq[contig])

    geneInfo = [
        (gene, contigList, len(set(contigList)))
        for gene, contigList in existGene2contigNames.items()
    ]
    sortGeneInfo = list(sorted(geneInfo, key=lambda x: x[-1], reverse=False))
    splitContigSetList = []
    splitRecordGenes = []
    # go through genes one by one
    for gene, contigList, _ in sortGeneInfo:
        curContigSet = set(contigList)
        # the information of current contigs
        infoList = []
        for curContig in curContigSet:
            if curContig in contigName2seq:
                length = len(contigName2seq[curContig])
                gene2num = existcontig2_gene2num[curContig]
                score = length / 10000.0 + len(gene2num)
                infoList.append((curContig, gene2num, length, score))
        sortedByLength = list(sorted(infoList, key=lambda x: x[3], reverse=True))  # greedy add but not the optimial
        for info in sortedByLength:
            allocate(splitContigSetList, splitRecordGenes, info, replication_times_threashold)

    if not splitContigSetList:
        return [notExistGeneContig2seq] if core else []
    totalN = len(existGene2contigNames)
    filtedContigList = []
    ratioSet = set()
    scoreSet = set()
    for i in range(len(splitContigSetList)):
        curNumGenes = len(splitRecordGenes[i])
        curSet = splitContigSetList[i].union(notExistGeneContig)
        curContig2seq = {}
        summedLength = 0.0
        for contigName in curSet:
            curContig2seq[contigName] = deepcopy(contigName2seq[contigName])
            summedLength += len(contigName2seq[contigName])
        ratio = curNumGenes / totalN + 0.0
        score = curNumGenes / totalN + 0.0 + math.log(summedLength) / 20.0
        if ratio not in ratioSet and score not in scoreSet:
            filtedContigList.append((curContig2seq, ratio, score))
            ratioSet.add(ratio)
            scoreSet.add(score)

    filtedContigList = sorted(filtedContigList, key=lambda x: x[-1], reverse=True)
    first = filtedContigList[0]
    if first[1] <= 0.72:
        return splitContigs(c1,
                            c2,
                            c3,
                            replication_times_threashold + 1,
                            estimate_completeness_threshold,
                            core)
    else:
        return [
            infoPair[0]
            for i, infoPair in enumerate(filtedContigList)
            if infoPair[1] >= estimate_completeness_threshold or i == 0
        ]


def modify(contigName2annot, coreNames):
    N = 0.0
    n = 0.0
    recordCount = [[0.0, ""] for _ in range(len(coreNames))]
    for _, annotLabel in contigName2annot.items():
        for i, coreName in enumerate(coreNames):
            if coreName == annotLabel:
                n += 1
                recordCount[i][0] += 1
                recordCount[i][1] = coreName
        N += 1
    if n / N >= 0.8:
        sortedRecordCount = list(sorted(recordCount, key=lambda x: x[0]))
        while (
            len(sortedRecordCount) > 1
            and summedRecord(sortedRecordCount) / N > 0.68
        ):
            sortedRecordCount.pop(0)
        newCoreNames = [coreNames[0]]
        for _, coreTaxo in sortedRecordCount:
            if coreTaxo != newCoreNames[0]:
                newCoreNames.append(coreTaxo)
        coreNames = deepcopy(newCoreNames)
    return coreNames


def filterContaminationOneBin(
    annotBinPath: str,
    binFastaPath: str,
    hmmFilePath: str,
    outputFastaFolder: str,
    taxoLevel: int,
    ratio_cutoff: float,
    acc_cutoff: float,
    estimate_completeness_threshold: float,
    seq_length_threshold: int,
    originalBinsCheckMPath,
    simulated_MAG=False
) -> None:
    assert 1 <= taxoLevel <= 6, ValueError("The taxoLevel must between 1 to 6.")
    assert 0.4 <= ratio_cutoff, ValueError("The ratio_cutoff value must bigger than 0.4")
    assert 0.6 <= acc_cutoff, ValueError("acc_cutoff must bigger than 0.6")

    contigName2annot, contigName2probs = readAnnotResult(annotBinPath)
    contigName2seq = readFasta(binFastaPath)
    annotRes = []
    probs = []
    length = []
    for key, val in contigName2annot.items():
        taxoInfo = val.split("@")
        annotRes.append(taxoInfo[0:taxoLevel])
        probs.append(contigName2probs[key][0:taxoLevel])
        length.append(len(contigName2seq[key]))
    coreList = getBestMultiLabelsForFiltering(annotRes, probs, length)
    coreNames = []
    for core in coreList:
        coreNames.append("@".join(core[1:]))

    filtedContigName2seq = {}
    annot2_contigName2seq = {}

    if taxoLevel == 6:
        coreNames = modify(contigName2annot, coreNames)

    for key, seq in contigName2seq.items():
        for coreName in coreNames:
            if coreName in contigName2annot[key]:
                filtedContigName2seq[key] = seq
        
        if key not in filtedContigName2seq:
            curAnnot = "@".join(contigName2annot[key].split("@")[0:taxoLevel])
            if curAnnot not in annot2_contigName2seq:
                newDict = dict()
                newDict[key] = seq
                annot2_contigName2seq[curAnnot] = newDict
            else:
                curDict = annot2_contigName2seq[curAnnot]
                curDict[key] = seq

    # write files
    binName = os.path.split(binFastaPath)[-1]
    annot2binNames = {}
    writeFasta(filtedContigName2seq, os.path.join(outputFastaFolder, binName))

    if simulated_MAG:
        return

    # using SCGs to exclude external contigs.
    gene2contigList, contigName2_gene2num = readHMMFile(hmmFilePath, ratio_cutoff, acc_cutoff)
    annot2binNames[coreNames[0]] = [binName]
    binNamePro, bin_suffix = os.path.splitext(binName)
    
    assert originalBinsCheckMPath is not None, ValueError("The checkm result of original MAGs is None.")
    res = readCheckMResultAndStat(originalBinsCheckMPath)[0]
    q = res[binNamePro]
    
    filtedContigName2seqList = \
    splitContigs(filtedContigName2seq, gene2contigList, contigName2_gene2num, 1, estimate_completeness_threshold, True)
    
    idx_k = 0
    for coreName2seqFilter in filtedContigName2seqList:
        summedLength = summedLengthCal(coreName2seqFilter)
        if summedLength >= seq_length_threshold:
            annot2binNames[coreNames[0]].append(binNamePro + "___" + str(idx_k) + bin_suffix)
            writeFasta(coreName2seqFilter, os.path.join(outputFastaFolder, binNamePro + "___" + str(idx_k) + bin_suffix))
            idx_k += 1

    gene2contigList, contigName2_gene2num = readHMMFile(hmmFilePath, ratio_cutoff, acc_cutoff)
    for annot, noCoreContigName2seq in annot2_contigName2seq.items():
        if summedLengthCal(noCoreContigName2seq) >= seq_length_threshold:
            if annot not in annot2binNames:
                annot2binNames[annot] = [binNamePro + "___" + str(idx_k) + bin_suffix]
            else:
                annot2binNames[annot].append(binNamePro + "___" + str(idx_k) + bin_suffix)
            writeFasta(noCoreContigName2seq, os.path.join(outputFastaFolder, binNamePro + "___" + str(idx_k) + bin_suffix))
            idx_k += 1
        
        curFilteredList = splitContigs(noCoreContigName2seq, gene2contigList, contigName2_gene2num, 1, estimate_completeness_threshold, False)
        for noCoreName2seqFilter in curFilteredList:
            summedLength = summedLengthCal(noCoreName2seqFilter)
            if summedLength >= seq_length_threshold:
                if annot not in annot2binNames:
                    annot2binNames[annot] = [binNamePro + "___" + str(idx_k) + bin_suffix]
                else:
                    annot2binNames[annot].append(binNamePro + "___" + str(idx_k) + bin_suffix)
                writeFasta(noCoreName2seqFilter, os.path.join(outputFastaFolder, binNamePro + "___" + str(idx_k) + bin_suffix))
                idx_k += 1

    writeAnnot2BinNames(annot2binNames, os.path.join(outputFastaFolder, binNamePro + "_BinNameToLineage.ann"))


def subProcessFilter(
    annotBinFolder: str,
    oriBinFolder: str,
    hmmOutFolder: str,
    outputFolder: str,
    bin_suffix: str,
    i: int,
    ratio_cutoff: float,
    acc_cutoff: float,
    estimate_completeness_threshold: float,
    seq_length_threshold: int,
    originalBinsCheckMPath,
    simulated_MAG=False
):
    binFiles = os.listdir(oriBinFolder)
    N = len(binFiles)
    for j, binFastaName in enumerate(binFiles):
        binName, suffix = os.path.splitext(binFastaName)
        hmmFilePath = os.path.join(hmmOutFolder, binName + ".HMM.txt")
        if suffix[1:] != bin_suffix:
            continue
        annotFile = binName + ".txt"
        statusStr = "          " + "{}, {} / {}".format(binFastaName, j + 1, N)
        cn = len(statusStr)
        if cn < 50:
            statusStr += "".join([" " for _ in range(50 - cn)])
        statusStr += "\r"
        sys.stderr.write("%s\r" % statusStr)
        sys.stderr.flush()
        filterContaminationOneBin(
            os.path.join(annotBinFolder, annotFile),
            os.path.join(oriBinFolder, binFastaName),
            hmmFilePath,
            os.path.join(outputFolder, index2Taxo[i + 1]),
            taxoLevel=i + 1,
            ratio_cutoff=ratio_cutoff,
            acc_cutoff=acc_cutoff,
            estimate_completeness_threshold=estimate_completeness_threshold,
            seq_length_threshold=seq_length_threshold,
            originalBinsCheckMPath = originalBinsCheckMPath,
            simulated_MAG=simulated_MAG
        )


def filterContaminationFolder(
    annotBinFolderInput: str,
    oriBinFolder: str,
    hmmOutFolder: str,
    outputFolder: str,
    bin_suffix: str,
    ratio_cutoff: float,
    acc_cutoff: float,
    estimate_completeness_threshold: float,
    seq_length_threshold: int,
    originalBinsCheckMPath: Union[str, None],
    simulated_MAG=False
):
    for i in range(6):
        if os.path.exists(os.path.join(outputFolder, index2Taxo[i + 1])) is False:
            os.mkdir(os.path.join(outputFolder, index2Taxo[i + 1]))
    res = []
    for i in range(6):
        res.append(
            Process(
                target=subProcessFilter,
                args=(
                    annotBinFolderInput,
                    oriBinFolder,
                    hmmOutFolder,
                    outputFolder,
                    bin_suffix,
                    i,
                    ratio_cutoff,
                    acc_cutoff,
                    estimate_completeness_threshold,
                    seq_length_threshold,
                    originalBinsCheckMPath,
                    simulated_MAG,
                ),
            )
        )
        res[-1].start()
    for p in res:
        p.join()
