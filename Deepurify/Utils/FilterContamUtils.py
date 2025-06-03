import math
import os
from copy import deepcopy
from multiprocessing import Process
from typing import Dict, List, Set, Tuple

import numpy as np
import psutil
from func_timeout import FunctionTimedOut, func_timeout
from sklearn.cluster import KMeans

from Deepurify.Utils.CallGenesUtils import splitListEqually
from Deepurify.Utils.HmmUtils import getHMMModels, processHits
from Deepurify.Utils.IOUtils import (progressBar, readAnnotResult, readFasta,
                                     readHMMFileReturnDict, readPickle,
                                     writeAnnot2BinNames, writeFasta)
from Deepurify.Utils.KMeans import COPKMeans
from Deepurify.Utils.LabelBinsUtils import getBestMultiLabelsForFiltering

index2Taxo = {
    0: "T0_filter",
    1: "T1_filter",
    2: "T2_filter",
    3: "T3_filter",
    4: "T4_filter",
    5: "T5_filter",
    6: "T6_filter"}

# os.environ["OMP_NUM_THREADS"] = "1"

def summedLengthCal(name2seq: Dict[str, str]) -> int:
    return sum(len(seq) for seq in name2seq.values())


def allocate(
    splitContigSetList: List[Set[str]],
    splitRecordGenes: List[Dict[str, int]],
    info: Tuple[str, Dict[str, int]],
    replication_times_threshold: int,
) -> None:
    if len(splitContigSetList) == 0:
        curSet = set()
        curSet.add(info[0])
        splitContigSetList.append(curSet)
        curDict = dict()
        curDict.update(info[1])
        splitRecordGenes.append(curDict)
    else:
        insertIndex = None
        for i, record in enumerate(splitRecordGenes):
            if_insert = True
            for gene, num in info[1].items():
                if gene in record:
                    recordNum = record[gene]
                    if (recordNum + num) > replication_times_threshold:
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
            curSet = set()
            curSet.add(info[0])
            splitContigSetList.append(curSet)
            curDict = dict()
            curDict.update(info[1])
            splitRecordGenes.append(curDict)


def summedRecord(recordList):
    summedValue = 0.0
    for num, _ in recordList:
        summedValue += num
    return summedValue


def cluster_kmeans(bin_cluster_num, X, cl, length_weights):
    kmeans_model = None
    try:
        kmeans_model = COPKMeans(bin_cluster_num)
        func_timeout(60 * 3, kmeans_model.fit, args=(X, None, [], cl,))
    except FunctionTimedOut:
        kmeans_model = KMeans(bin_cluster_num, n_init=20, max_iter=600)
        kmeans_model.fit(X, None, length_weights)
    except ValueError:
        kmeans_model = KMeans(bin_cluster_num, n_init=20, max_iter=600)
        kmeans_model.fit(X, None, length_weights)
    except:
        kmeans_model = KMeans(bin_cluster_num, n_init=20, max_iter=600)
        kmeans_model.fit(X, None, length_weights)
    return kmeans_model


# original split
def cluster_split(
    sub_contigName2seq: Dict[str, str],
    contigName2RepNormV,
    gene2contigNames: Dict[str, List[str]],
    contigName2_gene2num: Dict[str, Dict[str, int]]
) -> List[Dict[str, str]]:
    contigSeqPair = [(contigName, len(seq)) for contigName, seq in sub_contigName2seq.items()]
    if len(contigSeqPair) == 1:
        return [sub_contigName2seq]

    exist_contigs = [
        contig for contig, _ in sorted(contigSeqPair, key=lambda x: x[1], reverse=True)
    ]
    existGene2contigNames = {}  # subset of gene2contigNames
    existcontig2_gene2num = []
    existContig2RepNormV = {}
    notExistGeneContig = set()
    notExistGeneContig2seq = {}

    # find the exist genes in those input contigs
    for contig in exist_contigs:
        existContig2RepNormV[contig] = contigName2RepNormV[contig]
        if contig in contigName2_gene2num:
            curExistGenes2num = contigName2_gene2num[contig]
            existcontig2_gene2num.append((contig, deepcopy(curExistGenes2num)))
            for gene, _ in curExistGenes2num.items():
                if gene not in existGene2contigNames:
                    cur_list = []
                    for cur_contigName in gene2contigNames[gene]:
                        if cur_contigName in sub_contigName2seq:
                            cur_list.append(cur_contigName)
                    existGene2contigNames[gene] = cur_list
        else:
            notExistGeneContig.add(contig)
            notExistGeneContig2seq[contig] = deepcopy(sub_contigName2seq[contig])

    # go through contigs one by one
    splitContigSetList = []
    splitRecordGenes = []
    for info in existcontig2_gene2num:
        allocate(splitContigSetList, splitRecordGenes, info, 1)

    if len(splitContigSetList) == 0:
        return [notExistGeneContig2seq]
    
    bin_cluster_num = len(splitContigSetList) - 1
    if bin_cluster_num <= 1:
        # the bins number is smaller than 2.
        # only split by contigs
        totalN = len(existGene2contigNames)
        filteredContigList = []
        for i in range(len(splitContigSetList)):
            curNumGenes = len(splitRecordGenes[i])
            curSet = splitContigSetList[i].union(notExistGeneContig)
            curContig2seq = {}
            summedLength = 0.0
            for contigName in curSet:
                curContig2seq[contigName] = deepcopy(sub_contigName2seq[contigName])
                summedLength += len(sub_contigName2seq[contigName])
            ratio = curNumGenes / totalN + 0.0
            score = curNumGenes / totalN + 0.0 + math.log(summedLength) / 20.0
            filteredContigList.append((curContig2seq, ratio, score))
        filteredContigList = sorted(filteredContigList, key=lambda x: x[-1], reverse=True)
        return [infoPair[0] for i, infoPair in enumerate(filteredContigList)]

    # build can not link paris and X array
    X = []
    length_weights = []
    contigName2index = {}
    index2contigName = {}
    for j, (contigName, repNormVec) in enumerate(existContig2RepNormV.items()):
        X.append(repNormVec)
        contigName2index[contigName] = j
        index2contigName[j] = contigName
        length_weights.append(np.log(len(sub_contigName2seq[contigName])))
    X = np.array(X, dtype=np.float32)
    cl = []
    for _, contigsList in existGene2contigNames.items():
        for i in range(len(contigsList)):
            for j in range(i + 1, len(contigsList)):
                cl.append((contigName2index[contigsList[i]], contigName2index[contigsList[j]]))
    
    kmeans_model = None
    try:
        kmeans_model = cluster_kmeans(bin_cluster_num, X, cl, length_weights)
    except FunctionTimedOut:
        kmeans_model = KMeans(bin_cluster_num, n_init=20, max_iter=600)
        kmeans_model.fit(X, None, length_weights)
    except ValueError:
        kmeans_model = KMeans(bin_cluster_num, n_init=20, max_iter=600)
        kmeans_model.fit(X, None, length_weights)
    except:
        kmeans_model = KMeans(bin_cluster_num, n_init=20, max_iter=600)
        kmeans_model.fit(X, None, length_weights)
    
    cluster_out = {}
    for i, label in enumerate(kmeans_model.labels_):
        contigName = index2contigName[i]
        if label not in cluster_out:
            cur_name2seq = {}
            cur_name2seq[contigName] = sub_contigName2seq[contigName]
            cluster_out[label] = cur_name2seq
        else:
            cur_name2seq = cluster_out[label]
            cur_name2seq[contigName] = sub_contigName2seq[contigName]
    # result collection
    res = []
    for _, name2seq in cluster_out.items():
        res.append((name2seq, summedLengthCal(name2seq)))
    res_ord = []
    for name2seq, _ in list(sorted(res, key=lambda x: x[-1], reverse=True)):
        res_ord.append(name2seq)
    return res_ord


def adjust(contigName2annot, coreNames):
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


def oriFilterOneBin(
        binFastaPath: str,
        all_contigName2hits,
        hmmAcc2model,
        phy2accs_list,
        contigName2RepNormV,
        outputFastaFolder,
        seq_length_threshold: int):
    contigName2seq = readFasta(binFastaPath)
    gene2contigList, contigName2_gene2num = processHits(all_contigName2hits, hmmAcc2model, set(phy2accs_list["UNK"]))
    binName = os.path.split(binFastaPath)[-1]
    binNamePro, bin_suffix = os.path.splitext(binName)
    idx_k = 0
    filtedContigName2seqList = cluster_split(
        contigName2seq,
        contigName2RepNormV,
        gene2contigList,
        contigName2_gene2num
    )
    for i, coreName2seqFilter in enumerate(filtedContigName2seqList):
        if i == 0 or summedLengthCal(coreName2seqFilter) >= seq_length_threshold:
            out_name = f"{binNamePro}___s___{idx_k}{bin_suffix}"
            writeFasta(coreName2seqFilter, os.path.join(outputFastaFolder, out_name))
            idx_k += 1


def filterContaminationOneBin(
    annotBinPath: str,
    binFastaPath: str,
    all_contigName2hits,
    hmmAcc2model,
    phy2accs_list,
    contigName2RepNormV,
    outputFastaFolder: str,
    taxoLevel: int,
    seq_length_threshold: int,
    simulated_MAG=False
) -> None:
    assert 1 <= taxoLevel <= 6, ValueError("The taxoLevel must between 1 to 6.")

    contigName2seq = readFasta(binFastaPath)
    contigName2annot, contigName2probs = readAnnotResult(annotBinPath)
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
    if taxoLevel == 6:
        coreNames = adjust(contigName2annot, coreNames)

    filteredContigName2seq = {}
    annot2_contigName2seq = {}

    for key, seq in contigName2seq.items():
        for coreName in coreNames:
            if coreName in contigName2annot[key]:
                filteredContigName2seq[key] = seq

        if key not in filteredContigName2seq:
            curAnnot = "@".join(contigName2annot[key].split("@")[0: taxoLevel])
            if curAnnot not in annot2_contigName2seq:
                newDict = dict()
                newDict[key] = seq
                annot2_contigName2seq[curAnnot] = newDict
            else:
                curDict = annot2_contigName2seq[curAnnot]
                curDict[key] = seq

    if simulated_MAG:
        binName = os.path.split(binFastaPath)[-1]
        writeFasta(filteredContigName2seq, os.path.join(outputFastaFolder, binName))
        return 

    # using SCGs to exclude external contigs.
    idx_k = 0
    cur_core_name = coreNames[0]
    phy = cur_core_name.split("@")[0]
    gene2contigList, contigName2_gene2num = processHits(all_contigName2hits, hmmAcc2model, set(phy2accs_list[phy]))
    annot2binNames = {}
    binName = os.path.split(binFastaPath)[-1]
    binNamePro, bin_suffix = os.path.splitext(binName)
    out_name = f"{binNamePro}___o___{idx_k}{bin_suffix}"
    writeFasta(filteredContigName2seq, os.path.join(outputFastaFolder, out_name))
    idx_k += 1
    annot2binNames[cur_core_name] = [out_name]
    filteredContigName2seqList = cluster_split(
            filteredContigName2seq, contigName2RepNormV, 
            gene2contigList, contigName2_gene2num)

    for i, coreName2seqFilter in enumerate(filteredContigName2seqList):
        if i == 0 or summedLengthCal(coreName2seqFilter) >= seq_length_threshold:
            out_name = binNamePro + "___s___" + str(idx_k) + bin_suffix
            if cur_core_name not in annot2binNames:
                annot2binNames[cur_core_name] = [out_name]
            else:
                annot2binNames[cur_core_name].append(out_name)
            writeFasta(
                coreName2seqFilter,
                os.path.join(outputFastaFolder, out_name))
            idx_k += 1

    for annot, noCoreContigName2seq in annot2_contigName2seq.items():
        if summedLengthCal(noCoreContigName2seq) >= seq_length_threshold:
            out_name = binNamePro + "___o___" + str(idx_k) + bin_suffix
            if annot not in annot2binNames:
                annot2binNames[annot] = [out_name]
            else:
                annot2binNames[annot].append(out_name)
            writeFasta(noCoreContigName2seq, os.path.join(outputFastaFolder, out_name))
            idx_k += 1

        phy = annot.split("@")[0]
        gene2contigList, contigName2_gene2num = processHits(all_contigName2hits, hmmAcc2model, set(phy2accs_list[phy]))
        curFilteredList = cluster_split(
                noCoreContigName2seq, contigName2RepNormV, 
                gene2contigList, contigName2_gene2num)

        for noCoreName2seqFilter in curFilteredList:
            if summedLengthCal(noCoreName2seqFilter) >= seq_length_threshold:
                out_name = binNamePro + "___s___" + str(idx_k) + bin_suffix
                if annot not in annot2binNames:
                    annot2binNames[annot] = [out_name]
                else:
                    annot2binNames[annot].append(out_name)
                writeFasta(
                    noCoreName2seqFilter,
                    os.path.join(outputFastaFolder, out_name))
                idx_k += 1
    writeAnnot2BinNames(annot2binNames, os.path.join(outputFastaFolder, binNamePro + "_BinNameToLineage.ann"))


def subProcessFilter(
    oriBinFolder: str,
    annotBinFolder: str,
    binFiles: str,
    hmmOutFolder: str,
    outputFolder: str,
    bin_suffix: str,
    hmmAcc2model,
    phy2accs_list,
    contigName2RepNormV,
    j: int,
    seq_length_threshold: int,
    simulated_MAG=False
):
    N = len(binFiles)
    if N == 0:
        return
    for k, binFastaName in enumerate(binFiles):
        binName, suffix = os.path.splitext(binFastaName)
        if suffix[1:] != bin_suffix:
            continue
        if simulated_MAG is False:
            all_contigName2hits = readHMMFileReturnDict(os.path.join(hmmOutFolder, f"{binName}.HMM.txt"))
        else:
            all_contigName2hits = None
        annotFile = binName + ".txt"
        if j == 0:
            oriFilterOneBin(
                os.path.join(oriBinFolder, binFastaName),
                all_contigName2hits,
                hmmAcc2model,
                phy2accs_list,
                contigName2RepNormV,
                os.path.join(outputFolder, index2Taxo[j]),
                seq_length_threshold
            )
        else:
            filterContaminationOneBin(
                os.path.join(annotBinFolder, annotFile),
                os.path.join(oriBinFolder, binFastaName),
                all_contigName2hits,
                hmmAcc2model,
                phy2accs_list,
                contigName2RepNormV,
                os.path.join(outputFolder, index2Taxo[j]),
                j,
                seq_length_threshold,
                simulated_MAG
            )
        progressBar(k, N)


def filterContaminationFolder(
    annotBinFolderInput: str,
    oriBinFolder: str,
    hmmOutFolder: str,
    outputFolder: str,
    hmmModelPath: str,
    phy2accsPath: str,
    contigName2RepNormPath,
    bin_suffix: str,
    seq_length_threshold: int,
    simulated_MAG=False,
    cpu_num = None
):
    if cpu_num is None:
        cpu_num = psutil.cpu_count() + 8
    for i in range(7):
        if os.path.exists(os.path.join(outputFolder, index2Taxo[i])) is False:
            os.mkdir(os.path.join(outputFolder, index2Taxo[i]))
    res = []
    if simulated_MAG is False:
        hmmAcc2model = getHMMModels(hmmModelPath)
    else:
        hmmAcc2model = None

    if phy2accsPath is not None:
        phy2accs_list = readPickle(phy2accsPath)
    else:
        phy2accs_list = None
    
    if contigName2RepNormPath is not None:
        contigName2RepNormV = readPickle(contigName2RepNormPath)
    else:
        contigName2RepNormV = None

    binFiles = os.listdir(oriBinFolder)
    one_folder_cpu_num = cpu_num // 6 + 1
    binFiles_equal_list = splitListEqually(binFiles, one_folder_cpu_num)
    k = 0
    if simulated_MAG:
        process_indices_list = list(range(1, 7))
    else:
        process_indices_list = list(range(7))
    
    skip = True
    for i in process_indices_list:
        qu_path = os.path.join(outputFolder, index2Taxo[i] + "_checkm2_res", "quality_report.tsv")
        if os.path.exists(qu_path) is False:
            skip = False
            break
    if skip:
        return skip
    
    for i in range(len(binFiles_equal_list)):
        for j in process_indices_list:
            p = Process(
                    target=subProcessFilter,
                    args=(
                        oriBinFolder,
                        annotBinFolderInput,
                        binFiles_equal_list[i],
                        hmmOutFolder,
                        outputFolder,
                        bin_suffix,
                        hmmAcc2model,
                        phy2accs_list,
                        contigName2RepNormV,
                        j,
                        seq_length_threshold,
                        simulated_MAG,))
            p.start()
            res.append(p)
            k += 1
    for p in res:
        p.join()
    
    return False
    
    
    
    