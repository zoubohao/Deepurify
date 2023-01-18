import os
import subprocess
import sys
import torch
from multiprocessing import Process

from typing import List
from Deepurify.IOUtils import getNumberOfPhylum, loadTaxonomyTree, readVocabulary

from Deepurify.LabelContigTools.LabelBinUtils import buildTextsRepNormVector, labelBinsFolder
from Deepurify.CallGenesTools.CallGenesUtils import callMarkerGenes
from Deepurify.FilterBinsTools.FilterUtils import filterContaminationFolder
from Deepurify.Model.EncoderModels import SequenceCLIP
from Deepurify.SelectMAGsTools.SelectionUitls import findBestBinsAfterFiltering
import time


def runLabelFilterSplitBins(
    inputBinFolder: str,
    tempFileOutFolder: str,
    outputBinFolder: str,
    outputBinsMetaFilePath: str,
    modelWeightPath: str,
    hmmModelPath: str,
    taxoVocabPath: str,
    taxoTreePath: str,
    taxoName2RepNormVecPath: str,
    bin_suffix: str,
    mer3Path="./3Mer_vocabulary.txt",
    mer4Path="./4Mer_vocabulary.txt",
    gpus_work_ratio=list(),
    batch_size_per_gpu=list(),
    num_worker=2,
    overlapping_ratio=0.5,
    cutSeqLength=8192,
    num_cpus_call_genes=16,
    ratio_cutoff=0.4,
    acc_cutoff=0.6,
    estimate_completeness_threshold=0.5,
    seq_length_threshold=280000,
    checkM_parallel_num=3,
    num_cpus_per_checkm=25,
    dfsORgreedy="dfs",
    topK=3,
):
    print("Label each contig in bins...")
    print("\n")
    if os.path.exists(tempFileOutFolder) is False:
        os.mkdir(tempFileOutFolder)
    annotOutputFolder = os.path.join(tempFileOutFolder, "AnnotOutput")
    num_gpu = len(gpus_work_ratio)
    if os.path.exists(annotOutputFolder) is False:
        os.mkdir(annotOutputFolder)
    if os.path.exists(outputBinFolder) is False:
        os.mkdir(outputBinFolder)
    wht = open(os.path.join(outputBinFolder, "time.txt"), "w")
    startTime = time.clock_gettime(0)
    if os.path.exists(taxoName2RepNormVecPath) is False:
        modelConfig = {
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
        taxo_tree = loadTaxonomyTree(taxoTreePath)
        taxo_vocabulary = readVocabulary(taxoVocabPath)
        mer3_vocabulary = readVocabulary(mer3Path)
        mer4_vocabulary = readVocabulary(mer4Path)
        model = SequenceCLIP(
            max_model_len=modelConfig["max_model_len"],
            in_channels=modelConfig["inChannel"],
            taxo_dict_size=len(taxo_vocabulary),
            vocab_3Mer_size=len(mer3_vocabulary),
            vocab_4Mer_size=len(mer4_vocabulary),
            phylum_num=getNumberOfPhylum(taxo_tree),
            head_num=modelConfig["head_num"],
            d_model=modelConfig["d_model"],
            num_GeqEncoder=modelConfig["num_GeqEncoder"],
            num_lstm_layer=modelConfig["num_lstm_layers"],
            IRB_layers=modelConfig["IRB_num"],
            expand=modelConfig["expand"],
            feature_dim=modelConfig["feature_dim"],
            drop_connect_ratio=0.0,
            dropout=0.0,
        )
        print("DO NOT FIND taxoName2RepNormVecPath FILE. Start to build taxoName2RepNormVecPath file. ")
        with torch.no_grad():
            buildTextsRepNormVector(taxo_tree, model, taxo_vocabulary, "cpu", taxoName2RepNormVecPath)
    if num_gpu == 0:
        processList = []
        binFilesList = os.listdir(inputBinFolder)
        totalNum = len(binFilesList)
        nextIndex = 0
        for i in range(num_worker):
            if i != (num_worker) - 1:
                cutFileLength = totalNum // num_worker + 1
                curDataFilesList = binFilesList[nextIndex: nextIndex + cutFileLength]
                nextIndex += cutFileLength
            else:
                curDataFilesList = binFilesList[nextIndex:]
            print("Processer {} has {} files.".format(i, len(curDataFilesList)))
            processList.append(
                Process(
                    target=labelBinsFolder,
                    args=(
                        inputBinFolder,
                        annotOutputFolder,
                        "cpu",
                        modelWeightPath,
                        None,
                        mer3Path,
                        mer4Path,
                        taxoVocabPath,
                        taxoTreePath,
                        taxoName2RepNormVecPath,
                        8,
                        6,
                        bin_suffix,
                        curDataFilesList,
                        2,
                        overlapping_ratio,
                        cutSeqLength,
                        dfsORgreedy,
                        topK,
                    ),
                )
            )
            processList[-1].start()
        for p in processList:
            p.join()
    else:
        assert sum(gpus_work_ratio) == 1.0
        for b in batch_size_per_gpu:
            assert b % num_worker == 0, "The batch size number in batch_size_per_gpu can not divide num_worker."
        gpus = ["cuda:" + str(i) for i in range(num_gpu)]
        processList = []
        binFilesList = os.listdir(inputBinFolder)
        totalNum = len(binFilesList)
        nextIndex = 0
        for i in range(num_gpu * num_worker):
            if i != (num_gpu * num_worker) - 1:
                cutFileLength = int(totalNum * gpus_work_ratio[i // num_worker] / num_worker + 0.0) + 1
                curDataFilesList = binFilesList[nextIndex: nextIndex + cutFileLength]
                nextIndex += cutFileLength
            else:
                curDataFilesList = binFilesList[nextIndex:]
            print("Processer {} has {} files in device {} .".format(i, len(curDataFilesList), gpus[i // num_worker]))
            processList.append(
                Process(
                    target=labelBinsFolder,
                    args=(
                        inputBinFolder,
                        annotOutputFolder,
                        gpus[i // num_worker],
                        modelWeightPath,
                        None,
                        mer3Path,
                        mer4Path,
                        taxoVocabPath,
                        taxoTreePath,
                        taxoName2RepNormVecPath,
                        batch_size_per_gpu[i // num_worker] // num_worker,
                        6,
                        bin_suffix,
                        curDataFilesList,
                        2,
                        overlapping_ratio,
                        cutSeqLength,
                        dfsORgreedy,
                        topK,
                    ),
                )
            )
            processList[-1].start()
        for p in processList:
            p.join()
    filterOutputFolder = os.path.join(tempFileOutFolder, "FilterOutput")
    if os.path.exists(filterOutputFolder) is False:
        os.mkdir(filterOutputFolder)
    end1Time = time.clock_gettime(0)
    print("\n")
    print("Start Call Genes...")
    temp_folder_path = os.path.join(tempFileOutFolder, "CalledGenes")
    if os.path.exists(temp_folder_path) is False:
        os.mkdir(temp_folder_path)
    trueInputBinFolder = inputBinFolder
    callMarkerGenes(trueInputBinFolder, temp_folder_path, num_cpus_call_genes, hmmModelPath, bin_suffix)
    print("\n")
    print("Start Filter Contaminations and Split Bins...")
    filterContaminationFolder(
        annotOutputFolder,
        trueInputBinFolder,
        temp_folder_path,
        filterOutputFolder,
        None,
        bin_suffix,
        ratio_cutoff,
        acc_cutoff,
        estimate_completeness_threshold,
        seq_length_threshold,
    )
    print("\n")
    print("Start Run CheckM...")
    runCheckMsingle(inputBinFolder, os.path.join(filterOutputFolder, "original_checkm.txt"), num_cpus_per_checkm * checkM_parallel_num, bin_suffix)
    runCheckMParall(filterOutputFolder, bin_suffix, checkM_parallel_num, num_cpus_per_checkm)
    originalBinsCheckMPath = os.path.join(filterOutputFolder, "original_checkm.txt")
    print("\n")
    print("Start Gathering Result...")
    wh = open(outputBinsMetaFilePath, "w")
    binFilesList = os.listdir(inputBinFolder)
    tN = len(binFilesList)
    for i, binFileName in enumerate(binFilesList):
        statusStr = "        " + "{} / {}".format(i + 1, tN)
        cn = len(statusStr)
        if cn < 50:
            statusStr += "".join([" " for _ in range(50 - cn)])
        statusStr += "\r"
        sys.stderr.write("%s\r" % statusStr)
        sys.stderr.flush()
        _, suffix = os.path.splitext(binFileName)
        if suffix[1:] == bin_suffix:
            outInfo = findBestBinsAfterFiltering(
                binFileName, inputBinFolder, tempFileOutFolder, originalBinsCheckMPath, outputBinFolder
            )
            for outName, qualityValues, annotName in outInfo:
                wh.write(
                    str(outName)
                    + "\t"
                    + str(qualityValues[0])
                    + "\t"
                    + str(qualityValues[1])
                    + "\t"
                    + str(qualityValues[2])
                    + "\t"
                    + annotName
                    + "\n"
                )
    wh.close()
    print()
    end2Time = time.clock_gettime(0)
    wht.write(str(end1Time - startTime) + "\t" + str(end2Time - startTime) + "\n")
    wht.close()


### CheckM ###
def runCheckMsingle(binsFolder: str, checkmResFilePath: str, num_cpu: int, bin_suffix: str, repTime=0):
    if os.path.exists(checkmResFilePath):
        return
    res = subprocess.Popen(
        " checkm lineage_wf "
        + " -t "
        + str(num_cpu)
        + " --pplacer_threads "
        + str(num_cpu)
        + " -x "
        + bin_suffix
        + " -f "
        + checkmResFilePath
        + "  "
        + binsFolder
        + "  "
        + os.path.join(binsFolder, "checkmTempOut"),
        shell=True,
    )
    while res.poll() is None:
        if res.wait() != 0:
            print("CheckM running error has occur, we try again. Repeat time: ", repTime)
            if repTime >= 1:
                print("############################################")
                print("### Error Occured During CheckM Running  ###")
                print("############################################")
                raise RuntimeError("binFolder: {}, Checkm Result Path: {}".format(binsFolder, checkmResFilePath))
            runCheckMsingle(binsFolder, checkmResFilePath, num_cpu, bin_suffix, repTime + 1)
    # res.wait()
    res.kill()


index2Taxo = {1: "phylum_filter", 2: "class_filter", 3: "order_filter", 4: "family_filter", 5: "genus_filter", 6: "species_filter"}


def runCheckMForSixFilter(filterFolder, indexList: List, num_checkm_cpu: int, bin_suffix: str):
    for i in indexList:
        binsFolder = os.path.join(filterFolder, index2Taxo[i])
        checkMPath = os.path.join(filterFolder, index2Taxo[i].split("_")[0] + "_checkm.txt")
        runCheckMsingle(binsFolder, checkMPath, num_checkm_cpu, bin_suffix)


def runCheckMParall(filterFolder, bin_suffix, num_pall, num_cpu=40):
    assert 1 <= num_pall <= 6
    res = []
    indices = [1, 2, 3, 4, 5, 6]
    step = 6 // num_pall
    for i in range(num_pall):
        p = Process(
            target=runCheckMForSixFilter,
            args=(
                filterFolder,
                indices[step * i: step * (i + 1)],
                num_cpu,
                bin_suffix,
            ),
        )
        res.append(p)
        p.start()
    for p in res:
        p.join()


### GUNC ###
# def runGUNCsingle(binsFolder: str, outputFolder: str, num_cpu: int, bin_suffix: str):
#     res = subprocess.Popen(
#         "gunc run --threads " + str(num_cpu) + " --file_suffix  " + str(bin_suffix) + " --input_dir " + binsFolder + " --out_dir " + outputFolder,
#         shell=True,
#     )
#     res.wait()
#     res.kill()


# index2Taxo = {1: "phylum_filter", 2: "class_filter", 3: "order_filter", 4: "family_filter", 5: "genus_filter", 6: "species_filter"}


# def runGUNCForSixFilter(filterFolder, indexList: List, num_checkm_cpu: int, bin_suffix: str, guncOutFolder: str):
#     for i in indexList:
#         binsFolder = os.path.join(filterFolder, index2Taxo[i])
#         outFolder = os.path.join(guncOutFolder, "GUNC_" + index2Taxo[i].split("_")[0])
#         if os.path.exists(outFolder) is False:
#             os.makedirs(outFolder)
#         runGUNCsingle(binsFolder, outFolder, num_checkm_cpu, bin_suffix)


# def runGUNCParall(filterFolder, guncOutFolder, bin_suffix, num_pall, num_cpu=40):
#     assert 1 <= num_pall <= 6
#     res = []
#     indices = [1, 2, 3, 4, 5, 6]
#     step = 6 // num_pall
#     for i in range(num_pall):
#         p = Process(
#             target=runGUNCForSixFilter,
#             args=(filterFolder, indices[step * i: step * (i + 1)], num_cpu, bin_suffix, guncOutFolder),
#         )
#         res.append(p)
#         p.start()
#     for p in res:
#         p.join()
