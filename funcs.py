import os
from shutil import copy, rmtree
import subprocess
import sys
import time
from multiprocessing import Process, Queue
from typing import Dict, List

import torch

from Deepurify.CallGenesTools.CallGenesUtils import callMarkerGenes
from Deepurify.FilterBinsTools.FilterUtils import filterContaminationFolder
from Deepurify.IOUtils import (getNumberOfPhylum, loadTaxonomyTree, readCheckMResultAndStat,
                               readVocabulary, writePickle)
from Deepurify.LabelContigTools.LabelBinUtils import (buildTextsRepNormVector,
                                                      labelBinsFolder)
from Deepurify.Model.EncoderModels import SequenceCLIP
from Deepurify.SelectMAGsTools.SelectionUitls import findBestBinsAfterFiltering
from Deepurify.CallGenesTools.CallGenesUtils import splitListEqually


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
    mer3Path: str,
    mer4Path: str,
    gpus_work_ratio: List[float],
    batch_size_per_gpu: List[float],
    num_worker_per_device: int,
    overlapping_ratio: float,
    cutSeqLength: int,
    num_cpus_call_genes: int,
    ratio_cutoff: float,
    acc_cutoff: float,
    estimated_completeness_threshold: float,
    seq_length_threshold: int,
    checkM_parallel_num: int,
    num_cpus_per_checkm: int,
    dfsORgreedy: str,
    topK: int,
    model_config: Dict,
    stop_at_step2=False
):
    print("Labeling taxonomic lineage for each contig in bins...")
    print()
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
        taxo_tree = loadTaxonomyTree(taxoTreePath)
        taxo_vocabulary = readVocabulary(taxoVocabPath)
        mer3_vocabulary = readVocabulary(mer3Path)
        mer4_vocabulary = readVocabulary(mer4Path)
        model = SequenceCLIP(
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
        print("DO NOT FIND taxoName2RepNormVecPath FILE. Start to build taxoName2RepNormVecPath file. ")
        state = torch.load(modelWeightPath, map_location="cpu")
        model.load_state_dict(state, strict=True)
        with torch.no_grad():
            buildTextsRepNormVector(taxo_tree, model, taxo_vocabulary, "cpu", taxoName2RepNormVecPath)
    processList = []
    error_queue = Queue()
    if num_gpu == 0:
        binFilesList = os.listdir(inputBinFolder)
        totalNum = len(binFilesList)
        nextIndex = 0
        for i in range(num_worker_per_device):
            if i != (num_worker_per_device) - 1:
                cutFileLength = totalNum // num_worker_per_device + 1
                curDataFilesList = binFilesList[nextIndex: nextIndex + cutFileLength]
                nextIndex += cutFileLength
            else:
                curDataFilesList = binFilesList[nextIndex:]
            processList.append(
                Process(
                    target=labelBinsFolder,
                    args=(
                        inputBinFolder,
                        annotOutputFolder,
                        "cpu",
                        modelWeightPath,
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
                        error_queue,
                        model_config,
                    ),
                )
            )
            print("Processer {} has {} files.".format(i, len(curDataFilesList)))
            processList[-1].start()
    else:
        assert sum(gpus_work_ratio) == 1.0
        for b in batch_size_per_gpu:
            assert b % num_worker_per_device == 0, "The batch size number in batch_size_per_gpu can not divide num_worker_per_device."
        gpus = ["cuda:" + str(i) for i in range(num_gpu)]
        binFilesList = os.listdir(inputBinFolder)
        totalNum = len(binFilesList)
        nextIndex = 0
        for i in range(num_gpu * num_worker_per_device):
            if i != (num_gpu * num_worker_per_device) - 1:
                cutFileLength = int(totalNum * gpus_work_ratio[i //
                                    num_worker_per_device] / num_worker_per_device + 0.0) + 1
                curDataFilesList = binFilesList[nextIndex: nextIndex + cutFileLength]
                nextIndex += cutFileLength
            else:
                curDataFilesList = binFilesList[nextIndex:]
            processList.append(
                Process(
                    target=labelBinsFolder,
                    args=(
                        inputBinFolder,
                        annotOutputFolder,
                        gpus[i // num_worker_per_device],
                        modelWeightPath,
                        mer3Path,
                        mer4Path,
                        taxoVocabPath,
                        taxoTreePath,
                        taxoName2RepNormVecPath,
                        batch_size_per_gpu[i // num_worker_per_device] // num_worker_per_device,
                        6,
                        bin_suffix,
                        curDataFilesList,
                        2,
                        overlapping_ratio,
                        cutSeqLength,
                        dfsORgreedy,
                        topK,
                        error_queue,
                        model_config,
                    ),
                )
            )
            print("Processer {} has {} files in device {} .".format(
                i, len(curDataFilesList), gpus[i // num_worker_per_device]))
            processList[-1].start()

    # error collection
    queue_len = 0
    n = len(processList)
    while True:
        if not error_queue.empty():
            flag = error_queue.get()
            queue_len += 1
            if flag != None:
                for p in processList:
                    p.terminate()
                    p.join()
                print("\n")
                print("SOME ERROR DURING INFERENCE CONTIG LINEAGE WITH CPU OR GPU.")
                sys.exit(1)
            if queue_len >= n:
                for p in processList:
                    p.join()
                break

    end1Time = time.clock_gettime(0)

    filterOutputFolder = os.path.join(tempFileOutFolder, "FilterOutput")
    if os.path.exists(filterOutputFolder) is False:
        os.mkdir(filterOutputFolder)

    temp_folder_path = os.path.join(tempFileOutFolder, "CalledGenes")
    if os.path.exists(temp_folder_path) is False:
        os.mkdir(temp_folder_path)

    if stop_at_step2 is False:
        print("\n")
        print("Starting Call Genes...")
        callMarkerGenes(inputBinFolder, temp_folder_path, num_cpus_call_genes, hmmModelPath, bin_suffix)

    print("\n")
    print("Starting Filter Contaminations and Separate Bins...")
    filterContaminationFolder(
        annotOutputFolder,
        inputBinFolder,
        temp_folder_path,
        filterOutputFolder,
        bin_suffix,
        ratio_cutoff,
        acc_cutoff,
        estimated_completeness_threshold,
        seq_length_threshold,
        stop_at_step2=stop_at_step2
    )

    if stop_at_step2:
        print()
        return

    print("\n")
    print("Starting Run CheckM...")
    runCheckMsingle(inputBinFolder, os.path.join(filterOutputFolder, "original_checkm.txt"),
                    num_cpus_per_checkm * checkM_parallel_num, bin_suffix)
    runCheckMParall(filterOutputFolder, bin_suffix, checkM_parallel_num, num_cpus_per_checkm)
    originalBinsCheckMPath = os.path.join(filterOutputFolder, "original_checkm.txt")

    print("\n")
    print("Starting Gather Result...")
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
    print("\n")
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


index2Taxo = {1: "phylum_filter", 2: "class_filter", 3: "order_filter",
              4: "family_filter", 5: "genus_filter", 6: "species_filter"}


def runCheckMForSixFilter(filterFolder, indexList: List, num_checkm_cpu: int, bin_suffix: str):
    for i in indexList:
        binsFolder = os.path.join(filterFolder, index2Taxo[i])
        files = os.listdir(binsFolder)
        n = 0
        copyList = []
        for file in files:
            if os.path.splitext(file)[1][1:] == bin_suffix:
                n += 1
                copyList.append(file)

        k = n // 1000 + 1
        equalFilesList = splitListEqually(copyList, k)
        for j, equal_files in enumerate(equalFilesList):
            splitFolder = os.path.join(binsFolder, str(j))
            if os.path.exists(splitFolder) is False:
                os.mkdir(splitFolder)
            for file in equal_files:
                if os.path.exists(os.path.join(splitFolder, file)) is False:
                    copy(os.path.join(binsFolder, file), splitFolder)

        for j in range(k):
            splitFolder = os.path.join(binsFolder, str(j))
            checkMPath = os.path.join(filterFolder, index2Taxo[i].split("_")[0] + "_" + str(j) + "_checkm.txt")
            runCheckMsingle(splitFolder, checkMPath, num_checkm_cpu, bin_suffix)

        terCheckMres = {}
        for j in range(k):
            checkMPath = os.path.join(filterFolder, index2Taxo[i].split("_")[0] + "_" + str(j) + "_checkm.txt")
            thisCh = readCheckMResultAndStat(checkMPath)[0]
            for key, val in thisCh.items():
                terCheckMres[key] = val

        checkMPath = os.path.join(filterFolder, index2Taxo[i].split("_")[0] + "_checkm.pkl")
        writePickle(checkMPath, terCheckMres)

        for j in range(k):
            rmtree(os.path.join(binsFolder, str(j)), ignore_errors=True)


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
