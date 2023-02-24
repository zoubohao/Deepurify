import os
import subprocess
import sys
import time
from multiprocessing import Process, Queue
from typing import Dict, List, Union

import torch

from Deepurify.CallGenesTools.CallGenesUtils import callMarkerGenes
from Deepurify.FilterBinsTools.FilterUtils import filterContaminationFolder
from Deepurify.IOUtils import (getNumberOfPhylum, loadTaxonomyTree,
                               readVocabulary)
from Deepurify.LabelContigTools.LabelBinUtils import (buildTextsRepNormVector,
                                                      labelBinsFolder)
from Deepurify.Model.EncoderModels import SequenceCLIP
from Deepurify.SelectMAGsTools.SelectionUitls import findBestBinsAfterFiltering


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
    num_worker: int,
    overlapping_ratio: float,
    cutSeqLength: int,
    num_cpus_call_genes: int,
    ratio_cutoff: float,
    acc_cutoff: float,
    estimate_completeness_threshold: float,
    seq_length_threshold: int,
    checkM_parallel_num: int,
    num_cpus_per_checkm: int,
    dfsORgreedy: str,
    topK: int,
    model_config: Dict
):
    print("Labeling taxonomic lineage for each contig in bins...")
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
        with torch.no_grad():
            buildTextsRepNormVector(taxo_tree, model, taxo_vocabulary, "cpu", taxoName2RepNormVecPath)
    processList = []
    error_queue = Queue()
    if num_gpu == 0:
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
            processList[-1].start()
    else:
        assert sum(gpus_work_ratio) == 1.0
        for b in batch_size_per_gpu:
            assert b % num_worker == 0, "The batch size number in batch_size_per_gpu can not divide num_worker."
        gpus = ["cuda:" + str(i) for i in range(num_gpu)]
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
                        error_queue,
                        model_config,
                    ),
                )
            )
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

    filterOutputFolder = os.path.join(tempFileOutFolder, "FilterOutput")
    if os.path.exists(filterOutputFolder) is False:
        os.mkdir(filterOutputFolder)
    end1Time = time.clock_gettime(0)
    print("\n")
    print("Starting Call Genes...")
    temp_folder_path = os.path.join(tempFileOutFolder, "CalledGenes")
    if os.path.exists(temp_folder_path) is False:
        os.mkdir(temp_folder_path)
    trueInputBinFolder = inputBinFolder
    callMarkerGenes(trueInputBinFolder, temp_folder_path, num_cpus_call_genes, hmmModelPath, bin_suffix)
    print("\n")
    print("Starting Filter Contaminations and Separate Bins...")
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
    print("Starting Run CheckM...")
    runCheckMsingle(inputBinFolder, os.path.join(filterOutputFolder, "original_checkm.txt"), num_cpus_per_checkm * checkM_parallel_num, bin_suffix)
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


def cleanMAGs(
    input_bin_folder_path: str,
    output_bin_folder_path: str,
    bin_suffix: str,
    gpu_num: int,
    batch_size_per_gpu: int,
    num_worker: int,
    overlapping_ratio=0.5,
    cutSeqLength=8192,
    num_cpus_call_genes=64,
    hmm_acc_cutoff=0.7,
    hmm_align_ratio_cutoff=0.4,
    estimate_completeness_threshold=0.5,
    seq_length_threshold=550000,
    checkM_parallel_num=3,
    num_cpus_per_checkm=25,
    dfs_or_greedy="dfs",
    topK=3,
    temp_output_folder: Union[str, None] = None,
    output_bins_meta_info_path: Union[str, None] = None,
    info_files_path: Union[str, None] = None,
    modelWeightPath: Union[str, None] = None,
    taxoVocabPath: Union[str, None] = None,
    taxoTreePath: Union[str, None] = None,
    taxoName2RepNormVecPath: Union[str, None] = None,
    hmmModelPath: Union[str, None] = None,
    model_config: Union[Dict, None] = None
):
    """

    The main function to clean the MAGs.

    Args:
        input_bin_folder_path (str): The input path of MAGs.
        output_bin_folder_path (str): The output of clean MAGs.
        bin_suffix (str): The bin suffix of MAGs.
        gpu_num (int): The number of GPUs would be used. 0 means to use CPU. (ATTENTION: CPU is much slower !!!!)
        batch_size_per_gpu (int): The number of sequences would be loaded to one GPU. It is useless if gpu_num is 0.
        num_worker (int): The number of workers in one GPU or CPU. The batch size would divide this value for per worker.
        overlapping_ratio (float): The overlapping ratio if the length of contig exceeds the cutSeqLength. Defaults to 0.5.
        cutSeqLength (int, optional): The length to cut the contig if the length of it longer than this value. Defaults to 8192.
        num_cpus_call_genes (int, optional): The number of threads to call genes. Defaults to 64.
        hmm_acc_cutoff (float, optional): The threshold when the hmm model decides to treat the called gene's sequence as SCG. Defaults to 0.7.
        hmm_align_ratio_cutoff (float, optional): The threshold of alignment coverage when the called gene's sequence aligned to the SCG. Defaults to 0.4.
        estimate_completeness_threshold (float, optional): The threshold of estimated completeness for filtering bins generated by applying those SCGs. Defaults to 0.5.
        seq_length_threshold (int, optional): The threshold of a MAG's contigs' total length for filtering generated MAGs after applying SCGs.  Defaults to 550000.
        checkM_parallel_num (int, optional): The number of processes to run CheckM simultaneously. Defaults to 3.
        num_cpus_per_checkm (int, optional): The number of threads to run a CheckM process. Defaults to 25.
        dfs_or_greedy (str, optional): Depth first searching or greedy searching to label a contig. Defaults to "dfs".
        topK (int, optional): The Top-k nodes that have maximum cosine similarity with the contig encoded vector would be searched (Useless for greedy search). Defaults to 3.
        temp_output_folder (Union[str, None], optional): The path to store temporary files. Defaults to None.
        output_bins_meta_info_path (Union[str, None], optional): The path to record the meta informations of final clean MAGs. Defaults to None.
        info_files_path (Union[str, None]): The path of InfoFiles folder. Defaults to None.
        modelWeightPath (Union[str, None], optional): The path of model weight. (In InfoFiles folder) Defaults to None.
        taxoVocabPath (Union[str, None], optional): The path of taxon vocabulary. (In InfoFiles folder) Defaults to None.
        taxoTreePath (Union[str, None], optional): The path of taxonomic tree. (In InfoFiles folder) Defaults to None.
        taxoName2RepNormVecPath (Union[str, None], optional): The path of taxonomic lineage encoded vectors. (In InfoFiles folder) Defaults to None.
        hmmModelPath (Union[str, None], optional): The path of SCGs' hmm file. (In InfoFiles folder) Defaults to None.
        model_config (Union[Dict, None], optional): The config of model.
    """

    assert batch_size_per_gpu <= 20, "batch_size_per_gpu must smaller or equal with 20."
    assert num_worker <= 4, "num_worker must smaller or equal with 4."

    if "/" == input_bin_folder_path[-1]:
        input_bin_folder_path = input_bin_folder_path[0:-1]

    filesFolder = os.path.split(input_bin_folder_path)[0]
    if temp_output_folder is None:
        temp_output_folder = os.path.join(filesFolder, "TempOut")

    if output_bins_meta_info_path is None:
        output_bins_meta_info_path = os.path.join(output_bin_folder_path, "MetaInfo.txt")

    if gpu_num == 0:
        gpu_work_ratio = []
    else:
        gpu_work_ratio = [1.0 / gpu_num for _ in range(gpu_num - 1)]
        gpu_work_ratio = gpu_work_ratio + [1.0 - sum(gpu_work_ratio)]
    batch_size_per_gpu = [batch_size_per_gpu for _ in range(gpu_num)]

    if info_files_path is None:
        info_files_path = os.environ["DeepurifyInfoFiles"]
    if modelWeightPath is None:
        modelWeightPath = os.path.join(info_files_path, "CheckPoint", "Deepurify.ckpt")
    if taxoVocabPath is None:
        taxoVocabPath = os.path.join(info_files_path, "TaxonomyInfo",  "ProGenomesVocabulary.txt")
    if taxoTreePath is None:
        taxoTreePath = os.path.join(info_files_path, "TaxonomyInfo", "ProGenomesTaxonomyTree.pkl")
    if taxoName2RepNormVecPath is None:
        taxoName2RepNormVecPath = os.path.join(info_files_path, "PyObjs", "Deepurify_taxo_lineage_vector.pkl")
    if hmmModelPath is None:
        hmmModelPath = os.path.join(info_files_path, "HMM", "hmm_model.hmm")
    mer3Path = os.path.join(info_files_path, "3Mer_vocabulary.txt")
    mer4Path = os.path.join(info_files_path, "4Mer_vocabulary.txt")

    if os.path.exists(filesFolder) is False:
        print("Your input folder is not exist.")
        sys.exit(1)

    runLabelFilterSplitBins(
        inputBinFolder=input_bin_folder_path,
        tempFileOutFolder=temp_output_folder,
        outputBinFolder=output_bin_folder_path,
        outputBinsMetaFilePath=output_bins_meta_info_path,
        modelWeightPath=modelWeightPath,
        hmmModelPath=hmmModelPath,
        taxoVocabPath=taxoVocabPath,
        taxoTreePath=taxoTreePath,
        taxoName2RepNormVecPath=taxoName2RepNormVecPath,
        gpus_work_ratio=gpu_work_ratio,
        batch_size_per_gpu=batch_size_per_gpu,
        num_worker=num_worker,
        bin_suffix=bin_suffix,
        mer3Path=mer3Path,
        mer4Path=mer4Path,
        overlapping_ratio=overlapping_ratio,
        cutSeqLength=cutSeqLength,
        num_cpus_call_genes=num_cpus_call_genes,
        ratio_cutoff=hmm_align_ratio_cutoff,
        acc_cutoff=hmm_acc_cutoff,
        estimate_completeness_threshold=estimate_completeness_threshold,
        seq_length_threshold=seq_length_threshold,
        checkM_parallel_num=checkM_parallel_num,
        num_cpus_per_checkm=num_cpus_per_checkm,
        dfsORgreedy=dfs_or_greedy,
        topK=topK,
        model_config=model_config
    )


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
