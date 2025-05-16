import os
import sys
from multiprocessing import Process
import multiprocessing as mp
from shutil import rmtree
from typing import Dict, List, Union

import psutil
import torch
import time

from Deepurify.Utils.BuildFilesUtils import (build_taxonomic_file,
                                             buildAllConcatFiles,
                                             buildSubFastaFile)
from Deepurify.Utils.CallGenesUtils import callMarkerGenes
from Deepurify.Utils.FilterContamUtils import filterContaminationFolder
from Deepurify.Utils.IOUtils import (progressBar, readFasta, write_result,
                                     writeFasta)
from Deepurify.Utils.LabelBinsUtils import (estimateContigSimFromFile,
                                            estimateContigSimInBinsPall)
from Deepurify.Utils.RunCMDUtils import (buildCheckm2TmpFilesParall, cal_depth,
                                         runCheckm2Reuse, runCheckm2Single,
                                         runCONCOCT, runMetaBAT2, runSemibin)
from Deepurify.Utils.SelectBinsUitls import findBestBinsAfterFiltering


def run_all_deconta_steps(
    inputBinFolder: str,
    tempFileOutFolder: str,
    outputBinFolder: str,
    modelWeightPath: str,
    taxoVocabPath: str,
    taxoTreePath: str,
    taxoName2RepNormVecPath: str,
    hmmModelPath: str,
    phy2accsPath: str,
    bin_suffix: str,
    mer3Path: str,
    mer4Path: str,
    checkm2_db_path: str,
    gpus_work_ratio: List[float],
    batch_size_per_gpu: List[float],
    each_gpu_threads: int,
    overlapping_ratio: float,
    cut_seq_length: int,
    seq_length_threshold: int,
    topkORgreedy: str,
    topK: int,
    model_config: Union[Dict, None] = None,
    num_process = 64,
    build_concat_file=False,
    concat_annot_file_path = None,
    concat_vec_file_path = None,
    concat_TNF_vector_path = None,
    concat_contig_file_path = None,
    simulated_MAG=False,
    just_annot=False,
    
):
    mp.set_start_method("spawn", force=True)
    if os.path.exists(outputBinFolder) is False:
        os.mkdir(outputBinFolder)
    if os.path.exists(tempFileOutFolder) is False:
        os.mkdir(tempFileOutFolder)
    annotOutputFolder = os.path.join(tempFileOutFolder, "AnnotOutput")
    if os.path.exists(annotOutputFolder) is False:
        os.mkdir(annotOutputFolder)
    
    # check if there exist bins in input
    # check if there exist space in the contig names.
    N = 0
    binsNames = os.listdir(inputBinFolder)
    filtered_space_output_bin_folder = os.path.join(tempFileOutFolder, "filtered_space_in_contig_names_bins")
    has_space = False
    for binName in binsNames:
        pro, suffix = os.path.splitext(binName)
        if suffix[1:] == bin_suffix:
            # check if there exist space in the contig names.
            with open(os.path.join(inputBinFolder, binName), "r") as rh:
                for line in rh:
                    oneline = line.strip("\n")
                    if ">" in oneline and " " in oneline:
                        has_space = True
            N += 1
    if N == 0:
        raise ValueError("No bins in input folder.")
    if has_space:
        if os.path.exists(filtered_space_output_bin_folder) is False:
            os.mkdir(filtered_space_output_bin_folder)
        for binName in binsNames:
            pro, suffix = os.path.splitext(binName)
            if suffix[1:] == bin_suffix:
                with open(os.path.join(inputBinFolder, binName), "r") as rh, \
                    open(os.path.join(filtered_space_output_bin_folder, binName), "w") as wh:
                    for line in rh:
                        oneline = line.strip("\n")
                        if ">" in oneline and " " in oneline:
                            oneline = "_".join(oneline.split())
                        wh.write(oneline + "\n")
        inputBinFolder = filtered_space_output_bin_folder
    
    # TODO: BUG in this line.
    # after run build_taxonomic_file function, the following codes would be locked.
    if os.path.exists(taxoName2RepNormVecPath) is False:
        build_taxonomic_file(
            taxoTreePath,
            taxoVocabPath,
            mer3Path,
            mer4Path, modelWeightPath,
            taxoName2RepNormVecPath,
            model_config)
    
    print("========================================================================")
    print("--> Start to Estimate Taxonomic Similarity of Contigs in the MAG...")
    start_time = time.time()
    if build_concat_file:
        estimateContigSimInBinsPall(
            inputBinFolder,
            annotOutputFolder,
            modelWeightPath,
            mer3Path,
            mer4Path,
            taxoVocabPath,
            taxoTreePath,
            bin_suffix,
            overlapping_ratio,
            cut_seq_length,
            topkORgreedy,
            topK,
            taxoName2RepNormVecPath,
            gpus_work_ratio,
            each_gpu_threads,
            batch_size_per_gpu,
            model_config
        )
    else:
        estimateContigSimFromFile(
            inputBinFolder,
            annotOutputFolder,
            concat_annot_file_path,
            bin_suffix
        )
    end_time = time.time()
    with open(os.path.join(tempFileOutFolder, "Annot.time"), "w") as wh:
        wh.write(str(end_time - start_time) + "\n")

    if just_annot:
        return 
    
    mp.set_start_method("fork", force=True)
    print(f"--> The method for multiprocessing is {mp.get_start_method()}")
    if build_concat_file:
        buildAllConcatFiles(
            inputBinFolder,
            annotOutputFolder,
            concat_annot_file_path,
            concat_vec_file_path,
            concat_TNF_vector_path,
            concat_contig_file_path,
            bin_suffix
        )
        
    if num_process is None:
        cpu_num = psutil.cpu_count()
    else:
        cpu_num = num_process
    filterOutputFolder = os.path.join(tempFileOutFolder, "FilterOutput")
    if os.path.exists(filterOutputFolder) is False:
        os.mkdir(filterOutputFolder)

    temp_gene_folder_path = os.path.join(tempFileOutFolder, "CalledGenes")
    if os.path.exists(temp_gene_folder_path) is False:
        os.mkdir(temp_gene_folder_path)
    
    if simulated_MAG is False:
        print("========================================================================")
        print("--> Start to Call Genes...")
        s_time = time.time()
        callMarkerGenes(inputBinFolder, temp_gene_folder_path, cpu_num, hmmModelPath, bin_suffix)
        e_time = time.time()
        with open(os.path.join(tempFileOutFolder, "CallGene.time"), "w") as wh:
            wh.write(str(e_time - s_time) + "\n")
    
    print("========================================================================")
    print("--> Start to Filter Contaminations and Separate Bins...")
    
    s_time = time.time()
    skip = filterContaminationFolder(
        annotOutputFolder,
        inputBinFolder,
        temp_gene_folder_path,
        filterOutputFolder,
        hmmModelPath,
        phy2accsPath,
        concat_vec_file_path,
        bin_suffix,
        seq_length_threshold,
        simulated_MAG=simulated_MAG,
        cpu_num=cpu_num
    )
    e_time = time.time()
    with open(os.path.join(tempFileOutFolder, "Filter.time"), "w") as wh:
        wh.write(str(e_time - s_time) + "\n")
    
    if simulated_MAG:
        return
    
    print("========================================================================")
    print("--> Start to Run CheckM2...")
    originalBinsCheckMfolder = os.path.join(filterOutputFolder, "original_checkm2_res")
    originalBinsCheckMPath = os.path.join(originalBinsCheckMfolder, "quality_report.tsv")
    if os.path.exists(originalBinsCheckMfolder) is False:
        os.mkdir(originalBinsCheckMfolder)
    s_time = time.time()
    ### checkm2_db_path
    if os.path.exists(originalBinsCheckMPath) is False:
        runCheckm2Single(inputBinFolder, originalBinsCheckMfolder, bin_suffix, db_path=checkm2_db_path, num_cpu=cpu_num)
    e_time = time.time()
    with open(os.path.join(tempFileOutFolder, "Checkm2_first.time"), "w") as wh:
        wh.write(str(e_time - s_time) + "\n")
    
    print("========================================================================")
    print("--> Start to Reuse the CheckM2's Tmp Files...")
    s_time = time.time()
    buildCheckm2TmpFilesParall(filterOutputFolder, originalBinsCheckMfolder, bin_suffix)
    e_time = time.time()
    with open(os.path.join(tempFileOutFolder, "Reuse.time"), "w") as wh:
        wh.write(str(e_time - s_time) + "\n")
    
    print("========================================================================")
    print("--> Start to Run CheckM2...")
    s_time = time.time()
    runCheckm2Reuse(filterOutputFolder, bin_suffix, checkm2_db_path)
    e_time = time.time()
    with open(os.path.join(tempFileOutFolder, "Checkm2_second.time"), "w") as wh:
        wh.write(str(e_time - s_time) + "\n")
    
    print("========================================================================")
    print("--> Start to Select Result...")
    s_time = time.time()
    collected_list = []
    binFilesList = os.listdir(inputBinFolder)
    tN = len(binFilesList)
    for i, binFileName in enumerate(binFilesList):
        # display progress bar
        progressBar(i, tN)
        _, suffix = os.path.splitext(binFileName)
        if suffix[1:] == bin_suffix:
            outInfo = findBestBinsAfterFiltering(
                binFileName,
                inputBinFolder,
                tempFileOutFolder,
                originalBinsCheckMPath
            )
            for qualityValues, cor_path in outInfo:
                collected_list.append((qualityValues, cor_path))
    e_time = time.time()
    with open(os.path.join(tempFileOutFolder, "Select_Res.time"), "w") as wh:
        wh.write(str(e_time - s_time) + "\n")
    
    print("========================================================================")
    print("--> Start to Write Result...")
    outputBinsMetaFilePath = os.path.join(outputBinFolder, "MetaInfo.tsv")
    wh = open(outputBinsMetaFilePath, "w")
    if os.path.exists(outputBinFolder) is False:
        os.mkdir(outputBinFolder)
    write_result(outputBinFolder, collected_list, wh)
    wh.close()
    return N


####################################
########## Application #############
####################################

def bin_c(
    contigs_path: str,
    sorted_sorted_bam_file: str,
    res_output_path,
    depth_path,
    num_cpu
):
    if depth_path is not None and os.path.exists(depth_path) is False:
        cal_depth(sorted_sorted_bam_file, depth_path)
    concoct_folder = os.path.join(res_output_path, "concoct")
    if os.path.exists(concoct_folder) is False: 
        os.mkdir(concoct_folder)
    cur_depth_path = os.path.join(res_output_path, "cur_depth.txt")
    first_line = ""
    ori_depth_info = {}
    with open(depth_path, "r") as rh:
        i = 0
        for line in rh:
            if i == 0:
                first_line = line
            else:
                info = line.strip("\n").split("\t")
                ori_depth_info[info[0]] = info[1:]
            i += 1
    
    with open(contigs_path, "r") as rh, open(cur_depth_path, "w") as wh:
        wh.write(first_line)
        for line in rh:
            if line[0] == ">":
                contig_name = line.strip("\n")[1:]
                wh.write("\t".join([contig_name] + ori_depth_info[contig_name]) + "\n")
                
    p = Process(target=runCONCOCT, args=(contigs_path, cur_depth_path, concoct_folder, num_cpu))
    p.start()
    p.join()


def bin_m(
    contigs_path: str,
    sorted_sorted_bam_file: str,
    res_output_path,
    depth_path,
    num_cpu
):
    if depth_path is not None and os.path.exists(depth_path) is False:
        cal_depth(sorted_sorted_bam_file, depth_path)
    metabat2_folder = os.path.join(res_output_path, "metabat2")
    if os.path.exists(metabat2_folder) is False: os.mkdir(metabat2_folder)
    cur_depth_path = os.path.join(res_output_path, "cur_depth.txt")
    first_line = ""
    ori_depth_info = {}
    with open(depth_path, "r") as rh:
        i = 0
        for line in rh:
            if i == 0:
                first_line = line
            else:
                info = line.strip("\n").split("\t")
                ori_depth_info[info[0]] = info[1:]
            i += 1
    
    with open(contigs_path, "r") as rh, open(cur_depth_path, "w") as wh:
        wh.write(first_line)
        for line in rh:
            if line[0] == ">":
                contig_name = line.strip("\n")[1:]
                wh.write("\t".join([contig_name] + ori_depth_info[contig_name]) + "\n")
    p = Process(target=runMetaBAT2, args=(contigs_path, cur_depth_path, metabat2_folder, num_cpu))
    p.start()
    p.join()


def bin_c_m(
    contigs_path: str,
    sorted_sorted_bam_file: str,
    res_output_path,
    depth_path,
    num_cpu
):
    if depth_path is not None and os.path.exists(depth_path) is False:
        cal_depth(sorted_sorted_bam_file, depth_path)
    concoct_folder = os.path.join(res_output_path, "concoct")
    metabat2_folder = os.path.join(res_output_path, "metabat2")
    if os.path.exists(concoct_folder) is False: os.mkdir(concoct_folder)
    if os.path.exists(metabat2_folder) is False: os.mkdir(metabat2_folder)
    cur_depth_path = os.path.join(res_output_path, "cur_depth.txt")
    first_line = ""
    ori_depth_info = {}
    with open(depth_path, "r") as rh:
        i = 0
        for line in rh:
            if i == 0:
                first_line = line
            else:
                info = line.strip("\n").split("\t")
                ori_depth_info[info[0]] = info[1:]
            i += 1
    
    with open(contigs_path, "r") as rh, open(cur_depth_path, "w") as wh:
        wh.write(first_line)
        for line in rh:
            if line[0] == ">":
                contig_name = line.strip("\n")[1:]
                contig_name = contig_name.split()[0]
                if contig_name in ori_depth_info:
                    wh.write("\t".join([contig_name] + ori_depth_info[contig_name]) + "\n")
    
    process_list = []
    p = Process(target=runCONCOCT, args=(contigs_path, cur_depth_path, concoct_folder, num_cpu))
    process_list.append(p)
    p.start()
    p = Process(target=runMetaBAT2, args=(contigs_path, cur_depth_path, metabat2_folder, num_cpu))
    process_list.append(p)
    p.start()
    for p in process_list:
        p.join()


def binning(
    contigs_path: str,
    sorted_sorted_bam_file: str,
    res_output_folder,
    depth_path,
    num_cpu,
    binning_mode: str = None):
    s_time = time.time()
    if binning_mode is None:
        print(f"--> binning mode is 'Ensemble'...")
        pro_list = []
        semibin_out = os.path.join(res_output_folder, "semibin")
        if os.path.exists(semibin_out) is False: os.mkdir(semibin_out)
        p = Process(target=runSemibin, args=(contigs_path, sorted_sorted_bam_file, semibin_out, num_cpu))
        p.start()
        pro_list.append(p)
        p = Process(target=bin_c_m, args=(contigs_path, sorted_sorted_bam_file, res_output_folder, depth_path, num_cpu))
        p.start()
        pro_list.append(p)
        for p in pro_list:
            p.join()
    elif binning_mode.lower() == "semibin2":
        print(f"--> Binning mode is 'Semibin2'...")
        semibin_out = os.path.join(res_output_folder, "semibin")
        if os.path.exists(semibin_out) is False: os.mkdir(semibin_out)
        p = Process(target=runSemibin, args=(contigs_path, sorted_sorted_bam_file, semibin_out, num_cpu))
        p.start()
        p.join()
    elif binning_mode.lower() == "concoct":
        print(f"--> Binning mode is 'CONCOCT'...")
        bin_c(contigs_path, sorted_sorted_bam_file, res_output_folder, depth_path, num_cpu)
    elif binning_mode.lower() == "metabat2":
        print(f"--> Binning mode is 'MetaBAT2'...")
        bin_m(contigs_path, sorted_sorted_bam_file, res_output_folder, depth_path, num_cpu)
    else:
        raise ValueError("Only implement CONCOCT, MetaBAT2, SemiBin2 binning tools.")
    e_time = time.time()
    with open(os.path.join(res_output_folder, "binning.time"), "w") as wh:
        wh.write(str(e_time - s_time) + "\n")


def repeat_binning_purify(
    fasta_path: Dict,
    sorted_bam_file,
    temp_folder: str,
    depth_path,
    ####################
    modelWeightPath,
    taxoVocabPath,
    taxoTreePath,
    taxoName2RepNormVecPath,
    hmmModelPath,
    phy2accsPath,
    mer3Path,
    mer4Path,
    checkm2_db_path,
    gpus_work_ratio,
    batch_size_per_gpu,
    each_gpu_threads,
    overlapping_ratio,
    cut_seq_length,
    seq_length_threshold,
    topkORgreedy,
    topK,
    model_config,
    num_process,
    concat_annot_path,
    concat_vec_path,
    start_time = 0,
    binning_mode: str = None):
    re_times = start_time
    re_cluster_folder = os.path.join(temp_folder, f"re_cluster_{re_times}")
    if os.path.exists(re_cluster_folder) is False:
        os.mkdir(re_cluster_folder)
    de_tmp_folder = os.path.join(temp_folder, f"de_temp_{re_times}")
    if os.path.exists(de_tmp_folder) is False:
        os.mkdir(de_tmp_folder)
    de_out_folder = os.path.join(temp_folder, f"de_out_bins_{re_times}")
    if os.path.exists(de_out_folder) is False:
        os.mkdir(de_out_folder)
    if num_process is None:
        num_process = psutil.cpu_count()

    print("========================================================================")
    print(f"--> Start the 0-th re-binning...")
    if os.path.exists(os.path.join(re_cluster_folder, "Deepurify_Bin_0.fasta")) is False:
        binning(fasta_path, sorted_bam_file, re_cluster_folder, depth_path, num_process, binning_mode)
    
    index = 0
    for bin_name in os.listdir(re_cluster_folder):
        cur_f = os.path.join(re_cluster_folder, bin_name, "output_bins")
        if os.path.isdir(cur_f):
            for bin in os.listdir(cur_f):
                outName = f"Deepurify_Bin_{index}.fasta"
                writeFasta(readFasta(os.path.join(cur_f, bin)), 
                        os.path.join(re_cluster_folder, outName))
                index += 1
                
    if os.path.exists(os.path.join(de_out_folder, "Deepurify_Bin_0.fasta")) is False:
        run_all_deconta_steps(
            re_cluster_folder,
            de_tmp_folder,
            de_out_folder,
            modelWeightPath,
            taxoVocabPath,
            taxoTreePath,
            taxoName2RepNormVecPath,
            hmmModelPath,
            phy2accsPath,
            "fasta",
            mer3Path,
            mer4Path,
            checkm2_db_path,
            gpus_work_ratio,
            batch_size_per_gpu,
            each_gpu_threads,
            overlapping_ratio,
            cut_seq_length,
            seq_length_threshold,
            topkORgreedy,
            topK,
            model_config,
            num_process,
            False,
            concat_annot_path,
            concat_vec_path,
        )

    # re-binning
    prev_out_folder = de_out_folder
    for re_times in range(start_time + 1, start_time + 3):
        print("========================================================================")
        print(f"--> Start the {re_times}-th re-binning...")
        re_cluster_folder = os.path.join(temp_folder, f"re_cluster_{re_times}")
        if os.path.exists(re_cluster_folder) is False:
            os.mkdir(re_cluster_folder)
        de_tmp_folder = os.path.join(temp_folder, f"de_temp_{re_times}")
        if os.path.exists(de_tmp_folder) is False:
            os.mkdir(de_tmp_folder)
        de_out_folder = os.path.join(temp_folder, f"de_out_bins_{re_times}")
        if os.path.exists(de_out_folder) is False:
            os.mkdir(de_out_folder)
        sub_fasta_path = os.path.join(temp_folder, f"sub_concat_fasta_{re_times}.fasta")
        
        high_num = buildSubFastaFile(
            os.path.join(prev_out_folder, "MetaInfo.tsv"),
            prev_out_folder,
            sub_fasta_path,
            "fasta",
            None)
        if high_num == 0:
            rmtree(de_out_folder)
            break

        if os.path.exists(os.path.join(re_cluster_folder, "Deepurify_Bin_0.fasta")) is False:
            binning(sub_fasta_path, sorted_bam_file, re_cluster_folder, depth_path, num_process, binning_mode)

        index = 0
        for bin_name in os.listdir(re_cluster_folder):
            cur_f = os.path.join(re_cluster_folder, bin_name, "output_bins")
            if os.path.isdir(cur_f):
                for bin in os.listdir(cur_f):
                    outName = f"Deepurify_Bin_{index}.fasta"
                    writeFasta(readFasta(os.path.join(cur_f, bin)), 
                            os.path.join(re_cluster_folder, outName))
                    index += 1

        # re-decontamination
        if os.path.exists(os.path.join(de_out_folder, "Deepurify_Bin_0.fasta")) is False:
            run_all_deconta_steps(
                re_cluster_folder,
                de_tmp_folder,
                de_out_folder,
                modelWeightPath,
                taxoVocabPath,
                taxoTreePath,
                taxoName2RepNormVecPath,
                hmmModelPath,
                phy2accsPath,
                "fasta",
                mer3Path,
                mer4Path,
                checkm2_db_path,
                gpus_work_ratio,
                batch_size_per_gpu,
                each_gpu_threads,
                overlapping_ratio,
                cut_seq_length,
                seq_length_threshold,
                topkORgreedy,
                topK,
                model_config,
                num_process,
                False,
                concat_annot_path,
                concat_vec_path,
            )
        #### next prepare
        prev_out_folder = de_out_folder


def binning_purify(
    contig_fasta_path: str,
    all_tempFileOutFolder: str,
    cur_temp_folder: str,
    sorted_bam_file,
    modelWeightPath: str,
    taxoVocabPath: str,
    taxoTreePath: str,
    taxoName2RepNormVecPath: str,
    hmmModelPath: str,
    phy2accsPath: str,
    mer3Path: str,
    mer4Path: str,
    checkm2_db_path: str,
    gpus_work_ratio: List[float],
    batch_size_per_gpu: List[float],
    each_gpu_threads: int,
    overlapping_ratio: float,
    cut_seq_length: int,
    seq_length_threshold: int,
    topkORgreedy: str,
    topK: int,
    model_config: Union[Dict, None] = None,
    num_process: int = None,
    binning_mode = None
):
    de_tmp_folder = os.path.join(cur_temp_folder, f"de_temp_-1")
    if os.path.exists(de_tmp_folder) is False:
        os.mkdir(de_tmp_folder)
    de_out_folder = os.path.join(cur_temp_folder, f"de_out_bins_-1")
    if os.path.exists(de_out_folder) is False:
        os.mkdir(de_out_folder)
    re_cluster_folder = os.path.join(cur_temp_folder, f"re_cluster_-1")
    if os.path.exists(re_cluster_folder) is False:
        os.mkdir(re_cluster_folder)
    concat_vec_path = os.path.join(all_tempFileOutFolder, "all_concat_contigname2repNorm.pkl")
    concat_annot_path = os.path.join(all_tempFileOutFolder, "all_concat_annot.txt")
    concat_contig_file_path = os.path.join(all_tempFileOutFolder, "all_concat_contig_seq.txt")
    concat_TNF_vector_path = os.path.join(all_tempFileOutFolder, "all_concat_contigname2TNFNorm.pkl")
    depth_path = os.path.join(all_tempFileOutFolder, "contigs_depth.txt")
    if num_process is None:
        num_process = psutil.cpu_count()
    
    print("========================================================================")
    print("--> Start the initial binning...")
    if os.path.exists(os.path.join(re_cluster_folder, "Deepurify_Bin_0.fasta")) is False:
        binning(contig_fasta_path, sorted_bam_file, re_cluster_folder, depth_path, num_process, binning_mode)

    index = 0
    for bin_name in os.listdir(re_cluster_folder):
        cur_f = os.path.join(re_cluster_folder, bin_name, "output_bins")
        if os.path.isdir(cur_f):
            for bin in os.listdir(cur_f):
                outName = f"Deepurify_Bin_{index}.fasta"
                writeFasta(readFasta(os.path.join(cur_f, bin)), 
                        os.path.join(re_cluster_folder, outName))
                index += 1
    
    if os.path.exists(os.path.join(de_out_folder, "Deepurify_Bin_0.fasta")) is False:
        run_all_deconta_steps(
            re_cluster_folder,
            de_tmp_folder,
            de_out_folder,
            modelWeightPath,
            taxoVocabPath,
            taxoTreePath,
            taxoName2RepNormVecPath,
            hmmModelPath,
            phy2accsPath,
            "fasta",
            mer3Path,
            mer4Path,
            checkm2_db_path,
            gpus_work_ratio,
            batch_size_per_gpu,
            each_gpu_threads,
            overlapping_ratio,
            cut_seq_length,
            seq_length_threshold,
            topkORgreedy,
            topK,
            model_config,
            num_process,
            True,
            concat_annot_path,
            concat_vec_path,
            concat_TNF_vector_path,
            concat_contig_file_path,
            False,
            False
        )

    sub_fasta_path = os.path.join(cur_temp_folder, f"sub_concat_fasta_-1.fasta")
    buildSubFastaFile(
        os.path.join(de_out_folder, "MetaInfo.tsv"),
        de_out_folder,
        sub_fasta_path,
        "fasta",
        None)
    
    repeat_binning_purify(
        sub_fasta_path,
        sorted_bam_file,
        cur_temp_folder,
        depth_path,
        modelWeightPath,
        taxoVocabPath,
        taxoTreePath,
        taxoName2RepNormVecPath,
        hmmModelPath,
        phy2accsPath,
        mer3Path,
        mer4Path,
        checkm2_db_path,
        gpus_work_ratio,
        batch_size_per_gpu,
        each_gpu_threads,
        overlapping_ratio,
        cut_seq_length,
        seq_length_threshold,
        topkORgreedy,
        topK,
        model_config,
        num_process,
        concat_annot_path,
        concat_vec_path,
        0,
        binning_mode)
