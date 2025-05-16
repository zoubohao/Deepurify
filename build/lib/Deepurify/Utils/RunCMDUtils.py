

import os
import subprocess
from multiprocessing import Process
from shutil import copy, rmtree
from typing import List

import psutil

from Deepurify.Utils.CallGenesUtils import splitListEqually
from Deepurify.Utils.IOUtils import (progressBar, readCheckMResultAndStat,
                                     readDiamond, readFasta, writePickle)

index2Taxo = {
    0: "T0_filter",
    1: "T1_filter",
    2: "T2_filter",
    3: "T3_filter",
    4: "T4_filter",
    5: "T5_filter",
    6: "T6_filter"}


### CheckM ###
def runCheckMsingle(
        binsFolder: str,
        checkmResFilePath: str,
        num_cpu: int,
        bin_suffix: str,
        repTime=0):

    if os.path.exists(checkmResFilePath):
        return

    res = subprocess.Popen(
        " checkm lineage_wf "
        + " -t "
        + str(num_cpu)
        + " --pplacer_threads "
        + str(64)
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
                raise RuntimeError(
                    f"binFolder: {binsFolder}, Checkm Result Path: {checkmResFilePath}"
                )
            runCheckMsingle(binsFolder, checkmResFilePath, num_cpu, bin_suffix, repTime + 1)
    # res.wait()
    res.kill()


def runCheckMForSixFilter(
        filterFolder,
        indexList: List,
        num_checkm_cpu: int,
        bin_suffix: str):
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


def runCheckMParall(
        filterFolder,
        bin_suffix,
        num_pall,
        num_cpu=40):
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


def runCheckm2Single(
        input_bin_folder: str,
        output_bin_folder: str,
        bin_suffix: str,
        db_path: str,
        num_cpu: int):
    if os.path.exists(output_bin_folder) is False:
        os.makedirs(output_bin_folder)
    cmd = f"checkm2 predict -x {bin_suffix} --threads {num_cpu} -i {input_bin_folder} -o {output_bin_folder} --database_path {db_path}"
    os.system(cmd)


def runCheckM2ForSixFilter(
        filterFolder,
        indexList: List,
        num_checkm_cpu: int,
        bin_suffix: str):
    for i in indexList:
        binsFolder = os.path.join(filterFolder, index2Taxo[i])
        files = os.listdir(binsFolder)
        n = 0
        copyList = []
        for file in files:
            if os.path.splitext(file)[1][1:] == bin_suffix:
                n += 1
                copyList.append(file)

        ############## splitted to k parts #############
        k = n // 1000 + 1
        equalFilesList = splitListEqually(copyList, k)
        k = len(equalFilesList)
        for j, equal_files in enumerate(equalFilesList):
            splitFolder = os.path.join(binsFolder, str(j))
            if os.path.exists(splitFolder) is False:
                os.mkdir(splitFolder)

            for file in equal_files:
                if os.path.exists(os.path.join(splitFolder, file)) is False:
                    copy(os.path.join(binsFolder, file), splitFolder)

        cur_filter_name = index2Taxo[i].split("_")[0]
        for j in range(k):
            splitFolder = os.path.join(binsFolder, str(j))
            tmp_j_folder = os.path.join(filterFolder, cur_filter_name + "_" + str(j) + "_checkm2_tmp")
            runCheckm2Single(splitFolder, tmp_j_folder, bin_suffix, num_checkm_cpu)

        ## write res to disk ##
        final_cur_marker_file = os.path.join(filterFolder, cur_filter_name + "_" + "checkm2_res.tsv")
        with open(final_cur_marker_file, "w") as wh:
            for j in range(k):
                cur_marker_file = os.path.join(filterFolder, cur_filter_name + "_" + str(j) + "_checkm2_tmp", "quality_report.tsv")
                with open(cur_marker_file, "r") as rh:
                    for line in rh:
                        if "Name" in line:
                            continue
                        else:
                            wh.write(line)

        for j in range(k):
            rmtree(os.path.join(binsFolder, str(j)), ignore_errors=True)
            rmtree(os.path.join(filterFolder, cur_filter_name + "_" + str(j) + "_checkm2_tmp"), ignore_errors=True)


def runCheckM2Parall(
        filterFolder,
        bin_suffix):
    cpu_num = psutil.cpu_count()
    memory_info_obj = psutil.virtual_memory()
    total_mem = memory_info_obj.total / 1024 / 1024 / 1024  # GB
    used_mem = memory_info_obj.used / 1024 / 1024 / 1024
    free = total_mem - used_mem

    res = []
    num_pall = int(free // 20)
    if num_pall > 6:
        num_pall = 6
    elif num_pall < 1:
        num_pall = 1

    split_list = splitListEqually([1, 2, 3, 4, 5, 6], num_pall)

    for i in range(len(split_list)):
        p = Process(
            target=runCheckM2ForSixFilter,
            args=(
                filterFolder,
                split_list[i],
                cpu_num // len(split_list) + 8,
                bin_suffix,
            ),
        )
        res.append(p)
        p.start()
    for p in res:
        p.join()


def buildCheckm2TmpFiles(
        original_checkm2_res_folder: str,
        modified_bins_folder: str,
        modified_checkm2_tmp_folder: str,
        bin_suffix: str):

    if os.path.exists(modified_checkm2_tmp_folder) is False:
        os.mkdir(modified_checkm2_tmp_folder)

    output_faa_folder = os.path.join(modified_checkm2_tmp_folder, "protein_files")
    if os.path.exists(output_faa_folder) is False:
        os.mkdir(output_faa_folder)

    output_dimond_folder = os.path.join(modified_checkm2_tmp_folder, "diamond_output")
    if os.path.exists(output_dimond_folder) is False:
        os.mkdir(output_dimond_folder)

    output_dimond_file = os.path.join(output_dimond_folder, "DIAMOND_RESULTS.tsv")

    modified_bin_names = os.listdir(modified_bins_folder)

    faa_files_folder = os.path.join(original_checkm2_res_folder, "protein_files")
    diam_file = os.path.join(original_checkm2_res_folder, "diamond_output")
    diamond_info = {}
    for file in os.listdir(diam_file):
        readDiamond(os.path.join(diam_file, file), diamond_info)

    wdh = open(output_dimond_file, "w", encoding="utf-8")
    N = len(modified_bin_names)
    for j, modified_bin_name in enumerate(modified_bin_names):
        bin_name, suffix = os.path.splitext(modified_bin_name)
        if suffix[1:] != bin_suffix:
            continue
        progressBar(j, N)
        ori_bin_name = bin_name.split("___")[0]
        modified_contig2seq = readFasta(os.path.join(modified_bins_folder, modified_bin_name))
        contig_names = set(list(modified_contig2seq.keys()))

        if os.path.exists(os.path.join(faa_files_folder, ori_bin_name + ".faa")):
            faa_contig2seq = readFasta(os.path.join(faa_files_folder, ori_bin_name + ".faa"))
            with open(os.path.join(output_faa_folder, bin_name + ".faa"), "w") as wfh:
                for faa_contig_name, seq in faa_contig2seq.items():
                    cur_name = "_".join(faa_contig_name.split(" ")[0].split("_")[0:-1])
                    if cur_name in contig_names:
                        wfh.write(faa_contig_name + "\n")
                        wfh.write(seq + "\n")

        if ori_bin_name in diamond_info:
            cur_diamond_info = diamond_info[ori_bin_name]
            for dia_contig_name, dia_info in cur_diamond_info.items():
                cur_name = ">" + "_".join(dia_contig_name.split(" ")[0].split("_")[0:-1])
                if cur_name in contig_names:
                    wdh.write("\t".join([bin_name + "Ω" + dia_contig_name] + dia_info) + "\n")
    wdh.close()


def buildCheckm2TmpFilesParall(
    filterFolder: str,
    original_checkm2_res_folder: str,
    bin_suffix: str,
):
    res = []
    for i in range(7):
        modified_bins_folder = os.path.join(filterFolder, index2Taxo[i])
        modified_checkm2_tmp_folder = os.path.join(filterFolder, index2Taxo[i] + "_checkm2_res")
        p = Process(
            target=buildCheckm2TmpFiles,
            args=(
                original_checkm2_res_folder,
                modified_bins_folder,
                modified_checkm2_tmp_folder,
                bin_suffix,
            ),
        )
        res.append(p)
        p.start()
    for p in res:
        p.join()


def target(modified_bins_folder, modified_checkm2_tmp_folder, bin_suffix, db_path):
    num_cpu = psutil.cpu_count()
    res = subprocess.Popen(
        f"checkm2 predict -x {bin_suffix} --threads {num_cpu} --resume -i {modified_bins_folder} -o {modified_checkm2_tmp_folder} --database_path {db_path}",
        shell=True,
    )
    res.wait()
    res.kill()


def runCheckm2Reuse(filterFolder: str, bin_suffix: str, db_path: str):
    for i in range(7):
        modified_bins_folder = os.path.join(filterFolder, index2Taxo[i])
        modified_checkm2_tmp_folder = os.path.join(filterFolder, index2Taxo[i] + "_checkm2_res")
        if os.path.exists(os.path.join(modified_checkm2_tmp_folder, "quality_report.tsv")) is False:
            target(modified_bins_folder, modified_checkm2_tmp_folder, bin_suffix, db_path)


def runDeRep(drep_out_folder,
            genomes_input_folder,
            cpu_num):
    res = subprocess.Popen(
        f"dRep compare {drep_out_folder} -g {os.path.join(genomes_input_folder, '*.fasta')} --S_algorithm skani --processors {cpu_num}", 
        shell= True
    )
    res.wait()
    res.kill()

def runGalah(galah_out_folder,
             genomes_input_folder,
            cpu_num,
            bin_suffix):
    if not os.path.exists(galah_out_folder):
        os.mkdir(galah_out_folder)
    cur_out_files_txt = os.path.join(galah_out_folder, "files_path.txt")
    with open(cur_out_files_txt, "w") as wh:
        for i, file_name in enumerate(os.listdir(genomes_input_folder)):
            _, suffix = os.path.splitext(file_name)
            if suffix[1:] != bin_suffix:
                continue
            wh.write(os.path.join(genomes_input_folder, file_name) + "\n")
    cmd = f"galah cluster --ani 99 --precluster-ani 90 --genome-fasta-list {cur_out_files_txt}  " + \
        f"  --output-cluster-definition {os.path.join(galah_out_folder, 'clusters.tsv')}  -t {cpu_num}"
    os.system(cmd)


def runSemibin(input_contigs_path,
            bam_file: str,
            output_folder,
            num_cpu):
    
    cmd = f"SemiBin2 single_easy_bin --threads {num_cpu} -i {input_contigs_path} -o {output_folder} -b {bam_file} " + \
        " --compression none -m 1000  --no-recluster "
    run_hd = subprocess.Popen(cmd, shell = True)
    run_hd.communicate()
    run_hd.kill()


def cal_depth(sorted_bam_path, out_depth_file_path):
    res = subprocess.Popen(
        f"jgi_summarize_bam_contig_depths --outputDepth {out_depth_file_path} {sorted_bam_path}",
        shell=True,
    )
    res.wait()
    res.kill()


def convertMetabat2CONCOCT(depth_file_path, out_depth_path):
    with open(depth_file_path, "r") as rh, open(out_depth_path, "w") as wh:
        for line in rh:
            info = line.strip("\n").split("\t")
            wh.write("\t".join([info[0], info[3]]) + "\n")


def runCONCOCT(
    ori_contig_fasta,
    depth_file_path,
    output_folder,
    num_cpu
):
    if os.path.exists(output_folder) is False:
        os.makedirs(output_folder)
    output_cov_tsv = os.path.join(output_folder, 'coverage_table.tsv')
    output_clust_1000 = os.path.join(output_folder, 'clustering_gt1000.csv')
    output_merged = os.path.join(output_folder, 'clustering_merged.csv')
    bins_folder = os.path.join(output_folder, "output_bins")
    if os.path.exists(bins_folder) is False:
        os.mkdir(bins_folder)
    
    convertMetabat2CONCOCT(depth_file_path, output_cov_tsv)
    
    cmd3 = f"concoct --composition_file {ori_contig_fasta} --coverage_file {output_cov_tsv} -b {output_folder} -t {num_cpu} -l 1000 -i 200"
    print("CONCOCT Step 1.")
    run_hd = subprocess.Popen(cmd3, shell = True)
    run_hd.communicate()
    run_hd.kill()
    
    cmd4 = f"merge_cutup_clustering.py {output_clust_1000} > {output_merged}"
    print("CONCOCT Step 2.")
    run_hd = subprocess.Popen(cmd4, shell = True)
    run_hd.communicate()
    run_hd.kill()
    
    cmd5 = f"extract_fasta_bins.py {ori_contig_fasta} {output_merged} --output_path {bins_folder}"
    print("CONCOCT Step 3.")
    run_hd = subprocess.Popen(cmd5, shell = True)
    run_hd.communicate()
    run_hd.kill()


def runMetaBAT2(
    ori_contig_fasta,
    depth_file_path,
    output_folder,
    num_cpu
):
    if os.path.exists(output_folder) is False:
        os.makedirs(output_folder)
    out_bins_folder = os.path.join(output_folder, "output_bins")
    if os.path.exists(out_bins_folder) is False: 
        os.mkdir(out_bins_folder)
    out_bins_path = os.path.join(out_bins_folder, "metabat2_bin")
    cmd = f"metabat2  -o {out_bins_path} -t {num_cpu} --inFile {ori_contig_fasta}  --abdFile {depth_file_path} -m 1500"
    run_hd = subprocess.Popen(cmd, shell = True)
    run_hd.communicate()
    run_hd.kill()


def runGUNCsingle(binsFolder: str, outputFolder: str, num_cpu: int, bin_suffix: str):
    res = subprocess.Popen(
        "gunc run --threads " + str(num_cpu) + " --file_suffix  " + str(bin_suffix) + " --input_dir " + binsFolder + " --out_dir " + outputFolder,
        shell=True,
    )
    res.wait()
    res.kill()


