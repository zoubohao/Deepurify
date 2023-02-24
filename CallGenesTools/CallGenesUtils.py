import os
from multiprocessing import Process
from subprocess import Popen
from typing import List

from Deepurify.IOUtils import readFasta


def splitListEqually(input_list: List, num_parts: int) -> List[List[object]]:
    n = len(input_list)
    step = n // num_parts + 1
    out_list = []
    for i in range(num_parts):
        curList = input_list[i * step: (i + 1) * step]
        if curList:
            out_list.append(curList)
    return out_list


def runProgidalSingle(bin_path: str, ouputFAA_path: str):
    if os.path.exists(ouputFAA_path):
        return
    name2seq = readFasta(bin_path)
    length = 0
    mode = "single"
    for key, seq in name2seq.items():
        length += len(seq)
    if length <= 100000:
        mode = "meta"
    res = Popen("prodigal -p {} -q -m  -g 11 -a {} -i {} > /dev/null".format(mode, ouputFAA_path, bin_path), shell=True)
    res.wait()
    res.kill()


def subProcessProgidal(files: List[str], bin_folder_path: str, output_faa_folder_path: str):
    for file in files:
        binName = os.path.splitext(file)[0]
        bin_path = os.path.join(bin_folder_path, file)
        outFAA_path = os.path.join(output_faa_folder_path, binName + ".faa")
        runProgidalSingle(bin_path, outFAA_path)


def runProgidalFolder(bin_folder_path: str, output_faa_folder_path: str, num_cpu: int, bin_suffix: str):
    files = os.listdir(bin_folder_path)
    bin_files = []
    for file in files:
        if os.path.splitext(file)[-1][1:] == bin_suffix:
            bin_files.append(file)
    splited_files = splitListEqually(bin_files, num_cpu)
    n = len(splited_files)
    ps = []
    for i in range(n):
        p = Process(
            target=subProcessProgidal,
            args=(
                splited_files[i],
                bin_folder_path,
                output_faa_folder_path,
            ),
        )
        ps.append(p)
        p.start()
    for p in ps:
        p.join()


def runHMMsearchSinale(faa_path: str, ouput_path: str, hmm_model_path):
    if os.path.getsize(faa_path) == 0 or os.path.exists(faa_path) is False:
        wh = open(ouput_path, "w")
        wh.close()
        return
    if os.path.exists(ouput_path):
        return
    res = Popen("hmmsearch --domtblout {} --cpu 2 --notextw -E 0.1 --domE 0.1 --noali {} {} > /dev/null".format(ouput_path, hmm_model_path, faa_path),
                shell=True,
                )
    res.wait()
    res.kill()


def subProcessHMM(hmm_model_path: str, files: List[str], faa_folder_path: str, output_folder_path: str):
    for file in files:
        binName = os.path.splitext(file)[0]
        faa_path = os.path.join(faa_folder_path, file)
        output_path = os.path.join(output_folder_path, binName + ".HMM" + ".txt")
        runHMMsearchSinale(faa_path, output_path, hmm_model_path)


def runHMMsearchFolder(faa_folder_path: str, output_folder_path: str, hmm_model_path: str, num_cpu: int, faa_suffix: str):
    files = os.listdir(faa_folder_path)
    faa_files = []
    for file in files:
        if os.path.splitext(file)[-1][1:] == faa_suffix:
            faa_files.append(file)
    splited_files = splitListEqually(faa_files, num_cpu)
    n = len(splited_files)
    ps = []
    for i in range(n):
        p = Process(
            target=subProcessHMM,
            args=(
                hmm_model_path,
                splited_files[i],
                faa_folder_path,
                output_folder_path,
            ),
        )
        ps.append(p)
        p.start()
    for p in ps:
        p.join()


def callMarkerGenes(bin_folder_path: str, temp_folder_path: str, num_cpu: int, hmm_model_path: str, bin_suffix: str):
    if os.path.exists(temp_folder_path) is False:
        os.mkdir(temp_folder_path)
    print("Running Prodigal...")
    runProgidalFolder(bin_folder_path, temp_folder_path, num_cpu, bin_suffix)
    print("Running Hmm-Search...")
    runHMMsearchFolder(temp_folder_path, temp_folder_path, hmm_model_path, num_cpu, "faa")
