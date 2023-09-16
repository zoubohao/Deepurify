import os
from multiprocessing import Process
from subprocess import Popen
from typing import List
from Deepurify.CallGenesTools.prodigal import ProdigalRunner

from Deepurify.IOUtils import readFasta


def splitListEqually(input_list: List, num_parts: int) -> List[List[object]]:
    n = len(input_list)
    step = n // num_parts + 1
    out_list = []
    for i in range(num_parts):
        if curList := input_list[i * step : (i + 1) * step]:
            out_list.append(curList)
    return out_list


def runProgidalSingle(binName, bin_path: str, output_faa_folder_path: str) -> None:
    outFAA_path = os.path.join(output_faa_folder_path, f"{binName}.faa")
    if os.path.exists(outFAA_path):
        return
    runner = ProdigalRunner(binName, output_faa_folder_path)
    runner.run(bin_path)


def subProcessProgidal(files: List[str], bin_folder_path: str, output_faa_folder_path: str) -> None:
    for file in files:
        binName = os.path.splitext(file)[0]
        bin_path = os.path.join(bin_folder_path, file)
        runProgidalSingle(binName, bin_path, output_faa_folder_path)


def runProgidalFolder(bin_folder_path: str, output_faa_folder_path: str, num_cpu: int, bin_suffix: str) -> None:
    files = os.listdir(bin_folder_path)
    bin_files = [
        file for file in files if os.path.splitext(file)[-1][1:] == bin_suffix
    ]
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


def runHMMsearchSingle(faa_path: str, ouput_path: str, hmm_model_path) -> None:
    if os.path.getsize(faa_path) == 0 or os.path.exists(faa_path) is False:
        wh = open(ouput_path, "w")
        wh.close()
        return
    if os.path.exists(ouput_path):
        return
    res = Popen(
        f"hmmsearch --domtblout {ouput_path} --cpu 2 --notextw -E 0.01 --domE 0.01 --noali {hmm_model_path} {faa_path} > /dev/null",
        shell=True,
    )
    res.wait()
    res.kill()


def subProcessHMM(hmm_model_path: str, files: List[str], faa_folder_path: str, output_folder_path: str) -> None:
    for file in files:
        binName = os.path.splitext(file)[0]
        faa_path = os.path.join(faa_folder_path, file)
        output_path = os.path.join(output_folder_path, f"{binName}.HMM.txt")
        runHMMsearchSingle(faa_path, output_path, hmm_model_path)


def runHMMsearchFolder(faa_folder_path: str, output_folder_path: str, hmm_model_path: str, num_cpu: int, faa_suffix: str) -> None:
    files = os.listdir(faa_folder_path)
    faa_files = [
        file for file in files if os.path.splitext(file)[-1][1:] == faa_suffix
    ]
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


def callMarkerGenes(bin_folder_path: str, temp_folder_path: str, num_cpu: int, hmm_model_path: str, bin_suffix: str) -> None:
    if os.path.exists(temp_folder_path) is False:
        os.mkdir(temp_folder_path)
    print("Running Prodigal...")
    runProgidalFolder(bin_folder_path, temp_folder_path, num_cpu, bin_suffix)
    print("Running Hmm-Search...")
    runHMMsearchFolder(temp_folder_path, temp_folder_path, hmm_model_path, num_cpu, "faa")
