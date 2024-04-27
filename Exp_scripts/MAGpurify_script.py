from multiprocessing import Process
import os
from shutil import copy
import subprocess
import argparse
from func_timeout import func_timeout, FunctionTimedOut 
from multiprocessing.pool import Pool

# def runCheckM(binsFolder: str, checkmResFilePath: str, num_cpu: int, bin_suffix: str):
#     res = subprocess.Popen(" checkm lineage_wf " + " -t " + str(num_cpu) + " --pplacer_threads " + str(num_cpu) +
#                            " -x " + bin_suffix + " -f " + checkmResFilePath + " -r " +
#                            "  " + binsFolder + "  " + os.path.join(binsFolder, "checkmTempOut"), shell=True)
#     res.wait()
#     res.terminate()
#     res.kill()


def splitListEqually(input_list, num_parts: int):
    n = len(input_list)
    step = n // num_parts + 1
    out_list = []
    for i in range(num_parts):
        if curList := input_list[i * step: (i + 1) * step]:
            out_list.append(curList)
    return out_list


def runOneBin(binFolder, binFile, tempFolder, binName, cleanBins):
    # step 1
    res = subprocess.Popen("magpurify phylo-markers " + os.path.join(binFolder, binFile) + " " +
                           os.path.join(tempFolder, binName), shell=True)
    res.wait()
    res.terminate()
    res.kill()
    # step 2
    res = subprocess.Popen("magpurify clade-markers " + os.path.join(binFolder, binFile) + " " +
                           os.path.join(tempFolder, binName), shell=True)
    res.wait()
    res.terminate()
    res.kill()
    # step 3
    res = subprocess.Popen("magpurify tetra-freq " + os.path.join(binFolder, binFile) + " " +
                           os.path.join(tempFolder, binName), shell=True)
    res.wait()
    res.terminate()
    res.kill()
    # step 4
    res = subprocess.Popen("magpurify gc-content " + os.path.join(binFolder, binFile) + " " +
                           os.path.join(tempFolder, binName), shell=True)
    res.wait()
    res.terminate()
    res.kill()
    # step 5
    res = subprocess.Popen("magpurify known-contam " + os.path.join(binFolder, binFile) + " " +
                           os.path.join(tempFolder, binName), shell=True)
    res.wait()
    res.terminate()
    res.kill()
    # step 6
    res = subprocess.Popen("magpurify clean-bin " + os.path.join(binFolder, binFile) + " " +
                           os.path.join(tempFolder, binName) + " " + os.path.join(cleanBins, binFile), shell=True)
    res.wait()
    res.terminate()
    res.kill()


def runMagPurify(binFolder: str, tempFolder, cleanBins, binList, suffix_in):
    for binFile in binList:
        binName, suffix = os.path.splitext(binFile)
        if suffix[1:] != suffix_in:
            continue
        if os.path.exists(os.path.join(cleanBins, binFile)):
            print("exist,continue")
            continue
        try:
            runOneBin(binFolder, binFile, tempFolder, binName, cleanBins)
        except:
            copy(os.path.join(binFolder, binFile), cleanBins)


def run(folders: list, input_folder, output_folder):
    for folder in folders:
        print(f"{folder}")
        if "checkm2" in folder or "gunc" in folder:
            continue
        num_cpu = 100
        bin_suffix = "fasta"
        cur_input_folder = os.path.join(input_folder, folder, "DeepurifyTmpFiles/deconta_tmp/re_cluster_-1")
        binFiles = os.listdir(cur_input_folder)
        processList = []
        cur_output_folder = os.path.join(output_folder, f"{folder}_nodrep")
        if os.path.exists(cur_output_folder) is False:
            os.mkdir(cur_output_folder)
        cleanBinsNum = 0
        oriNum = 0
        for file in os.listdir(cur_input_folder):
            binName, suffix = os.path.splitext(file)
            if suffix[1:] == bin_suffix:
                oriNum += 1
        for file in os.listdir(cur_output_folder):
            binName, suffix = os.path.splitext(file)
            if suffix[1:] == bin_suffix:
                cleanBinsNum += 1
        # print(cleanBinsNum, oriNum)
        if cleanBinsNum == oriNum:
            continue
        tempFolder = os.path.join(cur_output_folder, "MagTemp")
        if os.path.exists(tempFolder) is False:
            os.mkdir(tempFolder)
        split_binFiles = splitListEqually(binFiles, num_cpu)
        for i in range(len(split_binFiles)):
            # print(f"bins number: {len(split_binFiles[i])}")
            processList.append(Process(target=runMagPurify, args=(cur_input_folder, tempFolder, cur_output_folder,
                                                                split_binFiles[i], bin_suffix,),))
            processList[-1].start()
        for p in processList:
            p.join()
        cleanBinsNum = 0
        oriNum = 0
        for file in os.listdir(cur_input_folder):
            binName, suffix = os.path.splitext(file)
            if suffix[1:] == bin_suffix:
                oriNum += 1
        for file in os.listdir(cur_output_folder):
            binName, suffix = os.path.splitext(file)
            if suffix[1:] == bin_suffix:
                cleanBinsNum += 1
        # print(cleanBinsNum, oriNum)
        assert cleanBinsNum == oriNum, ValueError("The number of bins in cleanBins and origianl folder is not equal.")



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-c", "--cpu")
    # args = parser.parse_args()
    input_folder = "/home/datasets/ZOUbohao/Proj1-Deepurify/plant-ensemble-Deepurify/"
    output_folder = "/home/datasets/ZOUbohao/Proj1-Deepurify/plant-ensemble-MAGpurify/"
    # ibs_id_path = "/home/datasets/ZOUbohao/Proj1-Deepurify/Deepurify-v2.2.0-ReBinning/IBS_ids_completed.txt"
    folders = os.listdir(input_folder)
    # with open(ibs_id_path, "r") as rh:
    #     for line in rh:
    #         folders.append(line.strip("\n"))
    if os.path.exists(output_folder) is False:
        os.mkdir(output_folder)
    res_list = []
    for folder in folders:
        res_list.append(Process(target=run, args=([folder], input_folder, output_folder)))
        res_list[-1].start()
    for p in res_list:
        p.join()

