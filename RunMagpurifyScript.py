from multiprocessing import Process
import os
from shutil import copy
import subprocess
import argparse

def runCheckM(binsFolder: str, checkmResFilePath: str, num_cpu: int, bin_suffix: str):
    res = subprocess.Popen(" checkm lineage_wf " + " -t " + str(num_cpu) + " --pplacer_threads " + str(num_cpu) +
                           " -x " + bin_suffix + " -f " + checkmResFilePath + " -r " +
                           "  " + binsFolder + "  " + os.path.join(binsFolder, "checkmTempOut"), shell=True)
    res.wait()
    res.terminate()
    res.kill()


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
        # print(binName, suffix)
        if suffix[1:] != suffix_in:
            continue
        try:
            runOneBin(binFolder, binFile, tempFolder, binName, cleanBins)
        except:
            copy(os.path.join(binFolder, binFile), cleanBins)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--numCPU", type=int, default=1)
    parser.add_argument("--bin_suffix", type=str, default="fasta")
    parser.add_argument("binInputFolder",  type=str)
    parser.add_argument("outputFolder", type=str)
    args = parser.parse_args()
    num_cpu = args.numCPU
    binFiles = os.listdir(args.binInputFolder)
    processList = []
    step = len(binFiles) // num_cpu + 1
    outputFolder = args.outputFolder
    if os.path.exists(outputFolder) is False:
        os.mkdir(outputFolder)
    tempFolder = os.path.join(outputFolder, "MagTemp")
    if os.path.exists(tempFolder) is False:
        os.mkdir(tempFolder)
    cleanBins = os.path.join(outputFolder, "CleanBins")
    if os.path.exists(cleanBins) is False:
        os.mkdir(cleanBins)
    for i in range(num_cpu):
        processList.append(Process(target=runMagPurify, args=(args.binInputFolder, tempFolder, cleanBins,
                                                              binFiles[step * i: step * (i + 1)], args.bin_suffix,),))
        processList[-1].start()
    for p in processList:
        p.join()
    cleanBinsNum = 0
    oriNum = 0
    for file in os.listdir(args.binInputFolder):
        binName, suffix = os.path.splitext(file)
        if suffix[1:] == args.bin_suffix:
            oriNum += 1
    for file in os.listdir(os.path.join(args.outputFolder, "CleanBins")):
        binName, suffix = os.path.splitext(file)
        if suffix[1:] == args.bin_suffix:
            cleanBinsNum += 1
    print(cleanBinsNum, oriNum)
    assert cleanBinsNum == oriNum, ValueError("The number of bins in cleanBins and origianl folder is not equal.")
    runCheckM(os.path.join(args.outputFolder, "CleanBins"),
              os.path.join(args.outputFolder, "magpurify_checkm.txt"),
              32, bin_suffix=args.bin_suffix)
