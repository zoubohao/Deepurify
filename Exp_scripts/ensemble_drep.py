import argparse
from shutil import copy
import os
import sys
from Deepurify.Utils.BuildFilesUtils import process_drep_result
from Deepurify.Utils.IOUtils import readCSV, readCheckm2Res, readMetaInfo
from Deepurify.Utils.RunCMDUtils import runDeRep, runCheckm2Single, runGUNCsingle

def splitListEqually(input_list, num_parts: int):
    n = len(input_list)
    step = n // num_parts + 1
    out_list = []
    for i in range(num_parts):
        if curList := input_list[i * step: (i + 1) * step]:
            out_list.append(curList)
    return out_list


def getScore(
    qualityValues
) -> float:
    if qualityValues[-1] == "HighQuality":
        score = qualityValues[0] - 5. * qualityValues[1] + 100.
    elif qualityValues[-1] == "MediumQuality":
        score = qualityValues[0] - 5. * qualityValues[1] + 50.
    else:
        score = qualityValues[0] - 5. * qualityValues[1]
    return score


if __name__ == "__main__":
    parse = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),)
    parse.add_argument("-m", "--mode")
    parse.add_argument("-index", "--index")
    args = parse.parse_args()
    mode = args.mode
    index = int(args.index)
    # cd /home/datasets/ZOUbohao/Proj1-Deepurify/Deepurify-v2.2.0-ReBinning/
    input_folders = "/home/datasets/ZOUbohao/Proj1-Deepurify/CAMI-Ensemble-Deepurify/"
    output_folder = "/home/datasets/ZOUbohao/Proj1-Deepurify/CAMI-Ensemble-Deepurify/"
    if os.path.exists(output_folder) is False:
        os.mkdir(output_folder)
    folders = []
    with open("./IBS_ids.txt", "r") as rh:
        for line in rh:
            folders.append(line.strip("\n"))
    # folders_list = splitListEqually(folders, 4)
    folders = os.listdir(input_folders)
    folders = ["CAMI_high_norebin", "CAMI_medium1_norebin", "CAMI_medium2_norebin", "CAMI_low_norebin"]
    skip = []
    for i, folder in enumerate(folders):
        if folder in skip or "checkm2" in folder or "gunc" in folder: 
            continue
        # folder = folder.split("_")[0]
        if mode.lower() == "o":
            cur_input = os.path.join(input_folders, folder, "DeepurifyTmpFiles/deconta_tmp/re_cluster_-1") # for original 
        elif mode.lower() == "md":
            cur_input = os.path.join(input_folders, f"{folder}")  # for magpurify and mdmcleaner
        else:
            cur_input = os.path.join(input_folders, f"{folder}_nodrep")
        print(f"##### {folder}, {cur_input} {i} / {len(folders)} ##### ")
        gunc_folder = os.path.join(output_folder, f"{folder}_gunc")
        if os.path.exists(os.path.join(gunc_folder, "GUNC.progenomes_2.1.maxCSS_level.tsv")):
            continue
        cur_drep_out = os.path.join(output_folder, f"{folder}_drep")
        if os.path.exists(cur_drep_out) is False:
            os.mkdir(cur_drep_out)
        runDeRep(cur_drep_out, cur_input, 64)
        # checkm2_folder = os.path.join(output_folder, f"{folder}_checkm2")
        # runCheckm2Single(cur_input, checkm2_folder, "fasta", 128)
        # meta_info = readCheckm2Res(os.path.join(checkm2_folder, "quality_report.tsv"))[0]
        meta_info = readMetaInfo(os.path.join(cur_input, "MetaInfo.tsv"))[0]
        ## drep 
        derep_csv = os.path.join(cur_drep_out, "data_tables", "Cdb.csv")
        collect = {}
        csv_info = readCSV(derep_csv)[1:]
        for info in csv_info:
            c = info[0]
            n = info[1]
            prefix, _ = os.path.splitext(n)
            prefix = n
            if prefix in meta_info:
                q = meta_info[prefix]
            else:
                q = (0., 0., "LowQuality")
            if c not in collect:
                collect[c] = [(n, q, getScore(q))]
            else:
                collect[c].append((n, q, getScore(q)))
        res = []
        for c, q_l in collect.items():
            res.append(list(sorted(q_l, key=lambda x: x[-1], reverse=True))[0])
        wh = open(os.path.join(cur_drep_out, "MetaInfo.tsv"), "w")
        for i, r in enumerate(res):
            outName = f"Deepurify_Bin_{i}.fasta"
            wh.write(outName
                    + "\t"
                    + str(r[1][0])
                    + "\t"
                    + str(r[1][1])
                    + "\t"
                    + str(r[1][2])
                    + "\n")
            copy(os.path.join(cur_input, r[0]), 
                os.path.join(cur_drep_out, outName))
        ## gunc
        wh.close()
        if os.path.exists(gunc_folder) is False:
            os.mkdir(gunc_folder)
        runGUNCsingle(cur_drep_out, gunc_folder, 128, "fasta")
        
