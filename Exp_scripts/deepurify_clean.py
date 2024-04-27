import argparse
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from Deepurify.clean_func import cleanMAGs


def splitListEqually(input_list, num_parts: int):
    n = len(input_list)
    step = n // num_parts + 1
    out_list = []
    for i in range(num_parts):
        if curList := input_list[i * step: (i + 1) * step]:
            out_list.append(curList)
    return out_list


if __name__ == "__main__":
    binnings = ["Concoct", "Metabeta2", "Semibin2"]
    datas = ["CAMI_high", "CAMI_medium1", "CAMI_medium2", "CAMI_low"]
    for binning in binnings:
        for data in datas:
            input_folder = os.path.join("/home/datasets/ZOUbohao/Proj1-Deepurify/CAMI-original", f"{binning}_{data}")
            output_folder = os.path.join("/home/datasets/ZOUbohao/Proj1-Deepurify/CAMI-original-deepurify-only-clean", f"{binning}_{data}")
            if os.path.exists(output_folder) is False:
                os.mkdir(output_folder)
            print(input_folder)
            print(output_folder)
            if os.path.exists(os.path.join(output_folder, "Deepurify_Bin_0.fasta")):
                continue
            # cd /home/datasets/ZOUbohao/Proj1-Deepurify/Deepurify-v2.2.0-ReBinning/
            # python deepurify.py -i ../plant_data/ -o ../plant-ensemble-Deepurify/ -n SRR10968246
            cleanMAGs(
                output_bin_folder_path=output_folder,
                batch_size_per_gpu=32,
                each_gpu_threads=2,
                # setting of contig inference stage
                input_bins_folder=input_folder,
                bin_suffix="fa",
                gpu_work_ratio=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125], # [0.25, 0.25, 0.25, 0.25]
                db_files_path="./GTDB_Taxa_Info/"
            )

