import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from shutil import rmtree
from Deepurify.clean_func import cleanMAGs

import numpy as np

if __name__ == "__main__":

    input_folder = "/home/datasets/ZOUbohao/Proj1-Deepurify/Deepurify_data_train_sim/SIM2"
    output_folder = "/home/datasets/ZOUbohao/Proj1-Deepurify/SIM2-RES/"

    if os.path.exists(output_folder) is False:
        os.mkdir(output_folder)

    folders = os.listdir(input_folder)
    for folder in folders:
        if "All_not_see_out" in folder:
            continue
        cur_input_folder = os.path.join(input_folder, folder)
        cur_output_folder = os.path.join(output_folder, folder)
        if os.path.join(cur_output_folder) is False:
            os.mkdir(cur_output_folder)
        print(f"{cur_output_folder}")

        cleanMAGs(
            output_bin_folder_path=cur_output_folder,
            batch_size_per_gpu=16,
            each_gpu_threads=2,
            # setting of contig inference stage
            input_bins_folder=cur_input_folder,
            bin_suffix="fasta",
            gpu_work_ratio=[0.25, 0.25, 0.25, 0.25],
            num_process=128,
            db_files_path="./GTDB_Taxa_Info/",
            model_weight_path="./GTDB_OOD_Taxa_info/GTDB_OOD_Mini_Model.pth",
            taxo_tree_path="./GTDB_OOD_Taxa_info/OOD_taxonomy_tree.pkl",
            taxo_vocab_path="./GTDB_OOD_Taxa_info/taxon_vocab.txt",
            simulated_MAG=True
        )
