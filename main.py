import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from shutil import rmtree
from Deepurify.clean_func import cleanMAGs

import numpy as np

if __name__ == "__main__":

    input_folder = ""
    output_folder = ""

    data_names = os.listdir(input_folder)
    if os.path.exists(output_folder) is False:
        os.mkdir(output_folder)

    for data_name in data_names:
        cur_input_contigs = os.path.join(input_folder, f"{data_name}.contigs.fasta")
        cur_bam = os.path.join(input_folder, f"{data_name}.sorted.bam")
        cur_output_folder = os.path.join(output_folder, f"{data_name}")
        if os.path.join(cur_output_folder) is False:
            os.mkdir(cur_output_folder)
        print(f"{len(cur_input_contigs)}, {cur_bam}, {cur_output_folder}")

        cleanMAGs(
            output_bin_folder_path=cur_output_folder,
            batch_size_per_gpu=48,
            each_gpu_threads=4,
            # setting of contig inference stage
            contig_fasta_path=cur_input_contigs,
            sorted_bam_file=cur_bam,
            gpu_work_ratio=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
            num_process=64,
            db_files_path="./GTDB_Taxa_Info/"
        )
