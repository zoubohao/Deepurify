import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
from shutil import rmtree
from Deepurify.clean_func import cleanMAGs

import numpy as np

if __name__ == "__main__":

    input_bins_folder = "/home/datasets/ZOUbohao/Proj1-Deepurify/CAMI/CAMI_low.contigs.fasta"
    sorted_bam_file = "/home/datasets/ZOUbohao/Proj1-Deepurify/CAMI/CAMI_low.sorted.bam"
    output_folder = "/home/datasets/ZOUbohao/Proj1-Deepurify/test-v2.4.0-cami-low"
    bin_suffix = "fa"

    cleanMAGs(
        output_bin_folder_path=output_folder,
        gpu_work_ratio=[0.5, 0.5],
        batch_size_per_gpu=2,
        each_gpu_threads=2,
        contig_fasta_path=input_bins_folder,
        sorted_bam_file=sorted_bam_file
    )
