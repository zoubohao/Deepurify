
from Deepurify.Utils.BuildFilesUtils import random_split_fasta
from Deepurify.Utils.IOUtils import readFasta
from Deepurify.clean_func import cleanMAGs

import numpy as np

if __name__ == "__main__":
    
    fasta_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Data-CAMI2-Marine-contigs-bam/marine-sample-0.contigs.fasta"
    contigname2seq = readFasta(fasta_path)
    new_contigname2seq = {}
    for contigname, seq in contigname2seq.items():
        if len(seq) >= 1000:
            new_contigname2seq[contigname] = seq
    random_split_fasta(new_contigname2seq, "/home/datasets/ZOUbohao/Proj1-Deepurify/test-deepurify-random-split")

    # input_bins_folder = "/home/datasets/ZOUbohao/Proj1-Deepurify/CAMI/CAMI_low.contigs.fasta"
    # sorted_bam_file = "/home/datasets/ZOUbohao/Proj1-Deepurify/CAMI/CAMI_low.sorted.bam"
    # output_folder = "/home/datasets/ZOUbohao/Proj1-Deepurify/test-deepurify-v2.4.3-cami-low"
    # bin_suffix = "fa"

    # cleanMAGs(
    #     output_bin_folder_path=output_folder,
    #     input_bins_folder = input_bins_folder,
    #     bin_suffix = "fa",
    #     cuda_device_list=["1", "2"],
    #     gpu_work_ratio=[0.5, 0.5],
    #     batch_size_per_gpu=2,
    #     each_gpu_threads=2,
    # )
