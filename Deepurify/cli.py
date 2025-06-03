#!/usr/bin/env python3


import argparse
import os
import sys
from typing import Dict, List

from Deepurify.clean_func import cleanMAGs
from Deepurify.Utils.DataUtils import insert
from Deepurify.version import deepurify_v

def bulid_tree(weight_file_path: str) -> Dict:
    def split_func(oneLine: str) -> List:
        levelsInfor = oneLine.split("@")
        return levelsInfor

    taxonomyTree = {"TaxoLevel": "root", "Name": "root", "Children": []}
    with open(weight_file_path, mode="r") as rh:
        k = 0
        for line in rh:
            info = line.strip("\n").split("\t")
            insert(split_func(info[0]), taxonomyTree)
        k += 1
    return taxonomyTree


def build_taxo_vocabulary(weight_file_path: str) -> Dict[str, int]:
    vocab_dict = {"[PAD]": 0}
    k = 1
    with open(weight_file_path, "r") as rh:
        for line in rh:
            split_info = line.strip("\n").split("@")
            for word in split_info:
                vocab_dict[word] = k
                k += 1
    return vocab_dict


def main():
    print(f"Deepurify version: *** {deepurify_v} ***")
    myparser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]), description="Deepurify is a tool to improving the quality of MAGs."
    )
    subparsers = myparser.add_subparsers(dest="command")

    clean_parser = subparsers.add_parser("clean", help="The **CLEAN** mode. Only clean the MAGs in the input folder.")

    # Add parameters
    clean_parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        help="The folder of input MAGs.")
    clean_parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="The folder used to output cleaned MAGs.")
    clean_parser.add_argument(
        "--bin_suffix",
        required=True,
        help="The bin suffix of MAG files.",
        type=str)

    ### optional ###
    clean_parser.add_argument(
        "-db",
        "--db_folder_path",
        default=None,
        help="""
        The path of database folder.
        By default, if no path is provided (i.e., set to None), it is expected that the environment variable 'DeepurifyInfoFiles' has been set to 
        point to the appropriate folder. 
        Please ensure that the 'DeepurifyInfoFiles' environment variable is correctly configured if the path is not explicitly provided.
        """,
        type=str
    )
    clean_parser.add_argument(
        "--gpu_num",
        default=1,
        help="""The number of GPUs to be used can be specified. Defaults to 1.
        If you set it to 0, the code will utilize the CPU. 
        However, please note that using the CPU can result in significantly slower processing speed. 
        It is recommended to provide at least one GPU (>= GTX-1060-6GB) for accelerating the speed.""",
        type=int
    )
    clean_parser.add_argument(
        "--cuda_device_list",
        default=None,
        type=str,
        nargs="+",
        help=" The gpu id that you want to apply. " + \
            "You can set '0 1' to use gpu0 and gpu1. The code would auto apply GPUs if it is None. Default to None.")
    clean_parser.add_argument(
        "--batch_size_per_gpu",
        default=4,
        help="""The batch size per GPU determines the number of sequences that will be loaded onto each GPU. 
        This parameter is only applicable if the --gpu_num option is set to a value greater than 0. 
        The default value is 4, meaning that one sequences will be loaded per GPU batch.
        The batch size for CPU is 4.
        """,
        type=int)
    clean_parser.add_argument(
        "--each_gpu_threads",
        default=1,
        type=int,
        help="""The number of threads per GPU (or CPU) determines the parallelism level during contigs' inference stage. 
        If the value of --gpu_num is greater than 0, each GPU will have a set number of threads to do inference. 
        Similarly, if --gpu_num is set to 0 and the code will run on CPU, the specified number of threads will be used. 
        By default, the number of threads per GPU or CPU is set to 1. 
        The --batch_size_per_gpu value will be divided by the number of threads to determine the batch size per thread.
        """
    )
    clean_parser.add_argument(
        "--overlapping_ratio",
        default=0.5,
        type=float,
        help="""The --overlapping_ratio is a parameter used when the length of a contig exceeds the specified --cut_seq_length. 
        By default, the overlapping ratio is set to 0.5. 
        This means that when a contig is longer than the --cut_seq_length, it will be split into overlapping subsequences with 0.5 overlap 
        between consecutive subsequences.
        """
    )
    clean_parser.add_argument(
        "--cut_seq_length",
        default=8192,
        type=int,
        help="""The --cut_seq_length parameter determines the length at which a contig will be cut if its length exceeds this value. 
        The default setting is 8192, which is also the maximum length allowed during training. 
        If a contig's length surpasses this threshold, it will be divided into smaller subsequences with lengths equal to or less 
        than the cut_seq_length.
        """)
    clean_parser.add_argument(
        "--mag_length_threshold",
        default=200000,
        type=int,
        help="""The threshold for the total length of a MAG's contigs is used to filter generated MAGs after applying single-copy genes (SCGs). 
        The default threshold is set to 200,000, which represents the total length of the contigs in base pairs (bp). 
        MAGs with a total contig length equal to or greater than this threshold will be considered for further analysis or inclusion, 
        while MAGs with a total contig length below the threshold may be filtered out.
        """,
    )
    clean_parser.add_argument(
        "--num_process",
        default=None,
        type=int,
        help="The maximum number of threads will be used. All CPUs will be used if it is None. Defaults to None")
    clean_parser.add_argument(
        "--topk_or_greedy_search",
        default="topk",
        choices=["topk", "greedy"],
        type=str,
        help="""Topk searching or greedy searching to label a contig. Defaults to "topk".
        """
    )
    clean_parser.add_argument(
        "--topK_num",
        default=3,
        type=int,
        help="""During the top-k searching approach, the default behavior is to search for the top-k nodes that exhibit the 
        highest cosine similarity with the contig's encoded vector. By default, the value of k is set to 3, meaning that the three most similar 
        nodes in terms of cosine similarity will be considered for labeling the contig. 
        Please note that this parameter does not have any effect when using the greedy search approach (topK_num=1). Defaults to 3.
        """)
    clean_parser.add_argument(
        "--temp_output_folder",
        default=None,
        type=str,
        help="""
        The temporary files generated during the process can be stored in a specified folder path. 
        By default, if no path is provided (i.e., set to None), the temporary files will be stored in the parent folder of the '--input_path' location. 
        However, you have the option to specify a different folder path to store these temporary files if needed.
        """,
    )

    ######################
    #### build parser ####
    ######################

    re_bin_parser = subparsers.add_parser("iter-clean", help="The **iter-clean** mode. Binning the contigs and cleaning the MAGs with applying the iter-clean strategy." +
                                          " This mode can ensemble (or apply single binner) the binning results from different binners. Make sure there is no space in the contigs' names.")
    # Add parameter
    re_bin_parser.add_argument(
        "-c",
        "--contigs_path",
        required=True,
        help="The contigs fasta path.")
    re_bin_parser.add_argument(
        "-b",
        "--sorted_bam_path",
        required=True,
        help="The sorted bam path.")
    re_bin_parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="The folder used to output cleaned MAGs.")

    ### optional ###
    re_bin_parser.add_argument(
        "-db",
        "--db_folder_path",
        default=None,
        help="""
        The path of database folder.
        By default, if no path is provided (i.e., set to None), it is expected that the environment variable 'DeepurifyInfoFiles' has been set to 
        point to the appropriate folder. 
        Please ensure that the 'DeepurifyInfoFiles' environment variable is correctly configured if the path is not explicitly provided.
        """,
        type=str
    )
    re_bin_parser.add_argument(
        "--binning_mode",
        default=None,
        help="The semibin2, concoct, metabat2 will all be run if this parameter is None." +
        " The other modes are: 'semibin2', 'concoct', and 'metabat2'. Defaults to None.",
        type=str)
    re_bin_parser.add_argument(
        "--gpu_num",
        default=1,
        help="""The number of GPUs to be used can be specified. Defaults to 1.
        If you set it to 0, the code will utilize the CPU. 
        However, please note that using the CPU can result in significantly slower processing speed. 
        It is recommended to provide at least one GPU (>= GTX-1060-6GB) for accelerating the speed.""",
        type=int
    )
    re_bin_parser.add_argument(
        "--cuda_device_list",
        default=None,
        type=str,
        nargs="+",
        help=" The gpu id that you want to apply. " + \
            "You can set '0 1' to use gpu0 and gpu1. The code would auto apply GPUs if it is None. Default to None.")
    re_bin_parser.add_argument(
        "--batch_size_per_gpu",
        default=4,
        help="""The batch size per GPU determines the number of sequences that will be loaded onto each GPU. 
        This parameter is only applicable if the --gpu_num option is set to a value greater than 0. 
        The default value is 4, meaning that one sequences will be loaded per GPU batch.
        The batch size for CPU is 4.
        """,
        type=int)
    re_bin_parser.add_argument(
        "--each_gpu_threads",
        default=1,
        type=int,
        help="""The number of threads per GPU (or CPU) determines the parallelism level during contigs' inference stage. 
        If the value of --gpu_num is greater than 0, each GPU will have a set number of threads to do inference. 
        Similarly, if --gpu_num is set to 0 and the code will run on CPU, the specified number of threads will be used. 
        By default, the number of threads per GPU or CPU is set to 1. 
        The --batch_size_per_gpu value will be divided by the number of threads to determine the batch size per thread.
        """
    )
    re_bin_parser.add_argument(
        "--overlapping_ratio",
        default=0.5,
        type=float,
        help="""The --overlapping_ratio is a parameter used when the length of a contig exceeds the specified --cut_seq_length. 
        By default, the overlapping ratio is set to 0.5. 
        This means that when a contig is longer than the --cut_seq_length, it will be split into overlapping subsequences with 0.5 overlap 
        between consecutive subsequences.
        """
    )
    re_bin_parser.add_argument(
        "--cut_seq_length",
        default=8192,
        type=int,
        help="""The --cut_seq_length parameter determines the length at which a contig will be cut if its length exceeds this value. 
        The default setting is 8192, which is also the maximum length allowed during training. 
        If a contig's length surpasses this threshold, it will be divided into smaller subsequences with lengths equal to or less 
        than the cut_seq_length.
        """)
    re_bin_parser.add_argument(
        "--mag_length_threshold",
        default=200000,
        type=int,
        help="""The threshold for the total length of a MAG's contigs is used to filter generated MAGs after applying single-copy genes (SCGs). 
        The default threshold is set to 200,000, which represents the total length of the contigs in base pairs (bp). 
        MAGs with a total contig length equal to or greater than this threshold will be considered for further analysis or inclusion, 
        while MAGs with a total contig length below the threshold may be filtered out.
        """,
    )
    re_bin_parser.add_argument(
        "--num_process",
        default=None,
        type=int,
        help="The maximum number of threads will be used. All CPUs will be used if it is None. Defaults to None")
    re_bin_parser.add_argument(
        "--topk_or_greedy_search",
        default="topk",
        choices=["topk", "greedy"],
        type=str,
        help="""Topk searching or greedy searching to label a contig. Defaults to "topk".
        """
    )
    re_bin_parser.add_argument(
        "--topK_num",
        default=3,
        type=int,
        help="""During the top-k searching approach, the default behavior is to search for the top-k nodes that exhibit the 
        highest cosine similarity with the contig's encoded vector. By default, the value of k is set to 3, meaning that the three most similar 
        nodes in terms of cosine similarity will be considered for labeling the contig. 
        Please note that this parameter does not have any effect when using the greedy search approach (topK_num=1). Defaults to 3.
        """)
    re_bin_parser.add_argument(
        "--temp_output_folder",
        default=None,
        type=str,
        help="""
        The temporary files generated during the process can be stored in a specified folder path. 
        By default, if no path is provided (i.e., set to None), the temporary files will be stored in the parent folder of the '--input_path' location. 
        However, you have the option to specify a different folder path to store these temporary files if needed.
        """,
    )

    ### main part ###
    args = myparser.parse_args()
    if args.command == "clean":
        gpu_num_int = int(args.gpu_num)
        if gpu_num_int == 0:
            gpu_work_ratio = []
        else:
            s_ratio = 1. / float(args.gpu_num)
            gpu_work_ratio = [s_ratio for _ in range(int(args.gpu_num) - 1)]
            gpu_work_ratio.append(1. - sum(gpu_work_ratio))
        cleanMAGs(
            output_bin_folder_path=args.output_path,
            cuda_device_list=args.cuda_device_list,
            gpu_work_ratio_list=gpu_work_ratio,
            batch_size_per_gpu=args.batch_size_per_gpu,
            each_gpu_threads=args.each_gpu_threads,
            input_bins_folder=args.input_path,
            bin_suffix=args.bin_suffix,
            contig_fasta_path=None,
            sorted_bam_file=None,
            binning_mode=None,
            overlapping_ratio=args.overlapping_ratio,
            cut_seq_length=args.cut_seq_length,
            seq_length_threshold=args.mag_length_threshold,
            topk_or_greedy=args.topk_or_greedy_search,
            topK_num=args.topK_num,
            num_process=args.num_process,
            temp_output_folder=args.temp_output_folder,
            db_files_path=args.db_folder_path,
        )
    elif args.command == "iter-clean":
        gpu_num_int = int(args.gpu_num)
        if gpu_num_int == 0:
            gpu_work_ratio = []
        else:
            s_ratio = 1. / float(args.gpu_num)
            gpu_work_ratio = [s_ratio for _ in range(int(args.gpu_num) - 1)]
            gpu_work_ratio.append(1. - sum(gpu_work_ratio))
        cleanMAGs(
            output_bin_folder_path=args.output_path,
            cuda_device_list=args.cuda_device_list,
            gpu_work_ratio_list=gpu_work_ratio,
            batch_size_per_gpu=args.batch_size_per_gpu,
            each_gpu_threads=args.each_gpu_threads,
            input_bins_folder=None,
            bin_suffix=None,
            contig_fasta_path=args.contigs_path,
            sorted_bam_file=args.sorted_bam_path,
            binning_mode=args.binning_mode,
            overlapping_ratio=args.overlapping_ratio,
            cut_seq_length=args.cut_seq_length,
            seq_length_threshold=args.mag_length_threshold,
            topk_or_greedy=args.topk_or_greedy_search,
            topK_num=args.topK_num,
            num_process=args.num_process,
            temp_output_folder=args.temp_output_folder,
            db_files_path=args.db_folder_path
        )
    else:
        print("#################################")
        print("### RUN THE DEEPURIFY PROJECT ###")
        print("#################################")
        print()
        print(f"Deepurify version: *** {deepurify_v} ***")
        print("Please use 'deepurify -h' for helping.")

# if __name__ == "__main__":
#     main()
