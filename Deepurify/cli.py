#!/usr/bin/env python3


import argparse
import os
import sys
from typing import Dict, List
from Deepurify.IOUtils import writePickle

from Deepurify.clean_func import cleanMAGs
from Deepurify.DataTools.DataUtils import insert


def bulid_tree(weight_file_path: str) -> Dict:
    def split_func(oneLine: str) -> List:
        levelsInfor = oneLine.split("@")
        return levelsInfor

    taxonomyTree = {"TaxoLevel": "superkingdom", "Name": "bacteria", "Children": []}
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


def cli():
    myparser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]), description="Deepurify is a tool to improving the quality of MAGs."
    )
    subparsers = myparser.add_subparsers(dest="command")

    clean_parser = subparsers.add_parser("clean", help="Filtering the contamination in MAGs.")

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
    clean_parser.add_argument(
        "--gpu_num",
        default=1,
        help="""The number of GPUs to be used can be specified. Defaults to 1.
        If you set it to 0, the code will utilize the CPU. 
        However, please note that using the CPU can result in significantly slower processing speed. 
        It is recommended to provide at least one GPU for better performance.""",
        type=int
    )
    clean_parser.add_argument(
        "--batch_size_per_gpu",
        default=2,
        help="""The batch size per GPU determines the number of sequences that will be loaded onto each GPU. 
        This parameter is only applicable if the --gpu_num option is set to a value greater than 0. 
        The default value is 2, meaning that two sequences will be loaded per GPU batch.
        The batch size for CPU is 2.
        """,
        type=int)
    clean_parser.add_argument(
        "--num_threads_per_device",
        default=1,
        type=int,
        help="""The number of threads per GPU or CPU determines the parallelism level during contigs' inference stage. 
        If the value of --gpu_num is greater than 0, each GPU will have a set number of threads to do inference. 
        Similarly, if --gpu_num is set to 0 and the code is running on CPU, the specified number of threads will be used. 
        By default, the number of threads per GPU or CPU is set to 1. 
        The --batch_size_per_gpu value will be divided by the number of threads to determine the batch size per thread.
        """
    )

    ### optional ###
    clean_parser.add_argument(
        "--overlapping_ratio",
        default=0.5,
        type=float,
        help="""The --overlapping_ratio is a parameter used when the length of a contig exceeds the specified --cut_seq_length. 
        By default, the overlapping ratio is set to 0.5. 
        This means that when a contig is longer than the --cut_seq_length, it will be split into overlapping subsequences with 50\%\ overlap between consecutive subsequences.
        """
    )
    clean_parser.add_argument(
        "--cut_seq_length",
        default=8192,
        type=int,
        help="""The --cut_seq_length parameter determines the length at which a contig will be cut if its length exceeds this value. 
        The default setting is 8192, which is also the maximum length allowed during training. 
        If a contig's length surpasses this threshold, it will be divided into smaller subsequences with lengths equal to or less than the cut_seq_length.
        """)
    clean_parser.add_argument(
        "--num_threads_call_genes",
        default=12,
        type=int,
        help="The number of threads to call genes. Defaults to 12.")
    clean_parser.add_argument(
        "--hmm_acc_cutoff",
        default=0.6,
        type=float,
        help="""If the acc score and the aligned ratio assigned by the HMM model for a gene sequence exceeds this threshold, it would be considered as a single-copy gene.
        It is set to 0.6 by default.
        """,
    )
    clean_parser.add_argument(
        "--hmm_align_ratio_cutoff",
        default=0.4,
        type=float,
        help="""If the acc score and the aligned ratio assigned by the HMM model for a gene sequence exceeds this threshold, it would be considered as a single-copy gene.
        It is set to 0.4 by default.
        """,
    )
    clean_parser.add_argument(
        "--estimate_completeness_threshold",
        default=0.55,
        type=float,
        help="""The --estimate_completeness_threshold is used as a criterion for filtering MAGs that are generated by applying specific single-copy genes (SCGs). 
        The default threshold is set to 0.55. 
        MAGs with an estimated completeness score equal to or higher than this threshold will be considered for further analysis or inclusion, 
        while those falling below the threshold may be filtered out.
        """,
    )
    clean_parser.add_argument(
        "--seq_length_threshold",
        default=550000,
        type=int,
        help="""The threshold for the total length of a MAG's contigs is used to filter generated MAGs after applying single-copy genes (SCGs). 
        The default threshold is set to 550,000, which represents the total length of the contigs in base pairs (bp). 
        MAGs with a total contig length equal to or greater than this threshold will be considered for further analysis or inclusion, 
        while MAGs with a total contig length below the threshold may be filtered out.
        """,
    )
    clean_parser.add_argument(
        "--checkM_process_num",
        default=1,
        choices=[1, 2, 3, 6],
        type=int,
        help="The number of processes to run CheckM simultaneously. Defaults to 1.")
    clean_parser.add_argument(
        "--num_threads_per_checkm",
        default=12,
        type=int,
        help="The number of threads to run a single CheckM process. Defaults to 12.")
    clean_parser.add_argument(
        "--topk_or_greedy_search",
        default="topk",
        choices=["topk", "greedy"],
        type=str,
        help="""Topk searching or greedy searching to label a contig. Defaults to 'topk'.
        The contig is assigned a label based on the top-k most relevant or similar taxonomic lineages. 
        The specific number of lineages considered for labeling can be determined by the value of k.
        """
    )
    clean_parser.add_argument(
        "--topK_num",
        default=3,
        type=int,
        help="""
        During the top-k searching approach, the default behavior is to search for the top-k nodes that exhibit the highest cosine similarity with the contig's encoded vector. 
        By default, the value of k is set to 3, meaning that the three most similar nodes in terms of cosine similarity will be considered for labeling the contig. 
        Please note that this parameter does not have any effect when using the greedy search approach (topK_num=1).
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
    clean_parser.add_argument(
        "--output_bins_meta_info_path",
        default=None,
        type=str,
        help="""
        The path of a text file can be provided to record the meta information, including the evaluated results, of the final cleaned MAGs. 
        By default, if no path is specified (i.e., set to None), the file will be created under the "--output_path" directory. 
        However, you have the flexibility to specify a different file path if desired.
        """,
    )
    clean_parser.add_argument(
        "--info_files_path",
        default=None,
        help="""
        The DeepurifyInfoFiles is essential for running Deepurify. 
        By default, if no path is provided (i.e., set to None), it is expected that the environment variable 'DeepurifyInfoFiles' has been set to point to the appropriate folder. 
        Please ensure that the 'DeepurifyInfoFiles' environment variable is correctly configured if the path is not explicitly provided.
        """,
        type=str
    )
    clean_parser.add_argument(
        "--model_weight_path",
        default=None,
        type=str,
        help="The path of model weight. (In DeepurifyInfoFiles folder) Defaults to None.")
    clean_parser.add_argument(
        "--taxo_vocab_path",
        default=None,
        type=str,
        help="The path of taxon vocabulary. (In DeepurifyInfoFiles folder) Defaults to None.",
    )
    clean_parser.add_argument(
        "--taxo_tree_path",
        default=None,
        type=str,
        help="The path of taxonomic tree. (In DeepurifyInfoFiles folder) Defaults to None.",
    )
    clean_parser.add_argument(
        "--taxo_lineage_vector_file_path",
        default=None,
        type=str,
        help="The path of taxonomic lineage encoded vectors. (In DeepurifyInfoFiles folder) Defaults to None. ",
    )
    clean_parser.add_argument(
        "--hmm_model_path",
        default=None,
        type=str,
        help="The path of SCGs' hmm file. (In DeepurifyInfoFiles folder) Defaults to None.",
    )
    clean_parser.add_argument(
        "--self_evaluate",
        default=False,
        type=bool,
        choices=[True, False],
        help="""Evaluate the results by the user. Defaults to False. 
        Set to True if you have knowledge of clean and contaminated contigs in the simulated dataset or you want to evaluate the outcomes by yourself.
        We would remove the outlier contigs and only keep clean contigs with different cosine similarity threshold for a MAG if this variable is True.
        The outputs will be stored in the following folder path: /temp_output_folder/FilterOutput/
        You should independently evaluate the outcomes from various similarity threshold and select the best output from the cleaned MAGs.
        """)

    #### build parser ####
    bulid_parser = subparsers.add_parser(
        "build", help="(Do not use this command at present.) Build the files like taxonomy tree and the taxonomy vocabulary for training.")
    # Add parameter
    bulid_parser.add_argument(
        "-i",
        "--input_taxo_lineage_weight_file_path",
        required=True,
        type=str,
        help="The path of the taxonomic lineages weights file. This file has two columns. " +
        "This first column is the taxonomic lineage of one species from phylum to species level, split with @ charactor. \n" +
        "The second colums is the weight value of the species." +
        "The two columns are split with '\\t'.")
    bulid_parser.add_argument(
        "-ot",
        "--output_tree_path",
        type=str,
        required=True,
        help="The output path of the taxonomy tree that build from your taxonomic lineages weights file.")
    bulid_parser.add_argument(
        "-ov",
        "--output_vocabulary_path",
        type=str,
        required=True,
        help="the output path of the taxonomy vocabulary that build from your taxonomic lineages weights file.")

    ### main part ###
    args = myparser.parse_args()

    if args.command == "clean":
        cleanMAGs(
            args.input_path,
            args.output_path,
            args.bin_suffix,
            args.gpu_num,
            args.batch_size_per_gpu,
            args.num_threads_per_device,
            args.overlapping_ratio,
            args.cut_seq_length,
            args.num_threads_call_genes,
            args.hmm_acc_cutoff,
            args.hmm_align_ratio_cutoff,
            args.estimate_completeness_threshold,
            args.seq_length_threshold,
            args.checkM_process_num,
            args.num_threads_per_checkm,
            args.topk_or_greedy_search,
            args.topK_num,
            args.temp_output_folder,
            args.output_bins_meta_info_path,
            args.info_files_path,
            args.model_weight_path,
            args.taxo_vocab_path,
            args.taxo_tree_path,
            args.taxo_lineage_vector_file_path,
            args.hmm_model_path,
            None,
            args.self_evaluate
        )

    elif args.command == "build":
        taxo_tree = bulid_tree(args.input_weight_file_path)
        writePickle(args.output_tree_path, taxo_tree)
        vocab = build_taxo_vocabulary(args.input_weight_file_path)
        with open(args.output_vocabulary_path, "w") as wh:
            for word, index in vocab.items():
                wh.write(word+"\t"+str(index) + "\n")
    else:
        print("#################################")
        print("### RUN THE DEEPURIFY PROJECT ###")
        print("#################################")
        print()
        print("Please use 'deepurify -h' or 'deepurify clean -h' for helping.")
