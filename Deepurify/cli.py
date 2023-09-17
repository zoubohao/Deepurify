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
        prog=os.path.basename(sys.argv[0]), description="Deepurify is a tool to remove the contamination in MAGs."
    )
    subparsers = myparser.add_subparsers(dest="command")

    clean_parser = subparsers.add_parser("clean", help="Filtering the contamination in MAGs.")

    # Add parameters
    clean_parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        help="The input folder containing MAGs")
    clean_parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="The output folder containing decontaminated MAGs.")
    clean_parser.add_argument(
        "--bin_suffix",
        required=True,
        help="The suffix of MAG files.",
        type=str)


    ### optional ###
    clean_parser.add_argument(
        "--gpu_num",
        default=1,
        help="""The number of GPUs to be used, with the default value being 1. 
        Setting it to 0 will make the code use the CPU, but it's important to note that using the CPU can result in significantly slower processing speeds. 
        For better performance, it is recommended to have at least one GPU with a memory capacity of 3GB or more.""",
        type=int
    )
    clean_parser.add_argument(
        "--batch_size_per_gpu",
        default=1,
        help="""The --batch_size_per_gpu defines the number of sequences loaded onto each GPU simultaneously. 
        This parameter is relevant only when the --gpu_num option is set to a value greater than 0. 
        The default batch size is 1, which means that one sequence will be loaded per GPU by default.
        """,
        type=int)
    clean_parser.add_argument(
        "--num_threads_per_device",
        default=1,
        type=int,
        help="""The --num_threads_per_device (GPU or CPU) controls the level of parallelism during the contigs' lineage inference stage. 
        If the --gpu_num option is set to a value greater than 0, each GPU will utilize this specified number of threads for inference. 
        Likewise, if --gpu_num is set to 0 and the code runs on a CPU, the same number of threads will be employed. 
        By default, each GPU or CPU uses 1 thread. 
        The --batch_size_per_gpu value will be divided by this value to calculate the batch size per thread.
        """
    )
    clean_parser.add_argument(
        "--num_threads_call_genes",
        default=12,
        type=int,
        help="The number of threads to call genes. Defaults to 12.")
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
        "--temp_output_folder",
        default=None,
        type=str,
        help="""
        The folder stores the temporary files, which are generated during the running Deepurify. 
        If no path is provided (set to None), the temporary files will be stored in the parent folder of the '--input_bin_folder_path' location by default.
        """,
    )
    clean_parser.add_argument(
        "--output_bins_meta_info_path",
        default=None,
        type=str,
        help="""
        The path for a text file to record meta information, including the evaluated results of the output MAGs.
        If no path is provided (set to None), the file will be automatically created in the '--output_bin_folder_path' directory by default.
        """,
    )
    clean_parser.add_argument(
        "--info_files_path",
        default=None,
        help="""
        The files in the 'DeepurifyInfoFiles' folder are a crucial requirement for running Deepurify. 
        If you don't provide a path explicitly (set to None), it is assumed that the 'DeepurifyInfoFiles' environment variable has been properly configured to point to the necessary folder. 
        Ensure that the 'DeepurifyInfoFiles' environment variable is correctly set up if you don't specify the path.
        """,
        type=str
    )
    clean_parser.add_argument(
        "--simulated_MAG",
        default=False,
        type=bool,
        choices=[True, False],
        help="""If the input MAGs are simulated MAGs. False by default.
        This option is valuable when you have prior knowledge of core and contaminated contigs in simulated MAGs or prefer to personally assess the results. 
        When it sets to True, we will exclude contaminated contigs and retain core contigs using varying cosine similarity thresholds for each MAG. 
        Multiple sets of results will be generated in different folders within the '/temp_output_folder/FilterOutput/' directory. 
        You should independently evaluate these different results and select the MAGs that exhibit the best performance.
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
            0.5,
            8192,
            args.num_threads_call_genes,
            0.6,
            0.4,
            0.55,
            550000,
            args.checkM_process_num,
            args.num_threads_per_checkm,
            "topk",
            3,
            args.temp_output_folder,
            args.output_bins_meta_info_path,
            args.info_files_path,
            None,
            None,
            None,
            None,
            None,
            None,
            args.simulated_MAG
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
