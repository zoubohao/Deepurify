#!/usr/bin/env python3


import argparse
import os
import sys
from typing import Dict, List, Union
from Deepurify.IOUtils import writePickle

from Deepurify.RUN_Functions import runLabelFilterSplitBins
from Deepurify.DataTools.DataUtils import insert


def clean(
    input_bin_folder_path: str,
    output_bin_folder_path: str,
    info_files_path: Union[str, None],
    bin_suffix: str,
    gpu_num: int,
    batch_size_per_gpu: int,
    num_worker: int,
    overlapping_ratio: float,
    cutSeqLength=8192,
    num_cpus_call_genes=64,
    hmm_acc_cutoff=0.7,
    hmm_align_ratio_cutoff=0.4,
    estimate_completeness_threshold=0.5,
    seq_length_threshold=320000,
    checkM_parallel_num=3,
    num_cpus_per_checkm=25,
    dfs_or_greedy="dfs",
    topK=3,
    temp_output_folder: Union[str, None] = None,
    output_bins_meta_info_path: Union[str, None] = None,
    modelWeightPath: Union[str, None] = None,
    taxoVocabPath: Union[str, None] = None,
    taxoTreePath: Union[str, None] = None,
    taxoName2RepNormVecPath: Union[str, None] = None,
    hmmModelPath: Union[str, None] = None,
):
    """_summary_

    Args:
        input_bin_folder_path (str): _description_
        output_bin_folder_path (str): _description_
        info_files_path (Union[str, None]): _description_
        bin_suffix (str): _description_
        gpu_num (int): _description_
        batch_size_per_gpu (int): _description_
        num_worker (int): _description_
        overlapping_ratio (float): _description_
        cutSeqLength (int, optional): _description_. Defaults to 8192.
        num_cpus_call_genes (int, optional): _description_. Defaults to 64.
        hmm_acc_cutoff (float, optional): _description_. Defaults to 0.7.
        hmm_align_ratio_cutoff (float, optional): _description_. Defaults to 0.4.
        estimate_completeness_threshold (float, optional): _description_. Defaults to 0.5.
        seq_length_threshold (int, optional): _description_. Defaults to 320000.
        checkM_parallel_num (int, optional): _description_. Defaults to 3.
        num_cpus_per_checkm (int, optional): _description_. Defaults to 25.
        dfs_or_greedy (str, optional): _description_. Defaults to "dfs".
        topK (int, optional): _description_. Defaults to 3.
        temp_output_folder (Union[str, None], optional): _description_. Defaults to None.
        output_bins_meta_info_path (Union[str, None], optional): _description_. Defaults to None.
        modelWeightPath (Union[str, None], optional): _description_. Defaults to None.
        taxoVocabPath (Union[str, None], optional): _description_. Defaults to None.
        taxoTreePath (Union[str, None], optional): _description_. Defaults to None.
        taxoName2RepNormVecPath (Union[str, None], optional): _description_. Defaults to None.
        hmmModelPath (Union[str, None], optional): _description_. Defaults to None.
    """
    if info_files_path is None:
        info_files_path = os.environ["DeepurifyInfoFiles"]
    if "/" == input_bin_folder_path[-1]:
        input_bin_folder_path = input_bin_folder_path[0:-1]
    filesFolder = os.path.split(input_bin_folder_path)[0]
    if temp_output_folder is None:
        temp_output_folder = os.path.join(filesFolder, "DeepurifyTempOut")
    if output_bins_meta_info_path is None:
        output_bins_meta_info_path = os.path.join(output_bin_folder_path, "MetaInfo.txt")
    if gpu_num == 0:
        gpu_work_ratio = []
    else:
        gpu_work_ratio = [1.0 / gpu_num for _ in range(gpu_num - 1)]
        gpu_work_ratio = gpu_work_ratio + [1.0 - sum(gpu_work_ratio)]
    batch_size_per_gpu = [batch_size_per_gpu for _ in range(gpu_num)]
    if modelWeightPath is None:
        modelWeightPath = os.path.join(info_files_path, "CheckPoint", "Deepurify.ckpt")
    if taxoVocabPath is None:
        taxoVocabPath = os.path.join(info_files_path, "TaxonomyInfo",  "ProGenomesVocabulary.txt")
    if taxoTreePath is None:
        taxoTreePath = os.path.join(info_files_path, "TaxonomyInfo", "ProGenomesTaxonomyTree.pkl")
    if taxoName2RepNormVecPath is None:
        taxoName2RepNormVecPath = os.path.join(info_files_path, "PyObjs", "Deepurify_taxo_lineage_vector.pkl")
    if hmmModelPath is None:
        hmmModelPath = os.path.join(info_files_path, "HMM", "hmm_model.hmm")
    mer3Path = os.path.join(info_files_path, "3Mer_vocabulary.txt")
    mer4Path = os.path.join(info_files_path, "4Mer_vocabulary.txt")

    if os.path.exists(filesFolder) is False:
        print("Your input folder is not exist.")
        sys.exit(1)

    runLabelFilterSplitBins(
        inputBinFolder=input_bin_folder_path,
        tempFileOutFolder=temp_output_folder,
        outputBinFolder=output_bin_folder_path,
        outputBinsMetaFilePath=output_bins_meta_info_path,
        modelWeightPath=modelWeightPath,
        hmmModelPath=hmmModelPath,
        taxoVocabPath=taxoVocabPath,
        taxoTreePath=taxoTreePath,
        taxoName2RepNormVecPath=taxoName2RepNormVecPath,
        gpus_work_ratio=gpu_work_ratio,
        batch_size_per_gpu=batch_size_per_gpu,
        num_worker=num_worker,
        bin_suffix=bin_suffix,
        mer3Path=mer3Path,
        mer4Path=mer4Path,
        overlapping_ratio=overlapping_ratio,
        cutSeqLength=cutSeqLength,
        num_cpus_call_genes=num_cpus_call_genes,
        ratio_cutoff=hmm_align_ratio_cutoff,
        acc_cutoff=hmm_acc_cutoff,
        estimate_completeness_threshold=estimate_completeness_threshold,
        seq_length_threshold=seq_length_threshold,
        checkM_parallel_num=checkM_parallel_num,
        num_cpus_per_checkm=num_cpus_per_checkm,
        dfsORgreedy=dfs_or_greedy,
        topK=topK,
    )


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


if __name__ == "__main__":
    myparser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]), description="Deepurify is a tool to filter and split contamination in MAGs."
    )
    subparsers = myparser.add_subparsers(dest="command")
    clean_parser = subparsers.add_parser("clean", help="Filtering and spliting the contamination in MAGs.")
    clean_parser.add_argument("-i", "--input_path", required=True, help="The input path of fastas folder.")
    clean_parser.add_argument("-o", "--output_path", required=True, help="The output path of the final filtered MAGs.")
    ### optional ###
    clean_parser.add_argument(
        "--bin_suffix",
        default="fasta",
        help="The suffix of the MAG file.",
        type=str)
    clean_parser.add_argument(
        "--info_files_path",
        default=None,
        help="The path of information files for Deepurify. It always contains the weight of Deepurify, the taxonomy tree pickle file and so on." +
        "You can set enviroment variable path of \"DeepurifyInfoFiles\" and omit this parameter.",
        type=str
    )
    clean_parser.add_argument(
        "--gpu_num",
        default=0,
        help="The number of GPUs you want to use to accelarate the inference. It may slow if there is no GPU.",
        type=int
    )
    clean_parser.add_argument(
        "--batch_size_per_gpu",
        default=4,
        help="The batch size for per GPU. It is useless if --gpu_num is 0.",
        type=int)
    clean_parser.add_argument(
        "--num_worker",
        default=2,
        type=int,
        help="The number of worker for CPU or per GPU. The batch size would divide this number for per worker."
    )
    clean_parser.add_argument(
        "--overlapping_ratio",
        default=0.5,
        type=float,
        help="The overlapping ratio for splitting the contig if the length of it is longer than 8192."
    )
    clean_parser.add_argument(
        "--cut_seq_length",
        default=8192,
        type=int,
        help="The length to cut the contig if its length is longer than 8192.")
    clean_parser.add_argument(
        "--num_cpus_call_genes",
        default=64,
        type=int,
        help="The number of threads to call genes by using prodigal and hmmsearch.")
    clean_parser.add_argument(
        "--hmm_acc_cutoff",
        default=0.7,
        type=float,
        help="The cutoff value of the confidence of hmm model to treat the called gene's sequence as the corresponding gene.",
    )
    clean_parser.add_argument(
        "--hmm_align_ratio_cutoff",
        default=0.4,
        type=float,
        help="The cutoff ratio for the called gene's sequence aligned to the origianl gene's sequence.",
    )
    clean_parser.add_argument(
        "--estimate_completeness_threshold",
        default=0.5,
        type=float,
        help="We estimate the completeness for each filtered and splitted bins by using those called genes. "
        + "We would not output the filtered or splitted MAGs if the estimated value smaller than the threshold.",
    )
    clean_parser.add_argument(
        "--seq_length_threshold",
        default=320000,
        type=int,
        help="We would not output the filtered or splitted MAGs if the summed sequences length smaller than the threshold.",
    )
    clean_parser.add_argument(
        "--checkM_parallel_num",
        default=3,
        choices=[1, 2, 3, 6],
        type=int,
        help="The number of CheckM run simultaneously.")
    clean_parser.add_argument(
        "--num_cpus_per_checkm",
        default=25,
        type=int,
        help="The number of threads for each CheckM.")
    clean_parser.add_argument(
        "--dfs_or_greedy",
        default="dfs",
        choices=["dfs", "greedy"],
        type=str,
        help="Use greedy search or DFS search to label the taxonomy lineage of contig."
    )
    clean_parser.add_argument(
        "--topK",
        default=3,
        type=int,
        help="Search the the topK's node. It is useless if used greedy.")
    clean_parser.add_argument(
        "--temp_output_folder",
        default=None,
        type=str,
        help="The path of the temporary folder to store the intermediate file. It is as same as the input fastas folder by default.",
    )
    clean_parser.add_argument(
        "--output_bins_meta_info_path",
        default=None,
        type=str,
        help="The path of meta information file. It records the completeness, contamination, quality, annotation of each MAG.",
    )
    clean_parser.add_argument(
        "--model_weight_path",
        default=None,
        type=str,
        help="The weight path of the model.")
    clean_parser.add_argument(
        "--taxo_vocab_path",
        default=None,
        type=str,
        help="The path of the taxonomy vocabulary file. It can be bulit by 'build' command or using the original file in the 'info_files_path'.",
    )
    clean_parser.add_argument(
        "--taxo_tree_path",
        default=None,
        type=str,
        help="The path of the taxonomy tree file. It is similar with JSON. It can be built by 'build' command or using the original file in the 'info_files_path'.",
    )
    clean_parser.add_argument(
        "--taxo_lineage_vector_file_path",
        default=None,
        type=str,
        help="This file can be built automaticaly. It would be built and stored in this path if it does not exist. ",
    )
    clean_parser.add_argument(
        "--hmm_model_path",
        default=None,
        type=str,
        help="The path of hmm model for hmmsearch. You can build your own hmm model or using the original file in the 'info_files_path'",
    )

    #### build parser ####
    bulid_parser = subparsers.add_parser("bulid", help="Build the files like taxonomy tree and the taxonomy vocabulary.")
    bulid_parser.add_argument(
        "-i",
        "--input_weight_file_path",
        required=True,
        type=str,
        help="The path of the weight file. This weight file has two columns. " +
        "This first column is the taxonomy lineage of one species from phylum to species level, split with @ charactor. The second colums is the weight value of the species." +
        "The two columns are split with '\\t'.")
    bulid_parser.add_argument(
        "-ot",
        "--output_tree_path",
        type=str,
        required=True,
        help="The output path of the taxonomy tree that build from your file.")
    bulid_parser.add_argument(
        "-ov",
        "--output_vocabulary_path",
        type=str,
        required=True,
        help="the output path of the taxonomy vocabulary that build from your file.")

    ### main part ###
    args = myparser.parse_args()

    if args.command == "clean":
        clean(
            args.input_path,
            args.output_path,
            args.info_files_path,
            args.bin_suffix,
            args.gpu_num,
            args.batch_size_per_gpu,
            args.num_worker,
            args.overlapping_ratio,
            args.cut_seq_length,
            args.num_cpus_call_genes,
            args.hmm_acc_cutoff,
            args.hmm_align_ratio_cutoff,
            args.estimate_completeness_threshold,
            args.seq_length_threshold,
            args.checkM_parallel_num,
            args.num_cpus_per_checkm,
            args.dfs_or_greedy,
            args.topK,
            args.temp_output_folder,
            args.output_bins_meta_info_path,
            args.model_weight_path,
            args.taxo_vocab_path,
            args.taxo_tree_path,
            args.taxo_lineage_vector_file_path,
            args.hmm_model_path
        )

    elif args.command == "build":
        taxo_tree = bulid_tree(args.input_weight_file_path)
        writePickle(args.output_tree_path, taxo_tree)
        vocab = build_taxo_vocabulary(args.input_weight_file_path)
        with open(args.output_vocabulary_path, "w") as wh:
            for word, index in vocab.items():
                wh.write(word+"\t"+str(index) + "\n")
    else:
        raise ValueError("Do not implement other command.")
