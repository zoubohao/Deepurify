

import os
from typing import Dict, Union, List

from Deepurify.Utils.BuildFilesUtils import (filterSpaceInFastaFile, 
                                             random_split_fasta, 
                                             build_contigname2fasta)
from Deepurify.Utils.IOUtils import readFasta
from .decontamination import run_all_deconta_steps

def inference_once_time(
    contig_fasta_path: str,
    inputBinFolder: str,
    tempFileOutFolder: str,
    modelWeightPath: str,
    taxoVocabPath: str,
    taxoTreePath: str,
    taxoName2RepNormVecPath: str,
    hmmModelPath: str,
    phy2accsPath: str,
    bin_suffix: str,
    mer3Path: str,
    mer4Path: str,
    gpus_work_ratio: List[float],
    batch_size_per_gpu: List[float],
    each_gpu_threads: int,
    overlapping_ratio: float,
    cut_seq_length: int,
    seq_length_threshold: int,
    topkORgreedy: str,
    topK: int,
    min_length_for_infer:int = 1000,
    num_process = 64,
    concat_annot_file_path = None,
    concat_vec_file_path = None,
):
    random_split_contigs_folder = os.path.join(tempFileOutFolder, "random_split_fasta")
    if os.path.exists(random_split_contigs_folder) is False:
        os.mkdir(random_split_contigs_folder)
    
    ## this means we can split contigs from contig fast
    output_fasta_path = None
    print("--> Start to filter the space in contig name. Make sure the first string of contig name is unique in fasta file.")
    if contig_fasta_path is not None:
        output_fasta_path = os.path.join(tempFileOutFolder, "filtered_space_in_name.contigs.fasta")
        filterSpaceInFastaFile(contig_fasta_path, output_fasta_path)
        contigname2seq = readFasta(output_fasta_path)
        new_contigname2seq = {}
        for contigname, seq in contigname2seq.items():
            if len(seq) >= min_length_for_infer:
                new_contigname2seq[contigname] = seq
        random_split_fasta(new_contigname2seq, random_split_contigs_folder)
    else:
        contigname2seq = build_contigname2fasta(inputBinFolder, bin_suffix)
        random_split_fasta(contigname2seq, random_split_contigs_folder)
    
    cur_tempFileOutFolder = os.path.join(tempFileOutFolder, "infer_temp")
    if os.path.exists(cur_tempFileOutFolder) is False:
        os.mkdir(cur_tempFileOutFolder)
    run_all_deconta_steps(
        inputBinFolder=random_split_contigs_folder,
        tempFileOutFolder=cur_tempFileOutFolder,
        outputBinFolder=None,
        modelWeightPath=modelWeightPath,
        taxoVocabPath=taxoVocabPath,
        taxoTreePath=taxoTreePath,
        taxoName2RepNormVecPath=taxoName2RepNormVecPath,
        hmmModelPath=hmmModelPath,
        phy2accsPath=phy2accsPath,
        bin_suffix="fasta",
        mer3Path=mer3Path,
        mer4Path=mer4Path,
        checkm2_db_path=None,
        gpus_work_ratio=gpus_work_ratio,
        batch_size_per_gpu=batch_size_per_gpu,
        each_gpu_threads=each_gpu_threads,
        overlapping_ratio=overlapping_ratio,
        cut_seq_length=cut_seq_length,
        seq_length_threshold=seq_length_threshold,
        topkORgreedy=topkORgreedy,
        topK=topK,
        num_process=num_process,
        build_concat_file=True,
        concat_annot_file_path=concat_annot_file_path,
        concat_vec_file_path=concat_vec_file_path,
        simulated_MAG=False,
        just_annot=True
    )
    return output_fasta_path



