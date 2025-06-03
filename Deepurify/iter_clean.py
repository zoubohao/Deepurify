
import os
from typing import Dict, List, Union

from Deepurify.decontamination import binning_purify
from Deepurify.Utils.BuildFilesUtils import (collect_all_deconta_results,
                                            filterSpaceInFastaFile, 
                                            process_galah_result)
from Deepurify.Utils.RunCMDUtils import runGalah


def run_iter_clean(
    contig_fasta_path: str,
    sorted_bam_file,
    concat_vec_path,
    concat_annot_path,
    tempFileOutFolder: str,
    outputBinFolder: str,
    modelWeightPath: str,
    taxoVocabPath: str,
    taxoTreePath: str,
    taxoName2RepNormVecPath: str,
    hmmModelPath: str,
    phy2accsPath: str,
    mer3Path: str,
    mer4Path: str,
    checkm2_db_path: str,
    gpus_work_ratio: List[float],
    batch_size_per_gpu: List[float],
    each_gpu_threads: int,
    overlapping_ratio: float,
    cut_seq_length: int,
    seq_length_threshold: int,
    topkORgreedy: str,
    topK: int,
    model_config: Union[Dict, None] = None,
    num_process: int = None,
    binning_mode = None
):
    de_temp_folder = os.path.join(tempFileOutFolder, "deconta_tmp")
    if os.path.exists(de_temp_folder) is False:
        os.mkdir(de_temp_folder)
    signal = False
    with open(contig_fasta_path, "r") as rh:
        for line in rh:
            if " " in line:
                signal = True
            break
    output_fasta_path = os.path.join(tempFileOutFolder, "filtered_space_in_name.contigs.fasta")
    if signal and os.path.exists(output_fasta_path) is False:
        print("========================================================================")
        print("--> !!! WARNING !!! Find space in the contig name. Make sure the first string of contig name is unique in fasta file.")
        filterSpaceInFastaFile(contig_fasta_path, output_fasta_path)
        contig_fasta_path = output_fasta_path

    binning_purify(
        contig_fasta_path,
        concat_annot_path,
        concat_vec_path,
        tempFileOutFolder,
        de_temp_folder,
        sorted_bam_file,
        modelWeightPath,
        taxoVocabPath,
        taxoTreePath,
        taxoName2RepNormVecPath,
        hmmModelPath,
        phy2accsPath,
        mer3Path,
        mer4Path,
        checkm2_db_path,
        gpus_work_ratio,
        batch_size_per_gpu,
        each_gpu_threads,
        overlapping_ratio,
        cut_seq_length,
        seq_length_threshold,
        topkORgreedy,
        topK,
        model_config,
        num_process,
        binning_mode
    )

    print("============================================================")
    print("--> Start Galah Filtering Replicated MAGs.")
    # Drep gather and filter results
    derep_g_folder = os.path.join(tempFileOutFolder, "derep_genomes")
    if os.path.exists(derep_g_folder) is False:
        os.mkdir(derep_g_folder)
    derep_out = os.path.join(tempFileOutFolder, "derep_out_info")
    if os.path.exists(derep_out) is False:
        os.mkdir(derep_out)
    collect_all_deconta_results(de_temp_folder, derep_g_folder, "fasta")
    galah_tsv = os.path.join(derep_out, "clusters.tsv")
    if os.path.exists(galah_tsv) is False:
        runGalah(derep_out, derep_g_folder, 64, "fasta")
    if os.path.exists(outputBinFolder) is False:
        os.makedirs(outputBinFolder)
    process_galah_result(derep_g_folder, galah_tsv, outputBinFolder)
    