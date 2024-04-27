
import os
from shutil import rmtree
from typing import Dict, List, Union

from Deepurify.decontamination import binning_purify
from Deepurify.Utils.BuildFilesUtils import (collect_all_deconta_results,
                                             process_drep_result, filterSpaceInFastaFile)
from Deepurify.Utils.RunCMDUtils import runDeRep


def run_integration(
    contig_fasta_path: str,
    sorted_bam_file,
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
    if signal:
        print("========================================================================")
        print("--> !!! WARNING !!! Find space in the contig name. Make sure the first string of contig name is unique in fasta file.")
        output_fasta_path = os.path.join(tempFileOutFolder, "filtered_space_in_name.contigs.fasta")
        filterSpaceInFastaFile(contig_fasta_path, output_fasta_path)
        contig_fasta_path = output_fasta_path
    
    binning_purify(
        contig_fasta_path,
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
    print("--> Start dRep Filtering.")
    # Drep gather and filter results
    derep_g_folder = os.path.join(tempFileOutFolder, "derep_genomes")
    if os.path.exists(derep_g_folder) is False:
        os.mkdir(derep_g_folder)
    derep_out = os.path.join(tempFileOutFolder, "derep_out_info")
    if os.path.exists(derep_out) is False:
        os.mkdir(derep_out)
    collect_all_deconta_results(de_temp_folder, derep_g_folder, "fasta")
    runDeRep(derep_out, derep_g_folder, 64)
    derep_csv = os.path.join(derep_out, "data_tables", "Cdb.csv")
    # res_out_path = os.path.join(tempFileOutFolder, "RESULT")
    if os.path.exists(outputBinFolder) is False:
        os.makedirs(outputBinFolder)
    process_drep_result(
        derep_g_folder,
        derep_csv,
        outputBinFolder # outputBinFolder
    )
    
    