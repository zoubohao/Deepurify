import os
import sys
from typing import Dict, Union

from Deepurify.funcs import runLabelFilterSplitBins


def cleanMAGs(
    input_bin_folder_path: str,
    output_bin_folder_path: str,
    bin_suffix: str,
    gpu_num: int = 1,
    batch_size_per_gpu: int = 1,
    num_threads_per_device: int = 1,
    overlapping_ratio=0.5,
    cut_seq_length=8192,
    num_threads_call_genes=12,
    hmm_acc_cutoff=0.6,
    hmm_align_ratio_cutoff=0.4,
    estimated_completeness_threshold=0.55,
    seq_length_threshold=550000,
    checkM_process_num=1,
    num_threads_per_checkm=12,
    topk_or_greedy="topk",
    topK_num=3,
    temp_output_folder: Union[str, None] = None,
    output_bins_meta_info_path: Union[str, None] = None,
    info_files_path: Union[str, None] = None,
    modelWeightPath: Union[str, None] = None,
    taxoVocabPath: Union[str, None] = None,
    taxoTreePath: Union[str, None] = None,
    taxoName2RepNormVecPath: Union[str, None] = None,
    hmmModelPath: Union[str, None] = None,
    model_config: Union[Dict, None] = None,
    simulated_MAG: bool = False
):
    """

    The main function to clean the MAGs.

    Args:
        input_bin_folder_path (str): The input folder containing MAGs

        output_bin_folder_path (str): The output folder containing decontaminated MAGs.

        bin_suffix (str): The suffix of MAG files.

        gpu_num (int): The number of GPUs to be used, with the default value being 1. 
        Setting it to 0 will make the code use the CPU, but it's important to note that using the CPU can result in significantly slower processing speeds. 
        For better performance, it is recommended to have at least one GPU with a memory capacity of 3GB or more.

        batch_size_per_gpu (int): The --batch_size_per_gpu defines the number of sequences loaded onto each GPU simultaneously. 
        This parameter is relevant only when the --gpu_num option is set to a value greater than 0. 
        The default batch size is 1, which means that one sequence will be loaded per GPU by default.

        num_threads_per_device (int): The --num_threads_per_device (GPU or CPU) controls the level of parallelism during the contigs' lineage inference stage. 
        If the --gpu_num option is set to a value greater than 0, each GPU will utilize this specified number of threads for inference. 
        Likewise, if --gpu_num is set to 0 and the code runs on a CPU, the same number of threads will be employed. 
        By default, each GPU or CPU uses 1 thread. 
        The --batch_size_per_gpu value will be divided by this value to calculate the batch size per thread.

        overlapping_ratio (float): The --overlapping_ratio parameter comes into play when a contig's length exceeds the specified --cut_seq_length. 
        The default value for the overlapping ratio is set at 0.5. 
        This implies that if a contig surpasses the --cut_seq_length, it will be divided into overlapping subsequences with a 0.5 overlap between each consecutive subsequence.

        cut_seq_length (int, optional): The --cut_seq_length parameter sets the maximum length for contigs. 
        The default value is 8192, which is also the maximum length allowed during training. 
        If a contig's length exceeds this threshold, it will be split into smaller subsequences, each with a length equal to or less than the specified cut_seq_length.
        
        num_threads_call_genes (int, optional): The number of threads to call genes. Defaults to 12.

        hmm_acc_cutoff (float, optional): A gene sequence will be classified as a single-copy gene if both its accuracy (acc) score and aligned ratio, 
        as determined by the HMM model, surpass a specified threshold. 
        The default threshold is set to 0.6.

        hmm_align_ratio_cutoff (float, optional): A gene sequence will be classified as a single-copy gene if both its accuracy (acc) score and aligned ratio, 
        as determined by the HMM model, surpass a specified threshold. 
        The default threshold is set to 0.4.

        estimated_completeness_threshold (float, optional): The --estimate_completeness_threshold serves as a filter criterion for MAGs obtained through the application of specific single-copy gene.
        The default threshold is 0.55, meaning that MAGs with an estimated completeness score equal to or greater than this value will be retained for further analysis, 
        while those scoring below it would be excluded.

        seq_length_threshold (int, optional): The threshold for the cumulative length of contigs within a MAG, which is used to filter MAGs. 
        The default threshold is 550,000 bps. 
        MAGs with a cumulative contig length equal to or exceeding this threshold will be retained for further analysis, whereas those falling below the threshold would be excluded.

        checkM_process_num (int, optional): The number of processes to run CheckM simultaneously. Defaults to 1.

        num_threads_per_checkm (int, optional): The number of threads to run a single CheckM process. Defaults to 12.

        topk_or_greedy (str, optional): Topk searching or greedy searching to assign taxonomic lineage for a contig. Defaults to 'topk'.

        topK_num (int, optional): The k setting for topk searching. Default to 3.

        temp_output_folder (Union[str, None], optional): The folder stores the temporary files, which are generated during the running Deepurify. 
        If no path is provided (set to None), the temporary files will be stored in the parent folder of the '--input_bin_folder_path' location by default.

        output_bins_meta_info_path (Union[str, None], optional): The path for a text file to record meta information, including the evaluated results of the output MAGs.
        If no path is provided (set to None), the file will be automatically created in the '--output_bin_folder_path' directory by default.

        info_files_path (Union[str, None]): The files in the 'DeepurifyInfoFiles' folder are a crucial requirement for running Deepurify. 
        If you don't provide a path explicitly (set to None), it is assumed that the 'DeepurifyInfoFiles' environment variable has been properly configured to point to the necessary folder. 
        Ensure that the 'DeepurifyInfoFiles' environment variable is correctly set up if you don't specify the path.
        If you set this variable to None and we can not find 'DeepurifyInfoFiles' environment variable either, than you should manually input the path of running files. 

        modelWeightPath (Union[str, None], optional): The path of model weight. Defaults to None. (In DeepurifyInfoFiles folder)

        taxoVocabPath (Union[str, None], optional): The path of taxon vocabulary. Defaults to None. (In DeepurifyInfoFiles folder)

        taxoTreePath (Union[str, None], optional): The path of taxonomic tree. Defaults to None. (In DeepurifyInfoFiles folder)

        taxoName2RepNormVecPath (Union[str, None], optional): The path of taxonomic lineage encoded vectors. Defaults to None. 
        We can generate this file if this variable sets None. (In DeepurifyInfoFiles folder)

        hmmModelPath (Union[str, None], optional): The path of SCGs' hmm file. Defaults to None. (In DeepurifyInfoFiles folder)

        model_config (Union[Dict, None], optional): The config of model. See the TrainScript.py to find more information. Defaults to None.
        It would be used if you trained a another model with different model_config. 
        Please set this variable equal with None at present.

        simulated_MAG (bool, optional): If the input MAGs are simulated MAGs. False by default.
        This option is valuable when you have prior knowledge of core and contaminated contigs in simulated MAGs or prefer to personally assess the results. 
        When it sets to True, we will exclude contaminated contigs and retain core contigs using varying cosine similarity thresholds for each MAG. 
        Multiple sets of results will be generated in different folders within the '/temp_output_folder/FilterOutput/' directory. 
        You should independently evaluate these different results and select the MAGs that exhibit the best performance.
    """

    print("##################################")
    print("###  WELCOME TO USE DEEPURIFY  ###")
    print("##################################")
    print()
    assert batch_size_per_gpu <= 32, "batch_size_per_gpu must smaller or equal with 32."
    assert num_threads_per_device <= 4, "num_threads_per_device must smaller or equal with 4."

    if input_bin_folder_path[-1] == "/":
        input_bin_folder_path = input_bin_folder_path[:-1]

    filesFolder = os.path.split(input_bin_folder_path)[0]
    if temp_output_folder is None:
        temp_output_folder = os.path.join(filesFolder, "DeepurifyTempFiles")

    if output_bins_meta_info_path is None:
        output_bins_meta_info_path = os.path.join(output_bin_folder_path, "MetaInfo.txt")

    if gpu_num == 0:
        gpu_work_ratio = []
    else:
        gpu_work_ratio = [1.0 / gpu_num for _ in range(gpu_num - 1)]
        gpu_work_ratio += [1.0 - sum(gpu_work_ratio)]
    batch_size_per_gpu = [batch_size_per_gpu for _ in range(gpu_num)]

    if isinstance(info_files_path, str):
        modelWeightPath = os.path.join(info_files_path, "CheckPoint", "Deepurify.ckpt")
        taxoVocabPath = os.path.join(info_files_path, "TaxonomyInfo", "ProGenomesVocabulary.txt")
        taxoTreePath = os.path.join(info_files_path, "TaxonomyInfo", "ProGenomesTaxonomyTree.pkl")
        taxoName2RepNormVecPath = os.path.join(info_files_path, "PyObjs", "Deepurify_taxo_lineage_vector.pkl")
        hmmModelPath = os.path.join(info_files_path, "HMM", "hmm_model.hmm")
        mer3Path = os.path.join(info_files_path, "3Mer_vocabulary.txt")
        mer4Path = os.path.join(info_files_path, "4Mer_vocabulary.txt")
    elif info_files_path is None:
        try:
            info_files_path = os.environ["DeepurifyInfoFiles"]
            modelWeightPath = os.path.join(info_files_path, "CheckPoint", "Deepurify.ckpt")
            taxoVocabPath = os.path.join(info_files_path, "TaxonomyInfo", "ProGenomesVocabulary.txt")
            taxoTreePath = os.path.join(info_files_path, "TaxonomyInfo", "ProGenomesTaxonomyTree.pkl")
            taxoName2RepNormVecPath = os.path.join(info_files_path, "PyObjs", "Deepurify_taxo_lineage_vector.pkl")
            hmmModelPath = os.path.join(info_files_path, "HMM", "hmm_model.hmm")
            mer3Path = os.path.join(info_files_path, "3Mer_vocabulary.txt")
            mer4Path = os.path.join(info_files_path, "4Mer_vocabulary.txt")
        except Exception as e:
            raise ValueError(
                "Error, can not find environment variable 'DeepurifyInfoFiles'. Make sure the variables info_files_path not None."
            ) from e
    elif taxoName2RepNormVecPath is None:
        taxoName2RepNormVecPath = \
            os.path.join(filesFolder, "DeepurifyTempOut" "Deepurify_taxo_lineage_vector.pkl")
        print(f"The variable 'taxoName2RepNormVecPath' is None, would build this file with this path: {taxoName2RepNormVecPath}")

    assert mer3Path is not None and mer4Path is not None, ValueError(
        "The variable mer3Path or mer4Path is None. Please check this file if it is in 'DeepurifyInfoFiles' folder.")
    assert modelWeightPath is not None, ValueError(
        "The variable modelWeightPath is None. Please check this file if it is in 'DeepurifyInfoFiles' folder.")
    assert taxoVocabPath is not None, ValueError(
        "The variable taxoVocabPath is None. Please check this file if it is in 'DeepurifyInfoFiles' folder.")
    assert taxoTreePath is not None, ValueError(
        "The variable taxoTreePath is None. Please check this file if it is in 'DeepurifyInfoFiles' folder.")
    if not simulated_MAG:
        assert hmmModelPath is not None, ValueError(
            "The variable hmmModelPath is None. Please check this file if it is in 'DeepurifyInfoFiles' folder.")

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
        num_threads_per_device=num_threads_per_device,
        bin_suffix=bin_suffix,
        mer3Path=mer3Path,
        mer4Path=mer4Path,
        overlapping_ratio=overlapping_ratio,
        cut_seq_length=cut_seq_length,
        num_threads_call_genes=num_threads_call_genes,
        ratio_cutoff=hmm_align_ratio_cutoff,
        acc_cutoff=hmm_acc_cutoff,
        estimated_completeness_threshold=estimated_completeness_threshold,
        seq_length_threshold=seq_length_threshold,
        checkM_process_num=checkM_process_num,
        num_threads_per_checkm=num_threads_per_checkm,
        topkORgreedy=topk_or_greedy,
        topK=topK_num,
        model_config=model_config,
        simulated_MAG=simulated_MAG
    )
