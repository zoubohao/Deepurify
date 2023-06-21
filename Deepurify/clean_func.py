import os
import sys
from typing import Dict, Union

from Deepurify.funcs import runLabelFilterSplitBins


def cleanMAGs(
    input_bin_folder_path: str,
    output_bin_folder_path: str,
    bin_suffix: str,
    gpu_num: int,
    batch_size_per_gpu: int,
    num_threads_per_device: int,
    overlapping_ratio=0.5,
    cutSeqLength=8192,
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
    self_evaluate: bool = False
):
    """

    The main function to clean the MAGs.

    Args:
        input_bin_folder_path (str): The folder of input MAGs

        output_bin_folder_path (str): The folder used to output cleaned MAGs.

        bin_suffix (str): The bin suffix of MAG files.

        gpu_num (int): The number of GPUs to be used can be specified. Defaults to 0.
        If you set it to 0, the code will utilize the CPU. 
        However, please note that using the CPU can result in significantly slower processing speed. 
        It is recommended to provide at least one GPU for better performance.

        batch_size_per_gpu (int): The batch size per GPU determines the number of sequences that will be loaded onto each GPU. 
        This parameter is only applicable if the --gpu_num option is set to a value greater than 0. 
        The default value is 2, meaning that two sequences will be loaded per GPU batch.
        The batch size for CPU is 2.

        num_threads_per_device (int): The number of threads per GPU or CPU determines the parallelism level during contigs' inference stage. 
        If the value of --gpu_num is greater than 0, each GPU will have a set number of threads to do inference. 
        Similarly, if --gpu_num is set to 0 and the code is running on CPU, the specified number of threads will be used. 
        By default, the number of threads per GPU or CPU is set to 1. 
        The --batch_size_per_gpu value will be divided by the number of threads to determine the batch size per thread.

        overlapping_ratio (float): The --overlapping_ratio is a parameter used when the length of a contig exceeds the specified --cut_seq_length. 
        By default, the overlapping ratio is set to 0.5. 
        This means that when a contig is longer than the --cut_seq_length, it will be split into overlapping subsequences with 50\%\ overlap between consecutive subsequences.

        cutSeqLength (int, optional): The --cut_seq_length parameter determines the length at which a contig will be cut if its length exceeds this value. 
        The default setting is 8192, which is also the maximum length allowed during training. 
        If a contig's length surpasses this threshold, it will be divided into smaller subsequences with lengths equal to or less than the cut_seq_length.
        
        num_threads_call_genes (int, optional): The number of threads to call genes. Defaults to 12.

        hmm_acc_cutoff (float, optional): If the acc score and the aligned ratio assigned by the HMM model for a gene sequence exceeds this threshold, 
        it would be considered as a single-copy gene. It is set to 0.6 by default.

        hmm_align_ratio_cutoff (float, optional): If the acc score and the aligned ratio assigned by the HMM model for a gene sequence exceeds this threshold, 
        it would be considered as a single-copy gene. It is set to 0.4 by default.

        estimated_completeness_threshold (float, optional): The --estimate_completeness_threshold is used as a criterion for filtering MAGs that are generated 
        by applying specific single-copy genes (SCGs). The default threshold is set to 0.55. 
        MAGs with an estimated completeness score equal to or higher than this threshold will be considered for further analysis or inclusion, 
        while those falling below the threshold may be filtered out.

        seq_length_threshold (int, optional): The threshold for the total length of a MAG's contigs is used to filter generated MAGs after applying single-copy genes (SCGs). 
        The default threshold is set to 550,000, which represents the total length of the contigs in base pairs (bp). 
        MAGs with a total contig length equal to or greater than this threshold will be considered for further analysis or inclusion, 
        while MAGs with a total contig length below the threshold may be filtered out.

        checkM_process_num (int, optional): The number of processes to run CheckM simultaneously. Defaults to 1.

        num_threads_per_checkm (int, optional): The number of threads to run a single CheckM process. Defaults to 12.

        topk_or_greedy (str, optional): Topk searching or greedy searching to label a contig. Defaults to 'topk'.
        The contig is assigned a label based on the top-k most relevant or similar taxonomic lineages. 
        The specific number of lineages considered for labeling can be determined by the value of k.

        topK_num (int, optional): During the top-k searching approach, the default behavior is to search for the top-k taxon nodes that exhibit the highest cosine 
        similarity with the contig's encoded vector. 
        By default, the value of k is set to 3, meaning that the three most similar nodes in terms of cosine similarity will be considered for labeling the contig. 
        Please note that this parameter does not have any effect when using the greedy search approach (topK_num=1).

        temp_output_folder (Union[str, None], optional): The temporary files generated during the process can be stored in a specified folder path. 
        By default, if no path is provided (i.e., set to None), the temporary files will be stored in the parent folder of the 'input_bin_folder_path' location. 
        However, you have the option to specify a different folder path to store these temporary files if needed.

        output_bins_meta_info_path (Union[str, None], optional): The path of a text file can be provided to record the meta information, including the evaluated results, of the final cleaned MAGs. 
        By default, if no path is specified (i.e., set to None), the file will be created under the 'output_bin_folder_path' directory. 
        However, you have the flexibility to specify a different file path if desired.

        info_files_path (Union[str, None]): The DeepurifyInfoFiles is essential for running Deepurify. 
        By default, if no path is provided (i.e., set to None), it is expected that the environment variable 'DeepurifyInfoFiles' has been set to point to the appropriate folder. 
        Please ensure that the 'DeepurifyInfoFiles' environment variable is correctly configured if the path is not explicitly provided.

        modelWeightPath (Union[str, None], optional): The path of model weight. (In DeepurifyInfoFiles folder) Defaults to None.

        taxoVocabPath (Union[str, None], optional): The path of taxon vocabulary. (In DeepurifyInfoFiles folder) Defaults to None.

        taxoTreePath (Union[str, None], optional): The path of taxonomic tree. (In DeepurifyInfoFiles folder) Defaults to None.

        taxoName2RepNormVecPath (Union[str, None], optional): The path of taxonomic lineage encoded vectors. (In DeepurifyInfoFiles folder) Defaults to None.

        hmmModelPath (Union[str, None], optional): The path of SCGs' hmm file. (In DeepurifyInfoFiles folder) Defaults to None.

        model_config (Union[Dict, None], optional): The config of model. See the TrainScript.py to find more information. Defaults to None.
        It would be used if you trained a another model with different model_config. Please set this variable equal with None at present.

        self_evaluate (bool, optional): Evaluate the results by the user. Defaults to False. 
        Set to True if you have knowledge of clean and contaminated contigs in the simulated dataset or you want to evaluate the outcomes by yourself.
        We would remove the outlier contigs and only keep clean contigs with different cosine similarity threshold for a MAG if this variable is True.
        The outputs will be stored in the following folder path: /temp_output_folder/FilterOutput/
        You should independently evaluate the outcomes from various similarity threshold and select the best output from the cleaned MAGs.
    """

    print("##################################")
    print("###  WELCOME TO USE DEEPURIFY  ###")
    print("##################################")
    print()
    assert batch_size_per_gpu <= 20, "batch_size_per_gpu must smaller or equal with 20."
    assert num_threads_per_device <= 4, "num_threads_per_device must smaller or equal with 4."

    if "/" == input_bin_folder_path[-1]:
        input_bin_folder_path = input_bin_folder_path[0:-1]

    filesFolder = os.path.split(input_bin_folder_path)[0]
    if temp_output_folder is None:
        temp_output_folder = os.path.join(filesFolder, "DeepurifyTempFiles")

    if output_bins_meta_info_path is None:
        output_bins_meta_info_path = os.path.join(output_bin_folder_path, "MetaInfo.txt")

    if gpu_num == 0:
        gpu_work_ratio = []
    else:
        gpu_work_ratio = [1.0 / gpu_num for _ in range(gpu_num - 1)]
        gpu_work_ratio = gpu_work_ratio + [1.0 - sum(gpu_work_ratio)]
    batch_size_per_gpu = [batch_size_per_gpu for _ in range(gpu_num)]

    if info_files_path is None:
        try:
            info_files_path = os.environ["DeepurifyInfoFiles"]
            modelWeightPath = os.path.join(info_files_path, "CheckPoint", "Deepurify.ckpt")
            taxoVocabPath = os.path.join(info_files_path, "TaxonomyInfo", "ProGenomesVocabulary.txt")
            taxoTreePath = os.path.join(info_files_path, "TaxonomyInfo", "ProGenomesTaxonomyTree.pkl")
            taxoName2RepNormVecPath = os.path.join(info_files_path, "PyObjs", "Deepurify_taxo_lineage_vector.pkl")
            hmmModelPath = os.path.join(info_files_path, "HMM", "hmm_model.hmm")
            mer3Path = os.path.join(info_files_path, "3Mer_vocabulary.txt")
            mer4Path = os.path.join(info_files_path, "4Mer_vocabulary.txt")
        except:
            print("Warnning !!!! Can not find environment variable 'DeepurifyInfoFiles', Make sure the variables of file paths are not None.")
            if taxoName2RepNormVecPath is None:
                print("The variable taxoName2RepNormVecPath is None, would build this file with this path: {}".format(os.path.join(
                    filesFolder, "DeepurifyTempOut" "Deepurify_taxo_lineage_vector.pkl")))
                taxoName2RepNormVecPath = os.path.join(
                    filesFolder, "DeepurifyTempOut" "Deepurify_taxo_lineage_vector.pkl")

    assert mer3Path is not None and mer4Path is not None, ValueError(
        "The variable mer3Path or mer4Path is None. Please check this file if it is in 'DeepurifyInfoFiles' folder.")
    assert modelWeightPath is not None, ValueError(
        "The variable modelWeightPath is None. Please check this file if it is in 'DeepurifyInfoFiles' folder.")
    assert taxoVocabPath is not None, ValueError(
        "The variable taxoVocabPath is None. Please check this file if it is in 'DeepurifyInfoFiles' folder.")
    assert taxoTreePath is not None, ValueError(
        "The variable taxoTreePath is None. Please check this file if it is in 'DeepurifyInfoFiles' folder.")
    if self_evaluate is False:
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
        cutSeqLength=cutSeqLength,
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
        self_evaluate=self_evaluate
    )
