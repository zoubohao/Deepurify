import os

from typing import List, Union

from Deepurify.Infer_uitls import inference_once_time
from Deepurify.decontamination import run_all_deconta_steps
from Deepurify.iter_clean import run_iter_clean

def cleanMAGs(
    output_bin_folder_path: str,
    cuda_device_list: List[float] = ["0"],
    gpu_work_ratio_list: List[float] = [1],
    batch_size_per_gpu: int = 1,
    each_gpu_threads: int = 1,
    input_bins_folder: str = None,
    bin_suffix: str = None,
    contig_fasta_path: str = None,
    sorted_bam_file: str = None,
    binning_mode = None,
    overlapping_ratio=0.5,
    cut_seq_length=8192,
    seq_length_threshold=200000,
    topk_or_greedy="topk",
    topK_num=3,
    num_process: int = None,
    temp_output_folder: Union[str, None] = None,
    db_files_path: Union[str, None] = None,
    model_weight_path = None,
    taxo_tree_path = None,
    taxo_vocab_path = None,
    taxo_name2rep_norm_vec_path = None
):
    """
    NOTE:
    We highly recommend you have at least one GPU (>= GTX-1060-6GB version) to run this function. 
    We further recommend your CPU has at least 16 cores 32 threads to run this function.
    This function does not need much memory. The memory bottleneck is running CheckM2.
    
    Deepurify has two modes for cleaning MAGs: 1. Only clean the MAGs; 2. Apply 'Re-binning' and 'Ensemble' strategies.
    
    MODE 1. Only clean the MAGs
    
    The parameters 'input_bins_folder' and 'bin_suffix' must be set if you want to use this mode. Please do not set 'contig_fasta_path', 
    'sorted_bam_file' for this mode.
    
    MODE 2. Apply 'Re-binning' and 'Ensemble' strategies.
    
    The parameters 'contig_fasta_path', 'sorted_bam_file' must be set if you want ot use this mode. Please do not set 'input_bins_folder', 
    'bin_suffix' for this mode.
    
    
    Args:
        output_bin_folder_path (str): The output folder of purified MAGs. It will be created if it does not exist.
        
        cuda_device_list (List[float], optional): The gpu id that you want to apply. 
        You can set ["0", "1"] to use gpu0 and gpu1.
        Defaults to ["0"].
        
        gpu_work_ratio_list (List[float], optional): The number of float elements in this list equals with the number of GPU will be used. 
        An empty list will apply CPU to do binning or decontamination. 
        For example, two GPUs will be used with different work ratio (CUDA:0 --> 0.6; CUDA:1 --> 0.4) if the input of this parameter is [0.6, 0.4]. 
        The summed value of this list must equal with 1. Defaults to [1].
        
        batch_size_per_gpu (int, optional): The batch size for a GPU. Defaults to 1.
        
        each_gpu_threads (int, optional): The number of threads for a GPU to do inference. Defaults to 1.
        
        input_bins_folder (str, optional): The input MAGs' folder. The parameter 'bin_suffix' must be set if this parameter is not None.
        This function will only **CLEAN** the MAGs in the input folder without 'Re-binning' and 'Ensemble' strategies if this parameter has been set. 
        Please do not set 'contig_fasta_path', 'sorted_bam_file', and 'binning_mode' if this parameter has been set. Defaults to None.
        
        bin_suffix (str, optional): The bin suffix of MAGs. Defaults to None.
        
        contig_fasta_path (str, optional): The path of contigs. The parameter 'sorted_bam_file' must be set if this parameter is not None.
        This function will apply 'Re-binning' strategies if this parameter has been set. Defaults to None.
        
        sorted_bam_file (str, optional): The path of the sorted BAM file. Defaults to None.
        
        binning_mode (str, optional): The semibin2, concoct, metabat2 will all be run if this parameter is None. 
        The other modes are: 'semibin2', 'concoct', and 'metabat2'. Defaults to None.
        
        overlapping_ratio (float, optional): This parameter will be used when the length of a contig exceeds the specified 'cut_seq_length'. 
        This means that when a contig is longer than the 'cut_seq_length', it will be split into overlapping subsequences with 50\%\ overlap 
        between consecutive subsequences. Defaults to 0.5.
        
        cut_seq_length (int, optional): The maximum length that the model can handle. We will cut the contig if it exceeds this length. 
        Defaults to 8192.
        
        seq_length_threshold (int, optional): The threshold for the total length of a MAG's contigs is used to filter generated MAGs after 
        applying single-copy genes (SCGs). Defaults to 200000.
        
        topk_or_greedy (str, optional): Topk searching or greedy searching to label a contig. Defaults to "topk".
        
        topK_num (int, optional): During the top-k searching approach, the default behavior is to search for the top-k nodes that exhibit the 
        highest cosine similarity with the contig's encoded vector. By default, the value of k is set to 3, meaning that the three most similar 
        nodes in terms of cosine similarity will be considered for labeling the contig. 
        Please note that this parameter does not have any effect when using the greedy search approach (topK_num=1). Defaults to 3.
        
        num_process (int, optional): The maximum number of threads will be used. All CPUs will be used if it is None. Defaults to None
        
        temp_output_folder (Union[str, None], optional): The path to store temporary files. Defaults to None.
        
        db_files_path (Union[str, None], optional): The database folder path. Defaults to None.
        
        model_weight_path (str, optional): The path of model weight. It should in database folder. Defaults to None.
        
        taxo_tree_path (str, optional): The path of taxonomic tree. It should in database folder. Defaults to None.
        
        taxo_vocab_path (str, optional): The path of taxonomic vocabulary. It should in database folder. Defaults to None.
        
        taxo_name2rep_norm_vec_path (str, optional): The path of taxnomic lineage with its normalized vector. 
        It should in database folder. Defaults to None.
    """

    print("##################################")
    print("###  WELCOME TO USE DEEPURIFY  ###")
    print("##################################")
    print()
    assert batch_size_per_gpu <= 64, "batch_size_per_gpu must smaller or equal with 64."
    assert each_gpu_threads <= 4, "each_gpu_threads must smaller or equal with 4."
    
    if cuda_device_list is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(cuda_device_list)
    
    if input_bins_folder is not None:
        assert bin_suffix is not None, ValueError("The bin_suffix is None.")
    else:
        assert contig_fasta_path is not None and sorted_bam_file is not None, ValueError("contig_fasta_path is None or sorted_bam_file is None.")

    if temp_output_folder is None:
        temp_output_folder = os.path.join(output_bin_folder_path, "DeepurifyTmpFiles")
    if os.path.exists(output_bin_folder_path) is False:
        os.mkdir(output_bin_folder_path)
    if os.path.exists(temp_output_folder) is False:
        os.mkdir(temp_output_folder)

    gpu_num = len(gpu_work_ratio_list)

    batch_size_per_gpu = [batch_size_per_gpu for _ in range(gpu_num)]

    if db_files_path is None:
        try:
            db_files_path = os.environ["DeepurifyInfoFiles"]
        except:
            print("Warnning !!!! Can not find environment variable 'DeepurifyInfoFiles', Make sure the variables of db_files_path not None.")

    assert db_files_path is not None, ValueError("The db_files_path is None. Please make sure your have set the database files properly.")

    if not model_weight_path or not taxo_vocab_path or not taxo_vocab_path or not taxo_name2rep_norm_vec_path:
        modelWeightPath = os.path.join(db_files_path, "CheckPoint", "GTDB-clu-last.pth")
        taxoVocabPath = os.path.join(db_files_path, "Vocabs", "taxa_vocabulary.txt")
        taxoTreePath = os.path.join(db_files_path, "PyObjs", "gtdb_taxonomy_tree.pkl")
        taxoName2RepNormVecPath = os.path.join(db_files_path, "PyObjs", "taxoName2RepNormVecPath.pkl")
    else:
        modelWeightPath = model_weight_path
        taxoVocabPath = taxo_vocab_path
        taxoTreePath = taxo_tree_path
        taxoName2RepNormVecPath = taxo_name2rep_norm_vec_path
    mer3Path = os.path.join(db_files_path, "Vocabs", "3Mer_vocabulary.txt")
    mer4Path = os.path.join(db_files_path, "Vocabs", "4Mer_vocabulary.txt")
    hmmModelPath = os.path.join(db_files_path, "HMM", "hmm_models.hmm")
    phy2accsPath = os.path.join(db_files_path, "HMM", "phy2accs_new.pkl")
    checkm2_db_path = os.path.join(db_files_path, 'Checkm', "checkm2_db.dmnd")
    
    ## fast inference 
    concat_vec_path = os.path.join(temp_output_folder, "all_concat_contigname2repNorm.pkl")
    concat_annot_path = os.path.join(temp_output_folder, "all_concat_annot.txt")
    output_fasta_path = None
    if os.path.exists(concat_vec_path) is False \
        or os.path.exists(concat_annot_path) is False:
        output_fasta_path = inference_once_time(
            contig_fasta_path=contig_fasta_path,
            inputBinFolder=input_bins_folder,
            tempFileOutFolder=temp_output_folder,
            modelWeightPath=modelWeightPath,
            taxoVocabPath=taxoVocabPath,
            taxoTreePath=taxoTreePath,
            taxoName2RepNormVecPath=taxoName2RepNormVecPath,
            hmmModelPath=hmmModelPath,
            phy2accsPath=phy2accsPath,
            bin_suffix=bin_suffix,
            mer3Path=mer3Path,
            mer4Path=mer4Path,
            gpus_work_ratio=gpu_work_ratio_list,
            batch_size_per_gpu=batch_size_per_gpu,
            each_gpu_threads=each_gpu_threads,
            overlapping_ratio=overlapping_ratio,
            cut_seq_length=cut_seq_length,
            seq_length_threshold=seq_length_threshold,
            topkORgreedy=topk_or_greedy,
            topK=topK_num,
            min_length_for_infer=1000,
            num_process=num_process,
            concat_annot_file_path=concat_annot_path,
            concat_vec_file_path=concat_vec_path,
        )
    
    ## decontamination
    if contig_fasta_path is not None and  sorted_bam_file is not None:
        run_iter_clean(
            contig_fasta_path=output_fasta_path,
            sorted_bam_file=sorted_bam_file,
            concat_vec_path=concat_vec_path,
            concat_annot_path=concat_annot_path,
            tempFileOutFolder=temp_output_folder,
            outputBinFolder=output_bin_folder_path,
            modelWeightPath=modelWeightPath,
            taxoVocabPath=taxoVocabPath,
            taxoTreePath=taxoTreePath,
            taxoName2RepNormVecPath=taxoName2RepNormVecPath,
            hmmModelPath=hmmModelPath,
            phy2accsPath=phy2accsPath,
            mer3Path=mer3Path,
            mer4Path=mer4Path,
            checkm2_db_path=checkm2_db_path,
            gpus_work_ratio=gpu_work_ratio_list,
            batch_size_per_gpu=batch_size_per_gpu,
            each_gpu_threads=each_gpu_threads,
            overlapping_ratio=overlapping_ratio,
            cut_seq_length=cut_seq_length,
            seq_length_threshold=seq_length_threshold,
            topkORgreedy=topk_or_greedy,
            topK=topK_num,
            num_process=num_process,
            binning_mode=binning_mode
        )
    else:
        run_all_deconta_steps(
            input_bins_folder,
            temp_output_folder,
            outputBinFolder=output_bin_folder_path,
            modelWeightPath=modelWeightPath,
            taxoVocabPath=taxoVocabPath,
            taxoTreePath=taxoTreePath,
            taxoName2RepNormVecPath=taxoName2RepNormVecPath,
            hmmModelPath=hmmModelPath,
            phy2accsPath=phy2accsPath,
            bin_suffix=bin_suffix,
            mer3Path=mer3Path,
            mer4Path=mer4Path,
            checkm2_db_path=checkm2_db_path,
            gpus_work_ratio=gpu_work_ratio_list,
            batch_size_per_gpu=batch_size_per_gpu,
            each_gpu_threads=each_gpu_threads,
            overlapping_ratio=overlapping_ratio,
            cut_seq_length=cut_seq_length,
            seq_length_threshold=seq_length_threshold,
            topkORgreedy=topk_or_greedy,
            topK=topK_num,
            num_process=num_process,
            build_concat_file=False,
            concat_vec_file_path=concat_vec_path,
            concat_annot_file_path=concat_annot_path,
            simulated_MAG=False,
            just_annot=False
        )
