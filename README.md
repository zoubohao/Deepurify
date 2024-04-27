# Deepurify_Project
**Paper --> Deepurify: a multi-modal deep language model to remove contamination from metagenome-assembled genomes**

Deepurify elevates metagenome-assembled genomes' (MAGs) quality by utilizing a multi-modal deep language model to filter contaminated contigs, and it can leverage GPU acceleration.

<div align=center> <img src="/deeplogo.png" alt="logo"></div>

## Dependencies:

Please independently install the following tools and ensure their proper functionality.

1. **[prodigal](https://github.com/hyattpd/Prodigal/wiki/installation)** v 2.6.3 (ORF/CDS-prediction)
2. **[hmmer](http://hmmer.org/download.html)** v.3.3.2 (Detecting conserved single-copy marker genes)
3. **[CheckM2](https://github.com/chklovski/CheckM2)** v 1.0.1 (Evaluate the quality of MAGs)
4. **[dRep](https://github.com/MrOlm/drep)** v3.5.0 (Filter replicated MAGs)
5. **[CONCOCT](https://github.com/BinPro/CONCOCT)** v1.1.0 (Binner)
6. **[MetaBAT2](https://bitbucket.org/berkeleylab/metabat/src/master/)** v2.15 (Binner)
7. **[Semibin2](https://github.com/BigDataBiology/SemiBin)** v2.1.0 (Binner)

**Note**: Ensure that all the listed dependencies above are installed and functioning without any errors.


## Installation:
#### FIRST STEP (Create Environment)
Create deepurify's conda environment by using this command:

```
conda env create -n deepurify -f deepurify-conda-env.yml
```

Do not forget to download the database files for **CheckM2 !!!**

and Please download PyTorch v2.0.1 -cu118 (or higher version) from **[http://pytorch.org/](http://pytorch.org/)** if you want to use GPUs (We highly recommend to use GPUs).

#### SECOND STEP (Install Codes)
After preparing the env, the code of Deepurify can be installed via pip. 

```
conda activate deepurify

pip install Deepurify==2.3.1
```


## Download Files and Set Environment Variable for Running
- Download the database (**Deepurify-DB.zip**) for running Deepurify from this **[LINK](https://drive.google.com/file/d/1FXpxoXFYHcX9QAFe7U6zfM8YjalxNLFk/view?usp=sharing)**.

- Unzip the downloaded file (**Deepurify-DB.zip**) and set an **environment variable** called "DeepurifyInfoFiles" by adding the following line to the last line of .bashrc file (The path of the file: ~/.bashrc):
```
export DeepurifyInfoFiles=/path/of/this/unzip/folder/
```
For example: 'export DeepurifyInfoFiles=/home/csbhzou/software/Deepurify-DB/'.

- Save the .bashrc file, and then execute:
```
source .bashrc
```

- **You can set the '--db_folder_path' in CLI to the path of 'Deepurify-DB' folder if you do not want to set the environment variable.**


## Running Deepurify
**1.  You can run the Deepurify with 'clean' mode through the **cleanMAGs** function.**
```
from Deepurify.clean_func import cleanMAGs

if __name__ == "__main__":
    input_folder = "./input_folder/"
    bin_suffix = "fasta"
    output_folder = "./output_folder/"
    cleanMAGs(
        output_bin_folder_path=output_folder,
        batch_size_per_gpu=32,
        each_gpu_threads=2,
        input_bins_folder=input_folder,
        bin_suffix=bin_suffix,
        gpu_work_ratio=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125], # enable 8 GPUs with equal work ratios.
        db_files_path="./GTDB_Taxa_Info/"
    )

```


**2.  You can run the Deepurify with 're-bin' mode through the **cleanMAGs** function.**
```
from Deepurify.clean_func import cleanMAGs

if __name__ == "__main__":
    contig_fasta_path = "./contigs.fasta"
    bam_file_path = "./sorted.bam"
    output_folder = "./output_folder/"
    cleanMAGs(
        output_bin_folder_path=output_folder,
        batch_size_per_gpu=32,
        each_gpu_threads=2,
        contig_fasta_path=contig_fasta_path,
        sorted_bam_file=bam_file_path,
        gpu_work_ratio=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.25], # enable 8 GPUs with equal work ratios.
        db_files_path="./Deepurify-DB/"
    )

```
Please refer to the documentation of this function for more details.


**3.  You can run Deepurify with 'clean' mode through the following command:**
```
deepurify clean  -i ./input_folder/ -o ./output_folder/ --bin_suffix fasta --gpu_num 1 --each_gpu_threads 1
```
----------------------------------------------------------------------------------------------------------------------------------------
```
usage: deepurify clean [-h] -i INPUT_PATH -o OUTPUT_PATH --bin_suffix BIN_SUFFIX [--gpu_num GPU_NUM] [--batch_size_per_gpu BATCH_SIZE_PER_GPU] [--each_gpu_threads EACH_GPU_THREADS]
                       [--overlapping_ratio OVERLAPPING_RATIO] [--cut_seq_length CUT_SEQ_LENGTH] [--mag_length_threshold MAG_LENGTH_THRESHOLD] [--num_process NUM_PROCESS]
                       [--topk_or_greedy_search {topk,greedy}] [--topK_num TOPK_NUM] [--temp_output_folder TEMP_OUTPUT_FOLDER] [--db_folder_path DB_FOLDER_PATH]
                       [--model_weight_path MODEL_WEIGHT_PATH] [--taxo_vocab_path TAXO_VOCAB_PATH] [--taxo_tree_path TAXO_TREE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input_path INPUT_PATH
                        The folder of input MAGs.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        The folder used to output cleaned MAGs.
  --bin_suffix BIN_SUFFIX
                        The bin suffix of MAG files.
  --gpu_num GPU_NUM     The number of GPUs to be used can be specified. Defaults to 1. If you set it to 0, the code will utilize the CPU. However, please note that using the CPU can
                        result in significantly slower processing speed. It is recommended to provide at least one GPU (>= GTX-1060-6GB) for accelerating the speed.
  --batch_size_per_gpu BATCH_SIZE_PER_GPU
                        The batch size per GPU determines the number of sequences that will be loaded onto each GPU. This parameter is only applicable if the --gpu_num option is set to a
                        value greater than 0. The default value is 4, meaning that one sequences will be loaded per GPU batch. The batch size for CPU is 4.
  --each_gpu_threads EACH_GPU_THREADS
                        The number of threads per GPU (or CPU) determines the parallelism level during contigs' inference stage. If the value of --gpu_num is greater than 0, each GPU will
                        have a set number of threads to do inference. Similarly, if --gpu_num is set to 0 and the code will run on CPU, the specified number of threads will be used. By
                        default, the number of threads per GPU or CPU is set to 1. The --batch_size_per_gpu value will be divided by the number of threads to determine the batch size per
                        thread.
  --overlapping_ratio OVERLAPPING_RATIO
                        The --overlapping_ratio is a parameter used when the length of a contig exceeds the specified --cut_seq_length. By default, the overlapping ratio is set to 0.5.
                        This means that when a contig is longer than the --cut_seq_length, it will be split into overlapping subsequences with 0.5 overlap between consecutive
                        subsequences.
  --cut_seq_length CUT_SEQ_LENGTH
                        The --cut_seq_length parameter determines the length at which a contig will be cut if its length exceeds this value. The default setting is 8192, which is also the
                        maximum length allowed during training. If a contig's length surpasses this threshold, it will be divided into smaller subsequences with lengths equal to or less
                        than the cut_seq_length.
  --mag_length_threshold MAG_LENGTH_THRESHOLD
                        The threshold for the total length of a MAG's contigs is used to filter generated MAGs after applying single-copy genes (SCGs). The default threshold is set to
                        200,000, which represents the total length of the contigs in base pairs (bp). MAGs with a total contig length equal to or greater than this threshold will be
                        considered for further analysis or inclusion, while MAGs with a total contig length below the threshold may be filtered out.
  --num_process NUM_PROCESS
                        The maximum number of threads will be used. All CPUs will be used if it is None. Defaults to None
  --topk_or_greedy_search {topk,greedy}
                        Topk searching or greedy searching to label a contig. Defaults to "topk".
  --topK_num TOPK_NUM   During the top-k searching approach, the default behavior is to search for the top-k nodes that exhibit the highest cosine similarity with the contig's encoded
                        vector. By default, the value of k is set to 3, meaning that the three most similar nodes in terms of cosine similarity will be considered for labeling the contig.
                        Please note that this parameter does not have any effect when using the greedy search approach (topK_num=1). Defaults to 3.
  --temp_output_folder TEMP_OUTPUT_FOLDER
                        The temporary files generated during the process can be stored in a specified folder path. By default, if no path is provided (i.e., set to None), the temporary
                        files will be stored in the parent folder of the '--input_path' location. However, you have the option to specify a different folder path to store these temporary
                        files if needed.
  --db_folder_path DB_FOLDER_PATH
                        The path of database folder. By default, if no path is provided (i.e., set to None), it is expected that the environment variable 'DeepurifyInfoFiles' has been set
                        to point to the appropriate folder. Please ensure that the 'DeepurifyInfoFiles' environment variable is correctly configured if the path is not explicitly
                        provided.
  --model_weight_path MODEL_WEIGHT_PATH
                        The path of model weight. (In database folder) Defaults to None.
  --taxo_vocab_path TAXO_VOCAB_PATH
                        The path of taxon vocabulary. (In database folder) Defaults to None.
  --taxo_tree_path TAXO_TREE_PATH
                        The path of taxonomic tree. (In database folder) Defaults to None.
```
Please run 'deepurify clean -h' for more details.


**4.  You can run Deepurify with 're-bin' mode through the following command:**
```
deepurify re-bin  -c ./contigs.fasta -o ./output_folder/ -s ./sorted.bam --gpu_num 1 --each_gpu_threads 1
```
----------------------------------------------------------------------------------------------------------------------------------------
```
usage: deepurify re-bin [-h] -c CONTIGS PATH -s SORTED_BAM_PATH -o OUTPUT_PATH [--binning_mode BINNING_MODE] [--gpu_num GPU_NUM] [--batch_size_per_gpu BATCH_SIZE_PER_GPU]
                        [--each_gpu_threads EACH_GPU_THREADS] [--overlapping_ratio OVERLAPPING_RATIO] [--cut_seq_length CUT_SEQ_LENGTH] [--mag_length_threshold MAG_LENGTH_THRESHOLD]
                        [--num_process NUM_PROCESS] [--topk_or_greedy_search {topk,greedy}] [--topK_num TOPK_NUM] [--temp_output_folder TEMP_OUTPUT_FOLDER]
                        [--db_folder_path DB_FOLDER_PATH] [--model_weight_path MODEL_WEIGHT_PATH] [--taxo_vocab_path TAXO_VOCAB_PATH] [--taxo_tree_path TAXO_TREE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -c CONTIGS PATH, --contigs path CONTIGS PATH
                        The contigs fasta path.
  -s SORTED_BAM_PATH, --sorted_bam_path SORTED_BAM_PATH
                        The sorted bam path.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        The folder used to output cleaned MAGs.
  --binning_mode BINNING_MODE
                        The semibin2, concoct, metabat2 will all be run if this parameter is None. The other modes are: 'semibin2', 'concoct', and 'metabat2'. Defaults to None.
  --gpu_num GPU_NUM     The number of GPUs to be used can be specified. Defaults to 1. If you set it to 0, the code will utilize the CPU. However, please note that using the CPU can
                        result in significantly slower processing speed. It is recommended to provide at least one GPU (>= GTX-1060-6GB) for accelerating the speed.
  --batch_size_per_gpu BATCH_SIZE_PER_GPU
                        The batch size per GPU determines the number of sequences that will be loaded onto each GPU. This parameter is only applicable if the --gpu_num option is set to a
                        value greater than 0. The default value is 4, meaning that one sequences will be loaded per GPU batch. The batch size for CPU is 4.
  --each_gpu_threads EACH_GPU_THREADS
                        The number of threads per GPU (or CPU) determines the parallelism level during contigs' inference stage. If the value of --gpu_num is greater than 0, each GPU will
                        have a set number of threads to do inference. Similarly, if --gpu_num is set to 0 and the code will run on CPU, the specified number of threads will be used. By
                        default, the number of threads per GPU or CPU is set to 1. The --batch_size_per_gpu value will be divided by the number of threads to determine the batch size per
                        thread.
  --overlapping_ratio OVERLAPPING_RATIO
                        The --overlapping_ratio is a parameter used when the length of a contig exceeds the specified --cut_seq_length. By default, the overlapping ratio is set to 0.5.
                        This means that when a contig is longer than the --cut_seq_length, it will be split into overlapping subsequences with 0.5 overlap between consecutive
                        subsequences.
  --cut_seq_length CUT_SEQ_LENGTH
                        The --cut_seq_length parameter determines the length at which a contig will be cut if its length exceeds this value. The default setting is 8192, which is also the
                        maximum length allowed during training. If a contig's length surpasses this threshold, it will be divided into smaller subsequences with lengths equal to or less
                        than the cut_seq_length.
  --mag_length_threshold MAG_LENGTH_THRESHOLD
                        The threshold for the total length of a MAG's contigs is used to filter generated MAGs after applying single-copy genes (SCGs). The default threshold is set to
                        200,000, which represents the total length of the contigs in base pairs (bp). MAGs with a total contig length equal to or greater than this threshold will be
                        considered for further analysis or inclusion, while MAGs with a total contig length below the threshold may be filtered out.
  --num_process NUM_PROCESS
                        The maximum number of threads will be used. All CPUs will be used if it is None. Defaults to None
  --topk_or_greedy_search {topk,greedy}
                        Topk searching or greedy searching to label a contig. Defaults to "topk".
  --topK_num TOPK_NUM   During the top-k searching approach, the default behavior is to search for the top-k nodes that exhibit the highest cosine similarity with the contig's encoded
                        vector. By default, the value of k is set to 3, meaning that the three most similar nodes in terms of cosine similarity will be considered for labeling the contig.
                        Please note that this parameter does not have any effect when using the greedy search approach (topK_num=1). Defaults to 3.
  --temp_output_folder TEMP_OUTPUT_FOLDER
                        The temporary files generated during the process can be stored in a specified folder path. By default, if no path is provided (i.e., set to None), the temporary
                        files will be stored in the parent folder of the '--input_path' location. However, you have the option to specify a different folder path to store these temporary
                        files if needed.
  --db_folder_path DB_FOLDER_PATH
                        The path of database folder. By default, if no path is provided (i.e., set to None), it is expected that the environment variable 'DeepurifyInfoFiles' has been set
                        to point to the appropriate folder. Please ensure that the 'DeepurifyInfoFiles' environment variable is correctly configured if the path is not explicitly
                        provided.
  --model_weight_path MODEL_WEIGHT_PATH
                        The path of model weight. (In database folder) Defaults to None.
  --taxo_vocab_path TAXO_VOCAB_PATH
                        The path of taxon vocabulary. (In database folder) Defaults to None.
  --taxo_tree_path TAXO_TREE_PATH
                        The path of taxonomic tree. (In database folder) Defaults to None.
```
Please run 'deepurify re-bin -h' for more details.


## Files in output directory
- #### The purified MAGs.

- #### MetaInfo.tsv
This file contains the following columns: 

1. MAG name (first column), 
2. completeness of MAG (second column), 
3. contamination of MAG (third column), 
4. MAG quality (fourth column),

## Minimum System Requirements for Running Deepurify
- System: Linux (>= Ubuntu 22.04.2 LTS)
- CPU: No restriction.
- RAM: >= 32 GB
- GPU: The GPU memory must be equal to or greater than 6GB. (5273MB)

This system can run the configuration in the **"Running Deepurify"** section.


## Our System Config
- System: NVIDIA DGX Server Version 5.5.1 (GNU/Linux 5.4.0-131-generic x86_64)
- CPU: AMD EPYC 7742 64-Core Processor (2 Sockets)
- RAM: 1TB
- GPU: 8 GPUs (A100-40GB)

This system can run the configuration in the main.py file.
