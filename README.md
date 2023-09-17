# Deepurify_Project

<div align=center> <img width="300" height="300" src="/deeplogo.png" alt="logo"></div>

  **Paper --> Deepurify: a multi-modal deep language model to remove contamination from metagenome-assembled genomes**
  
 Deepurify elevates metagenome-assembled genomes' (MAGs) quality by utilizing a multi-modal deep language model to filter contaminated contigs, and it can leverage GPU acceleration.


## Dependencies:
Please independently install the following tools and ensure their proper functionality.

1. **[Prodigal](https://github.com/hyattpd/Prodigal/wiki/installation)** v 2.6.3 (ORF/CDS-prediction)
2. **[Hmmer](http://hmmer.org/download.html)** v.3.3.1 (Detecting conserved single-copy marker genes)
3. **[CheckM](https://github.com/Ecogenomics/CheckM/wiki/)** v 1.2.2 (Evaluate the quality of MAGs)
**Note**: Installing the correct version of **pplacer** is essential to avoid errors when running CheckM. 
            Failure to do so may result in errors during execution CheckM. 
            We utilized pplacer version "v1.1.alpha19" in our work.
4. **[PyTorch](https://pytorch.org/)** v2.0.1 + cu118 (GPU version)

**Note**: Ensure that all the listed dependencies above are installed and functioning without any errors.


## Installation:
Deepurify can be installed using pip without dependencies. 
```
pip install Deepurify==1.2.4
```


## Download Files and Set Environment Variable for Running
- Download the required files for running Deepurify from this **[LINK](https://drive.google.com/file/d/1i-qNfxVmxDXymTuVoTPuNFSB6VdKIYjb/view?usp=sharing)**.

- Unzip the downloaded file and set an **environment variable** called "DeepurifyInfoFiles" by adding the following line to the last line of .bashrc file (~/.bashrc):
```
export DeepurifyInfoFiles=/path/of/this/unzip/folder/
```
For example: 'export DeepurifyInfoFiles=/home/csbhzou/software/DeepurifyInfoFiles/'.

- Save the .bashrc file, and then execute:
```
source .bashrc
```


## Running Deepurify
1.  You can run the Deepurify through the **cleanMAGs** function.
```
from Deepurify.clean_func import cleanMAGs

if __name__ == "__main__":
    inputBinFolderPath = "./demo_input/"
    outputBinFolderPath = "./demo_output/"
    
    cleanMAGs(
        input_bin_folder_path=inputBinFolderPath, # Input directory containing MAGs
        output_bin_folder_path=outputBinFolderPath, # Output directory containing purification MAGs
        bin_suffix="fa", # The file suffix of MAGs.
        gpu_num=1, # Specify the number of GPUs to be used (use '0' for CPU, considerably slower than using GPU).
        batch_size_per_gpu=1, # The number of batch size for each GPU. Useless with gpu_num=0
        num_threads_per_device=1, # The number of threads for labeling taxonomic lineage for contigs.
        num_threads_call_genes=1, # The number of threads to call genes.
        checkM_process_num=1, # The number of processes to run CheckM simultaneously.
        num_threads_per_checkm=1, # The number of threads to run a single CheckM process.
        temp_output_folder=None, # The directory to store temporary files.
    )

```
Please refer to the documentation of this function for more details.

2.  You can run Deepurify through the following command:
```
deepurify clean  -i ./demo_input/ -o ./demo_output/ --bin_suffix fa --gpu_num 1 --num_threads_per_device 1
```
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```
usage: deepurify clean [-h] -i INPUT_PATH -o OUTPUT_PATH --bin_suffix BIN_SUFFIX [--gpu_num GPU_NUM] [--batch_size_per_gpu BATCH_SIZE_PER_GPU]
                       [--num_threads_per_device NUM_THREADS_PER_DEVICE] [--num_threads_call_genes NUM_THREADS_CALL_GENES] [--checkM_process_num {1,2,3,6}]
                       [--num_threads_per_checkm NUM_THREADS_PER_CHECKM] [--temp_output_folder TEMP_OUTPUT_FOLDER] [--output_bins_meta_info_path OUTPUT_BINS_META_INFO_PATH]
                       [--info_files_path INFO_FILES_PATH] [--simulated_MAG {True,False}]

options:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input_path INPUT_PATH
                        The input folder containing MAGs
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        The output folder containing decontaminated MAGs.
  --bin_suffix BIN_SUFFIX
                        The suffix of MAG files.
  --gpu_num GPU_NUM     The number of GPUs to be used, with the default value being 1. Setting it to 0 will make the code use the CPU, but it's important to note that using
                        the CPU can result in significantly slower processing speeds. For better performance, it is recommended to have at least one GPU with a memory capacity
                        of 3GB or more.
  --batch_size_per_gpu BATCH_SIZE_PER_GPU
                        The --batch_size_per_gpu defines the number of sequences loaded onto each GPU simultaneously. This parameter is relevant only when the --gpu_num option
                        is set to a value greater than 0. The default batch size is 1, which means that one sequence will be loaded per GPU by default.
  --num_threads_per_device NUM_THREADS_PER_DEVICE
                        The --num_threads_per_device (GPU or CPU) controls the level of parallelism during the contigs' lineage inference stage. If the --gpu_num option is set
                        to a value greater than 0, each GPU will utilize this specified number of threads for inference. Likewise, if --gpu_num is set to 0 and the code runs
                        on a CPU, the same number of threads will be employed. By default, each GPU or CPU uses 1 thread. The --batch_size_per_gpu value will be divided by
                        this value to calculate the batch size per thread.
  --num_threads_call_genes NUM_THREADS_CALL_GENES
                        The number of threads to call genes. Defaults to 12.
  --checkM_process_num {1,2,3,6}
                        The number of processes to run CheckM simultaneously. Defaults to 1.
  --num_threads_per_checkm NUM_THREADS_PER_CHECKM
                        The number of threads to run a single CheckM process. Defaults to 12.
  --temp_output_folder TEMP_OUTPUT_FOLDER
                        The folder stores the temporary files, which are generated during the running Deepurify. If no path is provided (set to None), the temporary files will
                        be stored in the parent folder of the '--input_bin_folder_path' location by default.
  --output_bins_meta_info_path OUTPUT_BINS_META_INFO_PATH
                        The path for a text file to record meta information, including the evaluated results of the output MAGs. If no path is provided (set to None), the file
                        will be automatically created in the '--output_bin_folder_path' directory by default.
  --info_files_path INFO_FILES_PATH
                        The files in the 'DeepurifyInfoFiles' folder are a crucial requirement for running Deepurify. If you don't provide a path explicitly (set to None), it
                        is assumed that the 'DeepurifyInfoFiles' environment variable has been properly configured to point to the necessary folder. Ensure that the
                        'DeepurifyInfoFiles' environment variable is correctly set up if you don't specify the path.
  --simulated_MAG {True,False}
                        If the input MAGs are simulated MAGs. False by default. This option is valuable when you have prior knowledge of core and contaminated contigs in
                        simulated MAGs or prefer to personally assess the results. When it sets to True, we will exclude contaminated contigs and retain core contigs using
                        varying cosine similarity thresholds for each MAG. Multiple sets of results will be generated in different folders within the
                        '/temp_output_folder/FilterOutput/' directory. You should independently evaluate these different results and select the MAGs that exhibit the best
                        performance.
```


## Files in output directory
- #### The purified MAGs.
- #### time.txt 
The elapsed running time of Deepurify is shown in two columns. 

1. The first column represents the time (seconds) taken to infer the taxonomic lineage.
2. The second column represents the time (seconds) taken to evaluate the results.

- #### MetaInfo.txt 
This file contains the following columns: 

1. MAG name (first column), 
2. completeness of MAG (second column), 
3. contamination of MAG (third column), 
4. MAG quality (fourth column),
5. potential taxonomic lineage for MAG (fifth column).

## Minimum System Requirements for Running Deepurify
- System: Linux (>= Ubuntu 22.04.2 LTS)
- CPU: No restriction.
- RAM: > 45GB (Running CheckM with using the full reference genome tree required approximately 40 GB of memory.)
- GPU: The GPU memory must be equal to or greater than 3GB.
This system can run the configuration in the **"Running Deepurify"** section.


## Our System Config
- System: WSL (Ubuntu 22.04.2 LTS)
- CPU: AMD EPYC 7542 (32 cores 64 threads).
- RAM: 256GB
- GPU: Double GTX-3090 24GB
This system can run the configuration in the main.py file.
