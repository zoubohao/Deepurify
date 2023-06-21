# Deepurify_Project
  **Paper: Deepurfy: metagenome-assembled genomes contamination removal using deep language model.**
  
  Deepurify is a tool for improving the quality of MAGs by filtering contaminated contigs using a multi-modality deep language model that can be accelerated by GPUs.
  

## Dependencies:
Install the tools by yourself and make sure these tools can work properly.
- **prodigal** v 2.6.3 (ORF/CDS-prediction)
- **hmmer** v.3.3.1 (Detecting conserved single-copy marker genes (SCGs))
- **CheckM** v 1.2.2 (Evaluate the quality of MAGs)

  (**Please note**: Please installing the right version of **pplacer** is necessary for zero error during running CheckM. 
  Otherwise, CheckM may have ERROR during running. 
  The version of **pplacer** that we used is "v1.1.alpha19-0-g807f6f3".)

## Installtion:
The Deepurify can be installed via pip (without dependencies). 
In case of pip, all dependencies listed above need to be installed seperately.

```
pip install Deepurify==1.1.0.4
```

## Download Files for Running

Download the necessary files for running Deepurify via https://drive.google.com/file/d/1i-qNfxVmxDXymTuVoTPuNFSB6VdKIYjb/view?usp=sharing

Unzip this file and set an **enviroment variable** "DeepurifyInfoFiles" with
```
export DeepurifyInfoFiles=/path/of/this/unzip/folder/
```

## Usage of Deepurify
1). You can use the Deepurify from the **cleanMAGs** function.
```
from Deepurify.clean_func import cleanMAGs

if __name__ == "__main__":
    inputBinFolderPath = "/path/of/bins/folder/"
    outputBinFolderPath = "/path/of/output/folder/"
    
    cleanMAGs(
        input_bin_folder_path=inputBinFolderPath,
        output_bin_folder_path=outputBinFolderPath,
        # setting of contig inference stage 
        bin_suffix="fa",
        gpu_num=2,
        batch_size_per_gpu=20,
        num_threads_per_device=4,
        # setting of call gene stage
        num_threads_call_genes=64,
        # setting of running checkm stage
        checkM_process_num=2,
        num_threads_per_checkm=22,
        # others
        temp_output_folder="/path/of/output/folder/DeepTempFiles/",
        self_evaluate=False,
    )

```


2). You can use Deepurify from command:
```
deepurify clean  -i /path/to/your/mags/ -o /path/to/your/output/ --bin_suffix fa --gpu_num 1 --num_worker 1
```

