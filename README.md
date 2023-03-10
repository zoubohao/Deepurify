# Deepurify
  Deepurify is a tool for improving the quality of MAGs by filtering contaminated contigs using a multi-modality deep language model that can be accelerated by GPUs.

## Dependencies:
- **hmmer** v.3.3.1 (Detecting conserved single-copy marker genes (SCGs))
- **prodigal** v 2.6.3 (ORF/CDS-prediction)
- **CheckM** v 1.2.2 (Evaluate the quality of MAGs)

  (**Please note**: Please installing the right version of **pplacer** is necessary for zero error during running CheckM. 
  Otherwise, CheckM may have ERROR during running. 
  The version of **pplacer** that we used is "v1.1.alpha19-0-g807f6f3".)

## Installtion:
The Deepurify can be installed via pip (without dependencies). 
In case of pip, all dependencies listed above need to be installed seperately.

```
pip install Deepurify==1.1.0.2
```

#### NOTE

Download the necessary files for running Deepurify via https://drive.google.com/file/d/1i-qNfxVmxDXymTuVoTPuNFSB6VdKIYjb/view?usp=sharing

Unzip this file and set an enviroment variable "DeepurifyInfoFiles" with
```
export DeepurifyInfoFiles=/path/of/this/unzip/folder/
```

## Usage of Deepurify
1. You can use the Deepurify from the **cleanMAGs** function.
```
from Deepurify.RUN_Functions import cleanMAGs

if __name__ == "__main__":
    input_mag_foler = "/path/to/your/mags/"
    output_mag_foler = "/path/to/your/output/"
    cleanMAGs(
        input_bin_folder_path = input_mag_foler,
        output_bin_folder_path = output_mag_folder,
        bin_suffix = "fa",
        gpu_num = 1, ## it can set 0 to use CPU, but it is much slower.
        num_worker = 1 
        )
```


2. You can use Deepurify from command:
```
deepurify clean  -i /path/to/your/mags/ -o /path/to/your/output/ --bin_suffix fa --gpu_num 1 --num_worker 1
```

