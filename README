# Deepurify
 A tool to improve the quality of MAGs based on multi-modality deep learning model.

## None-Python Dependencies:
- hmmer v.3.3.1 (detecting conserved single-copy marker genes)
- prodigal v 2.6.3 (ORF/CDS-prediction)
- CheckM v 1.2.2 (evaluate the quality of MAGs)
  (Please note: For installing CheckM, installing the right version of **pplacer** is necessary. Otherwise, CheckM may have ERROR during running. 
  The version that I installed is "v1.1.alpha19-0-g807f6f3".)

## Installtion:
The Deepurify can be installed via pip (without dependencies). In case of pip, all dependencies listed above need to be installed seperately.

*OR*

Download the source code and running it via command.

*NOTE*
Download the necessary files for running Deepurify via https:\\
Unzip this file and set an enviroment variable "DeepurifyInfoFiles" with
```
export DeepurifyInfoFiles=/path/of/this/unzip/folder/
```

## Usage of Deepurify
You can use the Deepurify from the **clean** function.
```
from Deepurify import clean

if __name__ == "__main__":
    input_mag_foler = "/path/to/your/mags/"
    output_mag_foler = "/path/to/your/output/"
    clean(
        input_bin_folder_path = input_mag_foler,
        output_bin_folder_path = output_mag_folder,
        bin_suffix = "fa",
        gpu_num = 1, ## it can set 0 to use CPU, but it is much slower.
        num_worker = 4 
        )
```

*OR*

You can use Deepurify from command:
```
python Deepurify.py clean  -i /path/to/your/mags/ -o /path/to/your/output/ --bin_suffix fa --gpu_num 2
```

