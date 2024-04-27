import argparse
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'
from Deepurify.clean_func import cleanMAGs


def splitListEqually(input_list, num_parts: int):
    n = len(input_list)
    step = n // num_parts + 1
    out_list = []
    for i in range(num_parts):
        if curList := input_list[i * step: (i + 1) * step]:
            out_list.append(curList)
    return out_list


if __name__ == "__main__":
    parse = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),)
    parse.add_argument("-i", "--input_folder")
    parse.add_argument("-o", "--output_folder")
    parse.add_argument("-n", "--name")
    args = parse.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    name = args.name
    if os.path.exists(output_folder) is False:
        os.mkdir(output_folder)
    
    contig_fasta_path = os.path.join(input_folder, f"{name}.contigs.fasta")
    bam_file_path = os.path.join(input_folder, f"{name}.sorted.bam")
    cur_output_folder = os.path.join(output_folder, f"{name}")
    print(input_folder)
    print(cur_output_folder)
    if os.path.exists(os.path.join(cur_output_folder, "Deepurify_Bin_0.fasta")):
        sys.exit(0)

    # cd /home/datasets/ZOUbohao/Proj1-Deepurify/Deepurify-v2.2.0-ReBinning/ SRR25158210
    # python deepurify.py -i ../plant_data/ -o ../plant-ensemble-Deepurify/ -n SRR10968246
    cleanMAGs(
        output_bin_folder_path=cur_output_folder,
        batch_size_per_gpu=32,
        each_gpu_threads=2,
        # setting of contig inference stage
        contig_fasta_path=contig_fasta_path,
        sorted_bam_file=bam_file_path,
        gpu_work_ratio=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.25], # [0.25, 0.25, 0.25, 0.25]
        db_files_path="./GTDB_Taxa_Info/"
    )

