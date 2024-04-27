
from shutil import copy
from subprocess import Popen
import os
import argparse
import sys
from multiprocessing.pool import Pool


def runMDMcleaner(input_folder, output_folder):
    files = os.listdir(input_folder)
    sinal = True
    for file in files:
        id_name = os.path.splitext(file)[0]
        cur_output_folder = os.path.join(output_folder, id_name, f"{id_name}_filtered_kept_contigs.fasta.gz")
        print(cur_output_folder)
        if os.path.exists(cur_output_folder) is False:
            sinal = False
    if sinal:
        # print(f"######### {input_folder} ########")
        return
    cmd = f"mdmcleaner clean -i {os.path.join(input_folder, '*')} -o {output_folder}"
    run_hd = Popen(
        cmd,
        shell = True
    )
    run_hd.communicate()
    run_hd.kill()


def splitListEqually(input_list, num_parts: int):
    n = len(input_list)
    step = n // num_parts + 1
    out_list = []
    for i in range(num_parts):
        if curList := input_list[i * step: (i + 1) * step]:
            out_list.append(curList)
    return out_list

# cd /datahome/datasets/ericteam/csbhzou/Deepurify_review/mdmc_temp_dir/
# nohup python  mdmcleaner0.py -s 8600 -t 500 -c 32 > ibs_8600_500.log 2>&1 &
if __name__ == "__main__":
    input_folder = "/datahome/datasets/ericteam/csbhzou/Deepurify_review/soil-MDMcleaner-split-bins"
    output_folder = "/datahome/datasets/ericteam/csbhzou/Deepurify_review/soil-MDMcleaner-split-Results"
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("-s", "--start")
    parser.add_argument("-t", "--step")
    parser.add_argument("-c", "--cpu")
    
    args = parser.parse_args()
    start = int(args.start)
    step = int(args.step)
    final = 9078
    # 
    end = start + step
    if end > final:
        end = final
    
    if os.path.exists(output_folder) is False:
        os.mkdir(output_folder)
    res_list = []
    with Pool(int(args.cpu)) as p:
        for i in range(start, start + step):
            cur_input = os.path.join(input_folder, str(i))
            cur_output = os.path.join(output_folder, str(i))
            if os.path.exists(cur_output) is False:
                os.mkdir(cur_output)
            res = p.apply_async(runMDMcleaner, args=(cur_input, cur_output))
            res_list.append(res)
        p.close()
        p.join()

