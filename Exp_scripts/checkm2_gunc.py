
import argparse
import os
import subprocess
from multiprocessing import Process
import sys

def runCheckm2Single(
        input_bin_folder: str,
        output_bin_folder: str,
        bin_suffix: str,
        num_cpu: int):
    if os.path.exists(output_bin_folder) is False:
        os.makedirs(output_bin_folder)
    res = subprocess.Popen(
        f"checkm2 predict -x {bin_suffix} --threads {num_cpu} -i {input_bin_folder} -o {output_bin_folder}",
        shell=True,
    )
    res.wait()
    res.kill()


def run_gunc(input_folder, bin_suffix, output_folder):
    if os.path.exists(output_folder) is False:
        os.makedirs(output_folder)
    cmd = f"gunc run -d {input_folder} -e {bin_suffix} -t 128 -o {output_folder}"
    res = subprocess.Popen(
        cmd,
        shell=True,
    )
    res.wait()
    res.kill()


if __name__ == "__main__":
    parse = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),)
    parse.add_argument("-i", "--input_folders")
    parse.add_argument("-e", "--bin_suffix", default=None)
    args = parse.parse_args()
    input_folders = args.input_folders
    bin_suffix = args.bin_suffix
    
    folders = os.listdir(input_folders)
    for folder in folders:
        print(folder)
        # if "Semibin2" not in folder:
        #     continue
        if "gunc" in folder or "checkm2" in folder:
            continue
        cur_input_folder = os.path.join(input_folders, folder)
        if bin_suffix is None:
            _, bin_suffix = os.path.splitext(os.listdir(cur_input_folder)[0])
            bin_suffix = bin_suffix[1:]
        out_checkm2_folder = os.path.join(input_folders, folder + "_checkm2")
        if os.path.exists(out_checkm2_folder) is False:
            os.mkdir(out_checkm2_folder)
        out_gunc_folder = os.path.join(input_folders, folder + "_gunc")
        if os.path.exists(out_gunc_folder) is False:
            os.mkdir(out_gunc_folder)
        p1 = Process(target=runCheckm2Single, args=(cur_input_folder, out_checkm2_folder, bin_suffix, 128))
        p1.start()
        p2 = Process(target=run_gunc, args=(cur_input_folder, bin_suffix, out_gunc_folder))
        p2.start()
        p1.join()
        p2.join()









