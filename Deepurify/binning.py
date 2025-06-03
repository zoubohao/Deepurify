



from multiprocessing import Process
import os
import time

from Deepurify.Utils.RunCMDUtils import cal_depth, runCONCOCT, runMetaBAT2, runSemibin


def bin_c(
    contigs_path: str,
    sorted_sorted_bam_file: str,
    res_output_path,
    depth_path,
    num_cpu
):
    if depth_path is not None and os.path.exists(depth_path) is False:
        cal_depth(sorted_sorted_bam_file, depth_path)
    concoct_folder = os.path.join(res_output_path, "concoct")
    if os.path.exists(concoct_folder) is False: 
        os.mkdir(concoct_folder)
    cur_depth_path = os.path.join(res_output_path, "cur_depth.txt")
    first_line = ""
    ori_depth_info = {}
    with open(depth_path, "r") as rh:
        i = 0
        for line in rh:
            if i == 0:
                first_line = line
            else:
                info = line.strip("\n").split("\t")
                ori_depth_info[info[0]] = info[1:]
            i += 1
    
    with open(contigs_path, "r") as rh, open(cur_depth_path, "w") as wh:
        wh.write(first_line)
        for line in rh:
            if line[0] == ">":
                contig_name = line.strip("\n")[1:]
                wh.write("\t".join([contig_name] + ori_depth_info[contig_name]) + "\n")
                
    p = Process(target=runCONCOCT, args=(contigs_path, cur_depth_path, concoct_folder, num_cpu))
    p.start()
    p.join()


def bin_m(
    contigs_path: str,
    sorted_sorted_bam_file: str,
    res_output_path,
    depth_path,
    num_cpu
):
    if depth_path is not None and os.path.exists(depth_path) is False:
        cal_depth(sorted_sorted_bam_file, depth_path)
    metabat2_folder = os.path.join(res_output_path, "metabat2")
    if os.path.exists(metabat2_folder) is False: os.mkdir(metabat2_folder)
    cur_depth_path = os.path.join(res_output_path, "cur_depth.txt")
    first_line = ""
    ori_depth_info = {}
    with open(depth_path, "r") as rh:
        i = 0
        for line in rh:
            if i == 0:
                first_line = line
            else:
                info = line.strip("\n").split("\t")
                ori_depth_info[info[0]] = info[1:]
            i += 1
    
    with open(contigs_path, "r") as rh, open(cur_depth_path, "w") as wh:
        wh.write(first_line)
        for line in rh:
            if line[0] == ">":
                contig_name = line.strip("\n")[1:]
                wh.write("\t".join([contig_name] + ori_depth_info[contig_name]) + "\n")
    p = Process(target=runMetaBAT2, args=(contigs_path, cur_depth_path, metabat2_folder, num_cpu))
    p.start()
    p.join()


def bin_c_m(
    contigs_path: str,
    sorted_sorted_bam_file: str,
    res_output_path,
    depth_path,
    num_cpu
):
    if depth_path is not None and os.path.exists(depth_path) is False:
        cal_depth(sorted_sorted_bam_file, depth_path)
    concoct_folder = os.path.join(res_output_path, "concoct")
    metabat2_folder = os.path.join(res_output_path, "metabat2")
    if os.path.exists(concoct_folder) is False: os.mkdir(concoct_folder)
    if os.path.exists(metabat2_folder) is False: os.mkdir(metabat2_folder)
    cur_depth_path = os.path.join(res_output_path, "cur_depth.txt")
    first_line = ""
    ori_depth_info = {}
    with open(depth_path, "r") as rh:
        i = 0
        for line in rh:
            if i == 0:
                first_line = line
            else:
                info = line.strip("\n").split("\t")
                ori_depth_info[info[0]] = info[1:]
            i += 1
    
    with open(contigs_path, "r") as rh, open(cur_depth_path, "w") as wh:
        wh.write(first_line)
        for line in rh:
            if line[0] == ">":
                contig_name = line.strip("\n")[1:]
                contig_name = contig_name.split()[0]
                if contig_name in ori_depth_info:
                    wh.write("\t".join([contig_name] + ori_depth_info[contig_name]) + "\n")
    
    process_list = []
    p = Process(target=runCONCOCT, args=(contigs_path, cur_depth_path, concoct_folder, num_cpu))
    process_list.append(p)
    p.start()
    p = Process(target=runMetaBAT2, args=(contigs_path, cur_depth_path, metabat2_folder, num_cpu))
    process_list.append(p)
    p.start()
    for p in process_list:
        p.join()


def binning(
    contigs_path: str,
    sorted_sorted_bam_file: str,
    res_output_folder,
    depth_path,
    num_cpu,
    binning_mode: str = None):
    s_time = time.time()
    if binning_mode is None:
        print(f"--> binning mode is 'Ensemble'...")
        pro_list = []
        semibin_out = os.path.join(res_output_folder, "semibin")
        if os.path.exists(semibin_out) is False: os.mkdir(semibin_out)
        p = Process(target=runSemibin, args=(contigs_path, sorted_sorted_bam_file, semibin_out, num_cpu))
        p.start()
        pro_list.append(p)
        p = Process(target=bin_c_m, args=(contigs_path, sorted_sorted_bam_file, res_output_folder, depth_path, num_cpu))
        p.start()
        pro_list.append(p)
        for p in pro_list:
            p.join()
    elif binning_mode.lower() == "semibin2":
        print(f"--> Binning mode is 'Semibin2'...")
        semibin_out = os.path.join(res_output_folder, "semibin")
        if os.path.exists(semibin_out) is False: os.mkdir(semibin_out)
        p = Process(target=runSemibin, args=(contigs_path, sorted_sorted_bam_file, semibin_out, num_cpu))
        p.start()
        p.join()
    elif binning_mode.lower() == "concoct":
        print(f"--> Binning mode is 'CONCOCT'...")
        bin_c(contigs_path, sorted_sorted_bam_file, res_output_folder, depth_path, num_cpu)
    elif binning_mode.lower() == "metabat2":
        print(f"--> Binning mode is 'MetaBAT2'...")
        bin_m(contigs_path, sorted_sorted_bam_file, res_output_folder, depth_path, num_cpu)
    else:
        raise ValueError("Only implement CONCOCT, MetaBAT2, SemiBin2 binning tools.")
    e_time = time.time()
    with open(os.path.join(res_output_folder, "binning.time"), "w") as wh:
        wh.write(str(e_time - s_time) + "\n")