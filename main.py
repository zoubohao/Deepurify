from Deepurify.clean_func import cleanMAGs


def read(file_path: str):
    res = {}
    with open(file_path, "r") as rh:
        for line in rh:
            info = line.strip("\n").split("\t")
            idx = info[0].split("_")[0]
            if idx not in res:
                res[idx] = [tuple(info[0:-1])]
            else:
                res[idx].append(tuple(info[0:-1]))
    return res


if __name__ == "__main__":
    
    
    # file1 = "/mnt/e/OriMetaInfo.txt"
    # file2 = "/mnt/e/MetaInfo.txt"
    
    # res1 = read(file1)
    # res2 = read(file2)
    
    # for key in res1.keys():
    #     v1 = res1[key]
    #     v2 = res2[key]
    #     if len(v1) != len(v2):
    #         print("##########")
    #         print(v1)
    #         print(v2)
    
    inputBinFolderPath = "/root/Deepurify_Data/RealDataBinningModels/Concoct_Data/CAMI/high/fasta_bins/"
    outputBinFolderPath = "/root/Deepurify_Data/RealDataBinningModels/Concoct_Data/test_concoct_cami_high/"
    
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
        temp_output_folder="/root/Deepurify_Data/RealDataBinningModels/Concoct_Data/DeepTempFiles/",
        self_evaluate=False,
    )