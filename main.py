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
