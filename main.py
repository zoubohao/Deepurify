from Deepurify.clean_func import cleanMAGs


if __name__ == "__main__":
    # The following config is our setting for testing.
    # CPU: AMD EPYC 7542, 2 x GTX-3090 GPUs, 256GB RAM.
    inputBinFolderPath = "/home/csbhzou/Deepurify_test_data/hlj/MetaBat2/"
    outputBinFolderPath = "/home/csbhzou/Deepurify_test_data/MetaBat2_test/"
    
    cleanMAGs(
        input_bin_folder_path=inputBinFolderPath,
        output_bin_folder_path=outputBinFolderPath,
        # setting of contig inference stage 
        bin_suffix="fa",
        gpu_num=2,
        batch_size_per_gpu=24,
        num_threads_per_device=2,
        num_threads_call_genes=64,
        checkM_process_num=2,
        num_threads_per_checkm=22,
    )
