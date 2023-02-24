

from Deepurify.RUN_Functions import cleanMAGs


if __name__ == "__main__":
    input_mag_folder = "/path/of/input/MAG/folder/"
    output_mag_folder = "/path/of/output/cleanMAGs/folder/"

    cleanMAGs(
        input_bin_folder_path=input_mag_folder,
        output_bin_folder_path=output_mag_folder,
        bin_suffix="fa",
        gpu_num=2,  # it can set 0 to use CPU, but it is much slower.
        num_worker=2,
        batch_size_per_gpu=20
    )
