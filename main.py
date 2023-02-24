

from .RUN_Functions import cleanMAGs


if __name__ == "__main__":
    input_mag_folder = "/path/to/your/mags/"
    output_mag_folder = "/path/to/your/output/"

    cleanMAGs(
        input_bin_folder_path=input_mag_folder,
        output_bin_folder_path=output_mag_folder,
        bin_suffix="fa",
        gpu_num=1,  # it can set 0 to use CPU, but it is much slower.
        num_worker=2
    )
