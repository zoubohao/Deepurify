from Deepurify.Utils.RunCMDUtils import runGUNCsingle
import os




if __name__ == "__main__":
    input_folder = "/home/datasets/ZOUbohao/Proj1-Deepurify/CAMI-original-deepurify-only-clean"
    folders = os.listdir(input_folder)
    for folder in folders:
        print(folder)
        if "checkm2" in folder or "gunc" in folder:
            continue
        gunc_folder = os.path.join(input_folder, f"{folder}_gunc")
        if os.path.exists(gunc_folder) is False:
            os.mkdir(gunc_folder)
        runGUNCsingle(
            os.path.join(input_folder, folder),
            gunc_folder,
            128,
            "fasta"
        )