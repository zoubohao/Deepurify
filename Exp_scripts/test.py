
from Deepurify.Utils.RunCMDUtils import runCONCOCT, runMetaBAT2, runSemibin





if __name__ == "__main__":
    contig_path = "/home/datasets/ZOUbohao/Proj1-Deepurify/1021520/contigs.fasta"
    bam_path = "/home/datasets/ZOUbohao/Proj1-Deepurify/1021520/1021520_contigs.sorted.bam"
    output_path = "/home/datasets/ZOUbohao/Proj1-Deepurify/1021520_test_concoct_metabat2"
    runSemibin(contig_path, bam_path, output_path)