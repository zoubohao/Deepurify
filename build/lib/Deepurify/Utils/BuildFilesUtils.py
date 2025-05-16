
import os
from itertools import product
from shutil import copy

import numpy as np
import torch

from Deepurify.Model.EncoderModels import DeepurifyModel
from Deepurify.Utils.IOUtils import (getNumberOfPhylum, loadTaxonomyTree,
                                     progressBar, readCSV, readFasta,
                                     readMetaInfo, readPickle, readVocabulary,
                                     writeFasta, writePickle)
from Deepurify.Utils.LabelBinsUtils import buildTextsRepNormVector
from Deepurify.Utils.SelectBinsUitls import getScore


def generate_feature_mapping(kmer_len):
    BASE_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}
    kmer_hash = {}
    counter = 0
    for kmer in product("ATGC", repeat=kmer_len):
        kmer = ''.join(kmer)
        if kmer not in kmer_hash:
            kmer_hash[kmer] = counter
            rev_compl = tuple([BASE_COMPLEMENT[x] for x in reversed(kmer)])
            kmer_hash[''.join(rev_compl)] = counter
            counter += 1
    return kmer_hash, counter


def get_normlized_vec(seq: str):
    kmer_len = 4
    seq = seq.upper()
    kmer_dict, nr_features = generate_feature_mapping(kmer_len)
    kmers = [kmer_dict[seq[i:i+kmer_len]]
                for i in range(len(seq) - kmer_len + 1)
                    if seq[i:i+kmer_len] in kmer_dict] # ignore kmers with non-canonical bases
    res = np.bincount(np.array(kmers, dtype=np.int64), minlength=nr_features)
    summed = sum(res)
    res = np.array(res, dtype= np.float32) + 1.0
    return res / summed



def buildAllConcatFiles(
    input_bins_folder: str,
    tmp_annot_folder: str,
    concat_annot_path: str,
    concat_vectors_path: str,
    concat_TNF_vector_path: str,
    concat_contig_path: str,
    bin_suffix: str
):
    with open(concat_annot_path, "w") as wh:
        for file in os.listdir(tmp_annot_folder):
            pro, suffix = os.path.splitext(file)
            if suffix[1:] == "txt":
                with open(os.path.join(tmp_annot_folder, file), "r") as rh:
                    for line in rh:
                        wh.write(line)
    
    ## concat vectors
    contigName2repNormVec = {}
    for file in os.listdir(tmp_annot_folder):
        pro, suffix = os.path.splitext(file)
        if suffix[1:] == "pkl":
            contigName2repNormVec.update(readPickle(os.path.join(tmp_annot_folder, file)))
    writePickle(concat_vectors_path, contigName2repNormVec)
    
    contigName2seq = {}
    for file in os.listdir(input_bins_folder):
        _, suffix = os.path.splitext(file)
        if suffix[1:] == bin_suffix:
            cur_contigname2seq = readFasta(os.path.join(input_bins_folder, file))
            contigName2seq.update(cur_contigname2seq)
    writeFasta(contigName2seq, concat_contig_path)
    
    contigName2TNFV = {}
    i = 1
    n = len(contigName2seq)
    for contigName, seq in contigName2seq.items():
        contigName2TNFV[contigName] = get_normlized_vec(seq)
        progressBar(i, n)
        i += 1
    writePickle(concat_TNF_vector_path, contigName2TNFV)


def filterSpaceInFastaFile(input_fasta, output_fasta):
    with open(input_fasta, "r") as rh, open(output_fasta, "w") as wh:
        for line in rh:
            oneline = line.strip("\n")
            if ">" in oneline and " " in oneline:
                oneline = oneline.split()[0]
            wh.write(oneline + "\n")


def buildSubFastaFile(
    metaInfoPath: str,
    binTmpOutFolder: str,
    concat_fasta_path: str,
    bin_suffix: str,
    other_contigs_file = None
):
    ## only for medium low quality
    h_num = 0
    res2quality, _, _, _ = readMetaInfo(metaInfoPath)
    contigName2seq = {}
    for file in os.listdir(binTmpOutFolder):
        _, suffix = os.path.splitext(file)
        if suffix[1:] == bin_suffix and file in res2quality and res2quality[file][-1] == "HighQuality":
            h_num += 1
        if suffix[1:] == bin_suffix and file in res2quality and res2quality[file][-1] != "HighQuality":
            cur_contigname2seq = readFasta(os.path.join(binTmpOutFolder, file))
            contigName2seq.update(cur_contigname2seq)
    if other_contigs_file is not None:
        contigName2seq.update(readFasta(other_contigs_file))
    writeFasta(contigName2seq, concat_fasta_path)
    return h_num


def build_taxonomic_file(
    taxoTreePath: str,
    taxoVocabPath: str,
    mer3Path: str,
    mer4Path: str,
    modelWeightPath: str,
    taxoName2RepNormVecOutPath: str,
    model_config=None
):

    if model_config is None:
        model_config = {
            "min_model_len": 1000,
            "max_model_len": 8192,
            "inChannel": 108,
            "expand": 1.2,
            "IRB_num": 2,
            "head_num": 6,
            "d_model": 864,  # 1080
            "num_GeqEncoder": 6,
            "num_lstm_layers": 5,
            "feature_dim": 1024,
        }
    taxo_tree = loadTaxonomyTree(taxoTreePath)
    taxo_vocabulary = readVocabulary(taxoVocabPath)
    mer3_vocabulary = readVocabulary(mer3Path)
    mer4_vocabulary = readVocabulary(mer4Path)
    spe2index = {}
    index = 0
    for name, _ in taxo_vocabulary.items():
        if "s__" == name[0:3]:
            spe2index[name] = index
            index += 1
    model = DeepurifyModel(
        max_model_len=model_config["max_model_len"],
        in_channels=model_config["inChannel"],
        taxo_dict_size=len(taxo_vocabulary),
        vocab_3Mer_size=len(mer3_vocabulary),
        vocab_4Mer_size=len(mer4_vocabulary),
        phylum_num=getNumberOfPhylum(taxo_tree),
        species_num=len(spe2index),
        head_num=model_config["head_num"],
        d_model=model_config["d_model"],
        num_GeqEncoder=model_config["num_GeqEncoder"],
        num_lstm_layer=model_config["num_lstm_layers"],
        IRB_layers=model_config["IRB_num"],
        expand=model_config["expand"],
        feature_dim=model_config["feature_dim"],
        drop_connect_ratio=0.0,
        dropout=0.0,
    )
    print("Warning, DO NOT FIND taxoName2RepNormVecPath FILE. Start to build taxoName2RepNormVecPath file.")
    state = torch.load(modelWeightPath, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        buildTextsRepNormVector(taxo_tree, model, taxo_vocabulary, "cpu", taxoName2RepNormVecOutPath)
        


def collect_all_deconta_results(
    deconta_tmp, 
    output_folder,
    bin_suffix = "fasta"
):
    """_summary_

    Args:
        tempFileOutFolder (_type_): _description_
        bin_suffix (str, optional): _description_. Defaults to "fasta".
    """
    i = 0
    if os.path.exists(output_folder) is False:
        os.mkdir(output_folder)
    wh = open(os.path.join(output_folder, "MetaInfo.tsv"), "w")

    for de_temps_all in os.listdir(deconta_tmp):
        if "de_out_bins_" in de_temps_all:
            cur_bin_out_folder = os.path.join(deconta_tmp, de_temps_all)
            cur_meta_info = readMetaInfo(os.path.join(cur_bin_out_folder, "MetaInfo.tsv"))[0]
            for bin_file_name in os.listdir(cur_bin_out_folder):
                _, suffix = os.path.splitext(bin_file_name)
                if suffix[1:] == bin_suffix:
                    qualityValues = cur_meta_info[bin_file_name]
                    outName = f"Deepurify_Bin_{i}.fasta"
                    wh.write(outName
                            + "\t"
                            + str(qualityValues[0])
                            + "\t"
                            + str(qualityValues[1])
                            + "\t"
                            + str(qualityValues[2])
                            + "\n")
                    copy(os.path.join(cur_bin_out_folder, bin_file_name), 
                        os.path.join(output_folder, outName))
                    i += 1
    wh.close()


def process_drep_result(
    drep_genomes_folder: str,
    drep_Cdb_csv_path: str,
    output_folder: str
):
    collect = {}
    meta_info = readMetaInfo(os.path.join(drep_genomes_folder, "MetaInfo.tsv"))[0]
    csv_info = readCSV(drep_Cdb_csv_path)[1:]
    wh = open(os.path.join(output_folder, "MetaInfo.tsv"), "w")
    for info in csv_info:
        c = info[1]
        n = info[0]
        q = meta_info[n]
        if c not in collect:
            collect[c] = [(n, q, getScore(q))]
        else:
            collect[c].append((n, q, getScore(q)))
    res = []
    for c, q_l in collect.items():
        res.append(list(sorted(q_l, key=lambda x: x[-1], reverse=True))[0])
    for i, r in enumerate(res):
        outName = f"Deepurify_Bin_{i}.fasta"
        wh.write(outName
                + "\t"
                + str(r[1][0])
                + "\t"
                + str(r[1][1])
                + "\t"
                + str(r[1][2])
                + "\n")
        copy(os.path.join(drep_genomes_folder, r[0]), 
            os.path.join(output_folder, outName))
    wh.close()


def readGalahClusterTSV(tsv_path: str):
    res = {}
    with open(tsv_path, "r", encoding="utf-8") as rh:
        for line in rh:
            info = line.strip("\n").split("\t")
            if info[0] in res:
                res[info[0]].append(info[1])
            else:
                res[info[0]] = [info[1]]
    return res


def process_galah_result(
    drep_genomes_folder,
    galah_tsv_path: str,
    output_folder: str,
):
    collect = {}
    checkm2_meta_info = readMetaInfo(os.path.join(drep_genomes_folder, "MetaInfo.tsv"))[0]
    clu_res_info = readGalahClusterTSV(galah_tsv_path)
    wh = open(os.path.join(output_folder, "MetaInfo.tsv"), "w")
    # n: name, q: quality, v: path of file
    for c, vals in clu_res_info.items():
        for v in vals:
            n = os.path.split(v)[-1]
            q = checkm2_meta_info[n]
            if c not in collect:
                    collect[c] = [(n, q, v, getScore(q))]
            else:
                collect[c].append((n, q, v, getScore(q)))
    res = []
    for c, q_l in collect.items():
        res.append(list(sorted(q_l, key=lambda x: x[-1], reverse=True))[0])
    for i, r in enumerate(res):
        outName = f"Deepurify_Bin_{i}.fasta"
        wh.write(outName
                + "\t"
                + str(r[1][0])
                + "\t"
                + str(r[1][1])
                + "\t"
                + str(r[1][2])
                + "\n")
        copy(os.path.join(drep_genomes_folder, r[0]), 
            os.path.join(output_folder, outName))
    wh.close()
