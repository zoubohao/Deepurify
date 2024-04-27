import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from Deepurify.Utils.DataUtils import readVocabulary
from Deepurify.Model.EncoderModels import DeepurifyModel
from Deepurify.Dataset.SequenceDataset import SequenceSampledTestDataset, SequenceTrainingDataset
from Deepurify.Train.TrainUtils import train

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(local_rank, world_size):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    fileConfig = {
        "taxoVocabularyPath": "./TaxonomyInfo/GTDB/vocabulary.txt",
        "vocab3MerPath": "./TaxonomyInfo/3Mer_vocabulary.txt",
        "vocab4MerPath": "./TaxonomyInfo/4Mer_vocabulary.txt",
        "trainPath": "/home/datasets/ZOUbohao/Deepurify_data/GTDB_clu_rep_split/",
        "testPath": "/home/datasets/ZOUbohao/Deepurify_data/GTDB_testing_data/",
        "taxonomyTreePklPath": "./TaxonomyInfo/GTDB/gtdb_taxonomy_tree.pkl",
        "sampleWeightPath": "./TaxonomyInfo/GTDB/samples_weights.txt",
        "gmmModel": "./PyObjs/contig_GMM.pkl",
    }

    modelConfig = {
        "min_model_len": 1000,
        "max_model_len": 8192,
        # 12 for ori + rev, 16 * 2 for 3mer ori + rev, 32 * 2 for 4mer ori + rev. 12 + 16 * 2 + 32 * 2 = 108
        "inChannel": 108,
        "expand": 1.2,
        "IRB_num": 2,
        "head_num": 6,
        "d_model": 864,
        "num_GeqEncoder": 6,
        "num_lstm_layers": 5,
        "feature_dim": 1024,
        "dropout": 0.08,
        "drop_connect_ratio": 0.01,
    }

    trainingConfig = {
        "misMatchNum": 199,
        "if_weight": False,
        "epoch": 96,
        "batchSize": 10,
        "weightSavePath": "./CheckPoint/",
        "loadWeightPath": "./CheckPoint/Epoch_52_None",
        "reguLambda": 1e-5,
        "learningRate": 1e-6,
        "multiplier": 50.0,
        "warmEpoch": 4,
        "focal_gamma": 0.5,
        "modelName": "Deepurify",
        "loss_state": "mean",
        "finetune": False,
        "finetune_absThre": 0.05
    }

    print("Config Done.")
    taxo_vocabulary = readVocabulary(fileConfig["taxoVocabularyPath"])
    mer3_vocabulary = readVocabulary(fileConfig["vocab3MerPath"])
    mer4_vocabulary = readVocabulary(fileConfig["vocab4MerPath"])
    # Data buildd
    # Traing
    # Traing
    trainDataset = SequenceTrainingDataset(
        fileConfig["trainPath"],
        min_model_len=modelConfig["min_model_len"],
        max_model_len=modelConfig["max_model_len"],
        taxo_vocabulary=taxo_vocabulary,
        taxomonyTreePath=fileConfig["taxonomyTreePklPath"],
        vocab_3Mer=mer3_vocabulary,
        vocab_4Mer=mer4_vocabulary,
        sampleName2weightPath=fileConfig["sampleWeightPath"],
        misMatchNum=trainingConfig["misMatchNum"],
        gmmModelPath=fileConfig["gmmModel"],
        finetune=trainingConfig["finetune"],
    )
    # Test
    testDataset = SequenceSampledTestDataset(
        fileConfig["testPath"],
        min_model_len=modelConfig["min_model_len"],
        max_model_len=modelConfig["max_model_len"],
        taxo_vocabulary=taxo_vocabulary,
        taxomonyTreePath=fileConfig["taxonomyTreePklPath"],
        vocab_3Mer=mer3_vocabulary,
        vocab_4Mer=mer4_vocabulary,
        sampleName2weightPath=fileConfig["sampleWeightPath"],
        misMatchNum=trainingConfig["misMatchNum"],
        gmmModelPath=fileConfig["gmmModel"],
    )
    # Build Model
    # setup_seed(2048)
    approxNum = 3
    addNum = 0.05
    angleAroundAnchor = math.pi / approxNum
    gapAngle = angleAroundAnchor - math.pi / (approxNum + addNum)
    innerThre = math.cos(angleAroundAnchor - gapAngle)
    outerThre = math.cos(angleAroundAnchor + gapAngle)

    # those code must run first.
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    device = torch.device("cuda", local_rank)

    model = DeepurifyModel(
        max_model_len=modelConfig["max_model_len"],
        in_channels=modelConfig["inChannel"],
        taxo_dict_size=len(taxo_vocabulary),
        vocab_3Mer_size=len(mer3_vocabulary),
        vocab_4Mer_size=len(mer4_vocabulary),
        phylum_num=trainDataset.num_phylum,
        species_num=len(trainDataset.spe2index),
        head_num=modelConfig["head_num"],
        d_model=modelConfig["d_model"],
        num_GeqEncoder=modelConfig["num_GeqEncoder"],
        num_lstm_layer=modelConfig["num_lstm_layers"],
        IRB_layers=modelConfig["IRB_num"],
        expand=modelConfig["expand"],
        feature_dim=modelConfig["feature_dim"],
        drop_connect_ratio=modelConfig["drop_connect_ratio"],
        dropout=modelConfig["dropout"]
    )

    model = model.to(device)

    model = DistributedDataParallel(model, find_unused_parameters=True).to(device)

    print("Model Build Done, Start to Train.")
    train(trainingConfig,
          model,
          trainDataset,
          testDataset,
          innerThre,
          outerThre,
          device,
          local_rank)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    world_size = 8
    mp.set_start_method("spawn")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
