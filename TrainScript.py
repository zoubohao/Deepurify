import math

import torch
import torch.nn as nn

from Deepurify.DataTools.DataUtils import readVocabulary
from Deepurify.Model.EncoderModels import SequenceCLIP
from Deepurify.Model.Loss import FocalCrossEntropyLoss
from Deepurify.SeqProcessTools.SequenceDataset import SequenceSampledTestDataset, SequenceTrainingDataset
from Deepurify.TrainTools.TrainUtils import test, train


def config_train(fileConfig=None, modelConfig=None, trainingConfig=None):
    if fileConfig is None:
        fileConfig = {
            "taxoVocabularyPath": "./DeepurifyInfoFiles/ProGenomes/ProGenomesVocabulary.txt",
            "vocab3MerPath": "./3Mer_vocabulary.txt",
            "vocab4MerPath": "./4Mer_vocabulary.txt",
            "trainPath": "",
            "testPath": "",
            "taxonomyTreePklPath": "./DeepurifyInfoFiles/ProGenomes/ProGenomesTaxonomyTree.pkl",
            "sampleWeightPath": "./DeepurifyInfoFiles/ProGenomes/ProGenomesSamples2Weight.txt",
            "gmmModel": "./DeepurifyInfoFiles/PyObjs/contig_GMM.pkl",
        }
    if modelConfig is None:
        modelConfig = {
            "min_model_len": 1000,
            "max_model_len": 1024 * 8,
            # 12 for ori + rev, 16 * 2 for 3mer ori + rev, 32 * 2 for 4mer ori + rev. 12 + 16 * 2 + 32 * 2 = 108
            "inChannel": 108,
            "expand": 1.5,
            "IRB_num": 3,
            "head_num": 6,
            "d_model": 738,
            "num_GeqEncoder": 7,
            "num_lstm_layers": 5,
            "feature_dim": 1024,
            "dropout": 0.0,
            "drop_connect_ratio": 0.0,
        }
    if trainingConfig is None:
        trainingConfig = {
            "misMatchNum": 99,
            "if_weight": False,
            "epoch": 15,
            "batchSize": 16,
            "weightSavePath": "./DeepurifyInfoFiles/CheckPoint/",
            "loadWeightPath": None,
            "reguLambda": 5e-5,
            "learningRate": 1e-4,
            "multiplier": 1.5,
            "warmEpoch": 2,
            "focal_gamma": 1.0,
            "maskedTokens": 40,
            "modelName": "SequenceCLIP",
            "device": "cuda:0",
            "loss_state": "sum",
            "finetune": True,
            "finetune_absThre": 0.05,
            "eval": False,
        }
    taxo_vocabulary = readVocabulary(fileConfig["taxoVocabularyPath"])
    mer3_vocabulary = readVocabulary(fileConfig["vocab3MerPath"])
    mer4_vocabulary = readVocabulary(fileConfig["vocab4MerPath"])
    # Data buildd
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
        maskedLength=trainingConfig["maskedTokens"],
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
    print("Number of Phylums: ", trainDataset.num_phylum)
    approxNum = 3
    addNum = 0.05
    angleAroundAnchor = math.pi / approxNum
    gapAngle = angleAroundAnchor - math.pi / (approxNum + addNum)
    innerThre = math.cos(angleAroundAnchor - gapAngle)
    outerThre = math.cos(angleAroundAnchor + gapAngle)

    model = SequenceCLIP(
        max_model_len=modelConfig["max_model_len"],
        in_channels=modelConfig["inChannel"],
        taxo_dict_size=len(taxo_vocabulary),
        vocab_3Mer_size=len(mer3_vocabulary),
        vocab_4Mer_size=len(mer4_vocabulary),
        phylum_num=trainDataset.num_phylum,
        head_num=modelConfig["head_num"],
        d_model=modelConfig["d_model"],
        num_GeqEncoder=modelConfig["num_GeqEncoder"],
        num_lstm_layer=modelConfig["num_lstm_layers"],
        IRB_layers=modelConfig["IRB_num"],
        expand=modelConfig["expand"],
        feature_dim=modelConfig["feature_dim"],
        drop_connect_ratio=modelConfig["drop_connect_ratio"],
        dropout=modelConfig["dropout"],
        maskedLength=trainingConfig["maskedTokens"],
    )
    print("Model Build Done.")
    if trainingConfig["eval"]:
        if trainingConfig["loadWeightPath"] is not None:
            state = torch.load(trainingConfig["loadWeightPath"], map_location=trainingConfig["device"])
            model.load_state_dict(state, strict=True)
            print("Weight has been loaded !")
        model.eval()
        test(trainingConfig, modelConfig, model, None, trainingConfig["if_weight"], trainDataset, testDataset, innerThre, outerThre)
    else:
        if trainingConfig["finetune"] is False:
            print("Focal Loss Build Done.")
            loss_func = FocalCrossEntropyLoss(trainingConfig["focal_gamma"])
        else:
            loss_func = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.001)
        print("Start to Train.")
        train(trainingConfig, modelConfig, model, loss_func, trainingConfig["if_weight"], trainDataset, testDataset, innerThre, outerThre)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    config_train()
