import os

import numpy as np
import torch


def minMaxNorm(array):
    return (array - min(array)) / (max(array) - min(array))


def modelSoup(dataFolder, outputPath):
    modelCkptsPaths = os.listdir(dataFolder)
    samPairs = []
    newDict = {}
    for path in modelCkptsPaths:
        info = path.split("_")
        try:
            samPair = float(info[1][0:5])
        except:
            samPair = float(info[1][0:4])
        print(path)
        samPairs.append(samPair)
    samPairs = np.array(samPairs)
    samPairs = minMaxNorm(samPairs)
    finalWeight = samPairs
    finalWeight = finalWeight / sum(finalWeight)
    print(finalWeight, sum(finalWeight))
    for i, ckptPath in enumerate(modelCkptsPaths):
        print(i, finalWeight[i])
        ckpt = torch.load(os.path.join(dataFolder, ckptPath), "cpu")["model"]
        for key, val in ckpt.items():
            if key not in newDict:
                newDict[key] = val * finalWeight[i]
            else:
                newVal = newDict[key] + val * finalWeight[i]
                newDict[key] = newVal
    torch.save(newDict, outputPath)


def modelWeightAdjusment(vocab2index_old: dict, taxoTree_old: dict, vocab2index_new: dict, taxoTree_new: dict, modelCkptPath_old: str):
    state_dict = torch.load(modelCkptPath_old)
    new_dict = {}
    textEmbTensor = None
    textEmbTensor_new = None
    phyFcWeightTensor = None
    phyFcWeightTensor_new = None
    phyFcBiasTensor = None
    phyFcBiasTensor_new = None
    for key, val in state_dict.items():
        if key == "textEncoder.embedding.weight":
            textEmbTensor = val
        elif key == "phy_fc.weight":
            phyFcWeightTensor = val
        elif key == "phy_fc.bias":
            phyFcBiasTensor = val
        else:
            new_dict[key] = val
    phy2FcIndex_old = {}
    phy2FcIndex_new = {}
    for i, child in enumerate(taxoTree_old["Children"]):
        phy2FcIndex_old[child["Name"]] = i
    for i, child in enumerate(taxoTree_new["Children"]):
        phy2FcIndex_new[child["Name"]] = i

    def foo(key2index_old, key2index_new, tensor):
        new_tensor = [torch.zeros_like(tensor[0], dtype=torch.float32) for _ in range(len(key2index_new))]
        for key_o, index_o in key2index_old.items():
            if key_o in key2index_new:
                new_tensor[key2index_new[key_o]] = tensor[index_o]
        return torch.stack(new_tensor, dim=0)

    phyFcWeightTensor_new = foo(phy2FcIndex_old, phy2FcIndex_new, phyFcWeightTensor)
    phyFcBiasTensor_new = foo(phy2FcIndex_old, phy2FcIndex_new, phyFcBiasTensor)
    textEmbTensor_new = foo(vocab2index_old, vocab2index_new, textEmbTensor)
    # print(phyFcWeightTensor_new.shape)
    # print(phyFcBiasTensor_new.shape)
    # print(textEmbTensor_new.shape)
    new_dict["textEncoder.embedding.weight"] = textEmbTensor_new
    new_dict["phy_fc.weight"] = phyFcWeightTensor_new
    new_dict["phy_fc.bias"] = phyFcBiasTensor_new
    return new_dict
