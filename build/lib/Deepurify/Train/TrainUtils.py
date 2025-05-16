import time
from typing import Dict, TypeVar

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from Model.Loss import FocalCrossEntropyLoss, cosineLoss
from torch.amp import autocast_mode, grad_scaler
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from Train.Scheduler import GradualWarmupScheduler

Tensor = TypeVar("Tensor")


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k / batch_size)
    return res


def train(
    trainingConfig: Dict,
    model: nn.Module,
    trainDataSet: Dataset,
    testDataset,
    innerThre: float,
    outerThre: float,
    device,
    local_rank
):

    # Training config
    epoch = trainingConfig["epoch"]
    batchSize = trainingConfig["batchSize"]
    weightSavePath = trainingConfig["weightSavePath"]
    loadWeightPath = trainingConfig["loadWeightPath"]
    reguLambda = trainingConfig["reguLambda"]
    learningRate = trainingConfig["learningRate"]
    multiplier = trainingConfig["multiplier"]
    warmEpoch = trainingConfig["warmEpoch"]
    loss_state = trainingConfig["loss_state"]
    finetune = trainingConfig["finetune"]
    finetune_absThre = trainingConfig["finetune_absThre"]
    if_weight = trainingConfig["if_weight"]

    train_sampler = DistributedSampler(trainDataSet, shuffle=True)
    trainLoader = DataLoader(trainDataSet, batch_size=batchSize, num_workers=8, pin_memory=True, sampler=train_sampler)

    if finetune:
        optimizer = ZeroRedundancyOptimizer(
            list(model.module.visionEncoder.parameters()),
            optimizer_class=optim.AdamW,
            lr=learningRate,
            weight_decay=reguLambda,
            betas=(0.9, 0.99),
            eps=1e-5)
    else:
        optimizer = ZeroRedundancyOptimizer(
            list(model.module.parameters()),
            optimizer_class=optim.AdamW,
            lr=learningRate,
            weight_decay=reguLambda,
            betas=(0.9, 0.99),
            eps=1e-5)

    warmUpScheduler = GradualWarmupScheduler(
        optimizer, multiplier, warmEpoch, epoch - warmEpoch + 1)
    # checkpoint load
    startEpoch = 1
    if loadWeightPath is not None and loadWeightPath != "":
        state = torch.load(loadWeightPath, map_location=device)
        model.module.load_state_dict(state, strict=False)
        print("Weight has been loaded !")

    trainingStep = 0
    scaler = grad_scaler.GradScaler()
    loss_ce_func = nn.CrossEntropyLoss(reduction=loss_state, label_smoothing=0.01)
    nosiyLossFunc = nn.BCEWithLogitsLoss()
    loss_focal_func = FocalCrossEntropyLoss(trainingConfig["focal_gamma"])

    for e in range(startEpoch, epoch + 1):
        torch.cuda.empty_cache()
        model.train()
        epochLossRecord = []
        for r in range(1):
            with tqdm(trainLoader) as tqdmDataLoader:
                trainLoader.sampler.set_epoch(e + r)
                model.train()
                # We usually set the last element of this dataTuple as the labels.
                for (
                    _,
                    (b_ori_rev_tensor,
                    b_feature_3Mer,
                    b_feature_3Mer_rev_com,
                    b_feature_4Mer,
                    b_feature_4Mer_rev_com,
                    batch_texts,
                    batch_pairLabels,
                    batch_weights,
                    b_oriPhyTensor,
                    b_matchTextTensor,
                    b_outerMisMatchTextTensor,
                    b_nosiy_label,
                    b_phylabel,
                    b_spelabel),
                ) in enumerate(tqdmDataLoader):
                    # break
                    tqdmDataLoader.set_description("Epoch {}".format(e))
                    # load data to gpu
                    b_ori_rev_tensor = b_ori_rev_tensor.to(device)
                    b_feature_3Mer = b_feature_3Mer.to(device)
                    b_feature_3Mer_rev_com = b_feature_3Mer_rev_com.to(device)
                    b_feature_4Mer = b_feature_4Mer.to(device)
                    b_feature_4Mer_rev_com = b_feature_4Mer_rev_com.to(device)
                    batch_texts = batch_texts.to(device)
                    batch_weights = batch_weights.to(device)
                    batch_pairLabels = batch_pairLabels.to(device)
                    b_oriPhyTensor = b_oriPhyTensor.to(device)
                    b_matchTextTensor = b_matchTextTensor.to(device)
                    b_outerMisMatchTextTensor = b_outerMisMatchTextTensor.to(device)
                    b_nosiy_label = b_nosiy_label.to(device)
                    b_phylabel = b_phylabel.to(device)
                    b_spelabel = b_spelabel.to(device)
                    # optimizer
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_mode.autocast():
                        (batch_score,
                        batch_oriPhyNorm,
                        batch_matchNorm,
                        batchOuterMisMatchTextTensorNorm,
                        _, _,
                        batch_ifNoisyPred,
                        batch_phyPred,
                        batch_spePred) = model(
                            b_ori_rev_tensor,
                            b_feature_3Mer,
                            b_feature_3Mer_rev_com,
                            b_feature_4Mer,
                            b_feature_4Mer_rev_com,
                            batch_texts,
                            b_oriPhyTensor,
                            b_matchTextTensor,
                            b_outerMisMatchTextTensor)
                        # pair similarity loss, focal loss
                        with torch.no_grad():
                            pariAcc1 = accuracy(batch_score, batch_pairLabels)[0]
                            phyAcc1 = accuracy(batch_phyPred, b_phylabel)[0]
                            speAcc1 = accuracy(batch_spePred, b_spelabel)[0]
                        if finetune:
                            lossSimPair = loss_ce_func(batch_score, batch_pairLabels)
                            with torch.no_grad():
                                probs = torch.softmax(batch_score, dim=-1)
                                values, _ = torch.topk(probs, 2, dim=-1)
                                simLossMask = (torch.abs(values[:, 0] - values[:, 1]) > finetune_absThre).float()
                            lossSim = torch.sum(lossSimPair * simLossMask)
                        else:
                            if if_weight:
                                lossSim = loss_focal_func(batch_score, batch_pairLabels, batch_weights, loss_state)
                                lossPhy = loss_focal_func(batch_phyPred, b_phylabel, batch_weights, loss_state)
                            else:
                                lossSim = loss_focal_func(batch_score, batch_pairLabels, None, loss_state)
                                lossPhy = loss_focal_func(batch_phyPred, b_phylabel, None, loss_state)
                        lossSpe = loss_ce_func(batch_spePred, b_spelabel)
                        # cosine loss for constraining the texts label
                        cosLoss = cosineLoss(batch_oriPhyNorm, batch_matchNorm, innerThre, outerThre, "inner") + \
                            cosineLoss(batch_oriPhyNorm, batchOuterMisMatchTextTensorNorm, innerThre, outerThre, "outer")
                        if not finetune:
                            noisyLoss = nosiyLossFunc(batch_ifNoisyPred.squeeze(-1), b_nosiy_label)
                        else:
                            noisyLoss = 0.
                        # print(b_spelabel)
                        loss = lossSim * 2.0 + lossPhy + cosLoss + noisyLoss + lossSpe
                    scaler.scale(loss).backward()
                    for para in model.module.textEncoder.parameters():
                        if para.requires_grad is True and para.grad is not None:
                            para.grad.mul_(0.8)
                    for para in model.module.parameters():
                        if para.requires_grad is True and para.grad is not None and para.grad.isnan().float().sum() != 0:
                            para.grad = torch.zeros_like(para.grad).float().to(device)
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                    # Training Infor
                    if isinstance(noisyLoss, float):
                        nl = 0.
                    else:
                        nl = noisyLoss.item()
                    tqdmDataLoader.set_postfix(
                        ordered_dict={
                            "total_loss": loss.item(),
                            "pair_loss": lossSim.item(),
                            "pair_acc": pariAcc1.item(),
                            "phy_loss": lossPhy.item(),
                            "phy_acc": phyAcc1.item(),
                            "spe_loss": lossSpe.item(),
                            "spe_acc": speAcc1.item(),
                            "text_cos_loss": cosLoss.item(),
                            "noisy_loss": nl,
                            "LR": optimizer.param_groups[0]["lr"],
                        }
                    )
                    epochLossRecord.append(loss.item())
                    trainingStep += 1

        # validate the model
        dist.barrier()
        if e % 5 == 0 and local_rank in [0, -1]:
            model.eval()
            res = random_sampled_test(model.module, testDataset, device)
            save_model(e, weightSavePath, model, res)
        else:
            save_model(e, weightSavePath, model, None)
        dist.barrier()
        model.train()
        warmUpScheduler.step()


def random_sampled_test(model,
                        test_dataSet,
                        device):
    test_data_loader = DataLoader(test_dataSet, batch_size=8, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True)
    # validation loop
    total_num = 0
    pairCorrect = 0
    with torch.no_grad():
        for (
            idx,
            (
                b_ori_rev_tensor,
                b_feature_3Mer,
                b_feature_3Mer_rev_com,
                b_feature_4Mer,
                b_feature_4Mer_rev_com,
                batch_texts,
                batch_pairLabels,
            ),
        ) in enumerate(test_data_loader):
            b = batch_texts.shape[0]
            b_ori_rev_tensor = b_ori_rev_tensor.to(device)
            b_feature_3Mer = b_feature_3Mer.to(device)
            b_feature_3Mer_rev_com = b_feature_3Mer_rev_com.to(device)
            b_feature_4Mer = b_feature_4Mer.to(device)
            b_feature_4Mer_rev_com = b_feature_4Mer_rev_com.to(device)
            batch_score = model(
                b_ori_rev_tensor, b_feature_3Mer, b_feature_3Mer_rev_com, b_feature_4Mer, b_feature_4Mer_rev_com, batch_texts.to(
                    device)
            )
            batch_pairLabels = batch_pairLabels.to(device)
            pariAcc1 = accuracy(batch_score, batch_pairLabels)[0].item()
            print("idx: ", idx)
            print("Pair Acc1: ", pariAcc1)
            pairCorrect += pariAcc1 * b
            total_num += b
            print("Total Num: ", total_num)
    res = pairCorrect / total_num + 0.0
    print("Total Pair Acc1: ", res)
    return res


def save_model(e, save_dir, model, res):
    filename = save_dir + "Epoch_" + str(e) + "_" + f"{str(res)[:8]}"
    torch.save(model.module.state_dict(), filename)
