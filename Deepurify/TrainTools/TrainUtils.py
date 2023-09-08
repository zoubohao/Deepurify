import time
from typing import Dict, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tb
from Deepurify.Model.Loss import cosineLoss
from Deepurify.TrainTools.Scheduler import GradualWarmupScheduler
from torch.cuda.amp import autocast_mode, grad_scaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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


def SampledTestValid(model, test_data_loader, device=torch.device("cuda:0")):
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
    print("Total Pair Acc1: ", pairCorrect / total_num + 0.0)
    return pairCorrect / total_num + 0.0


def valid_epoch(model, sampled_test_data_loader, device=torch.device("cuda:0")):
    return SampledTestValid(model, sampled_test_data_loader, device)


def save_model(save_dir, model, samPair, optimizer):
    filename = f"{save_dir}samPair_{str(samPair)[:8]}.pth"
    torch.save(model.module.state_dict(), filename)


def train(
    trainingConfig: Dict,
    modelConfig: Dict,
    model: nn.Module,
    loss_func: nn.Module,
    if_weight: bool,
    trainDataSet: Dataset,
    samTestDataSet: Dataset,
    innerThre: float,
    outerThre: float,
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
    modelName = trainingConfig["modelName"]
    device = trainingConfig["device"]
    loss_state = trainingConfig["loss_state"]
    finetune = trainingConfig["finetune"]
    finetune_absThre = trainingConfig["finetune_absThre"]

    # Record current model's config and current training config.
    timeNow = str(time.asctime()).replace(" ", "_").replace(";", "").replace(":", "_")
    writer = tb.writer.SummaryWriter("./ModelLog/" + modelName + "/" + timeNow + "/")
    for key, val in trainingConfig.items():
        writer.add_text(key, str(val))
    for key, val in modelConfig.items():
        writer.add_text(key, str(val))

    trainLoader = DataLoader(trainDataSet, batch_size=batchSize, shuffle=True,
                             num_workers=10, pin_memory=True, drop_last=True)
    samTestLoader = DataLoader(samTestDataSet, batch_size=8, shuffle=True,
                               num_workers=4, pin_memory=True, drop_last=True)

    device = torch.device(device)
    model.to(device=device)
    model = nn.DataParallel(module=model, output_device=device)
    if finetune:
        optimizer = optim.AdamW(params=list(model.module.visionEncoder.parameters()),
                                lr=learningRate, weight_decay=reguLambda)
    else:
        optimizer = optim.AdamW(params=list(model.module.parameters()), lr=learningRate, weight_decay=reguLambda)
    warmUpScheduler = GradualWarmupScheduler(optimizer, multiplier, warmEpoch, epoch - warmEpoch + 1)
    # checkpoint load
    startEpoch = 1
    if loadWeightPath is not None or loadWeightPath != "":
        state = torch.load(loadWeightPath, map_location=device)
        model.module.load_state_dict(state, strict=False)
        print("Weight has been loaded !")

    bestsamPair = 0
    trainingStep = 0
    nosiyLossFunc = nn.BCEWithLogitsLoss()
    scaler = grad_scaler.GradScaler()
    loss_phy_func = nn.CrossEntropyLoss(reduction=loss_state, label_smoothing=0.005)
    loss_tokenProb_func = nn.CrossEntropyLoss(ignore_index=0)

    for e in range(startEpoch, epoch + 1):
        torch.cuda.empty_cache()
        model.train()
        epochLossRecord = []
        with tqdm(trainLoader) as tqdmDataLoader:
            # We usually set the last element of this dataTuple as the labels.
            for (
                i,
                (
                    b_ori_rev_tensor,
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
                    b_if_noisy,
                    b_phylabel,
                    b_generateLabel,
                    b_selectMask,
                ),
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
                b_if_noisyLabel = b_if_noisy.to(device)
                b_outerMisMatchTextTensor = b_outerMisMatchTextTensor.to(device)
                b_phylabel = b_phylabel.to(device)
                b_selectMask = b_selectMask.to(device)
                b_generateLabel = b_generateLabel.to(device)  # b, l
                # optimizer
                optimizer.zero_grad(set_to_none=True)
                with autocast_mode.autocast():
                    (
                        batch_score,
                        batch_oriPhyNorm,
                        batch_matchNorm,
                        batchOuterMisMatchTextTensorNorm,
                        _,
                        _,
                        batch_ifNoisyPred,
                        batch_phyPred,
                        batch_tokenProb,
                    ) = model(
                        b_ori_rev_tensor,
                        b_feature_3Mer,
                        b_feature_3Mer_rev_com,
                        b_feature_4Mer,
                        b_feature_4Mer_rev_com,
                        batch_texts,
                        b_oriPhyTensor,
                        b_matchTextTensor,
                        b_outerMisMatchTextTensor,
                        b_selectMask,
                    )
                    # pair similarity loss, focal loss
                    with torch.no_grad():
                        pariAcc1 = accuracy(batch_score, batch_pairLabels)[0]
                        phyAcc1 = accuracy(batch_phyPred, b_phylabel)[0]
                    if finetune:
                        lossSimPair = loss_func(batch_score, batch_pairLabels)
                        with torch.no_grad():
                            probs = torch.softmax(batch_score, dim=-1)
                            values, _ = torch.topk(probs, 2, dim=-1)
                            simLossMask = (torch.abs(values[:, 0] - values[:, 1]) > finetune_absThre).float()
                        lossSim = torch.sum(lossSimPair * simLossMask)
                    else:
                        if if_weight:
                            lossSim = loss_func(batch_score, batch_pairLabels, batch_weights, loss_state) * 2.
                        else:
                            lossSim = loss_func(batch_score, batch_pairLabels, None, loss_state) * 2.
                    lossPhy = loss_phy_func(batch_phyPred, b_phylabel.view(-1))
                    # cosine loss for constraining the texts label
                    if finetune:
                        with torch.no_grad():
                            cosLoss = cosineLoss(batch_oriPhyNorm, batch_matchNorm, innerThre, outerThre, "inner") + cosineLoss(
                                batch_oriPhyNorm, batchOuterMisMatchTextTensorNorm, innerThre, outerThre, "outer"
                            )
                            # noisy loss
                            noisyLoss = nosiyLossFunc(batch_ifNoisyPred.squeeze(-1), b_if_noisyLabel)
                        loss = lossSim + lossPhy
                    else:
                        cosLoss = cosineLoss(batch_oriPhyNorm, batch_matchNorm, innerThre, outerThre, "inner") + cosineLoss(
                            batch_oriPhyNorm, batchOuterMisMatchTextTensorNorm, innerThre, outerThre, "outer"
                        )
                        # noisy loss
                        noisyLoss = nosiyLossFunc(batch_ifNoisyPred.squeeze(-1), b_if_noisyLabel)
                        loss = lossSim + lossPhy + cosLoss + noisyLoss
                    assert batch_tokenProb.shape[-2] == b_generateLabel.shape[-1], "The length is not same"
                    ys = b_generateLabel.contiguous().view(-1)
                    c = batch_tokenProb.size(-1)
                    batch_tokenProb = batch_tokenProb.view(-1, c)
                    lossTokens = loss_tokenProb_func(batch_tokenProb, ys)
                    loss += lossTokens
                scaler.scale(loss).backward()
                for para in model.module.parameters():
                    if para.requires_grad is True and para.grad.isnan().float().sum() != 0:
                        para.grad = torch.zeros_like(para.grad).float().to(device)
                scaler.unscale_(optimizer)
                if not finetune:
                    for para in model.module.textEncoder.parameters():
                        if para.requires_grad is True:
                            para.grad.mul_(0.8)
                scaler.step(optimizer)
                scaler.update()
                # Training Infor
                tqdmDataLoader.set_postfix(
                    ordered_dict={
                        "TotalL": loss.item(),
                        "simL": lossSim.item(),
                        "pairAcc": pariAcc1.item(),
                        "phyL": lossPhy.item(),
                        "phyAcc": phyAcc1.item(),
                        "textCosL": cosLoss.item(),
                        "NL": noisyLoss.item(),
                        "TokenL": lossTokens.item(),
                        "LR": optimizer.state_dict()["param_groups"][0]["lr"],
                    }
                )
                epochLossRecord.append(loss.item())
                writer.add_scalar("Loss with training steps", loss.item(), trainingStep)
                trainingStep += 1
            writer.add_scalar("Loss with Epoch", np.array(epochLossRecord).mean(), e)
        # validate the model
        if e % 2 == 0 or e >= 7 * epoch // 8:
            model.eval()
            samPair = valid_epoch(model.module, samTestLoader, device)
            writer.add_scalar("Testing sam Pair Acc1", samPair, e)
            print("TESTING RESULT," + modelName + " at Epoch " + str(e) + ", samPair Acc1: " + str(samPair))
            if samPair >= bestsamPair or bestsamPair >= 0.715:
                save_model(weightSavePath, model, samPair, optimizer)
                bestsamPair = samPair
        model.train()
        warmUpScheduler.step()
    writer.close()


def test(
    trainingConfig: Dict,
    modelConfig: Dict,
    model: nn.Module,
    loss_func: nn.Module,
    if_weight: bool,
    trainDataSet: Dataset,
    samTestDataSet: Dataset,
    innerThre: float,
    outerThre: float,
):
    device = trainingConfig["device"]
    device = torch.device(device)
    model.to(device=device)
    samTestLoader = DataLoader(samTestDataSet, batch_size=8, shuffle=True,
                               num_workers=4, pin_memory=True, drop_last=True)
    model.eval()
    samPair = valid_epoch(model, samTestLoader, device)
    print("TESTING RESULT: " + ", samPair Acc1: " + str(samPair))
