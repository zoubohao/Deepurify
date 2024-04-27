import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCEwithLogitLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, outputs, targets, alphas=None):
        """
        outputs: [B], [B,1]
        targets: [B], [B,1]
        alphas: [B],, [B,1] weight for each samples
        """
        if len(outputs.shape) >= 2:
            outputs = outputs.squeeze(-1)
        if len(targets.shape) >= 2:
            targets = targets.squeeze(-1)
        if alphas is None:
            alphas = torch.ones(outputs.shape, device=outputs.device)
        else:
            if len(alphas.shape) >= 2:
                alphas = alphas.squeeze(-1)
        postiveProb = torch.sigmoid(outputs)
        postiveProb = torch.clip(postiveProb, min=1e-5, max=1.0 - 1e-5)
        negtiveProb = 1.0 - postiveProb
        outputs = torch.stack([negtiveProb, postiveProb], dim=-1)
        targets = targets.unsqueeze(dim=-1)
        probTruth = torch.clamp(outputs.gather(dim=-1, index=targets).flatten(0), max=-0.0)
        loss = -alphas * (1 - probTruth) ** self.gamma * torch.log(probTruth)
        assert torch.isnan(loss).sum() == 0
        return loss.sum()


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gamma=2.0):
        """
        It always return "sum" of each loss
        """
        super().__init__()
        self.gamma = gamma

    def forward(self, netOutput, labels, alphas=None, sum_mean="sum"):
        """
        :param netOutput: [B, Classes]
        :param labels: [B, 1]
        :param alphas: [B], the weight of each sample.
        """
        assert netOutput.sum().isnan().float() == 0, "Net Outputs have NaN value."
        b = labels.shape[0]
        logprobs = F.log_softmax(netOutput, dim=-1)
        if alphas is None:
            alphas = torch.ones(size=[b], device=netOutput.device)
        alphas = alphas[:, None]
        if len(labels.shape) != 2:
            labels = labels[:, None]
        logprobsTruth = torch.clamp(torch.gather(logprobs, dim=-1, index=labels), min=-100000.0, max=-1e-8)
        moderate = (1.0 - torch.exp(logprobsTruth)) ** self.gamma
        if moderate.sum().isnan().float() != 0:
            moderate = torch.ones_like(moderate).float().to(logprobsTruth.device)
        loss = -alphas * moderate * logprobsTruth
        if sum_mean == "sum":
            return loss.sum()
        return loss.sum() / alphas.sum()


def varianceLoss(z, val=1.0):
    """
    z: [B, D]
    """
    std_z = torch.sqrt(z.var(dim=0, keepdim=True) + 1e-4)
    std_loss = torch.mean(F.relu(val - std_z))
    return std_loss


def cosineLoss(oriPhyNormTensor, matchTextNormTensor, innerThre, outerThre, innerORouter="inner"):
    """
    oriPhyNormTensor: [B, C]
    matchTextNormTensor: [B, C]
    """
    cosSim = torch.sum(oriPhyNormTensor * matchTextNormTensor, dim=-1)  # [B]
    if innerORouter == "inner":
        return torch.sum(F.relu(innerThre - cosSim))
    else:
        return torch.sum(F.relu(cosSim - outerThre))


def covarianceLoss(z):
    """
    z: [B, feature_dim]
    """
    n = z.size(0)
    feature_dim = z.size(1)
    assert n >= 2
    z = z - z.mean(dim=0, keepdim=True)
    cov_z = (z.T @ z) / (n - 1)
    cov_mask = 1.0 - torch.eye(feature_dim)
    cov_mask = cov_mask.to(z.device)
    cov_loss = (cov_z * cov_mask).pow_(2).mean()
    return cov_loss


if __name__ == "__main__":
    # Test if the focal loss is correct by comparing with ce loss.
    testAlpha = torch.tensor([0.75, 1.0, 1.2, 1.5, 3.0], dtype=torch.float32)
    ceLoss = nn.CrossEntropyLoss(weight=testAlpha, reduction="mean")
    testInput = torch.randn(size=[5, 5])
    testLabel = torch.tensor([0, 1, 2, 3, 0])
    flLoss = FocalCrossEntropyLoss(gamma=0)
    print(ceLoss(testInput, testLabel))
    print(flLoss(testInput, testLabel.unsqueeze(-1), torch.tensor([0.75, 1.0, 1.2, 1.5, 0.75], dtype=torch.float32)))

    ###
    bceLoss = nn.BCEWithLogitsLoss(weight=torch.tensor([0.1, 0.2, 0.3]), reduction="sum")
    testInput = torch.randn(size=[3])
    testTarget = torch.tensor([0, 0, 1])
    bceFLloss = FocalBCEwithLogitLoss(gamma=0)
    print(bceLoss(testInput, testTarget.float()))
    print(bceFLloss(testInput, testTarget, torch.tensor([0.1, 0.2, 0.3])))

    z = torch.randn(size=[5, 1024])
    print(varianceLoss(z))
    # print(covarianceLoss(z, 1024))
