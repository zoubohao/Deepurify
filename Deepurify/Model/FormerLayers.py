import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Convolutions import Permute


class FeedForward(nn.Module):
    def __init__(self, d_model, drop_p=0.1):
        super(FeedForward, self).__init__()
        dff = d_model * 2
        self.ln = nn.LayerNorm(d_model, eps=1e-6)
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        x = self.ln(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class ConvFeedForward(nn.Module):
    def __init__(self, d_model: int, expand: float):
        super().__init__()
        in_cha = int(d_model * expand)
        self.ln = nn.LayerNorm(d_model, eps=1e-6)
        self.expension = nn.Sequential(
            Permute(),
            nn.Conv1d(
                d_model,
                in_cha,
                kernel_size=3,
                padding=1),
            nn.GELU())
        self.depthwise_conv = nn.Conv1d(
            in_cha, in_cha,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_cha,
            bias=False)
        self.pointwise_conv = nn.Sequential(
            nn.Conv1d(in_cha, d_model, kernel_size=1, stride=1),
            Permute())

    def forward(self, x):
        """
        x: [B, L, C]
        """
        x = self.ln(x)
        x = self.pointwise_conv(self.depthwise_conv(self.expension(x)))
        return x


class ConvAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=5,
            padding=2)

    def forward(self, x):
        """
        x: [B, H, L, L]
        """
        return self.conv(x)


class Conv3ChannelAttention(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv0 = ConvAttention(in_channels)
        self.conv1 = ConvAttention(in_channels)
        self.conv2 = ConvAttention(in_channels)

    def forward(self, x):
        """
        x: [B, 3, H, L, L]
        """
        x0 = self.conv0(x[:, 0, :, :, :])
        x1 = self.conv1(x[:, 1, :, :, :])
        x2 = self.conv2(x[:, 2, :, :, :])
        stacked = torch.stack([x0, x1, x2], dim=1)
        return torch.sum(stacked, dim=1, keepdim=True)


class RowWiseGateSelfAttention(nn.Module):
    def __init__(self, h, d_model, pairDim=128, dropout=0.1):
        super().__init__()
        assert d_model % h == 0, ValueError("Error with d_model and the number of heads.")
        assert d_model % 9 == 0, ValueError("The d_model does not divide 9")
        self.d_model = d_model
        self.d_k = d_model // h
        self.h = h
        self.div = math.sqrt(self.d_k)
        self.g = nn.parameter.Parameter(data=torch.ones(1), requires_grad=True)

        self.ln1 = nn.LayerNorm(d_model, eps=1e-6)
        self.ln2 = nn.LayerNorm(pairDim, eps=1e-6)
        self.q_linear = nn.Linear(d_model // 3, d_model // 3)
        self.v_linear = nn.Linear(d_model // 3, d_model // 3)
        self.conv3d = Conv3ChannelAttention(h)
        self.k_linear = nn.Linear(d_model // 3, d_model // 3)
        self.gate_linear = nn.Linear(d_model // 3, d_model // 3)
        self.pair_linear = nn.Linear(pairDim, h)
        self.pair_linear_rev = nn.Linear(h, pairDim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, pairX):
        """
        x: [B, L, C]
        pairX: [B, L, L, Cp]
        """
        ori_x = x
        _, L, C = x.shape
        x = self.ln1(x)
        pairX = self.ln2(pairX)
        # [B, 3, L, C//3]
        x = x.view([-1, L, 3, C // 3]).contiguous().permute([0, 2, 1, 3])
        k = self.k_linear(x).view([-1, 3, L, self.h, self.d_k // 3])
        q = self.q_linear(x).view([-1, 3, L, self.h, self.d_k // 3])
        v = self.v_linear(x).view([-1, 3, L, self.h, self.d_k // 3])
        g = self.gate_linear(x).view([-1, 3, L, self.h, self.d_k // 3]).permute([0, 1, 3, 2, 4])

        k = k.permute([0, 1, 3, 2, 4])  # [B, 3,  h,  L, d_k]
        q = q.permute([0, 1, 3, 2, 4])  # [B, 3,  h, L, d_k]
        v = v.permute([0, 1, 3, 2, 4])  # [B, 3,  h, L, d_k]

        qkMatrix = torch.matmul(q, k.transpose(-2, -1))  # [B, 3, h, L, L]
        pairBias = self.pair_linear(pairX).permute([0, 3, 1, 2]).unsqueeze(1)  # [B, 1, h, L, L]
        oriQK = qkMatrix + pairBias
        convQK = self.conv3d(oriQK)
        qk = (oriQK + convQK) / self.div
        score = F.softmax(qk, dim=-1)
        score = self.dropout1(score)
        h_outs = torch.matmul(score, v) * torch.sigmoid(g)  # [B, 3, h, L, d_k]
        h_outs = h_outs.permute([0, 3, 1, 2, 4]).contiguous().view([-1, L, self.d_model])
        rawScoreMean = F.gelu(torch.mean(qk, dim=1).permute([0, 2, 3, 1]).contiguous())
        return self.dropout2(self.out(h_outs)) + ori_x * self.g, \
            self.pair_linear_rev(rawScoreMean)


class ColWiseGateSelfAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0, ValueError("Error with d_model and the number of heads.")
        assert d_model % 9 == 0, ValueError("The d_model does not divide 9.")
        self.d_model = d_model
        self.d_k = d_model // h
        self.h = h
        self.div = math.sqrt(self.d_k)
        self.g = nn.parameter.Parameter(data=torch.ones(1), requires_grad=True)

        self.ln1 = nn.LayerNorm(d_model, eps=1e-6)
        self.q_linear = nn.Linear(d_model // 3, d_model // 3)
        self.v_linear = nn.Linear(d_model // 3, d_model // 3)
        self.k_linear = nn.Linear(d_model // 3, d_model // 3)
        self.gate_linear = nn.Linear(d_model // 3, d_model // 3)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: [B, L, C]
        """
        ori_x = x
        _, L, C = x.shape
        x = self.ln1(x)
        x = x.view([-1, L, 3, C // 3])
        k = self.k_linear(x).view([-1, L, 3, self.h, self.d_k // 3])
        q = self.q_linear(x).view([-1, L, 3, self.h, self.d_k // 3])
        v = self.v_linear(x).view([-1, L, 3, self.h, self.d_k // 3])
        g = self.gate_linear(x).view([-1, L, 3, self.h, self.d_k // 3]).permute([0, 1, 3, 2, 4])

        k = k.permute([0, 1, 3, 2, 4])
        q = q.permute([0, 1, 3, 2, 4])
        v = v.permute([0, 1, 3, 2, 4])

        rawScore = torch.matmul(q, k.transpose(-2, -1)) / self.div  # [B, L, h, 3, 3]
        score = F.softmax(rawScore, dim=-1)
        h_outs = torch.matmul(score, v) * torch.sigmoid(g)  # [B, L, h, 3, d_k]
        h_outs = h_outs.permute([0, 1, 3, 2, 4]).contiguous().view([-1, L, self.d_model])
        return self.dropout(self.out(h_outs)) + ori_x * self.g


class OuterProductPair(nn.Module):
    def __init__(self, d_model, pairDim=128):
        super().__init__()
        assert d_model % 3 == 0, ValueError("The d_model does not divide 3.")
        self.ln = nn.LayerNorm(d_model, eps=1e-6)
        self.trans1 = nn.Linear(d_model // 3, 256)
        self.trans2 = nn.Linear(32, pairDim)
        self.attenTrans = nn.Linear(64, 64)

    def forward(self, x):
        """
        x [B, L, C]
        return [B, L, L, pairDim]
        """
        B, L, C = x.shape
        x = self.ln(x)
        x = x.view([-1, L, 3, C // 3])
        trans = self.trans1(x).view([-1, L, 3, 32, 8])  # [B, L, 3, 32, 8]
        outer1 = trans.unsqueeze(-1)  # [B, L, 3, 32, 8, 1]
        outer2 = trans.unsqueeze(4)  # [B, L, 3, 32, 1, 8]
        outerproduct = torch.matmul(outer1, outer2)
        meanOut = torch.mean(outerproduct, dim=2, keepdim=False).\
            view([-1, L, 32, 64]).contiguous().permute([0, 2, 1, 3])  # [B, 32, L, 64]
        meanTrans = F.gelu(self.attenTrans(meanOut).contiguous())
        rawScore = torch.matmul(meanOut, meanTrans.transpose(-1, -2)).permute([0, 2, 3, 1])
        return self.trans2(rawScore)


class FormerBlock(nn.Module):
    def __init__(self, expand, h, d_model, pairDim, dropout, if_last_layer=False):
        super().__init__()
        # print("The dimension of model is: ", d_model)
        self.conv = ConvFeedForward(d_model, expand)
        self.outPair = OuterProductPair(d_model, pairDim)
        self.rowWiseAtten = RowWiseGateSelfAttention(h, d_model, pairDim, dropout)
        self.colWiseAtten = ColWiseGateSelfAttention(h, d_model, dropout)
        self.ffw = FeedForward(d_model, dropout)

        self.if_last_layer = if_last_layer
        if not if_last_layer:
            self.pairFFW = FeedForward(pairDim, dropout)

    def forward(self, x, pairX):
        """
        x:  [B, L, C]
        pairX: [B, L, L, pairDim]
        """
        # Conv
        x = self.conv(x) + x
        # Attention col
        x = self.colWiseAtten(x)
        # Attention row
        pairX = pairX + self.outPair(x)
        x, pairX = self.rowWiseAtten(x, pairX)
        # FFW
        if not self.if_last_layer:
            return self.ffw(x) + x, self.pairFFW(pairX)
        else:
            return self.ffw(x) + x


class FormerEncoder(nn.Module):
    def __init__(self, expand, h, d_model, pairDim, dropout, layers):
        super().__init__()
        block = []
        self.layers = layers
        self.pairDim = pairDim
        for _ in range(layers - 1):
            block.append(FormerBlock(expand, h, d_model, pairDim, dropout, False))
        block.append(FormerBlock(expand, h, d_model, pairDim, dropout, True))
        self.module_list = nn.ModuleList(block)

    def forward(self, x):
        """
        x:  [B, L, C]
        pairX: [B, L, L, pairDim]
        """
        b, l, _ = x.shape
        device = x.device
        pairX = torch.zeros([b, l, l, self.pairDim], device=device)
        for i in range(self.layers):
            if i != self.layers - 1:
                x, pairX = self.module_list[i](x, pairX)
            else:
                x = self.module_list[i](x, pairX)
        return x
