import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(p=keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class Permute(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute([0, 2, 1])


class LargeKernelAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.dw = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.dwd = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=3, groups=in_channels, bias=False, dilation=3)
        self.point = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        """
        x: [B, C, L]
        """
        atten = self.point(self.dwd(self.dw(x)))
        return x * atten


class DownSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.dw = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.dwd = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=3, groups=in_channels, bias=False, dilation=3)
        self.ln = nn.LayerNorm(in_channels, eps=1e-6)
        self.point = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.stridec_conv = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        self.c1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        conv = self.dwd(self.dw(x)).transpose(1, 2)
        conv = torch.transpose(self.ln(conv), 1, 2)
        h = self.point(conv) + x
        return self.stridec_conv(h) + self.c1(h)


class SqueezeFeedForward(nn.Module):
    def __init__(self, d_model):
        super(SqueezeFeedForward, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(d_model, d_model // 32)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.linear1(torch.flatten(self.avgpool(x), start_dim=1)))


class OmniDynamicConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_weight: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.conv_weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, num_weight), requires_grad=True)
        init.kaiming_uniform_(self.conv_weight)
        self.overall = SqueezeFeedForward(in_channels)
        self.kernel = nn.Linear(in_channels // 32, kernel_size, bias=False)
        self.inc = nn.Linear(in_channels // 32, in_channels, bias=False)
        self.outc = nn.Linear(in_channels // 32, out_channels, bias=False)
        self.nums = nn.Linear(in_channels // 32, num_weight, bias=False)

    def forward(self, x):
        overallBatch = self.overall(x)
        kernel = torch.sigmoid(self.kernel(overallBatch))
        inc = torch.sigmoid(self.inc(overallBatch))
        outc = torch.sigmoid(self.outc(overallBatch))
        nums_weight = torch.softmax(self.nums(overallBatch), dim=-1)

        def attention(kri, ini, outi, nums_wei, xi):
            xi = xi.unsqueeze(0)
            weight = (
                kri[None, None, :, None] * ini[None, :, None, None] * outi[:, None, None, None] * nums_wei[None, None, None, :] * self.conv_weight
            )
            weight = torch.sum(weight, dim=-1, keepdim=False)
            return F.conv1d(xi, weight, None, self.stride, self.padding)

        return torch.cat(list(map(attention, kernel, inc, outc, nums_weight, x)), dim=0)


class UnitBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=2, drop_connect_rate=0.25):
        super().__init__()
        mid_cha = int(in_channels * expansion_factor)
        self.expension = nn.Sequential(
            OmniDynamicConv1d(in_channels, mid_cha, num_weight=5, kernel_size=3, stride=1, padding=1),
            Permute(),
            nn.LayerNorm(mid_cha, eps=1e-6),
            Permute(),
            nn.GELU(),
        )
        self.depthwise_conv = nn.Conv1d(mid_cha, mid_cha, kernel_size=3, stride=1, padding=1, groups=mid_cha, bias=False)
        self.lka = LargeKernelAttention(mid_cha)
        self.pointwise_conv = nn.Sequential(
            nn.Conv1d(mid_cha, out_channels, kernel_size=1, stride=1), Permute(), nn.LayerNorm(out_channels, eps=1e-6), Permute()
        )
        self.drop_conn = drop_connect_rate
        self.gate = nn.parameter.Parameter(data=torch.ones(1), requires_grad=True)

    def forward(self, x):
        out = self.pointwise_conv(self.lka(self.depthwise_conv(self.expension(x))))
        if self.training and self.drop_conn > 0:
            out = drop_connect(out, self.drop_conn)
        return out + x * self.gate


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, layers, expand=2, drop_connect_rate=0.25):
        super().__init__()
        blocks = [DownSample(in_channels, out_channels)]
        blocks.extend(
            UnitBlock(out_channels, out_channels, expand, drop_connect_rate)
            for _ in range(layers)
        )
        self.bottleNeck = nn.Sequential(*blocks)

    def forward(self, x):
        return self.bottleNeck(x)


class MEfficientNet(nn.Module):
    def __init__(self, in_channels, out_channels, layers=2, expand=4, drop_connect_rate=0.25):
        super().__init__()
        self.conv1 = Bottleneck(in_channels, 128, 1, expand=expand, drop_connect_rate=drop_connect_rate)
        self.b1 = Bottleneck(128, 256, layers, expand=expand, drop_connect_rate=drop_connect_rate)
        self.b2 = Bottleneck(256, 320, layers, expand=expand, drop_connect_rate=drop_connect_rate)
        self.b3 = Bottleneck(320, out_channels, layers, expand=expand, drop_connect_rate=drop_connect_rate)

    def forward(self, x):
        """
        x: [b, C, L]
        out: x64, x32, x16: [B, C, L]
        """
        x2 = self.conv1(x)
        x4 = self.b1(x2)
        x8 = self.b2(x4)
        x16 = self.b3(x8)
        return x16, x8, x4, x2
