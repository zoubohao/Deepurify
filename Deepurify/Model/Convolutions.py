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
        self.dw = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False)

        self.dwd = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=3,
            groups=in_channels,
            bias=False,
            dilation=3)

        self.point = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=1)

    def forward(self, x):
        """
        x: [B, C, L]
        """
        atten = self.point(self.dwd(self.dw(x)))
        return x * atten


class DownSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.stridec_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=5,
            stride=2,
            padding=2)

        self.c1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): [b, c, l]

        Returns:
            _type_: _description_
        """
        return self.stridec_conv(x) + self.c1(x)


class Block(nn.Module):
    def __init__(self, 
                in_channels, 
                out_channels,
                expansion_factor=2, 
                drop_connect_rate=0.25):
        super().__init__()
        mid_cha = int(in_channels * expansion_factor)
        self.expan = nn.Sequential(
            nn.Conv1d(
                in_channels,
                mid_cha,
                kernel_size=3,
                stride=1,
                padding=1),
            Permute(),
            nn.LayerNorm(mid_cha, eps=1e-6),
            Permute(),
            nn.GELU(),
        )
        self.depthwise_conv = nn.Conv1d(
            mid_cha,
            mid_cha,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=mid_cha,
            bias=False)

        self.lka = LargeKernelAttention(mid_cha)

        self.pointwise_conv = nn.Sequential(
            nn.Conv1d(mid_cha, out_channels, kernel_size=1, stride=1),
            Permute(),
            nn.LayerNorm(out_channels, eps=1e-6),
            Permute()
        )
        self.drop_conn = drop_connect_rate
        self.gate = nn.parameter.Parameter(data=torch.ones(1), requires_grad=True)

    def forward(self, x):
        orix = x
        x = self.expan(x)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(self.lka(x))
        if self.training and self.drop_conn > 0:
            x = drop_connect(x, self.drop_conn)
        return x + orix * self.gate


class Bottleneck(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                layers,
                expand=2,
                drop_connect_rate=0.25):
        super().__init__()
        blocks = []
        blocks.append(DownSample(in_channels, out_channels))
        for _ in range(layers):
            blocks.append(
                Block(out_channels,
                      out_channels,
                      expand,
                      drop_connect_rate)
            )
        self.bottleNeck = nn.Sequential(*blocks)

    def forward(self, x):
        return self.bottleNeck(x)


class MEfficientNet(nn.Module):
    def __init__(self, in_channels, out_channels, layers=2, expand=4, drop_connect_rate=0.25):
        super().__init__()
        self.conv1 = Bottleneck(
            in_channels, 128, 1,
            expand=expand,
            drop_connect_rate=drop_connect_rate)

        self.b1 = Bottleneck(
            128, 256, layers,
            expand=expand,
            drop_connect_rate=drop_connect_rate)

        self.b2 = Bottleneck(
            256, 320, layers,
            expand=expand,
            drop_connect_rate=drop_connect_rate)

        self.b3 = Bottleneck(
            320, out_channels,
            layers, expand=expand,
            drop_connect_rate=drop_connect_rate)

    def forward(self, x):
        """
        x: [b, C, L]
        out: x64, x32, x16: [B, C, L]
        """
        x2 = self.conv1(x)
        x4 = self.b1(x2)
        x8 = self.b2(x4)
        x16 = self.b3(x8)
        return x16, x8, x4
