
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Parameter(nn.init.kaiming_normal_(torch.randn(1, max_len, d_model)), requires_grad=True)

    def forward(self, x):
        return x + torch.sigmoid(self.pos_embedding)
