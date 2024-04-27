import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Convolutions import MEfficientNet
from .FormerLayers import FormerEncoder


class Gseqformer(nn.Module):
    def __init__(
            self,
            in_channels,
            labels_num=None,
            head_num=6,
            d_model=528,
            num_GeqEncoder=2,
            IRB_layers=2,
            expand=2,
            feature_dim=1024,
            drop_connect_ratio=0.25,
            dropout=0.1,
            register_hook=False):

        super().__init__()
        self.d_model = d_model
        self.compressConv = MEfficientNet(
            in_channels,
            out_channels=d_model,
            layers=IRB_layers,
            expand=expand,
            drop_connect_rate=drop_connect_ratio)

        self.gSeqEncoder = FormerEncoder(
            expand,
            head_num,
            d_model,
            pairDim=64,
            dropout=dropout,
            layers=num_GeqEncoder)

        # 256
        self.conv16_32 = nn.Conv1d(256, 320, kernel_size=3, stride=2, padding=1)
        # 320
        self.conv32_64 = nn.Conv1d(320, d_model, kernel_size=3, stride=2, padding=1)
        self.feature = nn.Linear(d_model, feature_dim)

        self.reg = register_hook
        if register_hook:
            self.gradient = None
            self.feature_cam = None
            self.outAtten = None
            self.handle = None

    def save_gradient(self, grad):
        self.gradient = grad.clone().detach().cpu()

    def get(self):
        self.feature_cam = self.feature_cam.clone().detach().cpu()
        if self.gradient is None:
            raise "The gradient is None."
        return self.feature_cam, self.gradient, self.outAtten.clone().detach().cpu()

    def forward_features(self, x):
        x64, x32, x16 = self.compressConv(x)  # B, C, L

        eX64 = x64.permute([0, 2, 1])
        eX64 = self.gSeqEncoder(eX64)  # B,L,C
        eX64 = eX64.permute(dims=[0, 2, 1]).contiguous()  # B, C, L

        conv16_32 = F.gelu(self.conv16_32(x16))
        branchX32 = x32 + conv16_32
        conv32_64 = F.gelu(self.conv32_64(branchX32))
        rawGateScore = eX64 + conv32_64
        gateScore = torch.softmax(torch.mean(rawGateScore, dim=1, keepdim=True), dim=-1)  # B, 1, L
        eX64 = eX64 * gateScore
        eX64 = torch.sum(eX64, dim=-1)

        #### Inject hook to get attention tensor ####
        if self.reg:
            self.outAtten = gateScore
            if self.handle is None:
                print("######### Inject Hook.")
                self.handle = eX64.register_hook(self.save_gradient)
            self.feature_cam = eX64

        return self.feature(eX64)

    def forward(self, x):
        """
        :param x: [B, C, L] B: batch size,
        :return: [B, feature_num] 
        """
        rep = self.forward_features(x)
        return rep


class TaxaEncoder(nn.Module):
    def __init__(self, dict_size, embedding_dim, num_labels=None, feature_dim=1024, num_layers=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=dict_size, embedding_dim=embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim *
                            2, num_layers=num_layers, dropout=dropout)
        self.feature = nn.Linear(embedding_dim * 2, feature_dim)
        self.num_labels = num_labels
        if num_labels is not None:
            self.fc = nn.Linear(feature_dim, num_labels)

    def forward_features(self, x):
        x = self.embedding(x).permute([1, 0, 2])  # L, B, C
        x, _ = self.encoder(x)  # L, B, C
        x = x[-1]
        return self.feature(x)

    def forward(self, x):
        fea = self.forward_features(x)
        if self.num_labels is None:
            return fea
        else:
            return self.fc(fea)


class DeepurifyModel(nn.Module):
    def __init__(
        self,
        max_model_len: int,
        in_channels: int,
        taxo_dict_size: int,
        vocab_3Mer_size: int,
        vocab_4Mer_size: int,
        phylum_num: int,
        species_num: int,
        head_num=8,
        d_model=512,
        num_GeqEncoder=2,
        num_lstm_layer=4,
        IRB_layers=2,
        expand=2,
        feature_dim=1024,
        drop_connect_ratio=0.25,
        dropout=0.1,
        register_hook=False,
    ):
        super().__init__()
        self.visionEncoder = Gseqformer(in_channels, None, head_num, d_model, num_GeqEncoder,
                                        IRB_layers, expand, feature_dim, drop_connect_ratio,
                                        dropout, register_hook)
        self.textEncoder = TaxaEncoder(taxo_dict_size, d_model, None, feature_dim, num_lstm_layer, dropout)
        self.vocab3MerEmb = nn.Embedding(vocab_3Mer_size, 16, padding_idx=0)
        self.vocab4MerEmb = nn.Embedding(vocab_4Mer_size, 32, padding_idx=0)
        self.postionalEmb = nn.Parameter(
            nn.init.kaiming_normal_(torch.randn(1, in_channels, max_model_len)), requires_grad=True)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.phy_fc = nn.Linear(feature_dim, phylum_num)
        self.spe_fc = nn.Linear(feature_dim, species_num)
        self.if_noisy_c = nn.Linear(feature_dim, 1)

    def concatTensors(self, ori_rev_tensor, feature_3Mer, feature_3Mer_rev_com, feature_4Mer, feature_4Mer_rev_com):
        """
        ori_rev_tensor: [B, 12, max_model_len]
        feature_3Mer: [B, max_model_len]
        feature_3Mer_rev_com: [B, max_model_len]
        feature_4Mer: [B, max_model_len]
        feature_4Mer_rev_com: [B, max_model_len]
        """
        featrue3Mer = self.vocab3MerEmb(feature_3Mer).permute([0, 2, 1])  # [B, C, L]
        featrue3MerRevCom = self.vocab3MerEmb(feature_3Mer_rev_com).permute([0, 2, 1])  # [B, C, L]
        featrue4Mer = self.vocab4MerEmb(feature_4Mer).permute([0, 2, 1])  # [B, C, L]
        featrue4MerRevCom = self.vocab4MerEmb(feature_4Mer_rev_com).permute([0, 2, 1])  # [B, C, L]
        embedTensor = torch.cat([ori_rev_tensor, featrue3Mer, featrue3MerRevCom,
                                featrue4Mer, featrue4MerRevCom], dim=1) + self.postionalEmb
        return embedTensor  # [B, C, L] C = 108

    def gatherValues(self, v1, t2, num_labels):
        """
        t1 = torch.randn([3, 4])
        t2 = torch.randn([3, 5, 4])
        print(gatherValues(t1, t2, 5))
        t2v = t2.view(15, 4)
        print(t1 @ t2v.T)
        """
        b1 = v1.size(0)
        b2 = t2.size(0)
        assert b1 == b2, ValueError("Batch size is not equal.")
        dotTensor = torch.tensordot(v1, t2, dims=([1], [2])).permute([0, 2, 1])  # [b1, num_labels, b2]
        index = torch.arange(b1).expand([num_labels, b1]).transpose(1, 0).unsqueeze(-1).to(dotTensor.device)
        return torch.gather(dotTensor, dim=-1, index=index).squeeze(-1)

    def forward(
        self,
        ori_rev_tensor: torch.Tensor,
        feature_3Mer: torch.Tensor,
        feature_3Mer_rev_com: torch.Tensor,
        feature_4Mer: torch.Tensor,
        feature_4Mer_rev_com: torch.Tensor,
        texts: torch.Tensor,
        oriPhyTensor=None,
        matchTextTensor=None,
        outerMisMatchTextTensor=None
    ):
        """
        texts: [B, (misMatchNum + 1), L], 1 means the match text
        """
        if self.training:
            b = texts.size(0)
            b, num_labels, textLength = texts.shape
            images = self.concatTensors(ori_rev_tensor,
                                        feature_3Mer, feature_3Mer_rev_com,
                                        feature_4Mer, feature_4Mer_rev_com)
            image_features_ori = self.visionEncoder(images)  # [B, D] tokens [B,C,l]
            concatedTensor = torch.cat([oriPhyTensor, matchTextTensor, outerMisMatchTextTensor,
                                        texts.view([b * num_labels, textLength])], dim=0)
            text_features_ori = self.textEncoder(concatedTensor)

            # normalized features
            image_features_norm = image_features_ori / image_features_ori.norm(dim=-1, keepdim=True)
            text_features_norm = text_features_ori / text_features_ori.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            oriPhyTensorNorm = text_features_norm[0:b]
            matchTextTensorNorm = text_features_norm[b: b + b]
            outerMisMatchTextTensorNorm = text_features_norm[2 * b: 3 * b]
            pairTextTensor = text_features_norm[3 * b:]
            pairTextTensor = pairTextTensor.view([b, num_labels, -1])

            # print("token shape", tokens.shape)
            # print("token prob shape", tokenProb.shape)
            # return the result
            return (
                self.gatherValues(image_features_norm, pairTextTensor, num_labels) * logit_scale,
                oriPhyTensorNorm,
                matchTextTensorNorm,
                outerMisMatchTextTensorNorm,
                image_features_norm,
                text_features_norm,
                self.if_noisy_c(image_features_ori),
                self.phy_fc(image_features_norm),
                self.spe_fc(image_features_norm)
            )
        else:
            b = texts.size(0)
            b, num_labels, textLength = texts.shape
            images = self.concatTensors(ori_rev_tensor, feature_3Mer, feature_3Mer_rev_com,
                                        feature_4Mer, feature_4Mer_rev_com)
            image_features_ori = self.visionEncoder(images)  # [B, D]
            text_features_ori = self.textEncoder(texts.view([b * num_labels, textLength]))
            # normalized features
            image_features_norm = image_features_ori / image_features_ori.norm(dim=-1, keepdim=True)
            text_features_norm = text_features_ori / text_features_ori.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            pairTextTensor = text_features_norm.view([b, num_labels, -1])
            return self.gatherValues(image_features_norm, pairTextTensor, num_labels) * logit_scale


    ##### ANNOTATION PART #####
    def annotatedConcatTensors(self, ori_rev_tensor, feature_3Mer, feature_3Mer_rev_com, feature_4Mer, feature_4Mer_rev_com):
        """
        ori_rev_tensor: [12, max_model_len]
        feature_3Mer: [max_model_len]
        feature_3Mer_rev_com: [max_model_len]
        feature_4Mer: [max_model_len]
        feature_4Mer_rev_com: [max_model_len]
        """
        with torch.no_grad():
            featrue3Mer = self.vocab3MerEmb(feature_3Mer).permute([1, 0])  # [C, L]
            featrue3MerRevCom = self.vocab3MerEmb(feature_3Mer_rev_com).permute([1, 0])  # [C, L]
            featrue4Mer = self.vocab4MerEmb(feature_4Mer).permute([1, 0])  # [C, L]
            featrue4MerRevCom = self.vocab4MerEmb(feature_4Mer_rev_com).permute([1, 0])  # [C, L]
        return torch.cat([ori_rev_tensor, featrue3Mer, featrue3MerRevCom, featrue4Mer, featrue4MerRevCom], dim=0) + self.postionalEmb.squeeze(0)

    def visionRepNorm(self, images):
        with torch.no_grad():
            image_features_ori = self.visionEncoder(images)  # [B, D]
        return image_features_ori / image_features_ori.norm(dim=-1, keepdim=True)

    def visionRep(self, images):
        with torch.no_grad():
            image_features_ori = self.visionEncoder(images)  # [B, D]
        return image_features_ori

    def textRepNorm(self, texts):
        with torch.no_grad():
            b, num_labels, textLength = texts.shape
            text_features_ori = self.textEncoder(texts.view([b * num_labels, textLength]))  # [B * (misMatchNum + 1), D]
            text_features_norm = text_features_ori / text_features_ori.norm(dim=-1, keepdim=True)
            text_features_norm = text_features_norm.view([b, num_labels, -1])
        return text_features_norm

    def textRep(self, texts):
        with torch.no_grad():
            b, num_labels, textLength = texts.shape
            text_features_ori = self.textEncoder(texts.view([b * num_labels, textLength]))  # [B * (misMatchNum + 1), D]
            text_features_ori = text_features_ori.view([b, num_labels, -1])
        return text_features_ori
