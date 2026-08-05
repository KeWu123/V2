# -*- coding: utf-8 -*-
"""
UNet_XingTai: a light morphology-aware UNet for 2D semi-supervised segmentation.

The segmentation backbone keeps the same encoder-decoder topology as the provided
UNet baseline. The only added part is a one-channel morphology/uncertainty head
attached to the last decoder feature. By default forward(x) returns only the
segmentation logits, so existing validation/test code that expects a tensor will
not be broken. During training, call forward(x, return_aux=True) to get:
    segmentation_logits, morphology_logits, last_decoder_feature
"""
from __future__ import division, print_function

import torch
import torch.nn as nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Two convolution layers with batch norm, LeakyReLU, and dropout."""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock."""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upsampling followed by ConvBlock."""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p):
        super(UpBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params["in_chns"]
        self.ft_chns = self.params["feature_chns"]
        self.dropout = self.params["dropout"]
        assert len(self.ft_chns) == 5

        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.ft_chns = self.params["feature_chns"]
        self.n_class = self.params["class_num"]
        assert len(self.ft_chns) == 5

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0, x1, x2, x3, x4 = feature
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x_last = self.up4(x, x0)
        output = self.out_conv(x_last)
        return output, x_last


class MorphologyHead(nn.Module):
    """Predicts a one-channel boundary / conformal-margin likelihood map."""

    def __init__(self, in_channels=16):
        super(MorphologyHead, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)


class UNet_XingTai(nn.Module):
    """
    UNet with a morphology auxiliary head.

    Network name for net_factory: unet_xingtai
    """

    supports_morph_aux = True

    def __init__(self, in_chns, class_num):
        super(UNet_XingTai, self).__init__()

        params = {
            "in_chns": in_chns,
            "feature_chns": [16, 32, 64, 128, 256],
            "dropout": [0.05, 0.1, 0.2, 0.3, 0.5],
            "class_num": class_num,
            "acti_func": "relu",
        }

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.morph_head = MorphologyHead(in_channels=16)

        # Keep the projection/prediction heads from the original UNet family so
        # code that calls them will still work.
        dim_in = 16
        feat_dim = 32
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )
        for class_c in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1),
            )
            self.__setattr__("contrastive_class_selector_" + str(class_c), selector)

        for class_c in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1),
            )
            self.__setattr__("contrastive_class_selector_memory" + str(class_c), selector)

    def forward_projection_head(self, features):
        return self.projection_head(features)

    def forward_prediction_head(self, features):
        return self.prediction_head(features)

    def forward(self, x, return_aux=False):
        feature = self.encoder(x)
        output, last_feature = self.decoder(feature)
        if return_aux:
            morph_logits = self.morph_head(last_feature)
            return output, morph_logits, last_feature
        return output


# Common aliases, useful when importing manually.
UNet_xingtai = UNet_XingTai
UNetXingTai = UNet_XingTai


if __name__ == "__main__":
    model = UNet_XingTai(in_chns=1, class_num=2)
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    y_aux = model(x, return_aux=True)
    print(y.shape)
    print(y_aux[0].shape, y_aux[1].shape, y_aux[2].shape)
