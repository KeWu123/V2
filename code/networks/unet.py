# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
import pdb
from torch.nn import functional as F
from torch.distributions.uniform import Uniform


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upsampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p):
        super(UpBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
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
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

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
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x_last = self.up4(x, x0)
        output = self.out_conv(x_last)
        return output, x_last


class Decoder_tsne(nn.Module):
    def __init__(self, params):
        super(Decoder_tsne, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x5 = self.up1(x4, x3)  # 1, 128, 32, 32
        x6 = self.up2(x5, x2)  # 1, 64, 64, 64
        x7 = self.up3(x6, x1)  # 1, 32, 128, 128
        x_last = self.up4(x7, x0)  # 1, 16, 256, 256
        output = self.out_conv(x_last)
        return output, x_last


class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        dim_in = 16
        feat_dim = 32
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        for class_c in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_' + str(class_c), selector)

        for class_c in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_memory' + str(class_c), selector)

    def forward_projection_head(self, features):
        return self.projection_head(features)

    def forward_prediction_head(self, features):
        return self.prediction_head(features)

    def forward(self, x):
        feature = self.encoder(x)
        output, features = self.decoder(feature)
        return output, features


class UNet_2d(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_2d, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        dim_in = 16
        feat_dim = 32
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        for class_c in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_' + str(class_c), selector)

        for class_c in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_memory' + str(class_c), selector)

    def forward_projection_head(self, features):
        return self.projection_head(features)
        # return self.decoder(features)

    def forward_prediction_head(self, features):
        return self.prediction_head(features)

    def forward(self, x):
        feature = self.encoder(x)
        output, features = self.decoder(feature)
        return output


class Sep_UNet_2d(nn.Module):
    def __init__(self, in_chns, class_num):
        super(Sep_UNet_2d, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.pool = nn.MaxPool2d(3, stride=2)
        dim_in = 16
        feat_dim = 32
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        for class_c in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_' + str(class_c), selector)

        for class_c in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_memory' + str(class_c), selector)

    def forward_projection_head(self, features):
        return self.projection_head(features)
        # return self.decoder(features)

    def forward_prediction_head(self, features):
        return self.prediction_head(features)

    def forward(self, x):
        feature = self.encoder(x)
        output, features = self.decoder(feature)
        feature = self.pool(feature[4])
        feature = self.pool(feature)
        return feature, output


class GradReverse(torch.autograd.Function):
    def __init__(self):
        super(GradReverse, self).__init__()

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None


class GradReverseLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(GradReverseLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        lam = torch.tensor(self.lambd)
        return GradReverse.apply(x, lam)


class net_D(nn.Module):
    def __init__(self, b_size):
        super(net_D, self).__init__()
        # self.total_dim = int(total_dim)
        self.b_size = b_size
        self.total_dim = self.b_size * 256 * 3 * 3
        self.model = nn.Sequential(
            nn.Linear(self.total_dim, int(self.total_dim / 2)),
            nn.Tanh(),
            nn.Linear(int(self.total_dim / 2), int(self.total_dim / 4)),
            nn.Tanh(),
            nn.Linear(int(self.total_dim / 4), 1),
            nn.Sigmoid()
        )
        # self.GRL = GradReverseLayer(1)

    def forward(self, x):
        # x = self.GRL(x)
        x = x.view(1, -1)
        x = self.model(x)
        return x


class UNet_2dBCP(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_2dBCP, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output


class UNet_tsne(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_tsne, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder_tsne(params)
        dim_in = 16
        feat_dim = 32
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        for class_c in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_' + str(class_c), selector)

        for class_c in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_memory' + str(class_c), selector)

    def forward_projection_head(self, features):
        return self.projection_head(features)

    def forward_prediction_head(self, features):
        return self.prediction_head(features)

    def forward(self, x):
        feature = self.encoder(x)
        output, features = self.decoder(feature)
        return output, features


class UNet_3D(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=False):
        super(UNet_3D, self).__init__()
        self.training = training
        self.encoder1 = nn.Conv3d(in_channel, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2 = nn.Conv3d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3 = nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.encoder4 = nn.Conv3d(128, 256, 3, stride=1, padding=1)
        # self.encoder5=   nn.Conv3d(256, 512, 3, stride=1, padding=1)

        # self.decoder1 = nn.Conv3d(512, 256, 3, stride=1,padding=1)  # b, 16, 5, 5
        self.decoder2 = nn.Conv3d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 = nn.Conv3d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv3d(32, 2, 3, stride=1, padding=1)

        self.map4 = nn.Sequential(
            nn.Conv3d(2, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Softmax(dim=1)
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Softmax(dim=1)
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'),
            nn.Softmax(dim=1)
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(16, 32, 32), mode='trilinear'),
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        out = F.relu(F.max_pool3d(self.encoder1(x), 2, 2))
        t1 = out
        out = F.relu(F.max_pool3d(self.encoder2(out), 2, 2))
        t2 = out
        out = F.relu(F.max_pool3d(self.encoder3(out), 2, 2))
        t3 = out
        out = F.relu(F.max_pool3d(self.encoder4(out), 2, 2))
        # t4 = out
        # out = F.relu(F.max_pool3d(self.encoder5(out),2,2))

        # t2 = out
        # out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2,2),mode ='trilinear'))
        # print(out.shape,t4.shape)
        output1 = self.map1(out)
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t3)
        output2 = self.map2(out)
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t2)
        output3 = self.map3(out)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t1)

        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2, 2), mode='trilinear'))
        output4 = self.map4(out)
        # print(out.shape)
        # print(output1.shape,output2.shape,output3.shape,output4.shape)
        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4


# ================================================================
# FFM-inspired extension for semi-supervised Prostate segmentation
# Reference idea: ECCV 2024 "Fractal Feature Maps for Tubular Structure
# Segmentation". We keep the original UNet depth/channels unchanged.
# The image FFM is appended internally as an extra input channel, and
# an optional light edge head is attached to the last decoder feature.
# ================================================================
import math


def _normalize_minmax_per_sample(x, eps=1e-6):
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def fractal_feature_map_torch(x, scales=(3, 7, 15), eps=1e-6):
    """
    Fast pixel-level fractal feature map for 2D tensors.

    Args:
        x: Tensor [B, 1, H, W]. It can be an image, mask, or pseudo mask.
        scales: local window sizes used to approximate pixel-level fractal
                behaviour. Odd values are recommended.

    Returns:
        ffm: Tensor [B, 1, H, W], normalized to [0, 1] per sample.

    Note:
        This is a DBC-style local fractal complexity map implemented with
        max/min pooling for speed. It follows the ECCV-2024 FFM spirit:
        extend fractal descriptors to pixel level with sliding windows, but
        avoids Python loops over pixels.
    """
    if x.dim() != 4:
        raise ValueError("fractal_feature_map_torch expects [B, C, H, W].")
    if x.size(1) != 1:
        x = x[:, :1]

    x = x.float()
    x_norm = _normalize_minmax_per_sample(x, eps=eps)

    log_counts = []
    log_scales = []
    for k in scales:
        k = int(k)
        if k % 2 == 0:
            k += 1
        pad = k // 2
        max_v = F.max_pool2d(x_norm, kernel_size=k, stride=1, padding=pad)
        min_v = -F.max_pool2d(-x_norm, kernel_size=k, stride=1, padding=pad)
        local_range = (max_v - min_v).clamp_min(0.0)

        # Differential-box-counting style local count. Uniform regions approach
        # 1; structurally complex or boundary regions have larger counts.
        local_count = 1.0 + local_range * float(k)
        log_counts.append(torch.log(local_count + eps))
        log_scales.append(math.log(float(k)))

    y = torch.stack(log_counts, dim=1)  # [B, S, 1, H, W]
    s = torch.tensor(log_scales, dtype=x.dtype, device=x.device).view(1, -1, 1, 1, 1)
    s_centered = s - s.mean(dim=1, keepdim=True)
    y_centered = y - y.mean(dim=1, keepdim=True)
    slope = (s_centered * y_centered).sum(dim=1) / ((s_centered ** 2).sum(dim=1) + eps)
    slope = F.relu(slope)

    slope_min = slope.amin(dim=(2, 3), keepdim=True)
    slope_max = slope.amax(dim=(2, 3), keepdim=True)
    ffm = (slope - slope_min) / (slope_max - slope_min + eps)
    return ffm.detach()


def mask_to_boundary_torch(mask, kernel_size=3):
    """Create a binary boundary map from [B,H,W] or [B,1,H,W] masks."""
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    pad = kernel_size // 2
    max_v = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    min_v = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
    boundary = (max_v - min_v > 0).float()
    return boundary


class UNet_FFM_2d(nn.Module):
    """
    FFM-UNet for Prostate experiments.

    It keeps the original UNet encoder/decoder depth and channels unchanged:
    [16, 32, 64, 128, 256]. The only input-side change is that a pixel-level
    fractal feature map is computed internally and concatenated with the raw
    image, so the external dataloader/validation code still feeds a 1-channel
    image. Optionally, a 1x1 edge head is attached to the last decoder feature.
    """

    def __init__(self, in_chns, class_num, use_edge_head=True, ffm_scales=(3, 7, 15)):
        super(UNet_FFM_2d, self).__init__()
        self.raw_in_chns = in_chns
        self.class_num = class_num
        self.use_edge_head = use_edge_head
        self.ffm_scales = ffm_scales

        params = {'in_chns': in_chns + 1,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.edge_head = nn.Conv2d(16, 1, kernel_size=1) if use_edge_head else None

    def _append_ffm(self, x):
        # If a 2-channel tensor is already supplied, use it directly.
        if x.size(1) == self.raw_in_chns + 1:
            return x
        ffm = fractal_feature_map_torch(x[:, :1], scales=self.ffm_scales)
        if ffm.device != x.device:
            ffm = ffm.to(x.device)
        return torch.cat([x, ffm], dim=1)

    def forward(self, x, return_edge=False):
        # Safety against accidental CPU/GPU mismatch.
        if next(self.parameters()).device != x.device:
            self.to(x.device)
        x_aug = self._append_ffm(x)
        feature = self.encoder(x_aug)
        output, x_last = self.decoder(feature)
        if return_edge and self.edge_head is not None:
            edge_logits = self.edge_head(x_last)
            return output, edge_logits
        return output


# Backward-compatible aliases.
UNet_FFMInput_2d = UNet_FFM_2d
UNet_FDInput_2d = UNet_FFM_2d

if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from thop import profile
    from thop import clever_format

    model = UNet_3D(in_channel=1, out_channel=2)
    input = torch.randn(1, 1, 112, 112, 80)
    flops, params = profile(model, inputs=(input,))
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)

    from ptflops import get_model_complexity_info

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (1, 112, 112, 80), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # import pdb; pdb.set_trace()
