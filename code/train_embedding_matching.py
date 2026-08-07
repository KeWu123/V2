import argparse
import logging
import os
import random
import shutil
import sys
import time
from glob import glob

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from skimage.measure import label

# from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from utils import losses, ramps
# from val_2D import test_single_volume
from utils import val_2d
from embedding_matching import (embedding_matching_losses,
                                ensemble_embedding_classifier)


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

    def forward(self, x, need_fp=False, feature_dropout=0.5,
                return_features=False):
        feature = self.encoder(x)
        if need_fp:
            # Follow the official medical UniMatch U-Net implementation: pair
            # every encoder scale with an independently channel-dropped copy,
            # decode both streams in one batch, and split their logits.
            paired_feature = [
                torch.cat((feat, F.dropout2d(
                    feat, p=feature_dropout, training=True)), dim=0)
                for feat in feature
            ]
            paired_output, paired_decoder_features = self.decoder(paired_feature)
            regular_output, perturbed_output = paired_output.chunk(2, dim=0)
            if return_features:
                regular_features, _ = paired_decoder_features.chunk(2, dim=0)
                return regular_output, perturbed_output, regular_features
            return regular_output, perturbed_output
        output, features = self.decoder(feature)
        return output, features


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default=os.path.abspath(os.path.join(
                        os.path.dirname(__file__), '..', 'data', 'PROMISE12_h5')),
                    help='dataset root path')
parser.add_argument('--exp', type=str,
                    default='MT_PROMISE12_UniMatch_EmbeddingMatching_v2',
                    help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int,
                    default=1000, help='maximum epoch number to pre-train')
parser.add_argument('--max_iterations', type=int,
                    default=5000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=7,
                    help='number of labeled PROMISE12 cases')
# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--confidence_threshold', type=float, default=0.95,
                    help='UniMatch pseudo-label confidence threshold')
parser.add_argument('--feature_dropout', type=float, default=0.5,
                    help='UniMatch channel dropout probability')
parser.add_argument('--strong_aug_prob', type=float, default=0.8,
                    help='probability of MRI brightness/contrast perturbation')
parser.add_argument('--blur_prob', type=float, default=0.5,
                    help='probability of Gaussian blur per strong view')
parser.add_argument('--cutmix_prob', type=float, default=0.5,
                    help='probability of CutMix per sample and strong view')
parser.add_argument('--em_surface_radius', type=int, default=2,
                    help='2D inside/outside surface band radius for labeled references')
parser.add_argument('--em_references_per_class', type=int, default=16,
                    help='k labeled surface embeddings sampled per class and classifier')
parser.add_argument('--em_ensemble_size', type=int, default=5,
                    help='l independently sampled embedding classifiers')
parser.add_argument('--em_temperature', type=float, default=1.0,
                    help='softmax temperature for foreground/background similarities')
parser.add_argument('--em_mc_passes', type=int, default=5,
                    help='teacher MC-Dropout passes used for predictive entropy')
parser.add_argument('--em_mc_chunk_size', type=int, default=2,
                    help='number of MC passes evaluated together to limit GPU memory')
parser.add_argument('--em_mc_noise_std', type=float, default=0.01,
                    help='weak Gaussian noise standard deviation for MC teacher inputs')
parser.add_argument('--em_mc_noise_clip', type=float, default=0.02,
                    help='absolute clipping limit for MC teacher input noise')
parser.add_argument('--em_loss_weight', type=float, default=0.125,
                    help='mature paper weight for L_NN + L_EN')
parser.add_argument('--em_coverage_reference', type=float, default=0.05,
                    help='downscale EM only when active coverage is below this 2D reference; set 0 to disable')
args = parser.parse_args()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "PROMISE12" in dataset:
        train_list = os.path.join(dataset, "train.list")
        slices_dir = os.path.join(dataset, "data", "slices")
        if not os.path.isfile(train_list) or not os.path.isdir(slices_dir):
            raise FileNotFoundError(
                "PROMISE12 requires train.list and data/slices under {}".format(dataset)
            )
        with open(train_list, "r") as handle:
            labeled_cases = [line.strip() for line in handle if line.strip()][:patiens_num]
        if len(labeled_cases) != patiens_num:
            raise ValueError(
                "Requested {} labeled cases, but train.list contains only {}".format(
                    patiens_num, len(labeled_cases)
                )
            )
        labeled_slices = sum(
            len(glob(os.path.join(slices_dir, case + "_slice*.h5")))
            for case in labeled_cases
        )
        if labeled_slices == 0:
            raise ValueError(
                "No labeled PROMISE12 slices found for {}".format(labeled_cases)
            )
        return labeled_slices
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "2": 48, "3": 68, "5": 102, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate" in dataset:
        ref_dict = {"2": 27, "4": 53, "8": 120, "7": 191, "11": 306,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        raise ValueError("Unsupported dataset path: {}".format(dataset))
    return ref_dict[str(patiens_num)]


def get_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 2):
            temp_seg = segmentation[i]
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        n_batch = class_list[0]
        batch_list.append(n_batch)
    return torch.Tensor(np.array(batch_list)).cuda()


def get_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_2DLargestCC(probs)
    return probs


def gaussian_blur_2d(image, sigma):
    """Gaussian blur for one CxHxW floating-point MRI slice."""
    radius = max(1, int(round(2.0 * sigma)))
    kernel_size = 2 * radius + 1
    coords = torch.arange(
        kernel_size, device=image.device, dtype=image.dtype) - radius
    kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.expand(image.shape[0], 1, -1, -1)
    return F.conv2d(
        image.unsqueeze(0), kernel, padding=radius,
        groups=image.shape[0]).squeeze(0)


def strong_mri_augmentation(images):
    """Two-view strong augmentation adapted from official medical UniMatch.

    PROMISE12 is stored as z-score-normalized single-channel MRI. Therefore,
    applying PIL ColorJitter after uint8 conversion would clip/wrap valid signal.
    Brightness and contrast are applied directly in the floating-point domain;
    Gaussian blur follows the official medical implementation.
    """
    augmented = []
    for image in images:
        view = image.clone()
        if random.random() < args.strong_aug_prob:
            contrast = random.uniform(0.5, 1.5)
            brightness = random.uniform(-0.25, 0.25)
            mean = view.mean(dim=(-2, -1), keepdim=True)
            view = (view - mean) * contrast + mean + brightness
        if random.random() < args.blur_prob:
            view = gaussian_blur_2d(view, random.uniform(0.1, 2.0))
        augmented.append(view)
    return torch.stack(augmented, dim=0)


def obtain_cutmix_boxes(batch_size, height, width, device):
    """Official UniMatch CutMix area/aspect ranges, generalized to HxW."""
    boxes = torch.zeros(
        (batch_size, height, width), dtype=torch.bool, device=device)
    for index in range(batch_size):
        if random.random() > args.cutmix_prob:
            continue
        area = random.uniform(0.02, 0.4) * height * width
        ratio = random.uniform(0.3, 1.0 / 0.3)
        cut_width = max(1, min(width, int(round((area / ratio) ** 0.5))))
        cut_height = max(1, min(height, int(round((area * ratio) ** 0.5))))
        left = random.randint(0, width - cut_width)
        top = random.randint(0, height - cut_height)
        boxes[index, top:top + cut_height, left:left + cut_width] = True
    return boxes


def cutmix_tensor(base, donor, boxes):
    mask = boxes.unsqueeze(1) if base.ndim == 4 else boxes
    return torch.where(mask, donor, base)


def confidence_masked_baseline_loss(logits, targets, confidence, dice_loss):
    """Apply the baseline CE+Dice pseudo loss only at reliable pixels."""
    valid = confidence >= args.confidence_threshold
    valid_float = valid.float()
    per_pixel_ce = F.cross_entropy(logits, targets.long(), reduction='none')
    loss_ce = (per_pixel_ce * valid_float).sum() / valid_float.sum().clamp_min(1.0)
    loss_dice = dice_loss(
        torch.softmax(logits, dim=1), targets.unsqueeze(1),
        mask=valid_float.unsqueeze(1))
    return 0.5 * (loss_ce + loss_dice), valid


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 5 * args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


@torch.no_grad()
def mc_dropout_predictive_entropy(model, inputs, passes, chunk_size,
                                  noise_std, noise_clip, random_seed):
    """UA-MT-style predictive entropy with dropout but frozen BatchNorm.

    Only dropout layers are switched to training mode.  The dedicated RNG scope
    prevents these extra stochastic passes from changing UniMatch's original
    augmentation/dropout random sequence under the same experiment seed.
    """
    passes = int(passes)
    chunk_size = max(1, int(chunk_size))
    if passes <= 0:
        raise ValueError("em_mc_passes must be positive")

    # Open-source reference:
    # github.com/yulequan/UA-MT/blob/master/code/
    # train_LA_meanteacher_certainty_unlabel.py
    dropout_types = (nn.Dropout, nn.Dropout2d, nn.Dropout3d)
    dropout_modules = [
        module for module in model.modules()
        if isinstance(module, dropout_types)]
    previous_states = [module.training for module in dropout_modules]
    for module in dropout_modules:
        module.train(True)

    cuda_devices = []
    if inputs.is_cuda:
        device_index = inputs.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        cuda_devices = [device_index]

    probability_sum = None
    batch_size = inputs.shape[0]
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(random_seed))
            if inputs.is_cuda:
                torch.cuda.manual_seed(int(random_seed))
            completed = 0
            while completed < passes:
                current_chunk = min(chunk_size, passes - completed)
                repeated_inputs = inputs.repeat(
                    current_chunk, *([1] * (inputs.ndim - 1)))
                noise = torch.randn_like(repeated_inputs) * float(noise_std)
                if noise_clip > 0:
                    noise = noise.clamp(
                        -float(noise_clip), float(noise_clip))
                mc_output = model(repeated_inputs + noise)
                if isinstance(mc_output, tuple):
                    mc_output = mc_output[0]
                probability = torch.softmax(mc_output, dim=1)
                probability = probability.reshape(
                    current_chunk, batch_size, *probability.shape[1:])
                chunk_sum = probability.sum(dim=0)
                probability_sum = (
                    chunk_sum if probability_sum is None
                    else probability_sum + chunk_sum)
                completed += current_chunk
    finally:
        for module, training_state in zip(dropout_modules, previous_states):
            module.train(training_state)

    mean_probability = probability_sum / float(passes)
    return -(
        mean_probability * mean_probability.clamp_min(1e-6).log()
    ).sum(dim=1)


def get_embedding_matching_weight(iteration, max_iterations):
    """Paper Gaussian ramp-up to the mature L_NN + L_EN weight."""
    start_iteration = 1000
    if iteration < start_iteration:
        return 0.0
    ramp_length = max(1, int(max_iterations) - start_iteration)
    return args.em_loss_weight * ramps.sigmoid_rampup(
        iteration - start_iteration, ramp_length)


def get_embedding_uncertainty_threshold(iteration, max_iterations):
    """Threshold used by Embedding Matching and the official UA-MT code."""
    return (
        0.75 + 0.25 * ramps.sigmoid_rampup(iteration, max_iterations)
    ) * np.log(2.0)


# 【修复1】：替换为稳定且无警告的 EMA 更新机制，固定强动量
def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.pre_iterations

    model = UNet(in_chns=1, class_num=num_classes).cuda()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Pre-training Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre-training...")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    model.train()

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            labeled_volume_batch = volume_batch[:args.labeled_bs]
            labeled_label_batch = label_batch[:args.labeled_bs]

            outputs = model(labeled_volume_batch)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, labeled_label_batch.long())
            loss_dice = dice_loss(outputs_soft, labeled_label_batch.unsqueeze(1))
            loss = 0.5 * (loss_dice + loss_ce)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 移除衰减，保持基础学习率
            lr_ = base_lr

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            # === 验证逻辑 ===
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model,
                                                         classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))

                    # 【修复3】：同时保存模型权重和优化器动量状态
                    state_dict = {
                        'net': model.state_dict(),
                        'opt': optimizer.state_dict()
                    }
                    torch.save(state_dict, save_mode_path)
                    torch.save(state_dict, save_best_path)

                logging.info(
                    'iteration %d : mean_dice : %f, best_dice : %f' % (iter_num, performance, best_performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


def self_train(args, pre_snapshot_path, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    self_warmup_iterations = 1000

    def create_model(ema=False):
        model = UNet(in_chns=1, class_num=num_classes).cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)

    pre_trained_model = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model))


    try:
        checkpoint = torch.load(
            pre_trained_model, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(pre_trained_model, map_location='cpu')
    if 'net' in checkpoint:
        model.load_state_dict(checkpoint['net'])
        ema_model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['opt'])
        logging.info("Loaded pre-trained weights and optimizer from {}".format(pre_trained_model))
    else:
        # 兼容旧版本只存了 weight 的文件
        model.load_state_dict(checkpoint)
        ema_model.load_state_dict(checkpoint)
        logging.info("Loaded pre-trained weights from {}".format(pre_trained_model))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.eval()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    logging.info(
        "Baseline self schedule: total=%d, supervised_warmup=%d, pseudo_start=%d",
        max_iterations, self_warmup_iterations, self_warmup_iterations)
    logging.info(
        "UniMatch fusion: EMA+LCC teacher retained, tau=%.2f, "
        "feature_dropout=%.2f, branch_weights=(0.25, 0.25, 0.50), "
        "strong_aug_prob=%.2f, blur_prob=%.2f, cutmix_prob=%.2f",
        args.confidence_threshold, args.feature_dropout,
        args.strong_aug_prob, args.blur_prob, args.cutmix_prob)
    logging.info(
        "Embedding Matching (paper-faithful core): teacher-labeled -> "
        "student-unlabeled features, surface_radius=%d, k=%d, ensemble=%d, "
        "temperature=%.2f, MC_dropout=%d, MC_noise=(%.3f, clip %.3f), "
        "max_weight=%.3f, coverage_reference=%.3f",
        args.em_surface_radius, args.em_references_per_class,
        args.em_ensemble_size, args.em_temperature, args.em_mc_passes,
        args.em_mc_noise_std, args.em_mc_noise_clip,
        args.em_loss_weight, args.em_coverage_reference)

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            labeled_label_batch = label_batch[:args.labeled_bs]
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]
            zero = volume_batch.new_zeros(())
            loss_nn = zero
            loss_embedding_entropy = zero
            em_raw_weight = zero
            em_effective_weight = zero
            high_uncertainty_ratio = zero
            uncertainty_mean = zero
            uncertainty_threshold = zero
            em_stats = {
                "active_ratio": zero,
                "matching_foreground_ratio": zero,
                "teacher_disagreement_ratio": zero,
                "matching_entropy": zero,
            }
            reference_stats = {
                "foreground_reference_count": zero,
                "background_reference_count": zero,
                "reference_ready": zero,
            }

            # 【修复2】：彻底移除加噪逻辑，教师网络接收纯净输入，产出边界更清晰的伪标签
            # Preserve the supplied baseline's fixed 1000-iteration warm-up.
            if iter_num < self_warmup_iterations:
                model_output = model(volume_batch)
                outputs = model_output[0] if isinstance(model_output, tuple) else model_output
                consistency_loss = zero
                loss_u_s1 = zero
                loss_u_s2 = zero
                loss_u_fp = zero
                confident_ratio = zero
                confident_fg_ratio = zero
            else:
                strong_view1 = strong_mri_augmentation(unlabeled_volume_batch)
                strong_view2 = strong_mri_augmentation(unlabeled_volume_batch)

                unlabeled_count, _, height, width = unlabeled_volume_batch.shape
                permutation1 = torch.randperm(
                    unlabeled_count, device=unlabeled_volume_batch.device)
                permutation2 = torch.randperm(
                    unlabeled_count, device=unlabeled_volume_batch.device)
                cutmix_box1 = obtain_cutmix_boxes(
                    unlabeled_count, height, width,
                    unlabeled_volume_batch.device)
                cutmix_box2 = obtain_cutmix_boxes(
                    unlabeled_count, height, width,
                    unlabeled_volume_batch.device)
                strong_view1 = cutmix_tensor(
                    strong_view1, strong_view1[permutation1], cutmix_box1)
                strong_view2 = cutmix_tensor(
                    strong_view2, strong_view2[permutation2], cutmix_box2)

                # The corrected baseline EMA teacher and 2D largest connected
                # component pseudo masks are deliberately retained.
                with torch.no_grad():
                    ema_result = ema_model(volume_batch)
                    if not isinstance(ema_result, tuple):
                        raise RuntimeError(
                            "Embedding Matching requires EMA decoder features")
                    ema_all_output, ema_all_features = ema_result
                    teacher_labeled_features = ema_all_features[
                        :args.labeled_bs].detach()
                    ema_output = ema_all_output[args.labeled_bs:]
                    ema_probability = torch.softmax(ema_output, dim=1)
                    pseudo_labels = get_masks(ema_output, nms=1).long()
                    pseudo_confidence = ema_probability.gather(
                        1, pseudo_labels.unsqueeze(1)).squeeze(1)
                    pseudo_labels_s1 = cutmix_tensor(
                        pseudo_labels, pseudo_labels[permutation1], cutmix_box1)
                    pseudo_labels_s2 = cutmix_tensor(
                        pseudo_labels, pseudo_labels[permutation2], cutmix_box2)
                    confidence_s1 = cutmix_tensor(
                        pseudo_confidence,
                        pseudo_confidence[permutation1], cutmix_box1)
                    confidence_s2 = cutmix_tensor(
                        pseudo_confidence,
                        pseudo_confidence[permutation2], cutmix_box2)

                    # MC-Dropout uncertainty follows UA-MT.  It is used only
                    # to route pixels into Embedding Matching; the original
                    # UniMatch confidence/LCC targets above remain unchanged.
                    predictive_entropy = mc_dropout_predictive_entropy(
                        ema_model, unlabeled_volume_batch,
                        passes=args.em_mc_passes,
                        chunk_size=args.em_mc_chunk_size,
                        noise_std=args.em_mc_noise_std,
                        noise_clip=args.em_mc_noise_clip,
                        random_seed=(
                            args.seed + 104729 * (iter_num + 1)))
                    uncertainty_threshold_value = (
                        get_embedding_uncertainty_threshold(
                            iter_num, max_iterations))
                    high_uncertainty = (
                        predictive_entropy >= uncertainty_threshold_value)
                    # On top of UniMatch, matching fills only pixels that its
                    # confidence rule would otherwise ignore.  This avoids two
                    # pseudo-label sources supervising the same pixel.
                    ignored_by_unimatch = (
                        pseudo_confidence < args.confidence_threshold)
                    em_valid_mask = high_uncertainty & ignored_by_unimatch
                    high_uncertainty_ratio = (
                        high_uncertainty.float().mean())
                    uncertainty_mean = predictive_entropy.mean()
                    uncertainty_threshold = zero + uncertainty_threshold_value

                # The feature stream follows the official medical UniMatch
                # implementation and perturbs every U-Net encoder scale.
                outputs, outputs_fp, student_decoder_features = model(
                    volume_batch, need_fp=True,
                    feature_dropout=args.feature_dropout,
                    return_features=True)
                strong_output = model(
                    torch.cat((strong_view1, strong_view2), dim=0))
                if isinstance(strong_output, tuple):
                    strong_output = strong_output[0]
                outputs_s1, outputs_s2 = strong_output.chunk(2, dim=0)

                loss_u_s1, _ = confidence_masked_baseline_loss(
                    outputs_s1, pseudo_labels_s1, confidence_s1, dice_loss)
                loss_u_s2, _ = confidence_masked_baseline_loss(
                    outputs_s2, pseudo_labels_s2, confidence_s2, dice_loss)
                loss_u_fp, confident_mask = confidence_masked_baseline_loss(
                    outputs_fp[args.labeled_bs:], pseudo_labels,
                    pseudo_confidence, dice_loss)
                consistency_loss = (
                    0.25 * loss_u_s1 + 0.25 * loss_u_s2 +
                    0.50 * loss_u_fp)

                # Core paper direction: labeled embeddings come from Teacher,
                # while dense unlabeled embeddings come from Student.  L_NN is
                # applied once to the regular student logits and L_EN retains
                # the gradient through the student embedding classifier.
                (matching_probability, matching_target,
                 reference_stats) = ensemble_embedding_classifier(
                    teacher_labeled_features,
                    labeled_label_batch,
                    student_decoder_features[args.labeled_bs:],
                    surface_radius=args.em_surface_radius,
                    references_per_class=args.em_references_per_class,
                    ensemble_size=args.em_ensemble_size,
                    temperature=args.em_temperature,
                    random_seed=args.seed + 130363 * (iter_num + 1))
                (loss_nn, loss_embedding_entropy,
                 em_stats) = embedding_matching_losses(
                    outputs[args.labeled_bs:], matching_probability,
                    matching_target, em_valid_mask,
                    teacher_target=pseudo_labels)
                # Preserve the true routed coverage even if a rare batch has
                # no foreground surface reference and matching is skipped.
                em_stats['active_ratio'] = em_valid_mask.float().mean().detach()

                em_raw_weight = zero + get_embedding_matching_weight(
                    iter_num, max_iterations)
                if args.em_coverage_reference > 0:
                    coverage_scale = torch.clamp(
                        em_stats['active_ratio'] /
                        float(args.em_coverage_reference), max=1.0)
                else:
                    coverage_scale = zero + 1.0
                em_effective_weight = (
                    em_raw_weight * coverage_scale.detach())
                confident_ratio = confident_mask.float().mean()
                foreground = pseudo_labels == 1
                confident_fg_ratio = (
                    (confident_mask & foreground).float().sum() /
                    foreground.float().sum().clamp_min(1.0))

            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs[:args.labeled_bs],
                              label_batch[:][:args.labeled_bs].long())
            loss_dice = dice_loss(
                outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            loss = (
                supervised_loss + consistency_weight * consistency_loss +
                em_effective_weight *
                (loss_nn + loss_embedding_entropy))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 【修复1】：调用新的更新函数，传入固定的 ema_decay，保持强动量更新
            update_model_ema(model, ema_model, args.ema_decay)

            # 移除衰减，保持基础学习率
            lr_ = base_lr

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)
            writer.add_scalar('unimatch/loss_strong1', loss_u_s1, iter_num)
            writer.add_scalar('unimatch/loss_strong2', loss_u_s2, iter_num)
            writer.add_scalar('unimatch/loss_feature', loss_u_fp, iter_num)
            writer.add_scalar('unimatch/confident_ratio',
                              confident_ratio, iter_num)
            writer.add_scalar('unimatch/confident_foreground_ratio',
                              confident_fg_ratio, iter_num)
            writer.add_scalar('embedding/loss_nn', loss_nn, iter_num)
            writer.add_scalar('embedding/loss_entropy',
                              loss_embedding_entropy, iter_num)
            writer.add_scalar('embedding/raw_weight',
                              em_raw_weight, iter_num)
            writer.add_scalar('embedding/effective_weight',
                              em_effective_weight, iter_num)
            writer.add_scalar('embedding/uncertainty_mean',
                              uncertainty_mean, iter_num)
            writer.add_scalar('embedding/uncertainty_threshold',
                              uncertainty_threshold, iter_num)
            writer.add_scalar('embedding/high_uncertainty_ratio',
                              high_uncertainty_ratio, iter_num)
            writer.add_scalar('embedding/active_ratio',
                              em_stats['active_ratio'], iter_num)
            writer.add_scalar('embedding/matching_foreground_ratio',
                              em_stats['matching_foreground_ratio'], iter_num)
            writer.add_scalar('embedding/teacher_disagreement_ratio',
                              em_stats['teacher_disagreement_ratio'], iter_num)
            writer.add_scalar('embedding/matching_entropy',
                              em_stats['matching_entropy'], iter_num)
            writer.add_scalar('embedding/foreground_reference_count',
                              reference_stats['foreground_reference_count'], iter_num)
            writer.add_scalar('embedding/background_reference_count',
                              reference_stats['background_reference_count'], iter_num)
            writer.add_scalar('embedding/reference_ready',
                              reference_stats['reference_ready'], iter_num)

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs_img = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs_img[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
                logging.info(
                    'self iter %d/%d loss=%.6f sup=%.6f uni=%.6f '
                    's1=%.6f s2=%.6f fp=%.6f coverage=%.4f fg_coverage=%.4f '
                    'LNN=%.6f LEN=%.6f em_w=%.5f/%.5f '
                    'uncertain=%.4f active=%.4f nn_fg=%.4f disagree=%.4f '
                    'nn_entropy=%.4f refs=(%.0f,%.0f)',
                    iter_num, max_iterations, loss.item(),
                    supervised_loss.item(), consistency_loss.item(),
                    loss_u_s1.item(), loss_u_s2.item(), loss_u_fp.item(),
                    confident_ratio.item(), confident_fg_ratio.item(),
                    loss_nn.item(), loss_embedding_entropy.item(),
                    em_raw_weight.item(), em_effective_weight.item(),
                    high_uncertainty_ratio.item(),
                    em_stats['active_ratio'].item(),
                    em_stats['matching_foreground_ratio'].item(),
                    em_stats['teacher_disagreement_ratio'].item(),
                    em_stats['matching_entropy'].item(),
                    reference_stats['foreground_reference_count'].item(),
                    reference_stats['background_reference_count'].item())

            # === 验证逻辑 ===
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model,
                                                         classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                # 这句打印终于能在屏幕上看到了！
                logging.info(
                    'iteration %d : mean_dice : %f, best_dice : %f' % (iter_num, performance, best_performance))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    pre_snapshot_path = "../model/{}_{}_labeled/pre_train/{}".format(
        args.exp, args.labelnum, args.model)
    self_snapshot_path = "../model/{}_{}_labeled/self_train/{}".format(
        args.exp, args.labelnum, args.model)

    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)

    if os.path.exists(self_snapshot_path + '/code'):
        shutil.rmtree(self_snapshot_path + '/code')
    # shutil.copytree('.', self_snapshot_path + '/code',
    # shutil.ignore_patterns(['.git', '__pycache__']))

    # 【关键修复】：显式地将输出同时绑定到文件句柄和系统终端（stdout），避免被force=True抹掉
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S',
                        handlers=[logging.FileHandler(pre_snapshot_path + "/log.txt"),
                                  logging.StreamHandler(sys.stdout)],
                        force=True)
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    # 【关键修复】：再次显式绑定，确保进入自训练后依然能看到打印
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S',
                        handlers=[logging.FileHandler(self_snapshot_path + "/log.txt"),
                                  logging.StreamHandler(sys.stdout)],
                        force=True)
    logging.info("================ START SELF-TRAINING ================")
    self_train(args, pre_snapshot_path, self_snapshot_path)
