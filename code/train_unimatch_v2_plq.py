# UniMatch V2 (A0) + Pseudo-Label Quality (PLQ) on PROMISE12.
#
# Inherits UniMatch V2 A0 (two strong views + Complementary Channel-Wise
# Dropout, EMA teacher, confidence threshold, CutMix, strong augmentation,
# optimizer, validation-test protocol) and replaces ONLY the pseudo-label
# selection:
#
#   single threshold:  conf >= 0.95 (all pixels)
#   -> class-aware:    bg: conf >= threshold_bg ; fg: conf >= threshold_fg
#                      (default 0.95 / 0.80, foreground gets a lower bar)
#   + entropy gate:    keep pixels with teacher entropy <= entropy_threshold
#                      (0 = disabled)
#
# The EMA teacher, student network, augmentations and the loss body are
# unchanged; only the validity mask that selects pseudo-labeled pixels is
# replaced.

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
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from skimage.measure import label

# from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler, count_labeled_slices)
from utils import losses, metrics, ramps
# from val_2D import test_single_volume
from utils import losses, ramps, feature_memory, contrastive_losses, val_2d


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

        # UniMatch V2 Complementary Channel-Wise Dropout sampler. The dropout
        # ratio is controlled by --comp_drop_dropout_prob on the command line.
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)
        self.comp_drop_dropout_prob = args.comp_drop_dropout_prob

    def forward_projection_head(self, features):
        return self.projection_head(features)

    def forward_prediction_head(self, features):
        return self.prediction_head(features)

    def forward(self, x, comp_drop=False):
        feature = self.encoder(x)
        if comp_drop:
            # UniMatch V2 Complementary Channel-Wise Dropout.
            # x is [strong_view1(bs); strong_view2(bs)]. mask1/mask2 are two
            # complementary channel masks with values in {0, 2} (mask2 ==
            # 2 - mask1, so mask1 + mask2 == 2: every channel is used by
            # exactly one view, scaled by 2 to compensate the 50% drop). For a
            # random (1 - dropout_prob) fraction of the sample pairs, both
            # views keep all channels (mask == 1). Both views are decoded
            # together through the shared decoder; the caller chunks the
            # returned logits back into the two views.
            total, dim = feature[4].shape[0], feature[4].shape[1]
            if total % 2 != 0:
                raise ValueError(
                    "comp_drop requires an even batch size, got {}".format(total))
            half = total // 2
            mask1 = self.binomial.sample((half, dim)).to(x.device) * 2.0
            mask2 = 2.0 - mask1
            num_kept = int(half * (1.0 - self.comp_drop_dropout_prob))
            kept_indexes = torch.randperm(half, device=x.device)[:num_kept]
            mask1[kept_indexes, :] = 1.0
            mask2[kept_indexes, :] = 1.0
            mask = torch.cat((mask1, mask2))
            feature[4] = feature[4] * mask.unsqueeze(-1).unsqueeze(-1)
            output, _ = self.decoder(feature)
            return output
        output, features = self.decoder(feature)
        return output, features


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default=os.path.abspath(os.path.join(
                        os.path.dirname(__file__), '..', 'data', 'PROMISE12_h5')),
                    help='dataset root path')
parser.add_argument('--exp', type=str,
                    default='UniMatchV2_PLQ_label7_seed1337', help='experiment_name')
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
parser.add_argument('--ema_teacher_mode', type=str, default='eval',
                    choices=('eval', 'train'),
                    help='EMA teacher forward mode during self-training')
parser.add_argument('--skip_pretrain', action='store_true',
                    help='reuse the existing pretrain best checkpoint')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--confidence_threshold', type=float, default=0.95,
                    help='UniMatch pseudo-label confidence threshold')
parser.add_argument('--feature_dropout', type=float, default=0.5,
                    help='[V1 only, unused in V2 A0] channel dropout probability')
parser.add_argument('--comp_drop_dropout_prob', type=float, default=0.5,
                    help='UniMatch V2 complementary dropout: probability that a '
                         'sample pair drops complementary channels (1 - this '
                         'value = fraction of pairs that keep all channels)')
parser.add_argument('--strong_aug_prob', type=float, default=0.8,
                    help='probability of MRI brightness/contrast perturbation')
parser.add_argument('--blur_prob', type=float, default=0.5,
                    help='probability of Gaussian blur per strong view')
parser.add_argument('--cutmix_prob', type=float, default=0.5,
                    help='probability of CutMix per sample and strong view')
# Pseudo-label quality (PLQ) controls. Replaces the single fixed threshold
# with class-aware thresholds + optional teacher-entropy gate.
parser.add_argument('--threshold_bg', type=float, default=0.95,
                    help='PLQ: confidence threshold for background pseudo labels')
parser.add_argument('--threshold_fg', type=float, default=0.80,
                    help='PLQ: confidence threshold for foreground pseudo labels '
                         '(lower bar because the prostate foreground is tiny)')
parser.add_argument('--entropy_threshold', type=float, default=0.0,
                    help='PLQ: keep pseudo labels with teacher entropy <= this '
                         'value (natural log, 0 = disabled)')
args = parser.parse_args()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "PROMISE12" in dataset:
        return count_labeled_slices(dataset, patiens_num)
    elif "ACDC" in dataset:
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


def compute_plq_valid(pseudo_confidence, pseudo_labels, entropy):
    """Class-aware + entropy pseudo-label validity mask.

    valid = (label==1 ? conf>=thr_fg : conf>=thr_bg)  &  (entropy<=thr_ent or off)
    """
    foreground = pseudo_labels == 1
    per_pixel_thr = torch.where(
        foreground, torch.full_like(pseudo_confidence, args.threshold_fg),
        torch.full_like(pseudo_confidence, args.threshold_bg))
    valid = pseudo_confidence >= per_pixel_thr
    if args.entropy_threshold > 0:
        valid = valid & (entropy <= args.entropy_threshold)
    return valid


def masked_plq_loss(logits, targets, valid, dice_loss):
    """CE+Dice pseudo loss using a precomputed validity mask (PLQ)."""
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


# The UniMatch experiment inherits the EMA update from the supplied baseline.
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
    if args.ema_teacher_mode == 'train':
        ema_model.train()
    else:
        ema_model.eval()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    logging.info(
        "UniMatch V2 A0 + PLQ: EMA+LCC teacher retained, "
        "threshold_bg=%.2f, threshold_fg=%.2f, entropy_threshold=%.3f, "
        "comp_drop_dropout_prob=%.2f, branch_weights=(0.50, 0.50), "
        "strong_aug_prob=%.2f, blur_prob=%.2f, cutmix_prob=%.2f, "
        "ema_teacher_mode=%s",
        args.threshold_bg, args.threshold_fg, args.entropy_threshold,
        args.comp_drop_dropout_prob,
        args.strong_aug_prob, args.blur_prob, args.cutmix_prob,
        args.ema_teacher_mode)

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            if iter_num < 1000:
                model_output = model(volume_batch)
                outputs = model_output[0] if isinstance(model_output, tuple) else model_output
                consistency_loss = outputs.new_zeros(())
                loss_u_s1 = outputs.new_zeros(())
                loss_u_s2 = outputs.new_zeros(())
                confident_ratio = outputs.new_zeros(())
                confident_fg_ratio = outputs.new_zeros(())
                entropy_mean = 0.0
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

                with torch.no_grad():
                    ema_output = ema_model(unlabeled_volume_batch)
                    if isinstance(ema_output, tuple):
                        ema_output = ema_output[0]
                    ema_probability = torch.softmax(ema_output, dim=1)
                    pseudo_labels = get_masks(ema_output, nms=1).long()
                    pseudo_confidence = ema_probability.gather(
                        1, pseudo_labels.unsqueeze(1)).squeeze(1)
                    # PLQ: per-pixel Shannon entropy of the teacher
                    # distribution (natural log; binary class -> [0, log 2]).
                    log_prob = torch.log(ema_probability.clamp_min(1e-10))
                    entropy = -(ema_probability * log_prob).sum(dim=1)
                    # PLQ: class-aware thresholds + optional entropy gate.
                    valid_mask = compute_plq_valid(
                        pseudo_confidence, pseudo_labels, entropy)
                    pseudo_labels_s1 = cutmix_tensor(
                        pseudo_labels, pseudo_labels[permutation1], cutmix_box1)
                    pseudo_labels_s2 = cutmix_tensor(
                        pseudo_labels, pseudo_labels[permutation2], cutmix_box2)
                    valid_s1 = cutmix_tensor(
                        valid_mask, valid_mask[permutation1], cutmix_box1)
                    valid_s2 = cutmix_tensor(
                        valid_mask, valid_mask[permutation2], cutmix_box2)
                    entropy_mean = entropy.mean().item()

                # UniMatch V2 A0: the two strong views are decoded together in
                # one shared-decoder forward with Complementary Channel-Wise
                # Dropout, then split back into the two views.
                outputs = model(volume_batch)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                strong_output = model(
                    torch.cat((strong_view1, strong_view2), dim=0),
                    comp_drop=True)
                outputs_s1, outputs_s2 = strong_output.chunk(2, dim=0)

                loss_u_s1, _ = masked_plq_loss(
                    outputs_s1, pseudo_labels_s1, valid_s1, dice_loss)
                loss_u_s2, confident_mask = masked_plq_loss(
                    outputs_s2, pseudo_labels_s2, valid_s2, dice_loss)
                consistency_loss = (
                    0.5 * loss_u_s1 + 0.5 * loss_u_s2)
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

            loss = supervised_loss + consistency_weight * consistency_loss

            # Keep the restored SGD state and momentum unchanged. Only the
            # self-training learning rate follows UniMatch's polynomial decay.
            progress = min(float(iter_num) / float(max_iterations), 1.0)
            lr_ = base_lr * (1.0 - progress) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_model_ema(model, ema_model, args.ema_decay)

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)
            writer.add_scalar('v2/loss_view1', loss_u_s1, iter_num)
            writer.add_scalar('v2/loss_view2', loss_u_s2, iter_num)
            writer.add_scalar('v2/confident_ratio',
                              confident_ratio, iter_num)
            writer.add_scalar('v2/confident_foreground_ratio',
                              confident_fg_ratio, iter_num)

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
                    's1=%.6f s2=%.6f coverage=%.4f fg_coverage=%.4f '
                    'entropy=%.4f',
                    iter_num, max_iterations, loss.item(),
                    supervised_loss.item(), consistency_loss.item(),
                    loss_u_s1.item(), loss_u_s2.item(),
                    confident_ratio.item(), confident_fg_ratio.item(),
                    entropy_mean)

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
    pre_trained_model = os.path.join(
        pre_snapshot_path, '{}_best_model.pth'.format(args.model))
    if args.skip_pretrain:
        if not os.path.isfile(pre_trained_model):
            raise FileNotFoundError(
                "--skip_pretrain requires an existing checkpoint: {}".format(
                    pre_trained_model))
        logging.info("Skipping pre-training and reusing %s", pre_trained_model)
    else:
        pre_train(args, pre_snapshot_path)

    # 【关键修复】：再次显式绑定，确保进入自训练后依然能看到打印
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S',
                        handlers=[logging.FileHandler(self_snapshot_path + "/log.txt"),
                                  logging.StreamHandler(sys.stdout)],
                        force=True)
    logging.info("================ START SELF-TRAINING ================")
    self_train(args, pre_snapshot_path, self_snapshot_path)
