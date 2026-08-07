"""Stable v2 refinement with coverage-aware temporal-volume supervision."""

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.dataset import TwoStreamBatchSampler
from networks.unet import UNet
from temporal_volume_bank import (
    TemporalBankDataset,
    build_temporal_bank,
    extract_state_dict,
    load_temporal_bank,
    refresh_temporal_bank,
    save_temporal_bank,
)
from utils import losses, val_2d


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--unimatch_checkpoint", type=str, required=True)
    parser.add_argument("--history_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--refine_iterations", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--labeled_bs", type=int, default=12)
    parser.add_argument("--labelnum", type=int, default=7)
    parser.add_argument("--patch_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--base_lr", type=float, default=0.001)
    parser.add_argument("--lr_floor_ratio", type=float, default=0.1)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--feature_dropout", type=float, default=0.5)
    parser.add_argument("--strong_aug_prob", type=float, default=0.8)
    parser.add_argument("--blur_prob", type=float, default=0.5)
    parser.add_argument("--cutmix_prob", type=float, default=0.5)
    parser.add_argument("--hard_confidence", type=float, default=0.95)
    parser.add_argument("--hard_reliability", type=float, default=0.75)
    parser.add_argument("--soft_confidence", type=float, default=0.65)
    parser.add_argument("--soft_reliability", type=float, default=0.55)
    parser.add_argument("--boundary_soft_weight", type=float, default=0.25)
    parser.add_argument("--boundary_reference_fraction", type=float, default=0.05)
    parser.add_argument("--consistency_max", type=float, default=0.25)
    parser.add_argument("--consistency_rampup", type=int, default=1000)
    parser.add_argument("--case_fraction_start", type=float, default=0.4)
    parser.add_argument("--case_fraction_end", type=float, default=0.60)
    parser.add_argument("--history_count", type=int, default=3)
    parser.add_argument("--mc_passes", type=int, default=8)
    parser.add_argument("--bank_batch_size", type=int, default=12)
    parser.add_argument("--bank_refresh_interval", type=int, default=1000)
    parser.add_argument("--bank_update_margin", type=float, default=0.02)
    parser.add_argument("--bank_temporal_decay", type=float, default=0.8)
    parser.add_argument("--rebuild_bank", action="store_true")
    parser.add_argument("--validation_interval", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=1000)
    return parser


def configure_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    formatter = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S"
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)
    file_handler = logging.FileHandler(
        os.path.join(output_dir, "train.log"), mode="a", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False


def read_list(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def count_labeled_slices(root_path, labelnum):
    cases = read_list(os.path.join(root_path, "train.list"))[: int(labelnum)]
    if len(cases) != int(labelnum):
        raise ValueError("train.list does not contain {} labeled cases".format(labelnum))
    names = read_list(os.path.join(root_path, "train_slices.list"))
    prefixes = tuple(case + "_slice" for case in cases)
    count = 0
    for name in names:
        if name.startswith(prefixes):
            count += 1
        elif count:
            break
    if count <= 0:
        raise ValueError("No labeled slices found for {}".format(cases))
    return count


def load_raw_state(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state = extract_state_dict(checkpoint)
    return {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in state.items()
    }


def model_logits(model, images):
    output = model(images)
    return output[0] if isinstance(output, tuple) else output


def forward_with_feature_dropout(model, images, dropout_probability):
    features = model.encoder(images)
    paired_features = [
        torch.cat(
            (
                feature,
                F.dropout2d(
                    feature,
                    p=float(dropout_probability),
                    training=True,
                ),
            ),
            dim=0,
        )
        for feature in features
    ]
    paired_logits, _ = model.decoder(paired_features)
    return paired_logits.chunk(2, dim=0)


def gaussian_blur_2d(image, sigma):
    radius = max(1, int(round(2.0 * sigma)))
    size = 2 * radius + 1
    coordinates = (
        torch.arange(size, device=image.device, dtype=image.dtype) - radius
    )
    kernel_1d = torch.exp(-(coordinates ** 2) / (2.0 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.expand(image.shape[0], 1, -1, -1)
    return F.conv2d(
        image.unsqueeze(0),
        kernel,
        padding=radius,
        groups=image.shape[0],
    ).squeeze(0)


def strong_mri_augmentation(images, args):
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


def obtain_cutmix_boxes(batch_size, height, width, device, probability):
    boxes = torch.zeros(
        (batch_size, height, width), dtype=torch.bool, device=device
    )
    for index in range(batch_size):
        if random.random() > float(probability):
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


def masked_hard_loss(logits, targets, valid_mask, dice_loss):
    valid = valid_mask.float()
    if float(valid.sum().detach()) < 1.0:
        return logits.sum() * 0.0
    cross_entropy = F.cross_entropy(logits, targets.long(), reduction="none")
    cross_entropy = (cross_entropy * valid).sum() / valid.sum().clamp_min(1.0)
    dice = dice_loss(
        torch.softmax(logits, dim=1),
        targets.unsqueeze(1),
        mask=valid.unsqueeze(1),
    )
    return 0.5 * (cross_entropy + dice)


def masked_soft_loss(logits, target_probability, valid_mask):
    valid = valid_mask.float()
    if float(valid.sum().detach()) < 1.0:
        return logits.sum() * 0.0
    foreground_logit = logits[:, 1] - logits[:, 0]
    loss = F.binary_cross_entropy_with_logits(
        foreground_logit,
        target_probability.float(),
        reduction="none",
    )
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def routed_bank_loss(
    logits,
    target_probability,
    hard_mask,
    soft_mask,
    dice_loss,
    boundary_soft_weight,
    boundary_reference_fraction,
):
    hard_target = (target_probability >= 0.5).long()
    hard = masked_hard_loss(logits, hard_target, hard_mask, dice_loss)
    soft = masked_soft_loss(logits, target_probability, soft_mask)

    # A tiny boundary mask must not receive the same aggregate influence as
    # the much larger reliable core. Scale its independently normalized mean
    # loss by its coverage relative to a 5%-of-core reference region.
    hard_count = hard_mask.float().sum().detach()
    soft_count = soft_mask.float().sum().detach()
    reference_count = (
        float(boundary_reference_fraction) * hard_count
    ).clamp_min(1.0)
    soft_scale = (soft_count / reference_count).clamp(0.0, 1.0)
    combined = hard + float(boundary_soft_weight) * soft_scale * soft
    return combined, hard, soft, soft_scale


def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    ema_state = ema_model.state_dict()
    updated = {}
    for key in model_state:
        if torch.is_floating_point(model_state[key]):
            updated[key] = (
                float(alpha) * ema_state[key]
                + (1.0 - float(alpha)) * model_state[key]
            )
        else:
            updated[key] = model_state[key]
    ema_model.load_state_dict(updated)


def refinement_optimizer_step(
    model,
    optimizer,
    supervised_loss,
    unsupervised_loss,
    consistency_weight,
):
    """Default v2 optimization step.

    Kept as a small hook so an independent experiment can replace only the
    gradient-composition rule without duplicating or changing the temporal
    bank, data protocol, augmentation, validation, or checkpoint logic.
    """
    total_loss = (
        supervised_loss + float(consistency_weight) * unsupervised_loss
    )
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return total_loss, {}


def sigmoid_rampup(iteration, rampup_length):
    if rampup_length <= 0:
        return 1.0
    current = np.clip(float(iteration), 0.0, float(rampup_length))
    phase = 1.0 - current / float(rampup_length)
    return float(np.exp(-5.0 * phase * phase))


@torch.no_grad()
def evaluate(model, loader, num_classes=2):
    was_training = model.training
    model.eval()
    metric_list = 0.0
    for sampled_batch in loader:
        metric = val_2d.test_single_volume(
            sampled_batch["image"],
            sampled_batch["label"],
            model,
            classes=num_classes,
        )
        metric_list += np.asarray(metric)
    metric_list = metric_list / len(loader.dataset)
    if was_training:
        model.train()
    return float(np.mean(metric_list, axis=0)[0]), metric_list


def main(args):
    args.root_path = os.path.abspath(args.root_path)
    args.unimatch_checkpoint = os.path.abspath(args.unimatch_checkpoint)
    args.history_dir = os.path.abspath(args.history_dir) if args.history_dir else ""
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    configure_logging(args.output_dir)
    seed_everything(args.seed, args.deterministic)
    logging.info("Arguments: %s", args)

    if not os.path.isfile(args.unimatch_checkpoint):
        raise FileNotFoundError(args.unimatch_checkpoint)
    device = torch.device("cuda")
    state = load_raw_state(args.unimatch_checkpoint)
    model = UNet(in_chns=1, class_num=2).to(device)
    ema_model = UNet(in_chns=1, class_num=2).to(device)
    model.load_state_dict(state, strict=True)
    ema_model.load_state_dict(state, strict=True)
    for parameter in ema_model.parameters():
        parameter.detach_()
    ema_model.eval()

    train_names = read_list(os.path.join(args.root_path, "train_slices.list"))
    labeled_slice_count = count_labeled_slices(args.root_path, args.labelnum)
    unlabeled_names = train_names[labeled_slice_count:]
    logging.info(
        "Protocol: train_slices=%d labeled_cases=%d labeled_slices=%d unlabeled_slices=%d",
        len(train_names),
        args.labelnum,
        labeled_slice_count,
        len(unlabeled_names),
    )

    bank_dir = os.path.join(args.output_dir, "pseudo_bank")
    os.makedirs(bank_dir, exist_ok=True)
    bank_path = os.path.join(bank_dir, "temporal_bank.pth")
    if os.path.isfile(bank_path) and not args.rebuild_bank:
        bank = load_temporal_bank(bank_path)
        if os.path.abspath(bank.get("anchor_checkpoint", "")) != args.unimatch_checkpoint:
            raise RuntimeError(
                "Existing bank was built from a different anchor checkpoint. "
                "Use --rebuild_bank or a new output directory."
            )
        logging.info("Loaded existing temporal bank: %s", bank_path)
    else:
        bank = build_temporal_bank(
            root_path=args.root_path,
            sample_names=unlabeled_names,
            primary_checkpoint=args.unimatch_checkpoint,
            history_dir=args.history_dir,
            history_count=args.history_count,
            mc_passes=args.mc_passes,
            patch_size=args.patch_size,
            device=device,
            inference_batch_size=args.bank_batch_size,
            logger=logging.getLogger(),
        )
        save_temporal_bank(bank, bank_path)
        logging.info("Saved initial temporal bank: %s", bank_path)

    dataset = TemporalBankDataset(
        args.root_path,
        bank,
        labeled_slice_count,
        patch_size=args.patch_size,
    )
    labeled_indices = list(range(labeled_slice_count))
    unlabeled_indices = list(range(labeled_slice_count, len(dataset)))
    sampler = TwoStreamBatchSampler(
        labeled_indices,
        unlabeled_indices,
        args.batch_size,
        args.batch_size - args.labeled_bs,
    )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        np.random.seed(args.seed + worker_id)

    # num_workers=0 is intentional: conservative bank refreshes must become
    # visible to the dataset without stale worker copies.
    train_loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    from dataloaders.dataset import BaseDataSets

    val_dataset = BaseDataSets(base_dir=args.root_path, split="val")
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=1
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=0.0001,
    )
    logging.info(
        "Refinement optimizer: new SGD state, lr=%.6g momentum=0.9 wd=1e-4; "
        "the external UniMatch checkpoint contains weights only.",
        args.base_lr,
    )
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(2)
    writer = SummaryWriter(os.path.join(args.output_dir, "log"))

    best_path = os.path.join(args.output_dir, "{}_best_model.pth".format(args.model))
    initial_path = os.path.join(args.output_dir, "unimatch_anchor.pth")
    torch.save(model.state_dict(), initial_path)
    initial_performance, _ = evaluate(model, val_loader)
    best_performance = initial_performance
    torch.save(model.state_dict(), best_path)
    logging.info(
        "Initial fixed UniMatch val_dice=%.6f; best checkpoint initialized from anchor",
        initial_performance,
    )

    model.train()
    iter_num = 0
    last_bank_refresh_iter = 0
    max_epoch = args.refine_iterations // len(train_loader) + 1
    iterator = tqdm(range(max_epoch), ncols=90, desc="Temporal-bank refinement")

    for _ in iterator:
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            bank_probability = batch["bank_probability"].to(
                device, non_blocking=True
            )[args.labeled_bs:]
            bank_reliability = batch["bank_reliability"].to(
                device, non_blocking=True
            )[args.labeled_bs:]
            bank_boundary = batch["bank_boundary"].to(
                device, non_blocking=True
            )[args.labeled_bs:] > 0.5
            case_percentile = batch["case_percentile"].to(
                device, non_blocking=True
            )[args.labeled_bs:]

            progress = min(
                float(iter_num) / float(max(args.refine_iterations, 1)), 1.0
            )
            case_fraction = (
                args.case_fraction_start
                + (args.case_fraction_end - args.case_fraction_start) * progress
            )
            case_keep = (
                case_percentile <= case_fraction
            ).view(-1, 1, 1)
            confidence = torch.maximum(
                bank_probability, 1.0 - bank_probability
            )
            hard_mask = (
                case_keep
                & (~bank_boundary)
                & (confidence >= args.hard_confidence)
                & (bank_reliability >= args.hard_reliability)
            )
            soft_mask = (
                case_keep
                & bank_boundary
                & (confidence >= args.soft_confidence)
                & (bank_reliability >= args.soft_reliability)
            )

            unlabeled_images = images[args.labeled_bs:]
            strong_view1 = strong_mri_augmentation(unlabeled_images, args)
            strong_view2 = strong_mri_augmentation(unlabeled_images, args)
            count, _, height, width = unlabeled_images.shape
            permutation1 = torch.randperm(count, device=device)
            permutation2 = torch.randperm(count, device=device)
            boxes1 = obtain_cutmix_boxes(
                count, height, width, device, args.cutmix_prob
            )
            boxes2 = obtain_cutmix_boxes(
                count, height, width, device, args.cutmix_prob
            )
            strong_view1 = cutmix_tensor(
                strong_view1, strong_view1[permutation1], boxes1
            )
            strong_view2 = cutmix_tensor(
                strong_view2, strong_view2[permutation2], boxes2
            )

            probability1 = cutmix_tensor(
                bank_probability, bank_probability[permutation1], boxes1
            )
            probability2 = cutmix_tensor(
                bank_probability, bank_probability[permutation2], boxes2
            )
            hard_mask1 = cutmix_tensor(
                hard_mask, hard_mask[permutation1], boxes1
            )
            hard_mask2 = cutmix_tensor(
                hard_mask, hard_mask[permutation2], boxes2
            )
            soft_mask1 = cutmix_tensor(
                soft_mask, soft_mask[permutation1], boxes1
            )
            soft_mask2 = cutmix_tensor(
                soft_mask, soft_mask[permutation2], boxes2
            )

            outputs, outputs_fp = forward_with_feature_dropout(
                model, images, args.feature_dropout
            )
            strong_logits = model_logits(
                model, torch.cat((strong_view1, strong_view2), dim=0)
            )
            outputs_s1, outputs_s2 = strong_logits.chunk(2, dim=0)

            loss_s1, hard_s1, soft_s1, soft_scale_s1 = routed_bank_loss(
                outputs_s1,
                probability1,
                hard_mask1,
                soft_mask1,
                dice_loss,
                args.boundary_soft_weight,
                args.boundary_reference_fraction,
            )
            loss_s2, hard_s2, soft_s2, soft_scale_s2 = routed_bank_loss(
                outputs_s2,
                probability2,
                hard_mask2,
                soft_mask2,
                dice_loss,
                args.boundary_soft_weight,
                args.boundary_reference_fraction,
            )
            loss_fp, hard_fp, soft_fp, soft_scale_fp = routed_bank_loss(
                outputs_fp[args.labeled_bs:],
                bank_probability,
                hard_mask,
                soft_mask,
                dice_loss,
                args.boundary_soft_weight,
                args.boundary_reference_fraction,
            )
            unsupervised_loss = (
                0.25 * loss_s1 + 0.25 * loss_s2 + 0.50 * loss_fp
            )

            supervised_ce = ce_loss(
                outputs[:args.labeled_bs],
                labels[:args.labeled_bs].long(),
            )
            supervised_dice = dice_loss(
                torch.softmax(outputs[:args.labeled_bs], dim=1),
                labels[:args.labeled_bs].unsqueeze(1),
            )
            supervised_loss = 0.5 * (supervised_ce + supervised_dice)
            consistency_weight = (
                args.consistency_max
                * sigmoid_rampup(iter_num, args.consistency_rampup)
            )
            decay = (
                args.lr_floor_ratio
                + (1.0 - args.lr_floor_ratio)
                * (1.0 - progress) ** 0.9
            )
            learning_rate = args.base_lr * decay
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            total_loss, optimization_stats = refinement_optimizer_step(
                model,
                optimizer,
                supervised_loss,
                unsupervised_loss,
                consistency_weight,
            )
            update_model_ema(model, ema_model, args.ema_decay)
            iter_num += 1

            hard_coverage = hard_mask.float().mean()
            soft_coverage = soft_mask.float().mean()
            selected_case_ratio = case_keep.float().mean()
            writer.add_scalar("loss/total", total_loss.item(), iter_num)
            writer.add_scalar("loss/supervised", supervised_loss.item(), iter_num)
            writer.add_scalar("loss/unsupervised", unsupervised_loss.item(), iter_num)
            writer.add_scalar("loss/hard", hard_fp.item(), iter_num)
            writer.add_scalar("loss/soft", soft_fp.item(), iter_num)
            writer.add_scalar("loss/soft_scale", soft_scale_fp.item(), iter_num)
            writer.add_scalar("bank/hard_coverage", hard_coverage.item(), iter_num)
            writer.add_scalar("bank/soft_coverage", soft_coverage.item(), iter_num)
            writer.add_scalar(
                "bank/selected_case_ratio", selected_case_ratio.item(), iter_num
            )
            writer.add_scalar("bank/case_fraction", case_fraction, iter_num)
            writer.add_scalar("train/lr", learning_rate, iter_num)
            writer.add_scalar(
                "train/consistency_weight", consistency_weight, iter_num
            )
            for name, value in optimization_stats.items():
                writer.add_scalar(
                    "gradient_composition/{}".format(name), value, iter_num
                )

            if iter_num % 20 == 0:
                logging.info(
                    "refine iter %d/%d loss=%.6f sup=%.6f unsup=%.6f "
                    "hard=%.6f soft=%.6f soft_scale=%.4f "
                    "hard_cov=%.4f soft_cov=%.4f case_keep=%.4f lr=%.7f",
                    iter_num,
                    args.refine_iterations,
                    total_loss.item(),
                    supervised_loss.item(),
                    unsupervised_loss.item(),
                    hard_fp.item(),
                    soft_fp.item(),
                    soft_scale_fp.item(),
                    hard_coverage.item(),
                    soft_coverage.item(),
                    selected_case_ratio.item(),
                    learning_rate,
                )
                if optimization_stats:
                    logging.info(
                        "POS/MEO alpha_sup=%.4f alpha_unsup=%.4f "
                        "cosine=%.4f conflict_rate=%.4f "
                        "norm_sup=%.6f norm_unsup=%.6f "
                        "norm_uniform=%.6f norm_final=%.6f scale=%.4f",
                        optimization_stats["alpha_supervised"],
                        optimization_stats["alpha_unsupervised"],
                        optimization_stats["gradient_cosine"],
                        optimization_stats["conflict_rate"],
                        optimization_stats["norm_supervised"],
                        optimization_stats["norm_unsupervised"],
                        optimization_stats["norm_uniform"],
                        optimization_stats["norm_final"],
                        optimization_stats["meo_scale"],
                    )

            if iter_num % args.validation_interval == 0:
                performance, _ = evaluate(model, val_loader)
                writer.add_scalar("validation/dice", performance, iter_num)
                is_current_new_best = performance > best_performance
                if is_current_new_best:
                    best_performance = performance
                    torch.save(model.state_dict(), best_path)
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            args.output_dir,
                            "iter_{}_dice_{:.4f}.pth".format(
                                iter_num, best_performance
                            ),
                        ),
                    )
                logging.info(
                    "refine iter %d val_dice=%.6f best_dice=%.6f anchor_dice=%.6f",
                    iter_num,
                    performance,
                    best_performance,
                    initial_performance,
                )
                model.train()

                # Refresh only from the EMA corresponding to this exact new
                # validation best. A permission is never carried into a later,
                # degraded checkpoint. Enforce a minimum interval between
                # expensive whole-bank refreshes.
                if (
                    is_current_new_best
                    and args.bank_refresh_interval > 0
                    and iter_num - last_bank_refresh_iter
                    >= args.bank_refresh_interval
                ):
                    accepted_fraction = refresh_temporal_bank(
                        bank=bank,
                        model=ema_model,
                        root_path=args.root_path,
                        sample_names=unlabeled_names,
                        patch_size=args.patch_size,
                        device=device,
                        mc_passes=args.mc_passes,
                        update_margin=args.bank_update_margin,
                        temporal_decay=args.bank_temporal_decay,
                        inference_batch_size=args.bank_batch_size,
                        logger=logging.getLogger(),
                    )
                    refresh_path = os.path.join(
                        bank_dir,
                        "temporal_bank_iter_{}.pth".format(iter_num),
                    )
                    save_temporal_bank(bank, refresh_path)
                    save_temporal_bank(bank, bank_path)
                    writer.add_scalar(
                        "bank/refresh_accepted_fraction",
                        accepted_fraction,
                        iter_num,
                    )
                    last_bank_refresh_iter = iter_num

            if iter_num % args.save_interval == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(args.output_dir, "iter_{}.pth".format(iter_num)),
                )
                torch.save(
                    {
                        "iter": iter_num,
                        "net": model.state_dict(),
                        "ema": ema_model.state_dict(),
                        "opt": optimizer.state_dict(),
                        "best_performance": best_performance,
                        "anchor_performance": initial_performance,
                    },
                    os.path.join(args.output_dir, "last_checkpoint.pth"),
                )

            if iter_num >= args.refine_iterations:
                break
        if iter_num >= args.refine_iterations:
            iterator.close()
            break

    torch.save(model.state_dict(), os.path.join(args.output_dir, "unet_last_model.pth"))
    writer.close()
    logging.info(
        "Temporal-bank refinement completed: iter=%d anchor_val=%.6f best_val=%.6f",
        iter_num,
        initial_performance,
        best_performance,
    )
    logging.info("Best checkpoint: %s", best_path)


if __name__ == "__main__":
    main(build_parser().parse_args())
