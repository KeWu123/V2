"""UtilityMatch on the fixed PROMISE12 UniMatch training protocol.

The U-Net, EMA pseudo-label path, confidence mask, feature perturbation,
optimizer, learning-rate schedule, validation, and checkpoint format are kept
identical to the current UniMatch entry. Only the two strong branches change:
four exact UniMatch candidates are ranked by their output-head pseudo-gradient
projection onto the clean labeled gradient, and the best two are trained.
"""

import argparse
import hashlib
import json
import logging
import os
import random
import sys

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.backends import cudnn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utilitymatch import (
    freeze_batchnorm_running_stats,
    gradient_projection_utility,
    head_gradient,
    select_top_candidates,
)

# train_unimatch parses arguments during import. Hide this entry point's
# arguments, then make the imported functions use UtilityMatch's locked args.
_entry_argv = sys.argv[:]
try:
    sys.argv = [sys.argv[0]]
    import train_unimatch as base
finally:
    sys.argv = _entry_argv


def build_parser():
    parser = argparse.ArgumentParser(
        description="UtilityMatch on fixed PROMISE12 35/5/10 UniMatch"
    )
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--anchor_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--max_iterations", type=int, default=30000)
    parser.add_argument("--warmup_iterations", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--labeled_bs", type=int, default=12)
    parser.add_argument("--labelnum", type=int, default=7)
    parser.add_argument("--patch_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--consistency", type=float, default=0.1)
    parser.add_argument("--consistency_rampup", type=float, default=200.0)
    parser.add_argument("--confidence_threshold", type=float, default=0.95)
    parser.add_argument("--feature_dropout", type=float, default=0.5)
    parser.add_argument("--strong_aug_prob", type=float, default=0.8)
    parser.add_argument("--blur_prob", type=float, default=0.5)
    parser.add_argument("--cutmix_prob", type=float, default=0.5)
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--selected_views", type=int, default=2)
    parser.add_argument("--utility_epsilon", type=float, default=1e-12)
    parser.add_argument("--validation_interval", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=3000)
    parser.add_argument("--log_interval", type=int, default=20)
    return parser


def model_logits(model, images):
    output = model(images)
    return output[0] if isinstance(output, tuple) else output


def read_nonempty(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_locked_protocol(args):
    expected_values = {
        "seed": 1337,
        "labelnum": 7,
        "batch_size": 24,
        "labeled_bs": 12,
        "max_iterations": 30000,
        "warmup_iterations": 1000,
        "num_classes": 2,
        "num_candidates": 4,
        "selected_views": 2,
    }
    for name, expected in expected_values.items():
        actual = int(getattr(args, name))
        if actual != expected:
            raise ValueError(
                "Locked UtilityMatch protocol requires {}={}, got {}".format(
                    name, expected, actual
                )
            )
    expected_float = {
        "base_lr": 0.01,
        "ema_decay": 0.99,
        "consistency": 0.1,
        "consistency_rampup": 200.0,
        "confidence_threshold": 0.95,
        "feature_dropout": 0.5,
        "strong_aug_prob": 0.8,
        "blur_prob": 0.5,
        "cutmix_prob": 0.5,
        "utility_epsilon": 1e-12,
    }
    for name, expected in expected_float.items():
        actual = float(getattr(args, name))
        if abs(actual - expected) > max(1e-15, abs(expected) * 1e-12):
            raise ValueError(
                "Locked UtilityMatch protocol requires {}={}, got {}".format(
                    name, expected, actual
                )
            )
    if list(args.patch_size) != [256, 256]:
        raise ValueError("Locked protocol requires patch_size 256 256")
    if args.model != "unet":
        raise ValueError("Locked protocol requires the original U-Net")

    split_cases = {}
    for split, expected_count in (("train", 35), ("val", 5), ("test", 10)):
        path = os.path.join(args.root_path, split + ".list")
        if not os.path.isfile(path):
            raise FileNotFoundError("Missing fixed split: {}".format(path))
        cases = [value.split(".")[0] for value in read_nonempty(path)]
        if len(cases) != expected_count:
            raise ValueError(
                "Expected {} {} cases, found {}".format(
                    expected_count, split, len(cases)
                )
            )
        split_cases[split] = cases
    if len(set(sum(split_cases.values(), []))) != 50:
        raise ValueError("The fixed train/val/test lists overlap")

    slice_path = os.path.join(args.root_path, "train_slices.list")
    train_slices = read_nonempty(slice_path)
    if len(train_slices) != 940:
        raise ValueError(
            "Expected 940 training slices, found {}".format(len(train_slices))
        )
    labeled_cases = split_cases["train"][:7]
    counts = {
        case: sum(name.startswith(case + "_slice") for name in train_slices)
        for case in labeled_cases
    }
    if any(value == 0 for value in counts.values()) or sum(counts.values()) != 191:
        raise ValueError(
            "Expected 191 slices across first seven cases, found {}".format(counts)
        )
    prefixes = tuple(case + "_slice" for case in labeled_cases)
    if not all(name.startswith(prefixes) for name in train_slices[:191]):
        raise ValueError(
            "train_slices.list does not begin with the fixed labeled cases"
        )
    if any(name.startswith(prefixes) for name in train_slices[191:]):
        raise ValueError("A labeled-case slice occurs after index 190")

    for path in (args.pretrained_model_path, args.anchor_checkpoint):
        if not os.path.isfile(path):
            raise FileNotFoundError("Missing original checkpoint: {}".format(path))
    return split_cases, counts


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
    torch.cuda.manual_seed(seed)
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False


def load_initial_state(model, ema_model, optimizer, checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "net" in checkpoint:
        model.load_state_dict(checkpoint["net"])
        ema_model.load_state_dict(checkpoint["net"])
        if "opt" in checkpoint:
            optimizer.load_state_dict(checkpoint["opt"])
            logging.info("Restored optimizer state from original pretrain")
    else:
        model.load_state_dict(checkpoint)
        ema_model.load_state_dict(checkpoint)


def evaluate(model, valloader, dataset_size, num_classes):
    model.eval()
    metric_list = 0.0
    with torch.no_grad():
        for sampled_batch in valloader:
            metric_i = base.val_2d.test_single_volume(
                sampled_batch["image"],
                sampled_batch["label"],
                model,
                classes=num_classes,
            )
            metric_list += np.asarray(metric_i)
    metric_list = metric_list / dataset_size
    model.train()
    return metric_list


def save_config(args, split_cases, labeled_counts):
    config = vars(args).copy()
    config.update(
        {
            "hypothesis": "H-UTILITYMATCH",
            "protocol": "PROMISE12_35_5_10_first7_191_seed1337",
            "train_cases": split_cases["train"],
            "val_cases": split_cases["val"],
            "test_cases": split_cases["test"],
            "labeled_slice_counts": labeled_counts,
            "effective_labeled_slices": sum(labeled_counts.values()),
            "labeled_index_source": "validated_train_slices.list_prefix",
            "pretrained_sha256": sha256(args.pretrained_model_path),
            "anchor_sha256": sha256(args.anchor_checkpoint),
            "base_training_entry": os.path.abspath(base.__file__),
            "utility_parameter_block": "model.decoder.out_conv",
            "candidate_bn_policy": (
                "train-mode batch statistics with running-buffer tracking disabled; "
                "selected views are re-forwarded normally"
            ),
            "interpretation_limit": (
                "one-seed optimization run with extra candidate-scoring compute; "
                "not an equal-wall-clock causal claim"
            ),
        }
    )
    with open(
        os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, indent=2, sort_keys=True)


def make_candidates(unlabeled_images, args):
    count, _, height, width = unlabeled_images.shape
    candidates = []
    for _ in range(args.num_candidates):
        images = base.strong_mri_augmentation(unlabeled_images)
        permutation = torch.randperm(count, device=unlabeled_images.device)
        boxes = base.obtain_cutmix_boxes(count, height, width, unlabeled_images.device)
        images = base.cutmix_tensor(images, images[permutation], boxes)
        candidates.append(
            {
                "images": images,
                "permutation": permutation,
                "boxes": boxes,
            }
        )
    return candidates


def attach_transported_targets(candidates, pseudo_labels, pseudo_confidence):
    for candidate in candidates:
        permutation = candidate["permutation"]
        boxes = candidate["boxes"]
        candidate["targets"] = base.cutmix_tensor(
            pseudo_labels, pseudo_labels[permutation], boxes
        )
        candidate["confidence"] = base.cutmix_tensor(
            pseudo_confidence, pseudo_confidence[permutation], boxes
        )


def rank_candidates(
    model, candidates, reference_gradient, head_parameters, dice_loss, args
):
    """Score candidates with detached features and unchanged BN buffers."""
    candidate_images = torch.cat(
        [candidate["images"] for candidate in candidates], dim=0
    )
    with torch.no_grad(), freeze_batchnorm_running_stats(model):
        candidate_output = model(candidate_images)
        if not isinstance(candidate_output, tuple) or len(candidate_output) < 2:
            raise RuntimeError("The fixed U-Net must return (logits, decoder_features)")
        detached_features = candidate_output[1].detach()

    scoring_logits = model.decoder.out_conv(detached_features)
    logits_per_candidate = scoring_logits.chunk(args.num_candidates, dim=0)
    scoring_losses = []
    for logits, candidate in zip(logits_per_candidate, candidates, strict=True):
        loss, _ = base.confidence_masked_baseline_loss(
            logits, candidate["targets"], candidate["confidence"], dice_loss
        )
        scoring_losses.append(loss)

    utilities = []
    for index, loss in enumerate(scoring_losses):
        candidate_gradient = head_gradient(
            loss,
            head_parameters,
            retain_graph=index < len(scoring_losses) - 1,
        )
        utilities.append(
            gradient_projection_utility(
                candidate_gradient,
                reference_gradient,
                epsilon=args.utility_epsilon,
            )
        )
    utilities = torch.stack(utilities)
    selected = select_top_candidates(utilities, keep=args.selected_views)
    return utilities.detach(), selected.detach()


def train(args, split_cases, labeled_counts):
    base.args = args
    num_classes = args.num_classes

    def create_model(ema=False):
        network = base.UNet(in_chns=1, class_num=num_classes).cuda()
        if ema:
            for parameter in network.parameters():
                parameter.detach_()
        return network

    model = create_model()
    ema_model = create_model(ema=True)
    optimizer = optim.SGD(
        model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001
    )
    load_initial_state(model, ema_model, optimizer, args.pretrained_model_path)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = base.BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,
        transform=transforms.Compose([base.RandomGenerator(args.patch_size)]),
    )
    db_val = base.BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = int(sum(labeled_counts.values()))
    if labeled_slice != 191:
        raise ValueError(
            "Locked UtilityMatch protocol requires exactly 191 labeled slices, got "
            "{}".format(labeled_slice)
        )
    sample_names = [str(name).strip().split(".")[0] for name in db_train.sample_list]
    labeled_prefixes = tuple(
        case + "_slice" for case in split_cases["train"][: args.labelnum]
    )
    if len(sample_names) != 940:
        raise ValueError(
            "Locked UtilityMatch protocol requires 940 training slices, got {}".format(
                len(sample_names)
            )
        )
    if not all(
        name.startswith(labeled_prefixes) for name in sample_names[:labeled_slice]
    ):
        raise ValueError(
            "The first 191 DataLoader entries are not all from the fixed seven "
            "labeled cases"
        )
    if any(
        name.startswith(labeled_prefixes) for name in sample_names[labeled_slice:]
    ):
        raise ValueError(
            "A fixed labeled-case slice occurs outside the first 191 DataLoader entries"
        )
    labeled_idxs = list(range(labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = base.TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs
    )
    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    expected_batches = labeled_slice // args.labeled_bs
    if len(labeled_idxs) != 191 or len(unlabeled_idxs) != 749:
        raise RuntimeError(
            "Sampler partition must be labeled=191/unlabeled=749, got {}/{}".format(
                len(labeled_idxs), len(unlabeled_idxs)
            )
        )
    if len(trainloader) != expected_batches or expected_batches != 15:
        raise RuntimeError(
            "Sampler must expose 15 batches per epoch for 191//12, got {}".format(
                len(trainloader)
            )
        )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    ce_loss = CrossEntropyLoss()
    dice_loss = base.losses.DiceLoss(num_classes)
    writer = SummaryWriter(os.path.join(args.output_dir, "log"))
    model.train()
    # Preserve the original supplied train-mode EMA pseudo-label behavior.
    ema_model.train()
    head_parameters = tuple(model.decoder.out_conv.parameters())
    logging.info(
        "UtilityMatch start: total_slices=%d labeled_slices=%d iterations=%d "
        "batches_per_epoch=%d",
        total_slices,
        labeled_slice,
        args.max_iterations,
        len(trainloader),
    )
    logging.info(
        "Protocol-locked sampler: labeled=191 unlabeled=749 batches=15 "
        "source=train_slices.list prefix"
    )
    logging.info(
        "Fixed UniMatch: tau=%.2f dropout=%.2f strong/blur/CutMix="
        "%.2f/%.2f/%.2f branch_weights=(0.25,0.25,0.50)",
        args.confidence_threshold,
        args.feature_dropout,
        args.strong_aug_prob,
        args.blur_prob,
        args.cutmix_prob,
    )
    logging.info(
        "UtilityMatch: candidates=%d selected=%d gradient_block="
        "decoder.out_conv candidate_BN_buffers=frozen",
        args.num_candidates,
        args.selected_views,
    )

    iter_num = 0
    best_performance = 0.0
    final_performance = 0.0
    best_iteration = 0
    max_epoch = args.max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=100, dynamic_ncols=True)
    for _ in iterator:
        for sampled_batch in trainloader:
            volume_batch = sampled_batch["image"].cuda()
            label_batch = sampled_batch["label"].cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs :]

            utility_mean = volume_batch.new_zeros(())
            utility_minimum = volume_batch.new_zeros(())
            utility_maximum = volume_batch.new_zeros(())
            selected_utility_mean = volume_batch.new_zeros(())
            utility_positive_fraction = volume_batch.new_zeros(())
            selected_cutmix_ratio = volume_batch.new_zeros(())
            selected_indices_text = "warmup"

            if iter_num < args.warmup_iterations:
                outputs = model_logits(model, volume_batch)
                outputs_soft = torch.softmax(outputs, dim=1)
                loss_ce = ce_loss(
                    outputs[: args.labeled_bs], label_batch[: args.labeled_bs].long()
                )
                loss_dice = dice_loss(
                    outputs_soft[: args.labeled_bs],
                    label_batch[: args.labeled_bs].unsqueeze(1),
                )
                supervised_loss = 0.5 * (loss_dice + loss_ce)
                consistency_loss = outputs.new_zeros(())
                loss_u_s1 = outputs.new_zeros(())
                loss_u_s2 = outputs.new_zeros(())
                loss_u_fp = outputs.new_zeros(())
                confident_ratio = outputs.new_zeros(())
                confident_fg_ratio = outputs.new_zeros(())
            else:
                candidates = make_candidates(unlabeled_volume_batch, args)
                with torch.no_grad():
                    ema_output = model_logits(ema_model, unlabeled_volume_batch)
                    ema_probability = torch.softmax(ema_output, dim=1)
                    pseudo_labels = base.get_masks(ema_output, nms=1).long()
                    pseudo_confidence = ema_probability.gather(
                        1, pseudo_labels.unsqueeze(1)
                    ).squeeze(1)
                    attach_transported_targets(
                        candidates, pseudo_labels, pseudo_confidence
                    )

                outputs, outputs_fp = model(
                    volume_batch, need_fp=True, feature_dropout=args.feature_dropout
                )
                outputs_soft = torch.softmax(outputs, dim=1)
                loss_ce = ce_loss(
                    outputs[: args.labeled_bs], label_batch[: args.labeled_bs].long()
                )
                loss_dice = dice_loss(
                    outputs_soft[: args.labeled_bs],
                    label_batch[: args.labeled_bs].unsqueeze(1),
                )
                supervised_loss = 0.5 * (loss_dice + loss_ce)
                reference_gradient = head_gradient(
                    supervised_loss, head_parameters, retain_graph=True
                )

                utilities, selected_indices = rank_candidates(
                    model,
                    candidates,
                    reference_gradient,
                    head_parameters,
                    dice_loss,
                    args,
                )
                selected_ids = selected_indices.cpu().tolist()
                selected = [candidates[index] for index in selected_ids]
                selected_indices_text = ",".join(str(index) for index in selected_ids)

                strong_output = model_logits(
                    model,
                    torch.cat([candidate["images"] for candidate in selected], dim=0),
                )
                outputs_s1, outputs_s2 = strong_output.chunk(2, dim=0)
                loss_u_s1, _ = base.confidence_masked_baseline_loss(
                    outputs_s1,
                    selected[0]["targets"],
                    selected[0]["confidence"],
                    dice_loss,
                )
                loss_u_s2, _ = base.confidence_masked_baseline_loss(
                    outputs_s2,
                    selected[1]["targets"],
                    selected[1]["confidence"],
                    dice_loss,
                )
                loss_u_fp, confident_mask = base.confidence_masked_baseline_loss(
                    outputs_fp[args.labeled_bs :],
                    pseudo_labels,
                    pseudo_confidence,
                    dice_loss,
                )
                consistency_loss = 0.25 * loss_u_s1 + 0.25 * loss_u_s2 + 0.5 * loss_u_fp
                confident_ratio = confident_mask.float().mean()
                foreground = pseudo_labels == 1
                confident_fg_ratio = (
                    confident_mask & foreground
                ).float().sum() / foreground.float().sum().clamp_min(1.0)

                utility_mean = utilities.mean()
                utility_minimum = utilities.min()
                utility_maximum = utilities.max()
                selected_utility_mean = utilities[selected_indices].mean()
                utility_positive_fraction = (utilities > 0).float().mean()
                selected_cutmix_ratio = (
                    torch.cat(
                        [
                            candidate["boxes"].flatten(1).any(dim=1)
                            for candidate in selected
                        ]
                    )
                    .float()
                    .mean()
                )

            consistency_weight = base.get_current_consistency_weight(iter_num // 150)
            total_loss = supervised_loss + consistency_weight * consistency_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    "Non-finite total loss at iteration {}".format(iter_num)
                )

            progress = min(float(iter_num) / float(args.max_iterations), 1.0)
            learning_rate = args.base_lr * (1.0 - progress) ** 0.9
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            base.update_model_ema(model, ema_model, args.ema_decay)

            iter_num += 1
            writer.add_scalar("info/lr", learning_rate, iter_num)
            writer.add_scalar("info/total_loss", total_loss, iter_num)
            writer.add_scalar("info/loss_ce", loss_ce, iter_num)
            writer.add_scalar("info/loss_dice", loss_dice, iter_num)
            writer.add_scalar("info/consistency_loss", consistency_loss, iter_num)
            writer.add_scalar("info/consistency_weight", consistency_weight, iter_num)
            writer.add_scalar("unimatch/loss_strong1", loss_u_s1, iter_num)
            writer.add_scalar("unimatch/loss_strong2", loss_u_s2, iter_num)
            writer.add_scalar("unimatch/loss_feature", loss_u_fp, iter_num)
            writer.add_scalar("unimatch/confident_ratio", confident_ratio, iter_num)
            writer.add_scalar(
                "unimatch/confident_foreground_ratio", confident_fg_ratio, iter_num
            )
            writer.add_scalar("utility/mean", utility_mean, iter_num)
            writer.add_scalar("utility/min", utility_minimum, iter_num)
            writer.add_scalar("utility/max", utility_maximum, iter_num)
            writer.add_scalar("utility/selected_mean", selected_utility_mean, iter_num)
            writer.add_scalar(
                "utility/positive_fraction", utility_positive_fraction, iter_num
            )
            writer.add_scalar(
                "utility/selected_cutmix_ratio", selected_cutmix_ratio, iter_num
            )

            iterator.set_postfix(
                iteration=iter_num,
                loss="{:.4f}".format(float(total_loss.detach())),
                utility="{:.3g}".format(float(selected_utility_mean)),
                val="{:.4f}".format(final_performance),
                best="{:.4f}".format(best_performance),
            )
            if iter_num % args.log_interval == 0:
                logging.info(
                    "iter %d/%d loss=%.6f sup=%.6f uni=%.6f s1=%.6f "
                    "s2=%.6f fp=%.6f coverage=%.4f utility="
                    "[%.6g,%.6g,%.6g] selected=%s positive=%.3f "
                    "selected_cutmix=%.3f",
                    iter_num,
                    args.max_iterations,
                    total_loss.item(),
                    supervised_loss.item(),
                    consistency_loss.item(),
                    loss_u_s1.item(),
                    loss_u_s2.item(),
                    loss_u_fp.item(),
                    confident_ratio.item(),
                    utility_minimum.item(),
                    utility_mean.item(),
                    utility_maximum.item(),
                    selected_indices_text,
                    utility_positive_fraction.item(),
                    selected_cutmix_ratio.item(),
                )

            if iter_num % args.validation_interval == 0:
                metric_list = evaluate(model, valloader, len(db_val), num_classes)
                for class_index in range(num_classes - 1):
                    writer.add_scalar(
                        "info/val_{}_dice".format(class_index + 1),
                        metric_list[class_index, 0],
                        iter_num,
                    )
                    writer.add_scalar(
                        "info/val_{}_hd95".format(class_index + 1),
                        metric_list[class_index, 1],
                        iter_num,
                    )
                final_performance = float(np.mean(metric_list, axis=0)[0])
                writer.add_scalar("info/val_mean_dice", final_performance, iter_num)
                if final_performance > best_performance:
                    best_performance = final_performance
                    best_iteration = iter_num
                    iteration_path = os.path.join(
                        args.output_dir,
                        "iter_{}_dice_{}.pth".format(
                            iter_num, round(best_performance, 4)
                        ),
                    )
                    best_path = os.path.join(
                        args.output_dir, args.model + "_best_model.pth"
                    )
                    torch.save(model.state_dict(), iteration_path)
                    torch.save(model.state_dict(), best_path)
                logging.info(
                    "validation iter=%d mean_dice=%.6f best=%.6f best_iter=%d",
                    iter_num,
                    final_performance,
                    best_performance,
                    best_iteration,
                )

            if iter_num % args.save_interval == 0:
                checkpoint_path = os.path.join(
                    args.output_dir, "iter_{}.pth".format(iter_num)
                )
                torch.save(model.state_dict(), checkpoint_path)
                logging.info("saved checkpoint %s", checkpoint_path)
            if iter_num >= args.max_iterations:
                break
        if iter_num >= args.max_iterations:
            iterator.close()
            break

    writer.close()
    summary = {
        "hypothesis": "H-UTILITYMATCH",
        "best_validation_dice": best_performance,
        "best_iteration": best_iteration,
        "final_validation_dice": final_performance,
        "iterations": iter_num,
        "num_candidates": args.num_candidates,
        "selected_views": args.selected_views,
        "labeled_slices": 191,
        "unlabeled_slices": 749,
        "batches_per_epoch": 15,
        "labeled_index_source": "validated_train_slices.list_prefix",
    }
    with open(
        os.path.join(args.output_dir, "training_summary.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    logging.info("Training completed: %s", summary)


def main(args):
    args.root_path = os.path.abspath(args.root_path)
    args.pretrained_model_path = os.path.abspath(args.pretrained_model_path)
    args.anchor_checkpoint = os.path.abspath(args.anchor_checkpoint)
    args.output_dir = os.path.abspath(args.output_dir)
    split_cases, labeled_counts = validate_locked_protocol(args)
    configure_logging(args.output_dir)
    seed_everything(args.seed, bool(args.deterministic))
    save_config(args, split_cases, labeled_counts)
    logging.info("Arguments: %s", args)
    logging.info("Labeled cases/slices: %s", labeled_counts)
    logging.info("Original pretrain SHA256: %s", sha256(args.pretrained_model_path))
    logging.info("Original anchor SHA256:   %s", sha256(args.anchor_checkpoint))
    train(args, split_cases, labeled_counts)


if __name__ == "__main__":
    main(build_parser().parse_args())
