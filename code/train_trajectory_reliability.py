"""Trajectory-reliability controlled UniMatch self-training on PROMISE12."""

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import train_utilitymatch as common
from trajectory_reliability import (
    TeacherTrajectory,
    adaptive_unsupervised_scale,
    soft_boundary_target,
    trajectory_statistics,
    weighted_pseudo_loss,
    weighted_soft_binary_loss,
)


base = common.base
MODES = ("baseline", "weighting", "adaptive", "weighting_adaptive", "full")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Trajectory-aware reliability controlled UniMatch"
    )
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--anchor_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
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
    parser.add_argument("--history_size", type=int, default=4)
    parser.add_argument("--history_interval", type=int, default=200)
    parser.add_argument("--stable_threshold", type=float, default=0.75)
    parser.add_argument("--minimum_pixel_weight", type=float, default=0.05)
    parser.add_argument("--readiness_start", type=float, default=0.45)
    parser.add_argument("--readiness_full", type=float, default=0.80)
    parser.add_argument("--minimum_unsupervised_scale", type=float, default=0.02)
    parser.add_argument("--readiness_momentum", type=float, default=0.90)
    parser.add_argument("--boundary_radius", type=int, default=2)
    parser.add_argument("--boundary_recovery_strength", type=float, default=0.50)
    parser.add_argument("--boundary_minimum_weight", type=float, default=0.10)
    parser.add_argument("--boundary_loss_weight", type=float, default=0.10)
    parser.add_argument("--validation_interval", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=3000)
    parser.add_argument("--log_interval", type=int, default=20)

    # Values retained only so the shared locked-protocol validator can verify
    # that data and the original UniMatch recipe have not drifted.
    parser.set_defaults(num_candidates=4, selected_views=2, utility_epsilon=1e-12)
    return parser


def mode_flags(mode):
    return {
        "weighting": mode in ("weighting", "weighting_adaptive", "full"),
        "adaptive": mode in ("adaptive", "weighting_adaptive", "full"),
        "boundary": mode == "full",
    }


def validate_trajectory_args(args):
    common.validate_locked_protocol(args)
    if args.history_size < 2:
        raise ValueError("history_size must be at least two")
    if args.history_interval <= 0:
        raise ValueError("history_interval must be positive")
    if args.warmup_iterations % args.history_interval != 0:
        raise ValueError("warmup_iterations must be divisible by history_interval")
    if not 0.0 <= args.minimum_pixel_weight <= 1.0:
        raise ValueError("minimum_pixel_weight must be in [0, 1]")
    if not args.readiness_start < args.readiness_full:
        raise ValueError("readiness_start must be smaller than readiness_full")
    if not 0.0 <= args.readiness_momentum < 1.0:
        raise ValueError("readiness_momentum must be in [0, 1)")


def student_forward(model, images, feature_dropout):
    features = model.encoder(images)
    paired_features = [
        torch.cat(
            (
                feature,
                F.dropout2d(
                    feature, p=float(feature_dropout), training=True
                ),
            ),
            dim=0,
        )
        for feature in features
    ]
    paired_logits, paired_decoder_features = model.decoder(paired_features)
    logits, feature_logits = paired_logits.chunk(2, dim=0)
    decoder_features = paired_decoder_features[: images.shape[0]]
    return logits, feature_logits, decoder_features


def make_strong_views(images):
    count, _, height, width = images.shape
    augmented_views = [
        base.strong_mri_augmentation(images),
        base.strong_mri_augmentation(images),
    ]
    permutations = [
        torch.randperm(count, device=images.device),
        torch.randperm(count, device=images.device),
    ]
    boxes_per_view = [
        base.obtain_cutmix_boxes(count, height, width, images.device),
        base.obtain_cutmix_boxes(count, height, width, images.device),
    ]
    views = []
    for augmented, permutation, boxes in zip(
        augmented_views, permutations, boxes_per_view
    ):
        views.append(
            {
                "images": base.cutmix_tensor(
                    augmented, augmented[permutation], boxes
                ),
                "permutation": permutation,
                "boxes": boxes,
            }
        )
    return views


def transport(value, view):
    return base.cutmix_tensor(
        value, value[view["permutation"]], view["boxes"]
    )


def fixed_consistency_weight(args, iteration):
    epoch_like = iteration // 150
    return 5.0 * float(args.consistency) * base.ramps.sigmoid_rampup(
        epoch_like, args.consistency_rampup
    )


def save_config(args, split_cases, labeled_counts):
    config = vars(args).copy()
    config.update(
        {
            "hypothesis": "trajectory_reliability_controls_trust",
            "protocol": "PROMISE12_35_5_10_first7_191_seed1337",
            "train_cases": split_cases["train"],
            "val_cases": split_cases["val"],
            "test_cases": split_cases["test"],
            "labeled_slice_counts": labeled_counts,
            "pretrained_sha256": common.sha256(args.pretrained_model_path),
            "anchor_sha256": common.sha256(args.anchor_checkpoint),
            "target_source": "current_train_mode_EMA_teacher_only",
            "history_role": "reliability_estimation_only_no_target_ensemble",
            "trajectory_signals": [
                "class_agreement", "probability_variance", "prediction_flip_rate"
            ],
        }
    )
    with open(
        os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, indent=2, sort_keys=True)


def train(args, split_cases, labeled_counts):
    base.args = args
    flags = mode_flags(args.mode)

    def create_model(ema=False):
        network = base.UNet(in_chns=1, class_num=args.num_classes).cuda()
        if ema:
            for parameter in network.parameters():
                parameter.detach_()
        return network

    model = create_model()
    ema_model = create_model(ema=True)
    optimizer = optim.SGD(
        model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001
    )
    common.load_initial_state(
        model, ema_model, optimizer, args.pretrained_model_path
    )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = base.BaseDataSets(
        base_dir=args.root_path,
        split="train",
        transform=transforms.Compose([base.RandomGenerator(args.patch_size)]),
    )
    db_val = base.BaseDataSets(base_dir=args.root_path, split="val")
    labeled_slice = int(sum(labeled_counts.values()))
    labeled_idxs = list(range(labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, len(db_train)))
    sampler = base.TwoStreamBatchSampler(
        labeled_idxs,
        unlabeled_idxs,
        args.batch_size,
        args.batch_size - args.labeled_bs,
    )
    trainloader = DataLoader(
        db_train,
        batch_sampler=sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    if (labeled_slice, len(unlabeled_idxs), len(trainloader)) != (191, 749, 15):
        raise RuntimeError(
            "Expected labeled/unlabeled/batches=191/749/15, got {}/{}/{}".format(
                labeled_slice, len(unlabeled_idxs), len(trainloader)
            )
        )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    ce_loss = CrossEntropyLoss()
    dice_loss = base.losses.DiceLoss(args.num_classes)
    writer = SummaryWriter(os.path.join(args.output_dir, "log"))
    trajectory = TeacherTrajectory(args.history_size)
    trajectory.append(ema_model)
    smoothed_scale = None

    model.train()
    ema_model.train()
    logging.info("TRAJECTORY RELIABILITY ENTRY ACTIVE mode=%s", args.mode)
    logging.info(
        "History K=%d interval=%d target=current EMA only flags=%s",
        args.history_size,
        args.history_interval,
        flags,
    )

    iter_num = 0
    best_performance = 0.0
    final_performance = 0.0
    best_iteration = 0
    max_epoch = args.max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=110, dynamic_ncols=True)

    for _ in iterator:
        for sampled_batch in trainloader:
            volume_batch = sampled_batch["image"].cuda(non_blocking=True)
            label_batch = sampled_batch["label"].cuda(non_blocking=True)
            unlabeled_images = volume_batch[args.labeled_bs :]

            zero = volume_batch.new_zeros(())
            loss_s1 = zero
            loss_s2 = zero
            loss_fp = zero
            boundary_loss = zero
            reliability_mean = zero
            foreground_reliability = zero
            flip_rate_mean = zero
            readiness = zero
            trust_scale = zero
            pixel_weight_mean = zero

            if iter_num < args.warmup_iterations:
                model_output = model(volume_batch)
                outputs = (
                    model_output[0] if isinstance(model_output, tuple) else model_output
                )
                consistency_loss = zero
                consistency_weight = 0.0
            else:
                strong_views = make_strong_views(unlabeled_images)
                with torch.no_grad():
                    ema_output = common.model_logits(ema_model, unlabeled_images)
                    ema_probability = torch.softmax(ema_output, dim=1)
                    teacher_fg_probability = ema_probability[:, 1]
                    raw_pseudo_labels = torch.argmax(ema_probability, dim=1)
                    pseudo_labels = base.get_masks(ema_output, nms=1).long()
                    pseudo_confidence = ema_probability.gather(
                        1, pseudo_labels.unsqueeze(1)
                    ).squeeze(1)
                    history = torch.cat(
                        (
                            trajectory.probabilities(unlabeled_images),
                            teacher_fg_probability.unsqueeze(0),
                        ),
                        dim=0,
                    )
                    stats = trajectory_statistics(history)
                    target_consistency = raw_pseudo_labels == pseudo_labels
                    reliability = (
                        stats["reliability"] * target_consistency.float()
                    )
                    reliability_mean = reliability.mean()
                    flip_rate_mean = stats["flip_rate"].mean()
                    foreground = pseudo_labels == 1
                    if foreground.any():
                        foreground_reliability = reliability[foreground].mean()

                    if flags["weighting"]:
                        pseudo_weight = (
                            float(args.minimum_pixel_weight)
                            + (1.0 - float(args.minimum_pixel_weight)) * reliability
                        ) * target_consistency.float()
                    else:
                        pseudo_weight = (
                            pseudo_confidence >= args.confidence_threshold
                        ).float()
                    pixel_weight_mean = pseudo_weight.mean()

                    raw_scale, readiness = adaptive_unsupervised_scale(
                        reliability,
                        pseudo_labels,
                        stable_threshold=args.stable_threshold,
                        readiness_start=args.readiness_start,
                        readiness_full=args.readiness_full,
                        minimum_scale=args.minimum_unsupervised_scale,
                    )
                    if smoothed_scale is None:
                        smoothed_scale = raw_scale.detach()
                    else:
                        smoothed_scale = (
                            float(args.readiness_momentum) * smoothed_scale
                            + (1.0 - float(args.readiness_momentum))
                            * raw_scale.detach()
                        )
                    trust_scale = smoothed_scale

                    for view in strong_views:
                        view["targets"] = transport(pseudo_labels, view)
                        view["weights"] = transport(pseudo_weight, view)

                outputs, outputs_fp, decoder_features = student_forward(
                    model, volume_batch, args.feature_dropout
                )
                strong_output = common.model_logits(
                    model,
                    torch.cat(
                        (strong_views[0]["images"], strong_views[1]["images"]),
                        dim=0,
                    ),
                )
                outputs_s1, outputs_s2 = strong_output.chunk(2, dim=0)
                loss_s1, _ = weighted_pseudo_loss(
                    outputs_s1,
                    strong_views[0]["targets"],
                    strong_views[0]["weights"],
                    dice_loss,
                )
                loss_s2, _ = weighted_pseudo_loss(
                    outputs_s2,
                    strong_views[1]["targets"],
                    strong_views[1]["weights"],
                    dice_loss,
                )
                loss_fp, _ = weighted_pseudo_loss(
                    outputs_fp[args.labeled_bs :],
                    pseudo_labels,
                    pseudo_weight,
                    dice_loss,
                )
                consistency_loss = 0.25 * loss_s1 + 0.25 * loss_s2 + 0.5 * loss_fp

                boundary_coverage = zero
                core_coverage = zero
                if flags["boundary"]:
                    soft_target, boundary_weight, boundary, stable_core, _ = (
                        soft_boundary_target(
                            decoder_features[args.labeled_bs :],
                            teacher_fg_probability,
                            pseudo_labels,
                            reliability,
                            core_threshold=args.stable_threshold,
                            radius=args.boundary_radius,
                            recovery_strength=args.boundary_recovery_strength,
                            minimum_weight=args.boundary_minimum_weight,
                        )
                    )
                    boundary_loss = weighted_soft_binary_loss(
                        outputs_fp[args.labeled_bs :],
                        soft_target,
                        boundary_weight,
                    )
                    consistency_loss = (
                        consistency_loss
                        + float(args.boundary_loss_weight) * boundary_loss
                    )
                    boundary_coverage = boundary.float().mean()
                    core_coverage = stable_core.float().mean()
                    writer.add_scalar(
                        "trajectory/boundary_coverage", boundary_coverage, iter_num + 1
                    )
                    writer.add_scalar(
                        "trajectory/stable_core_coverage", core_coverage, iter_num + 1
                    )

                if flags["adaptive"]:
                    consistency_weight = 5.0 * float(args.consistency) * float(
                        trust_scale
                    )
                else:
                    consistency_weight = fixed_consistency_weight(args, iter_num)

            supervised_ce = ce_loss(
                outputs[: args.labeled_bs], label_batch[: args.labeled_bs].long()
            )
            supervised_dice = dice_loss(
                torch.softmax(outputs[: args.labeled_bs], dim=1),
                label_batch[: args.labeled_bs].unsqueeze(1),
            )
            supervised_loss = 0.5 * (supervised_ce + supervised_dice)
            total_loss = supervised_loss + consistency_weight * consistency_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    "Non-finite loss at iteration {}".format(iter_num + 1)
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
            if iter_num % args.history_interval == 0:
                trajectory.append(ema_model)
                logging.info(
                    "trajectory snapshot iter=%d members=%d",
                    iter_num,
                    len(trajectory),
                )

            writer.add_scalar("info/lr", learning_rate, iter_num)
            writer.add_scalar("info/total_loss", total_loss.item(), iter_num)
            writer.add_scalar("info/supervised_loss", supervised_loss.item(), iter_num)
            writer.add_scalar("info/unsupervised_loss", consistency_loss.item(), iter_num)
            writer.add_scalar("info/consistency_weight", consistency_weight, iter_num)
            writer.add_scalar("trajectory/reliability_mean", reliability_mean, iter_num)
            writer.add_scalar(
                "trajectory/foreground_reliability", foreground_reliability, iter_num
            )
            writer.add_scalar("trajectory/flip_rate", flip_rate_mean, iter_num)
            writer.add_scalar("trajectory/readiness", readiness, iter_num)
            writer.add_scalar("trajectory/trust_scale", trust_scale, iter_num)
            writer.add_scalar("trajectory/pixel_weight", pixel_weight_mean, iter_num)
            writer.add_scalar("trajectory/boundary_loss", boundary_loss, iter_num)

            iterator.set_postfix(
                iteration=iter_num,
                loss="{:.4f}".format(float(total_loss.detach())),
                trust="{:.3f}".format(float(trust_scale)),
                val="{:.4f}".format(final_performance),
                best="{:.4f}".format(best_performance),
            )
            if iter_num % args.log_interval == 0:
                logging.info(
                    "iter %d/%d mode=%s loss=%.6f sup=%.6f unsup=%.6f "
                    "lambda=%.6f R=%.4f R_fg=%.4f flip=%.4f readiness=%.4f "
                    "trust=%.4f pixel_w=%.4f boundary=%.6f",
                    iter_num,
                    args.max_iterations,
                    args.mode,
                    total_loss.item(),
                    supervised_loss.item(),
                    consistency_loss.item(),
                    consistency_weight,
                    reliability_mean.item(),
                    foreground_reliability.item(),
                    flip_rate_mean.item(),
                    readiness.item(),
                    trust_scale.item(),
                    pixel_weight_mean.item(),
                    boundary_loss.item(),
                )

            if iter_num % args.validation_interval == 0:
                metric_list = common.evaluate(
                    model, valloader, len(db_val), args.num_classes
                )
                final_performance = float(np.mean(metric_list, axis=0)[0])
                writer.add_scalar("info/val_mean_dice", final_performance, iter_num)
                if final_performance > best_performance:
                    best_performance = final_performance
                    best_iteration = iter_num
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            args.output_dir,
                            "iter_{}_dice_{}.pth".format(
                                iter_num, round(best_performance, 4)
                            ),
                        ),
                    )
                    torch.save(
                        model.state_dict(),
                        os.path.join(args.output_dir, "unet_best_model.pth"),
                    )
                logging.info(
                    "validation iter=%d mean_dice=%.6f best=%.6f best_iter=%d",
                    iter_num,
                    final_performance,
                    best_performance,
                    best_iteration,
                )

            if iter_num % args.save_interval == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(args.output_dir, "iter_{}.pth".format(iter_num)),
                )
            if iter_num >= args.max_iterations:
                break
        if iter_num >= args.max_iterations:
            iterator.close()
            break

    writer.close()
    summary = {
        "hypothesis": "trajectory_reliability_controls_trust",
        "mode": args.mode,
        "best_validation_dice": best_performance,
        "best_iteration": best_iteration,
        "final_validation_dice": final_performance,
        "iterations": iter_num,
        "history_size": args.history_size,
        "history_interval": args.history_interval,
        "target_source": "current_ema_only",
    }
    with open(
        os.path.join(args.output_dir, "training_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    logging.info("Training completed: %s", summary)


def main(args):
    args.root_path = os.path.abspath(args.root_path)
    args.pretrained_model_path = os.path.abspath(args.pretrained_model_path)
    args.anchor_checkpoint = os.path.abspath(args.anchor_checkpoint)
    args.output_dir = os.path.abspath(args.output_dir)
    split_cases, labeled_counts = common.validate_locked_protocol(args)
    validate_trajectory_args(args)
    common.configure_logging(args.output_dir)
    common.seed_everything(args.seed, bool(args.deterministic))
    save_config(args, split_cases, labeled_counts)
    logging.info("Arguments: %s", args)
    logging.info("Original pretrain SHA256: %s", common.sha256(args.pretrained_model_path))
    logging.info("Reference anchor SHA256: %s", common.sha256(args.anchor_checkpoint))
    train(args, split_cases, labeled_counts)


if __name__ == "__main__":
    main(build_parser().parse_args())
