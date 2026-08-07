"""Training stages for full PROMISE12 Uni-MedSAM/SAMatch."""

import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import (
    BaseDataSets,
    RandomGenerator,
    TwoStreamBatchSampler,
)
from samatch_medsam import (
    binary_iou,
    build_medsam_lite,
    extract_model_state,
    masks_to_boxes,
    safe_torch_load,
    sam_mask_loss,
)
from utils import losses, val_2d


def import_baseline():
    """Import the unchanged UniMatch module without consuming this CLI."""
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        import train_unimatch as module
    finally:
        sys.argv = original_argv
    return module


baseline = import_baseline()


def configure_logging(log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(str(log_path)),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def worker_init_fn(worker_id):
    random.seed(baseline.args.seed + worker_id)
    np.random.seed(baseline.args.seed + worker_id)


def build_training_loader(config):
    dataset = BaseDataSets(
        base_dir=config.root_path,
        split="train",
        transform=transforms.Compose([RandomGenerator(config.patch_size)]),
    )
    labeled_slices = baseline.patients_to_slices(
        config.root_path, config.labelnum)
    labeled_indices = list(range(labeled_slices))
    unlabeled_indices = list(range(labeled_slices, len(dataset)))
    sampler = TwoStreamBatchSampler(
        labeled_indices,
        unlabeled_indices,
        config.batch_size,
        config.batch_size - config.labeled_bs,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    return dataset, loader, labeled_slices


def logits_only(output):
    return output[0] if isinstance(output, tuple) else output


def load_unet_checkpoint(model, checkpoint_path):
    checkpoint = safe_torch_load(
        checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(extract_model_state(checkpoint), strict=True)
    return checkpoint


def run_match_warmup(config, pre_dir, self_dir):
    """Run the existing project UniMatch protocol without redefining it."""
    pre_checkpoint = pre_dir / "{}_best_model.pth".format(config.model)
    self_checkpoint = self_dir / "{}_best_model.pth".format(config.model)
    if (config.reuse_warmup and pre_checkpoint.is_file() and
            self_checkpoint.is_file()):
        logging.info("Reusing complete Match warm-up: %s", self_checkpoint)
        return

    config.pre_iterations = config.match_pre_iterations
    config.max_iterations = config.match_self_iterations
    baseline.args = config

    configure_logging(pre_dir / "log.txt")
    logging.info(
        "SAMatch stage 1a, unchanged Match branch: pre=%d self=%d",
        config.match_pre_iterations, config.match_self_iterations)
    baseline.pre_train(config, str(pre_dir))

    configure_logging(self_dir / "log.txt")
    logging.info("SAMatch stage 1a: existing UniMatch self-training")
    baseline.self_train(config, str(pre_dir), str(self_dir))


def run_medsam_warmup(config, snapshot_dir):
    """Official SAMatch labeled LiteMedSAM adaptation stage."""
    best_path = snapshot_dir / "medsam_lite_best.pth"
    if config.reuse_warmup and best_path.is_file():
        logging.info("Reusing complete MedSAM warm-up: %s", best_path)
        return

    configure_logging(snapshot_dir / "log.txt")
    logging.info("SAMatch stage 1b: labeled LiteMedSAM adaptation")
    device = torch.device("cuda")
    model = build_medsam_lite(
        config.medsam_pretrained, config.samatch_source_dir, device)
    model.train()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.medsam_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=5, cooldown=0)
    iou_loss = nn.MSELoss(reduction="mean")
    _, trainloader, labeled_slices = build_training_loader(config)
    logging.info(
        "labeled_slices=%d, target_medsam_updates=%d",
        labeled_slices, config.medsam_warmup_iterations)

    writer = SummaryWriter(str(snapshot_dir / "tensorboard"))
    iteration = 0
    epoch = 0
    best_loss = float("inf")
    progress = tqdm(
        total=config.medsam_warmup_iterations,
        ncols=90,
        desc="MedSAM warm-up",
    )
    while iteration < config.medsam_warmup_iterations:
        epoch_losses = []
        for batch in trainloader:
            images = batch["image"][:config.labeled_bs].cuda(non_blocking=True)
            labels = batch["label"][:config.labeled_bs].cuda(non_blocking=True)
            boxes, valid = masks_to_boxes(labels == 1, config.bbox_shift)
            # PROMISE12 includes true background-only slices. A box prompt is
            # undefined for them, so they are not sent through the box encoder.
            if not valid.any():
                continue

            inputs = images[valid].repeat(1, 3, 1, 1)
            targets = (labels[valid] == 1).float().unsqueeze(1)
            logits, predicted_iou = model(inputs, boxes[valid])
            predicted_masks = torch.sigmoid(logits) > 0.5
            target_iou = binary_iou(predicted_masks, targets)
            mask_loss = sam_mask_loss(logits, targets)
            loss = 2.0 * mask_loss + iou_loss(predicted_iou, target_iou)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iteration += 1
            epoch_losses.append(loss.item())
            writer.add_scalar("medsam_warmup/loss", loss.item(), iteration)
            writer.add_scalar(
                "medsam_warmup/lr",
                optimizer.param_groups[0]["lr"],
                iteration,
            )
            if iteration % 20 == 0:
                logging.info(
                    "medsam warm-up iter %d/%d loss=%.6f prompts=%d",
                    iteration, config.medsam_warmup_iterations,
                    loss.item(), int(valid.sum().item()))
            progress.update(1)
            if iteration >= config.medsam_warmup_iterations:
                break

        if not epoch_losses:
            raise RuntimeError(
                "No positive foreground prompt found in a complete epoch")
        mean_loss = float(np.mean(epoch_losses))
        scheduler.step(mean_loss)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "epoch": epoch,
            "loss": mean_loss,
        }
        torch.save(checkpoint, snapshot_dir / "medsam_lite_latest.pth")
        if mean_loss < best_loss:
            best_loss = mean_loss
            checkpoint["best_loss"] = best_loss
            torch.save(checkpoint, best_path)
            logging.info(
                "Saved best MedSAM warm-up checkpoint: loss=%.6f", best_loss)
        epoch += 1
    progress.close()
    writer.close()


def evaluate_unet(model, valloader, dataset, num_classes):
    model.eval()
    metric_list = 0.0
    with torch.no_grad():
        for batch in valloader:
            metric_i = val_2d.test_single_volume(
                batch["image"],
                batch["label"],
                model,
                classes=num_classes,
            )
            metric_list += np.asarray(metric_i)
    metric_list = metric_list / len(dataset)
    return float(np.mean(metric_list, axis=0)[0])


def update_sam(
        medsam,
        optimizer_sam,
        labeled_images,
        labeled_targets,
        unlabeled_images,
        teacher_pseudo,
        pseudo_boxes,
        pseudo_valid,
        config,
):
    """Update all SAMatch SAM losses and return detached refined masks."""
    refined_pseudo = torch.zeros_like(teacher_pseudo)
    sam_logits_u = None
    if pseudo_valid.any():
        sam_logits_u, _ = medsam(
            unlabeled_images[pseudo_valid].repeat(1, 3, 1, 1),
            pseudo_boxes[pseudo_valid],
        )
        refined_pseudo[pseudo_valid] = (
            torch.sigmoid(sam_logits_u.detach()).squeeze(1) > 0.5).long()

    gt_boxes, gt_valid = masks_to_boxes(
        labeled_targets == 1, config.bbox_shift)
    sam_losses = []
    if gt_valid.any():
        sam_logits_x, _ = medsam(
            labeled_images[gt_valid].repeat(1, 3, 1, 1),
            gt_boxes[gt_valid],
        )
        sam_losses.append(sam_mask_loss(
            sam_logits_x,
            (labeled_targets[gt_valid] == 1).float().unsqueeze(1),
        ))
    if sam_logits_u is not None:
        sam_losses.append(
            config.sam_unlabeled_weight * sam_mask_loss(
                sam_logits_u,
                (teacher_pseudo[pseudo_valid] == 1).float().unsqueeze(1),
            ))

    if sam_losses:
        loss_sam = torch.stack(sam_losses).sum()
        optimizer_sam.zero_grad()
        loss_sam.backward()
        optimizer_sam.step()
    else:
        loss_sam = unlabeled_images.new_zeros(())
    return refined_pseudo, loss_sam


def make_strong_streams(
        unlabeled_images,
        refined_pseudo,
        teacher_confidence,
        device,
):
    """Apply the unchanged UniMatch augmentation and aligned CutMix."""
    view1 = baseline.strong_mri_augmentation(unlabeled_images)
    view2 = baseline.strong_mri_augmentation(unlabeled_images)
    count, _, height, width = unlabeled_images.shape
    permutation1 = torch.randperm(count, device=device)
    permutation2 = torch.randperm(count, device=device)
    box1 = baseline.obtain_cutmix_boxes(count, height, width, device)
    box2 = baseline.obtain_cutmix_boxes(count, height, width, device)
    view1 = baseline.cutmix_tensor(view1, view1[permutation1], box1)
    view2 = baseline.cutmix_tensor(view2, view2[permutation2], box2)
    pseudo1 = baseline.cutmix_tensor(
        refined_pseudo, refined_pseudo[permutation1], box1)
    pseudo2 = baseline.cutmix_tensor(
        refined_pseudo, refined_pseudo[permutation2], box2)
    confidence1 = baseline.cutmix_tensor(
        teacher_confidence, teacher_confidence[permutation1], box1)
    confidence2 = baseline.cutmix_tensor(
        teacher_confidence, teacher_confidence[permutation2], box2)
    return view1, view2, pseudo1, pseudo2, confidence1, confidence2


def save_interactive_state(
        path,
        student,
        teacher,
        medsam,
        optimizer,
        optimizer_sam,
        iteration,
        best_dice,
):
    torch.save(
        {
            "model": student.state_dict(),
            "teacher": teacher.state_dict(),
            "medsam": medsam.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_sam": optimizer_sam.state_dict(),
            "iteration": iteration,
            "best_dice": best_dice,
        },
        path,
    )


def run_interactive(config, match_checkpoint, medsam_checkpoint, snapshot_dir):
    """Full joint UniMatch/LiteMedSAM interaction stage."""
    configure_logging(snapshot_dir / "log.txt")
    logging.info(
        "SAMatch stage 2: interactive=%d Match_lr=%.6g SAM_lr=%.6g",
        config.interactive_iterations,
        config.base_lr,
        config.interactive_sam_lr,
    )
    device = torch.device("cuda")
    student = baseline.UNet(
        in_chns=1, class_num=config.num_classes).to(device)
    teacher = baseline.UNet(
        in_chns=1, class_num=config.num_classes).to(device)
    load_unet_checkpoint(student, match_checkpoint)
    load_unet_checkpoint(teacher, match_checkpoint)
    for parameter in teacher.parameters():
        parameter.detach_()
    teacher.eval()

    medsam = build_medsam_lite(
        medsam_checkpoint, config.samatch_source_dir, device)
    medsam.train()

    # Official SAMatch starts new optimizers at the interactive stage.
    optimizer = optim.SGD(
        student.parameters(),
        lr=config.base_lr,
        momentum=0.9,
        weight_decay=0.0001,
    )
    optimizer_sam = optim.SGD(
        medsam.parameters(),
        lr=config.interactive_sam_lr,
        momentum=0.9,
        weight_decay=0.0001,
    )

    dataset, trainloader, labeled_slices = build_training_loader(config)
    val_dataset = BaseDataSets(base_dir=config.root_path, split="val")
    valloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=1)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(config.num_classes)
    writer = SummaryWriter(str(snapshot_dir / "tensorboard"))
    logging.info(
        "train_slices=%d labeled_slices=%d iterations_per_epoch=%d",
        len(dataset), labeled_slices, len(trainloader))

    baseline.args = config
    student.train()
    iteration = 0
    best_dice = 0.0
    max_epoch = config.interactive_iterations // len(trainloader) + 1
    progress = tqdm(range(max_epoch), ncols=90, desc="SAMatch interactive")
    for _ in progress:
        for batch in trainloader:
            images = batch["image"].cuda(non_blocking=True)
            labels = batch["label"].cuda(non_blocking=True)
            labeled_images = images[:config.labeled_bs]
            labeled_targets = labels[:config.labeled_bs]
            unlabeled_images = images[config.labeled_bs:]

            with torch.no_grad():
                teacher_logits = logits_only(teacher(unlabeled_images))
                teacher_probability = torch.softmax(teacher_logits, dim=1)
                teacher_pseudo = baseline.get_masks(
                    teacher_logits, nms=1).long()
                teacher_confidence = teacher_probability.gather(
                    1, teacher_pseudo.unsqueeze(1)).squeeze(1)

            pseudo_boxes, pseudo_valid = masks_to_boxes(
                teacher_pseudo == 1, config.bbox_shift)
            refined_pseudo, loss_sam = update_sam(
                medsam,
                optimizer_sam,
                labeled_images,
                labeled_targets,
                unlabeled_images,
                teacher_pseudo,
                pseudo_boxes,
                pseudo_valid,
                config,
            )
            (
                strong1,
                strong2,
                pseudo1,
                pseudo2,
                confidence1,
                confidence2,
            ) = make_strong_streams(
                unlabeled_images,
                refined_pseudo,
                teacher_confidence,
                device,
            )

            outputs, outputs_fp = student(
                images,
                need_fp=True,
                feature_dropout=config.feature_dropout,
            )
            strong_output = logits_only(
                student(torch.cat((strong1, strong2), dim=0)))
            outputs_s1, outputs_s2 = strong_output.chunk(2, dim=0)

            loss_u_s1, _ = baseline.confidence_masked_baseline_loss(
                outputs_s1, pseudo1, confidence1, dice_loss)
            loss_u_s2, _ = baseline.confidence_masked_baseline_loss(
                outputs_s2, pseudo2, confidence2, dice_loss)
            loss_u_fp, valid_fp = baseline.confidence_masked_baseline_loss(
                outputs_fp[config.labeled_bs:],
                refined_pseudo,
                teacher_confidence,
                dice_loss,
            )
            consistency_loss = (
                0.25 * loss_u_s1 +
                0.25 * loss_u_s2 +
                0.50 * loss_u_fp)

            loss_ce = ce_loss(
                outputs[:config.labeled_bs], labeled_targets.long())
            loss_dice = dice_loss(
                torch.softmax(outputs[:config.labeled_bs], dim=1),
                labeled_targets.unsqueeze(1),
            )
            supervised_loss = 0.5 * (loss_ce + loss_dice)
            consistency_weight = baseline.get_current_consistency_weight(
                iteration // 150)
            total_loss = (
                supervised_loss + consistency_weight * consistency_loss)

            ratio = min(
                float(iteration) /
                float(config.interactive_iterations),
                1.0,
            )
            current_lr = config.base_lr * (1.0 - ratio) ** 0.9
            current_sam_lr = (
                config.interactive_sam_lr * (1.0 - ratio) ** 0.9)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            for group in optimizer_sam.param_groups:
                group["lr"] = current_sam_lr

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            baseline.update_model_ema(
                student, teacher, config.ema_decay)
            iteration += 1

            coverage = valid_fp.float().mean()
            teacher_fg = (teacher_pseudo == 1).float().mean()
            refined_fg = (refined_pseudo == 1).float().mean()
            writer.add_scalar(
                "interactive/total_loss", total_loss.item(), iteration)
            writer.add_scalar(
                "interactive/supervised_loss",
                supervised_loss.item(),
                iteration,
            )
            writer.add_scalar(
                "interactive/consistency_loss",
                consistency_loss.item(),
                iteration,
            )
            writer.add_scalar(
                "interactive/sam_loss", loss_sam.item(), iteration)
            writer.add_scalar(
                "interactive/coverage", coverage.item(), iteration)
            writer.add_scalar(
                "interactive/teacher_fg", teacher_fg.item(), iteration)
            writer.add_scalar(
                "interactive/refined_fg", refined_fg.item(), iteration)
            writer.add_scalar("interactive/lr", current_lr, iteration)
            writer.add_scalar(
                "interactive/sam_lr", current_sam_lr, iteration)

            if iteration % 20 == 0:
                logging.info(
                    "interactive iter %d/%d loss=%.6f sup=%.6f "
                    "unsup=%.6f sam=%.6f coverage=%.4f "
                    "teacher_fg=%.4f refined_fg=%.4f prompts=%d",
                    iteration,
                    config.interactive_iterations,
                    total_loss.item(),
                    supervised_loss.item(),
                    consistency_loss.item(),
                    loss_sam.item(),
                    coverage.item(),
                    teacher_fg.item(),
                    refined_fg.item(),
                    int(pseudo_valid.sum().item()),
                )

            if (iteration > 0 and
                    iteration % config.validation_interval == 0):
                performance = evaluate_unet(
                    student, valloader, val_dataset, config.num_classes)
                writer.add_scalar(
                    "interactive/val_mean_dice",
                    performance,
                    iteration,
                )
                if performance > best_dice:
                    best_dice = performance
                    torch.save(
                        student.state_dict(),
                        snapshot_dir /
                        "{}_best_model.pth".format(config.model),
                    )
                    torch.save(
                        {
                            "model": medsam.state_dict(),
                            "optimizer": optimizer_sam.state_dict(),
                            "iteration": iteration,
                            "val_dice": best_dice,
                        },
                        snapshot_dir / "medsam_lite_best.pth",
                    )
                logging.info(
                    "interactive iter %d val_dice=%.6f best_dice=%.6f",
                    iteration, performance, best_dice)
                student.train()
                teacher.eval()
                medsam.train()

            if iteration % config.save_interval == 0:
                save_interactive_state(
                    snapshot_dir / "interactive_latest.pth",
                    student,
                    teacher,
                    medsam,
                    optimizer,
                    optimizer_sam,
                    iteration,
                    best_dice,
                )

            if iteration >= config.interactive_iterations:
                break
        if iteration >= config.interactive_iterations:
            break
    progress.close()
    writer.close()
    logging.info(
        "Full interactive SAMatch completed: best_dice=%.6f", best_dice)
