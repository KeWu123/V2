"""PROMISE12 UniMatch with non-destructive foreground pseudo-label rescue.

The original UniMatch hard pseudo-label path is kept unchanged.  A second,
low-weight soft loss is added only for uncertain foreground pixels that are
stable under a left-right weak-view transform and connected to an existing
high-confidence foreground seed.
"""

import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import train_unimatch as baseline


# Short-schedule main-experiment settings.  These deliberately are not exposed
# as a large tuning surface: the baseline training configuration stays fixed.
RESCUE_START = 500
RESCUE_RAMP = 1000
RESCUE_MAX_WEIGHT = 0.15
RESCUE_FOREGROUND_THRESHOLD = 0.80
RESCUE_MAX_VIEW_DISAGREEMENT = 0.10
RESCUE_RADIUS = 5


def foreground_rescue_route(probability, flipped_probability,
                            pseudo_labels, pseudo_confidence):
    """Return a soft target and a conservative foreground-only rescue mask."""
    soft_target = 0.5 * (probability + flipped_probability)
    label_a = probability.argmax(dim=1)
    label_b = flipped_probability.argmax(dim=1)

    high_confidence_foreground = (
        (pseudo_labels == 1) &
        (pseudo_confidence >= baseline.args.confidence_threshold)
    )
    kernel_size = 2 * RESCUE_RADIUS + 1
    connected_support = F.max_pool2d(
        high_confidence_foreground.float().unsqueeze(1),
        kernel_size=kernel_size, stride=1,
        padding=RESCUE_RADIUS).squeeze(1).bool()

    view_disagreement = (
        probability[:, 1] - flipped_probability[:, 1]).abs()
    baseline_valid = pseudo_confidence >= baseline.args.confidence_threshold
    rescue_mask = (
        (label_a == 1) &
        (label_b == 1) &
        (soft_target[:, 1] >= RESCUE_FOREGROUND_THRESHOLD) &
        (view_disagreement <= RESCUE_MAX_VIEW_DISAGREEMENT) &
        connected_support &
        (~baseline_valid)
    )
    return soft_target.detach(), rescue_mask, view_disagreement


def masked_soft_kl(logits, soft_target, mask):
    """KL loss on selected pixels; remains differentiable for an empty mask."""
    per_pixel = F.kl_div(
        F.log_softmax(logits, dim=1), soft_target,
        reduction='none').sum(dim=1)
    mask_float = mask.float()
    return ((per_pixel * mask_float).sum() /
            mask_float.sum().clamp_min(1.0))


def rescue_weight(iteration):
    if iteration < RESCUE_START:
        return 0.0
    progress = min(1.0, (iteration - RESCUE_START + 1) / RESCUE_RAMP)
    return RESCUE_MAX_WEIGHT * progress


def self_train(args, pre_snapshot_path, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    self_warmup_iterations = 1000

    def create_model(ema=False):
        model = baseline.UNet(in_chns=1, class_num=num_classes).cuda()
        if ema:
            for parameter in model.parameters():
                parameter.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)
    optimizer = baseline.optim.SGD(
        model.parameters(), lr=base_lr, momentum=0.9,
        weight_decay=0.0001)

    pre_trained_model = os.path.join(
        pre_snapshot_path, '{}_best_model.pth'.format(args.model))
    try:
        checkpoint = torch.load(
            pre_trained_model, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(pre_trained_model, map_location='cpu')
    if 'net' in checkpoint:
        model.load_state_dict(checkpoint['net'])
        ema_model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['opt'])
        logging.info(
            "Loaded pre-trained weights and optimizer from %s",
            pre_trained_model)
    else:
        model.load_state_dict(checkpoint)
        ema_model.load_state_dict(checkpoint)
        logging.info("Loaded pre-trained weights from %s", pre_trained_model)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = baseline.BaseDataSets(
        base_dir=args.root_path, split="train", num=None,
        transform=baseline.transforms.Compose([
            baseline.RandomGenerator(args.patch_size)]))
    db_val = baseline.BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = baseline.patients_to_slices(
        args.root_path, args.labelnum)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = baseline.TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size,
        batch_size - args.labeled_bs)
    trainloader = baseline.DataLoader(
        db_train, batch_sampler=batch_sampler, num_workers=4,
        pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = baseline.DataLoader(
        db_val, batch_size=1, shuffle=False, num_workers=1)

    model.train()
    ema_model.eval()
    ce_loss = baseline.CrossEntropyLoss()
    dice_loss = baseline.losses.DiceLoss(num_classes)
    writer = baseline.SummaryWriter(snapshot_path + '/log')
    logging.info("%d iterations per epoch", len(trainloader))
    logging.info(
        "Baseline self schedule: total=%d, supervised_warmup=%d, "
        "pseudo_start=%d", max_iterations, self_warmup_iterations,
        self_warmup_iterations)
    logging.info(
        "UniMatch retained unchanged: tau=%.2f, feature_dropout=%.2f, "
        "branch_weights=(0.25, 0.25, 0.50)",
        args.confidence_threshold, args.feature_dropout)
    logging.info(
        "Foreground rescue: start=%d ramp=%d max_weight=%.3f "
        "fg_tau=%.2f max_disagreement=%.2f radius=%d",
        RESCUE_START, RESCUE_RAMP, RESCUE_MAX_WEIGHT,
        RESCUE_FOREGROUND_THRESHOLD, RESCUE_MAX_VIEW_DISAGREEMENT,
        RESCUE_RADIUS)

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = baseline.tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for sampled_batch in trainloader:
            volume_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            if iter_num < self_warmup_iterations:
                model_output = model(volume_batch)
                outputs = (model_output[0] if isinstance(
                    model_output, tuple) else model_output)
                consistency_loss = outputs.new_zeros(())
                baseline_consistency_loss = outputs.new_zeros(())
                rescue_loss = outputs.new_zeros(())
                loss_u_s1 = outputs.new_zeros(())
                loss_u_s2 = outputs.new_zeros(())
                loss_u_fp = outputs.new_zeros(())
                confident_ratio = outputs.new_zeros(())
                confident_fg_ratio = outputs.new_zeros(())
                rescue_coverage = outputs.new_zeros(())
                rescue_disagreement = outputs.new_zeros(())
                current_rescue_weight = 0.0
            else:
                strong_view1 = baseline.strong_mri_augmentation(
                    unlabeled_volume_batch)
                strong_view2 = baseline.strong_mri_augmentation(
                    unlabeled_volume_batch)
                unlabeled_count, _, height, width = (
                    unlabeled_volume_batch.shape)
                permutation1 = torch.randperm(
                    unlabeled_count, device=unlabeled_volume_batch.device)
                permutation2 = torch.randperm(
                    unlabeled_count, device=unlabeled_volume_batch.device)
                cutmix_box1 = baseline.obtain_cutmix_boxes(
                    unlabeled_count, height, width,
                    unlabeled_volume_batch.device)
                cutmix_box2 = baseline.obtain_cutmix_boxes(
                    unlabeled_count, height, width,
                    unlabeled_volume_batch.device)
                strong_view1 = baseline.cutmix_tensor(
                    strong_view1, strong_view1[permutation1], cutmix_box1)
                strong_view2 = baseline.cutmix_tensor(
                    strong_view2, strong_view2[permutation2], cutmix_box2)

                current_rescue_weight = rescue_weight(iter_num)
                with torch.no_grad():
                    ema_output = ema_model(unlabeled_volume_batch)
                    if isinstance(ema_output, tuple):
                        ema_output = ema_output[0]
                    ema_probability = torch.softmax(ema_output, dim=1)

                    # These are exactly the original UniMatch targets/masks.
                    pseudo_labels = baseline.get_masks(
                        ema_output, nms=1).long()
                    pseudo_confidence = ema_probability.gather(
                        1, pseudo_labels.unsqueeze(1)).squeeze(1)
                    pseudo_labels_s1 = baseline.cutmix_tensor(
                        pseudo_labels, pseudo_labels[permutation1],
                        cutmix_box1)
                    pseudo_labels_s2 = baseline.cutmix_tensor(
                        pseudo_labels, pseudo_labels[permutation2],
                        cutmix_box2)
                    confidence_s1 = baseline.cutmix_tensor(
                        pseudo_confidence,
                        pseudo_confidence[permutation1], cutmix_box1)
                    confidence_s2 = baseline.cutmix_tensor(
                        pseudo_confidence,
                        pseudo_confidence[permutation2], cutmix_box2)

                    if current_rescue_weight > 0.0:
                        flipped_input = torch.flip(
                            unlabeled_volume_batch, dims=(-1,))
                        flipped_output = ema_model(flipped_input)
                        if isinstance(flipped_output, tuple):
                            flipped_output = flipped_output[0]
                        flipped_probability = torch.flip(
                            torch.softmax(flipped_output, dim=1), dims=(-1,))
                        rescue_target, rescue_mask, disagreement = (
                            foreground_rescue_route(
                                ema_probability, flipped_probability,
                                pseudo_labels, pseudo_confidence))
                    else:
                        rescue_target = ema_probability.detach()
                        rescue_mask = torch.zeros_like(
                            pseudo_labels, dtype=torch.bool)
                        disagreement = torch.zeros_like(
                            pseudo_confidence)

                    rescue_target_s1 = baseline.cutmix_tensor(
                        rescue_target, rescue_target[permutation1],
                        cutmix_box1)
                    rescue_target_s2 = baseline.cutmix_tensor(
                        rescue_target, rescue_target[permutation2],
                        cutmix_box2)
                    rescue_mask_s1 = baseline.cutmix_tensor(
                        rescue_mask, rescue_mask[permutation1], cutmix_box1)
                    rescue_mask_s2 = baseline.cutmix_tensor(
                        rescue_mask, rescue_mask[permutation2], cutmix_box2)

                outputs, outputs_fp = model(
                    volume_batch, need_fp=True,
                    feature_dropout=args.feature_dropout)
                strong_output = model(torch.cat(
                    (strong_view1, strong_view2), dim=0))
                if isinstance(strong_output, tuple):
                    strong_output = strong_output[0]
                outputs_s1, outputs_s2 = strong_output.chunk(2, dim=0)

                loss_u_s1, _ = baseline.confidence_masked_baseline_loss(
                    outputs_s1, pseudo_labels_s1, confidence_s1,
                    dice_loss)
                loss_u_s2, _ = baseline.confidence_masked_baseline_loss(
                    outputs_s2, pseudo_labels_s2, confidence_s2,
                    dice_loss)
                loss_u_fp, confident_mask = (
                    baseline.confidence_masked_baseline_loss(
                        outputs_fp[args.labeled_bs:], pseudo_labels,
                        pseudo_confidence, dice_loss))
                baseline_consistency_loss = (
                    0.25 * loss_u_s1 + 0.25 * loss_u_s2 +
                    0.5 * loss_u_fp)

                rescue_s1 = masked_soft_kl(
                    outputs_s1, rescue_target_s1, rescue_mask_s1)
                rescue_s2 = masked_soft_kl(
                    outputs_s2, rescue_target_s2, rescue_mask_s2)
                rescue_fp = masked_soft_kl(
                    outputs_fp[args.labeled_bs:], rescue_target,
                    rescue_mask)
                rescue_loss = (
                    0.25 * rescue_s1 + 0.25 * rescue_s2 +
                    0.5 * rescue_fp)
                consistency_loss = (
                    baseline_consistency_loss +
                    current_rescue_weight * rescue_loss)

                confident_ratio = confident_mask.float().mean()
                foreground = pseudo_labels == 1
                confident_fg_ratio = (
                    (confident_mask & foreground).float().sum() /
                    foreground.float().sum().clamp_min(1.0))
                rescue_coverage = rescue_mask.float().mean()
                if rescue_mask.any():
                    rescue_disagreement = disagreement[rescue_mask].mean()
                else:
                    rescue_disagreement = disagreement.new_zeros(())

            outputs_soft = torch.softmax(outputs, dim=1)
            loss_ce = ce_loss(
                outputs[:args.labeled_bs],
                label_batch[:args.labeled_bs].long())
            loss_dice = dice_loss(
                outputs_soft[:args.labeled_bs],
                label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)

            consistency_weight = baseline.get_current_consistency_weight(
                iter_num // 150)
            loss = supervised_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            baseline.update_model_ema(model, ema_model, args.ema_decay)
            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar(
                'info/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar(
                'unimatch/baseline_consistency_loss',
                baseline_consistency_loss, iter_num)
            writer.add_scalar('fg_rescue/loss', rescue_loss, iter_num)
            writer.add_scalar(
                'fg_rescue/weight', current_rescue_weight, iter_num)
            writer.add_scalar(
                'fg_rescue/coverage', rescue_coverage, iter_num)
            writer.add_scalar(
                'fg_rescue/view_disagreement',
                rescue_disagreement, iter_num)

            if iter_num % 20 == 0:
                logging.info(
                    'self iter %d/%d loss=%.6f sup=%.6f uni=%.6f '
                    'base=%.6f rescue=%.6f rescue_w=%.4f '
                    'coverage=%.4f fg_coverage=%.4f rescue_cov=%.5f '
                    'rescue_disagree=%.4f',
                    iter_num, max_iterations, loss.item(),
                    supervised_loss.item(), consistency_loss.item(),
                    baseline_consistency_loss.item(), rescue_loss.item(),
                    current_rescue_weight, confident_ratio.item(),
                    confident_fg_ratio.item(), rescue_coverage.item(),
                    rescue_disagreement.item())

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for sampled_val in valloader:
                    metric_i = baseline.val_2d.test_single_volume(
                        sampled_val["image"], sampled_val["label"],
                        model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar(
                    'info/val_mean_dice', performance, iter_num)
                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(
                        snapshot_path,
                        'iter_{}_dice_{}.pth'.format(
                            iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(
                        snapshot_path,
                        '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                logging.info(
                    'iteration %d : mean_dice : %f, best_dice : %f',
                    iter_num, performance, best_performance)
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_{}.pth'.format(iter_num))
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to %s", save_mode_path)
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    args = baseline.args
    if args.exp == 'MT_PROMISE12_UniMatch':
        args.exp = 'MT_PROMISE12_UniMatch_FGRescue'
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
    for snapshot_path in (pre_snapshot_path, self_snapshot_path):
        os.makedirs(snapshot_path, exist_ok=True)
    code_snapshot = os.path.join(self_snapshot_path, 'code')
    if os.path.exists(code_snapshot):
        shutil.rmtree(code_snapshot)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(pre_snapshot_path + "/log.txt"),
            logging.StreamHandler(sys.stdout)], force=True)
    logging.info(str(args))
    baseline.pre_train(args, pre_snapshot_path)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(self_snapshot_path + "/log.txt"),
            logging.StreamHandler(sys.stdout)], force=True)
    logging.info("================ START SELF-TRAINING ================")
    self_train(args, pre_snapshot_path, self_snapshot_path)
