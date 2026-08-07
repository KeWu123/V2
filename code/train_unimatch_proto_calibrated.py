"""UniMatch with labeled-prototype-calibrated pseudo-label weights.

The supplied UniMatch implementation is imported unchanged.  This file only
replaces self-training so that the original EMA+LCC pseudo labels are weighted
by their cosine agreement with class prototypes computed exclusively from
ground-truth labeled pixels.

Implementation references:
  * UniMatch: https://github.com/LiheYoung/UniMatch
  * U2PL prototype/cosine implementation:
    https://github.com/Haochen-Wang409/U2PL
  * SCP-Net medical prototypical learning:
    https://github.com/Medsemiseg/SCP-Net

This is a conservative adaptation, not a reproduction of PECL: it adds no
contrastive loss, no second teacher, and no trainable projection head.
"""

from train_unimatch import *


PROTOTYPE_MOMENTUM = float(os.environ.get("PROTOTYPE_MOMENTUM", "0.9"))
PROTOTYPE_TEMPERATURE = float(os.environ.get("PROTOTYPE_TEMPERATURE", "0.2"))
PROTOTYPE_MIN_WEIGHT = float(os.environ.get("PROTOTYPE_MIN_WEIGHT", "0.5"))


@torch.no_grad()
def update_labeled_prototypes(features, labels, prototypes, initialized):
    """Update class prototypes from EMA features and ground-truth pixels only."""
    if labels.shape[-2:] != features.shape[-2:]:
        labels = F.interpolate(
            labels.unsqueeze(1).float(), size=features.shape[-2:],
            mode="nearest").squeeze(1).long()
    else:
        labels = labels.long()

    normalized_features = F.normalize(features, p=2, dim=1, eps=1e-6)
    pixel_features = normalized_features.permute(0, 2, 3, 1)

    for class_id in range(prototypes.shape[0]):
        class_mask = labels.eq(class_id)
        if not class_mask.any():
            continue
        batch_prototype = pixel_features[class_mask].mean(dim=0)
        batch_prototype = F.normalize(
            batch_prototype.unsqueeze(0), p=2, dim=1,
            eps=1e-6).squeeze(0)
        if initialized[class_id]:
            batch_prototype = (
                PROTOTYPE_MOMENTUM * prototypes[class_id] +
                (1.0 - PROTOTYPE_MOMENTUM) * batch_prototype)
            batch_prototype = F.normalize(
                batch_prototype.unsqueeze(0), p=2, dim=1,
                eps=1e-6).squeeze(0)
        prototypes[class_id].copy_(batch_prototype)
        initialized[class_id] = True


@torch.no_grad()
def prototype_reliability(features, pseudo_labels, prototypes, initialized):
    """Return a soft reliability weight without changing pseudo-label classes."""
    if not bool(initialized.all()):
        ones = torch.ones_like(pseudo_labels, dtype=features.dtype)
        return ones, pseudo_labels.clone()

    if pseudo_labels.shape[-2:] != features.shape[-2:]:
        pseudo_at_feature_size = F.interpolate(
            pseudo_labels.unsqueeze(1).float(),
            size=features.shape[-2:], mode="nearest").squeeze(1).long()
    else:
        pseudo_at_feature_size = pseudo_labels.long()

    normalized_features = F.normalize(features, p=2, dim=1, eps=1e-6)
    normalized_prototypes = F.normalize(
        prototypes, p=2, dim=1, eps=1e-6)
    cosine_logits = torch.einsum(
        "bchw,kc->bkhw", normalized_features, normalized_prototypes)
    prototype_probability = torch.softmax(
        cosine_logits / PROTOTYPE_TEMPERATURE, dim=1)
    matched_probability = prototype_probability.gather(
        1, pseudo_at_feature_size.unsqueeze(1)).squeeze(1)
    prototype_prediction = prototype_probability.argmax(dim=1)

    reliability = (
        PROTOTYPE_MIN_WEIGHT +
        (1.0 - PROTOTYPE_MIN_WEIGHT) * matched_probability)
    if reliability.shape[-2:] != pseudo_labels.shape[-2:]:
        reliability = F.interpolate(
            reliability.unsqueeze(1), size=pseudo_labels.shape[-2:],
            mode="bilinear", align_corners=False).squeeze(1)
        prototype_prediction = F.interpolate(
            prototype_prediction.unsqueeze(1).float(),
            size=pseudo_labels.shape[-2:],
            mode="nearest").squeeze(1).long()
    return reliability.clamp_(PROTOTYPE_MIN_WEIGHT, 1.0), prototype_prediction


def prototype_weighted_pseudo_loss(
        logits, targets, confidence, reliability, dice_loss):
    """Keep the UniMatch confidence gate and softly weight accepted pixels."""
    valid = confidence >= args.confidence_threshold
    pixel_weight = valid.float() * reliability.detach()
    per_pixel_ce = F.cross_entropy(
        logits, targets.long(), reduction="none")
    loss_ce = ((per_pixel_ce * pixel_weight).sum() /
               pixel_weight.sum().clamp_min(1.0))
    loss_dice = dice_loss(
        torch.softmax(logits, dim=1), targets.unsqueeze(1),
        mask=pixel_weight.unsqueeze(1))
    return 0.5 * (loss_ce + loss_dice), valid


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
    optimizer = optim.SGD(
        model.parameters(), lr=base_lr,
        momentum=0.9, weight_decay=0.0001)

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

    db_train = BaseDataSets(
        base_dir=args.root_path, split="train", num=None,
        transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size,
        batch_size - args.labeled_bs)
    trainloader = DataLoader(
        db_train, batch_sampler=batch_sampler, num_workers=4,
        pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.eval()
    valloader = DataLoader(
        db_val, batch_size=1, shuffle=False, num_workers=1)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    writer = SummaryWriter(snapshot_path + '/log')

    # Decoder features have 16 channels in the supplied U-Net.  The prototype
    # state is diagnostic/training-only and does not alter the checkpoint.
    prototype_channels = model.decoder.ft_chns[0]
    prototypes = torch.zeros(
        num_classes, prototype_channels, device='cuda')
    prototype_initialized = torch.zeros(
        num_classes, dtype=torch.bool, device='cuda')

    logging.info("%d iterations per epoch", len(trainloader))
    logging.info(
        "UniMatch unchanged: EMA+LCC, tau=%.2f, feature_dropout=%.2f, "
        "branch_weights=(0.25, 0.25, 0.50), strong/blur/CutMix=%.2f/%.2f/%.2f",
        args.confidence_threshold, args.feature_dropout,
        args.strong_aug_prob, args.blur_prob, args.cutmix_prob)
    logging.info(
        "GT prototype calibration only: momentum=%.3f temperature=%.3f "
        "min_weight=%.3f; no hard prototype filtering and no pseudo-class replacement",
        PROTOTYPE_MOMENTUM, PROTOTYPE_TEMPERATURE,
        PROTOTYPE_MIN_WEIGHT)

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            if iter_num < 1000:
                model_output = model(volume_batch)
                outputs = (model_output[0] if isinstance(model_output, tuple)
                           else model_output)
                consistency_loss = outputs.new_zeros(())
                loss_u_s1 = outputs.new_zeros(())
                loss_u_s2 = outputs.new_zeros(())
                loss_u_fp = outputs.new_zeros(())
                confident_ratio = outputs.new_zeros(())
                confident_fg_ratio = outputs.new_zeros(())
                prototype_quality_mean = outputs.new_zeros(())
                prototype_quality_fg = outputs.new_zeros(())
                prototype_disagreement = outputs.new_zeros(())
            else:
                strong_view1 = strong_mri_augmentation(
                    unlabeled_volume_batch)
                strong_view2 = strong_mri_augmentation(
                    unlabeled_volume_batch)

                unlabeled_count, _, height, width = (
                    unlabeled_volume_batch.shape)
                permutation1 = torch.randperm(
                    unlabeled_count,
                    device=unlabeled_volume_batch.device)
                permutation2 = torch.randperm(
                    unlabeled_count,
                    device=unlabeled_volume_batch.device)
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
                    ema_output, ema_features = ema_model(volume_batch)
                    ema_unlabeled_output = ema_output[args.labeled_bs:]
                    ema_unlabeled_features = ema_features[args.labeled_bs:]
                    update_labeled_prototypes(
                        ema_features[:args.labeled_bs],
                        label_batch[:args.labeled_bs],
                        prototypes, prototype_initialized)

                    ema_probability = torch.softmax(
                        ema_unlabeled_output, dim=1)
                    pseudo_labels = get_masks(
                        ema_unlabeled_output, nms=1).long()
                    pseudo_confidence = ema_probability.gather(
                        1, pseudo_labels.unsqueeze(1)).squeeze(1)
                    quality_weight, prototype_prediction = (
                        prototype_reliability(
                            ema_unlabeled_features, pseudo_labels,
                            prototypes, prototype_initialized))

                    pseudo_labels_s1 = cutmix_tensor(
                        pseudo_labels, pseudo_labels[permutation1],
                        cutmix_box1)
                    pseudo_labels_s2 = cutmix_tensor(
                        pseudo_labels, pseudo_labels[permutation2],
                        cutmix_box2)
                    confidence_s1 = cutmix_tensor(
                        pseudo_confidence,
                        pseudo_confidence[permutation1], cutmix_box1)
                    confidence_s2 = cutmix_tensor(
                        pseudo_confidence,
                        pseudo_confidence[permutation2], cutmix_box2)
                    quality_s1 = cutmix_tensor(
                        quality_weight, quality_weight[permutation1],
                        cutmix_box1)
                    quality_s2 = cutmix_tensor(
                        quality_weight, quality_weight[permutation2],
                        cutmix_box2)

                # These are exactly the original UniMatch student branches.
                outputs, outputs_fp = model(
                    volume_batch, need_fp=True,
                    feature_dropout=args.feature_dropout)
                strong_output = model(
                    torch.cat((strong_view1, strong_view2), dim=0))
                if isinstance(strong_output, tuple):
                    strong_output = strong_output[0]
                outputs_s1, outputs_s2 = strong_output.chunk(2, dim=0)

                loss_u_s1, _ = prototype_weighted_pseudo_loss(
                    outputs_s1, pseudo_labels_s1, confidence_s1,
                    quality_s1, dice_loss)
                loss_u_s2, _ = prototype_weighted_pseudo_loss(
                    outputs_s2, pseudo_labels_s2, confidence_s2,
                    quality_s2, dice_loss)
                loss_u_fp, confident_mask = prototype_weighted_pseudo_loss(
                    outputs_fp[args.labeled_bs:], pseudo_labels,
                    pseudo_confidence, quality_weight, dice_loss)
                consistency_loss = (
                    0.25 * loss_u_s1 + 0.25 * loss_u_s2 +
                    0.5 * loss_u_fp)

                confident_ratio = confident_mask.float().mean()
                foreground = pseudo_labels.eq(1)
                confident_fg = confident_mask & foreground
                confident_fg_ratio = (
                    confident_fg.float().sum() /
                    foreground.float().sum().clamp_min(1.0))
                prototype_quality_mean = (
                    (quality_weight * confident_mask.float()).sum() /
                    confident_mask.float().sum().clamp_min(1.0))
                prototype_quality_fg = (
                    (quality_weight * confident_fg.float()).sum() /
                    confident_fg.float().sum().clamp_min(1.0))
                prototype_disagreement = (
                    ((prototype_prediction != pseudo_labels) &
                     confident_mask).float().sum() /
                    confident_mask.float().sum().clamp_min(1.0))

            outputs_soft = torch.softmax(outputs, dim=1)
            loss_ce = ce_loss(
                outputs[:args.labeled_bs],
                label_batch[:args.labeled_bs].long())
            loss_dice = dice_loss(
                outputs_soft[:args.labeled_bs],
                label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)
            consistency_weight = get_current_consistency_weight(
                iter_num // 150)
            loss = supervised_loss + consistency_weight * consistency_loss

            # Preserve the current UniMatch self-training LR behavior.
            progress = min(float(iter_num) / float(max_iterations), 1.0)
            lr_ = base_lr * (1.0 - progress) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_model_ema(model, ema_model, args.ema_decay)

            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar(
                'info/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar(
                'info/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar(
                'unimatch/loss_strong1', loss_u_s1, iter_num)
            writer.add_scalar(
                'unimatch/loss_strong2', loss_u_s2, iter_num)
            writer.add_scalar(
                'unimatch/loss_feature', loss_u_fp, iter_num)
            writer.add_scalar(
                'unimatch/confident_ratio', confident_ratio, iter_num)
            writer.add_scalar(
                'unimatch/confident_foreground_ratio',
                confident_fg_ratio, iter_num)
            writer.add_scalar(
                'prototype/quality_mean', prototype_quality_mean, iter_num)
            writer.add_scalar(
                'prototype/quality_foreground',
                prototype_quality_fg, iter_num)
            writer.add_scalar(
                'prototype/disagreement_ratio',
                prototype_disagreement, iter_num)

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs_img = torch.argmax(
                    torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image(
                    'train/Prediction', outputs_img[1, ...] * 50,
                    iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
                logging.info(
                    'self iter %d/%d loss=%.6f sup=%.6f uni=%.6f '
                    's1=%.6f s2=%.6f fp=%.6f coverage=%.4f '
                    'fg_coverage=%.4f proto_q=%.4f proto_fg_q=%.4f '
                    'proto_disagree=%.4f',
                    iter_num, max_iterations, loss.item(),
                    supervised_loss.item(), consistency_loss.item(),
                    loss_u_s1.item(), loss_u_s2.item(), loss_u_fp.item(),
                    confident_ratio.item(), confident_fg_ratio.item(),
                    prototype_quality_mean.item(),
                    prototype_quality_fg.item(),
                    prototype_disagreement.item())

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, val_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(
                        val_batch["image"], val_batch["label"], model,
                        classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar(
                        'info/val_{}_dice'.format(class_i + 1),
                        metric_list[class_i, 0], iter_num)
                    writer.add_scalar(
                        'info/val_{}_hd95'.format(class_i + 1),
                        metric_list[class_i, 1], iter_num)
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
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
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
    if args.exp == 'MT_PROMISE12_UniMatch':
        args.exp = 'MT_PROMISE12_UniMatch_ProtoCal'

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

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(pre_snapshot_path + "/log.txt"),
            logging.StreamHandler(sys.stdout)],
        force=True)
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(self_snapshot_path + "/log.txt"),
            logging.StreamHandler(sys.stdout)],
        force=True)
    logging.info("================ START SELF-TRAINING ================")
    self_train(args, pre_snapshot_path, self_snapshot_path)
