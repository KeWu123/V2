"""Embedding matching losses for the PROMISE12 UniMatch experiment.

The implementation follows *Semi-Supervised Segmentation via Embedding
Matching* (MIDL 2024): teacher features from labeled surface voxels are
matched to dense student features from unlabeled images.  Several randomly
sampled nearest-neighbour classifiers are averaged, L_NN supervises unreliable
pixels, and L_EN separates the foreground/background embedding similarities.

There is intentionally no cross-iteration feature memory and no additional
similarity/confidence acceptance threshold.  Those two choices in the previous
prototype deviated from the paper and left only a handful of background
targets per batch.
"""

import torch
import torch.nn.functional as F

# Algorithm: https://openreview.net/pdf?id=xkqLQoFQbl
# Normalized class-feature handling reference:
# https://github.com/IsYuchenYuan/PPC/blob/main/code/networks/unet_proto.py


def _binary_dilate(mask, radius):
    radius = int(radius)
    if radius <= 0:
        return mask.bool()
    kernel_size = 2 * radius + 1
    return F.max_pool2d(
        mask.float().unsqueeze(1), kernel_size,
        stride=1, padding=radius).squeeze(1) > 0.5


def _binary_erode(mask, radius):
    radius = int(radius)
    if radius <= 0:
        return mask.bool()
    kernel_size = 2 * radius + 1
    eroded = 1.0 - F.max_pool2d(
        (~mask.bool()).float().unsqueeze(1), kernel_size,
        stride=1, padding=radius).squeeze(1)
    return eroded > 0.5


def surface_reference_masks(labels, radius):
    """Return equally defined foreground-inside/background-outside bands."""
    foreground = labels.long() == 1
    eroded = _binary_erode(foreground, radius)
    dilated = _binary_dilate(foreground, radius)
    foreground_inside = foreground & (~eroded)
    background_outside = (~foreground) & dilated
    return foreground_inside, background_outside


def _sample_rows(vectors, count, generator):
    """Randomly sample rows, repeating only when a surface has fewer than k."""
    available = int(vectors.shape[0])
    count = int(count)
    if available >= count:
        indices = torch.randperm(
            available, generator=generator, device=vectors.device)[:count]
    else:
        indices = torch.randint(
            available, (count,), generator=generator, device=vectors.device)
    return vectors[indices]


def ensemble_embedding_classifier(teacher_labeled_features,
                                  labeled_targets,
                                  student_unlabeled_features,
                                  surface_radius=2,
                                  references_per_class=16,
                                  ensemble_size=5,
                                  temperature=1.0,
                                  random_seed=1337):
    """Build the paper's ensemble of dense cosine-similarity classifiers.

    Teacher reference vectors are detached.  Student query vectors retain their
    graph because L_EN must improve their separation.  Averaging the k pairwise
    cosine similarities is computed efficiently as a dot product with the mean
    of k already-normalized reference vectors; the operations are equivalent.
    """
    if teacher_labeled_features.ndim != 4:
        raise ValueError("teacher_labeled_features must be [B, C, H, W]")
    if student_unlabeled_features.ndim != 4:
        raise ValueError("student_unlabeled_features must be [B, C, H, W]")
    if teacher_labeled_features.shape[1] != student_unlabeled_features.shape[1]:
        raise ValueError("teacher and student embedding dimensions must match")
    if references_per_class <= 0 or ensemble_size <= 0:
        raise ValueError("references_per_class and ensemble_size must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    reference_size = teacher_labeled_features.shape[-2:]
    if labeled_targets.shape[-2:] != reference_size:
        labeled_targets = F.interpolate(
            labeled_targets.float().unsqueeze(1), size=reference_size,
            mode="nearest").squeeze(1).long()

    teacher_features = F.normalize(
        teacher_labeled_features.detach().float(), dim=1, eps=1e-6)
    student_features = F.normalize(
        student_unlabeled_features.float(), dim=1, eps=1e-6)
    teacher_vectors = teacher_features.permute(0, 2, 3, 1)
    foreground_mask, background_mask = surface_reference_masks(
        labeled_targets, surface_radius)
    foreground_references = teacher_vectors[foreground_mask]
    background_references = teacher_vectors[background_mask]

    zero = student_features.new_zeros(())
    stats = {
        "foreground_reference_count": zero + foreground_references.shape[0],
        "background_reference_count": zero + background_references.shape[0],
        "reference_ready": zero,
    }
    if foreground_references.shape[0] == 0 or background_references.shape[0] == 0:
        return None, None, stats

    # A dedicated generator keeps reference sampling reproducible without
    # changing the random stream used by UniMatch augmentation and dropout.
    generator = torch.Generator(device=teacher_features.device)
    generator.manual_seed(int(random_seed))
    similarity_sum = None
    for _ in range(int(ensemble_size)):
        foreground_sample = _sample_rows(
            foreground_references, references_per_class, generator)
        background_sample = _sample_rows(
            background_references, references_per_class, generator)

        # Class order is [background, foreground], matching the segmentation
        # logits.  Do not renormalize the means: q @ mean(r_k) is exactly the
        # mean of the k cosine similarities required by the paper.
        class_references = torch.stack((
            background_sample.mean(dim=0),
            foreground_sample.mean(dim=0)), dim=0)
        similarity = torch.einsum(
            "bchw,kc->bkhw", student_features, class_references)
        similarity_sum = (
            similarity if similarity_sum is None
            else similarity_sum + similarity)

    mean_similarity = similarity_sum / float(ensemble_size)
    matching_probability = torch.softmax(
        mean_similarity / float(temperature), dim=1)
    # The paper explicitly cuts pseudo-label generation off from autograd.
    matching_target = mean_similarity.detach().argmax(dim=1)
    stats["reference_ready"] = zero + 1.0
    return matching_probability, matching_target, stats


def embedding_matching_losses(student_logits,
                              matching_probability,
                              matching_target,
                              valid_mask,
                              teacher_target=None):
    """Compute L_NN and L_EN on unreliable pixels.

    L_NN uses detached hard nearest-neighbour targets.  L_EN deliberately keeps
    the gradient through ``matching_probability`` so the student's unlabeled
    embedding space is separated, which is a required part of the paper rather
    than an optional diagnostic term.
    """
    zero = student_logits.sum() * 0.0
    empty_stats = {
        "active_ratio": zero.detach(),
        "matching_foreground_ratio": zero.detach(),
        "teacher_disagreement_ratio": zero.detach(),
        "matching_entropy": zero.detach(),
    }
    if matching_probability is None or matching_target is None:
        return zero, zero, empty_stats

    valid_mask = valid_mask.bool()
    active_count = valid_mask.float().sum()
    if active_count.detach().item() <= 0:
        return zero, zero, empty_stats

    per_pixel_nn = F.cross_entropy(
        student_logits, matching_target.detach(), reduction="none")
    loss_nn = (per_pixel_nn * valid_mask.float()).sum() / active_count

    probability = matching_probability.clamp_min(1e-6)
    per_pixel_entropy = -(probability * probability.log()).sum(dim=1)
    loss_entropy = (
        per_pixel_entropy * valid_mask.float()).sum() / active_count

    matching_foreground = matching_target == 1
    if teacher_target is None:
        disagreement = zero.detach()
    else:
        disagreement = (
            ((matching_target != teacher_target.long()) & valid_mask)
            .float().sum() / active_count)

    total_pixels = float(valid_mask.numel())
    stats = {
        "active_ratio": (active_count / total_pixels).detach(),
        "matching_foreground_ratio": (
            (matching_foreground & valid_mask).float().sum() /
            active_count).detach(),
        "teacher_disagreement_ratio": disagreement.detach(),
        "matching_entropy": loss_entropy.detach(),
    }
    return loss_nn, loss_entropy, stats
