"""Trajectory-derived trust signals for semi-supervised segmentation.

Historical teachers estimate reliability only. The current EMA teacher remains
the sole source of pseudo-label targets, which keeps this distinct from
temporal ensembling.
"""

import copy
from collections import deque

import torch
import torch.nn.functional as F


def trajectory_statistics(probability_history):
    """Return agreement, variance stability, flip stability, and reliability."""
    if probability_history.ndim != 4:
        raise ValueError("Expected history with shape [K, B, H, W]")
    if probability_history.shape[0] < 2:
        raise ValueError("At least two historical predictions are required")

    probabilities = probability_history.float().clamp(0.0, 1.0)
    classes = probabilities >= 0.5
    latest_class = classes[-1:]
    agreement = (classes == latest_class).float().mean(dim=0)
    variance = probabilities.var(dim=0, unbiased=False)
    variance_stability = (1.0 - 4.0 * variance).clamp(0.0, 1.0)
    flip_rate = (classes[1:] != classes[:-1]).float().mean(dim=0)
    flip_stability = 1.0 - flip_rate
    reliability = (
        agreement + variance_stability + flip_stability
    ) / 3.0
    return {
        "agreement": agreement,
        "variance": variance,
        "variance_stability": variance_stability,
        "flip_rate": flip_rate,
        "flip_stability": flip_stability,
        "reliability": reliability.clamp(0.0, 1.0),
    }


def boundary_band(mask, radius=2):
    """Return a symmetric morphological boundary band for a binary mask."""
    if mask.ndim != 3:
        raise ValueError("Expected mask with shape [B, H, W]")
    radius = int(radius)
    if radius < 1:
        raise ValueError("Boundary radius must be positive")
    foreground = mask.float().unsqueeze(1)
    kernel_size = 2 * radius + 1
    dilated = F.max_pool2d(
        foreground, kernel_size=kernel_size, stride=1, padding=radius
    )
    eroded = 1.0 - F.max_pool2d(
        1.0 - foreground, kernel_size=kernel_size, stride=1, padding=radius
    )
    return ((dilated - eroded) > 0).squeeze(1)


def adaptive_unsupervised_scale(
    reliability,
    pseudo_labels,
    stable_threshold=0.75,
    readiness_start=0.45,
    readiness_full=0.80,
    minimum_scale=0.02,
):
    """Map foreground trajectory maturity to an unsupervised loss scale."""
    foreground = pseudo_labels == 1
    if not foreground.any():
        readiness = reliability.new_zeros(())
    else:
        foreground_reliability = reliability[foreground]
        mean_reliability = foreground_reliability.mean()
        stable_fraction = (
            foreground_reliability >= float(stable_threshold)
        ).float().mean()
        readiness = 0.5 * (mean_reliability + stable_fraction)

    denominator = max(float(readiness_full) - float(readiness_start), 1e-6)
    normalized = (
        (readiness - float(readiness_start)) / denominator
    ).clamp(0.0, 1.0)
    scale = float(minimum_scale) + (1.0 - float(minimum_scale)) * normalized
    return scale, readiness


def weighted_pseudo_loss(logits, targets, weights, dice_loss):
    """Continuous-weight counterpart of confidence-masked CE plus Dice."""
    weights = weights.float().clamp(0.0, 1.0)
    per_pixel_ce = F.cross_entropy(logits, targets.long(), reduction="none")
    mass = weights.sum()
    loss_ce = (per_pixel_ce * weights).sum() / mass.clamp_min(1.0)
    loss_dice = dice_loss(
        torch.softmax(logits, dim=1),
        targets.unsqueeze(1),
        mask=weights.unsqueeze(1),
    )
    return 0.5 * (loss_ce + loss_dice), mass


def soft_boundary_target(
    decoder_features,
    teacher_foreground_probability,
    pseudo_labels,
    reliability,
    core_threshold=0.75,
    radius=2,
    recovery_strength=0.50,
    minimum_weight=0.10,
):
    """Use a stable foreground prototype to form a soft boundary target.

    No boundary pixel is hard-converted to foreground. Prototype similarity is
    detached and softly blended with the current EMA probability only where a
    stable foreground core exists.
    """
    features = F.normalize(decoder_features.detach().float(), dim=1)
    boundary = boundary_band(pseudo_labels, radius=radius)
    stable_core = (
        (pseudo_labels == 1)
        & (~boundary)
        & (reliability >= float(core_threshold))
    )

    batch_size, channels, _, _ = features.shape
    core_float = stable_core.float().unsqueeze(1)
    core_count = core_float.sum(dim=(2, 3), keepdim=True)
    prototype = (features * core_float).sum(dim=(2, 3), keepdim=True)
    prototype = prototype / core_count.clamp_min(1.0)
    prototype = F.normalize(prototype, dim=1)
    similarity = (features * prototype).sum(dim=1).clamp(-1.0, 1.0)
    similarity_probability = 0.5 * (similarity + 1.0)

    valid_core = (core_count.reshape(batch_size) > 0).view(-1, 1, 1)
    blend = (
        float(recovery_strength) * (1.0 - reliability)
    ).clamp(0.0, 1.0)
    blend = blend * boundary.float() * valid_core.float()
    target = (
        (1.0 - blend) * teacher_foreground_probability.detach()
        + blend * similarity_probability.detach()
    ).clamp(0.0, 1.0)
    weight = boundary.float() * valid_core.float() * torch.maximum(
        reliability.detach(),
        reliability.new_full((), float(minimum_weight)),
    )
    return target, weight, boundary, stable_core, similarity_probability


def weighted_soft_binary_loss(logits, target_probability, weights):
    foreground_logit = logits[:, 1] - logits[:, 0]
    per_pixel = F.binary_cross_entropy_with_logits(
        foreground_logit, target_probability.float(), reduction="none"
    )
    weights = weights.float().clamp_min(0.0)
    return (per_pixel * weights).sum() / weights.sum().clamp_min(1.0)


class TeacherTrajectory:
    """Small on-device queue of deterministic historical EMA teachers."""

    def __init__(self, history_size=4):
        if int(history_size) < 2:
            raise ValueError("history_size must be at least two")
        self.history_size = int(history_size)
        self.models = deque()

    def __len__(self):
        return len(self.models)

    def append(self, teacher):
        snapshot = copy.deepcopy(teacher)
        snapshot.eval()
        for parameter in snapshot.parameters():
            parameter.requires_grad_(False)
        self.models.append(snapshot)
        while len(self.models) > self.history_size:
            expired = self.models.popleft()
            del expired

    @torch.no_grad()
    def probabilities(self, images):
        if len(self.models) < 2:
            raise RuntimeError("Teacher trajectory is not ready")
        predictions = []
        for model in self.models:
            output = model(images)
            logits = output[0] if isinstance(output, tuple) else output
            predictions.append(torch.softmax(logits, dim=1)[:, 1])
        return torch.stack(predictions, dim=0)

