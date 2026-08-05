"""Small, testable primitives for clean-gradient strong-view selection."""

from contextlib import contextmanager

import torch
from torch.nn.modules.batchnorm import _BatchNorm


def flatten_gradients(gradients):
    """Flatten a non-empty gradient tuple without changing its values."""
    gradients = tuple(gradients)
    if not gradients:
        raise ValueError("At least one gradient tensor is required")
    if any(gradient is None for gradient in gradients):
        raise ValueError("UtilityMatch does not allow unused head parameters")
    return torch.cat([gradient.reshape(-1) for gradient in gradients])


def head_gradient(loss, parameters, retain_graph=False):
    """Return d(loss)/d(parameters) without writing parameter ``.grad``."""
    parameters = tuple(parameters)
    if loss.ndim != 0:
        raise ValueError("UtilityMatch losses must be scalar")
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=bool(retain_graph),
        create_graph=False,
        allow_unused=False,
    )
    return flatten_gradients(gradients).detach()


def gradient_projection_utility(candidate_gradient, reference_gradient, epsilon=1e-12):
    """Signed candidate-gradient projection onto the clean gradient direction."""
    candidate = candidate_gradient.reshape(-1)
    reference = reference_gradient.reshape(-1)
    if candidate.shape != reference.shape:
        raise ValueError("Candidate and reference gradients must have identical shapes")
    denominator = reference.norm().clamp_min(float(epsilon))
    utility = torch.dot(candidate, reference) / denominator
    if not torch.isfinite(utility):
        raise FloatingPointError("Non-finite gradient utility")
    return utility


def select_top_candidates(utilities, keep=2):
    """Return candidate indices sorted from greatest to smallest utility."""
    if utilities.ndim != 1:
        raise ValueError("utilities must be one-dimensional")
    keep = int(keep)
    if keep < 1 or keep > utilities.numel():
        raise ValueError("keep must be in [1, number of candidates]")
    if not torch.isfinite(utilities).all():
        raise FloatingPointError("Candidate utilities contain NaN or Inf")
    return torch.topk(utilities.detach(), k=keep, largest=True, sorted=True).indices


@contextmanager
def freeze_batchnorm_running_stats(model):
    """Use batch statistics without mutating any BatchNorm running buffer.

    The surrounding model remains in train mode, so dropout behavior is not
    silently converted to evaluation behavior. Only ``track_running_stats`` is
    temporarily disabled for BatchNorm modules and is restored on exit.
    """
    states = []
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            states.append((module, module.track_running_stats))
            module.track_running_stats = False
    try:
        yield
    finally:
        for module, track_running_stats in states:
            module.track_running_stats = track_running_stats
