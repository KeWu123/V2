"""CPU smoke tests for trajectory reliability and soft boundary recovery."""

import torch

from trajectory_reliability import (
    adaptive_unsupervised_scale,
    boundary_band,
    soft_boundary_target,
    trajectory_statistics,
)


def check_trajectory_ordering():
    stable = torch.tensor([0.80, 0.84, 0.87, 0.90]).view(4, 1, 1, 1)
    flipping = torch.tensor([0.20, 0.80, 0.20, 0.80]).view(4, 1, 1, 1)
    history = torch.cat((stable, flipping), dim=2)
    stats = trajectory_statistics(history)
    if not stats["reliability"][0, 0, 0] > stats["reliability"][0, 1, 0]:
        raise AssertionError("Stable trajectory must outrank a flipping trajectory")


def check_adaptive_scale():
    labels = torch.ones(1, 4, 4, dtype=torch.long)
    low_scale, low_readiness = adaptive_unsupervised_scale(
        torch.full((1, 4, 4), 0.2), labels
    )
    high_scale, high_readiness = adaptive_unsupervised_scale(
        torch.full((1, 4, 4), 0.95), labels
    )
    if not high_readiness > low_readiness or not high_scale > low_scale:
        raise AssertionError("Higher foreground reliability must increase trust")


def check_soft_boundary():
    torch.manual_seed(1337)
    labels = torch.zeros(1, 16, 16, dtype=torch.long)
    labels[:, 4:12, 4:12] = 1
    boundary = boundary_band(labels, radius=1)
    if not boundary.any() or boundary.all():
        raise AssertionError("Boundary band is malformed")
    features = torch.randn(1, 8, 16, 16)
    probability = labels.float() * 0.7 + 0.15
    reliability = torch.full((1, 16, 16), 0.8)
    target, weight, _, core, _ = soft_boundary_target(
        features, probability, labels, reliability, radius=1
    )
    if target.min() < 0 or target.max() > 1:
        raise AssertionError("Soft target left the probability range")
    if not core.any() or weight.sum() <= 0:
        raise AssertionError("Stable-core boundary guidance is inactive")


def main():
    check_trajectory_ordering()
    check_adaptive_scale()
    check_soft_boundary()
    print(
        "Trajectory reliability smoke test passed: ordering, adaptive trust, "
        "and soft boundary guidance"
    )


if __name__ == "__main__":
    main()
