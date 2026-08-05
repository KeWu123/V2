"""Deterministic UtilityMatch checks that do not read PROMISE12 data."""

from pathlib import Path

import torch
from torch import nn
from utilitymatch import (
    freeze_batchnorm_running_stats,
    gradient_projection_utility,
    head_gradient,
    select_top_candidates,
)


def check_gradient_ranking():
    parameter = nn.Parameter(torch.tensor([1.0, -1.0]))
    reference_loss = (parameter[0] - 0.0).square()
    reference = head_gradient(reference_loss, (parameter,))

    candidate_losses = (
        (parameter[0] - 0.5).square(),
        (parameter[0] - 2.0).square(),
        0.5 * (parameter[0] - 0.8).square(),
        (parameter[1] + 1.0).square(),
    )
    candidates = [head_gradient(loss, (parameter,)) for loss in candidate_losses]
    utilities = torch.stack(
        [gradient_projection_utility(candidate, reference) for candidate in candidates]
    )
    torch.testing.assert_close(utilities, torch.tensor([1.0, -2.0, 0.2, 0.0]))
    selected = select_top_candidates(utilities, keep=2)
    if selected.tolist() != [0, 2]:
        raise AssertionError(
            "Expected positive clean-gradient candidates [0, 2], got "
            f"{selected.tolist()}"
        )
    if parameter.grad is not None:
        raise AssertionError("head_gradient must not populate parameter.grad")


def check_batchnorm_buffers():
    torch.manual_seed(1337)
    model = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.BatchNorm2d(4))
    model.train()
    images = torch.randn(8, 1, 16, 16) + 3.0
    batch_norm = model[1]
    mean_before = batch_norm.running_mean.clone()
    count_before = batch_norm.num_batches_tracked.clone()
    with freeze_batchnorm_running_stats(model):
        model(images)
    torch.testing.assert_close(batch_norm.running_mean, mean_before)
    torch.testing.assert_close(batch_norm.num_batches_tracked, count_before)
    if not batch_norm.track_running_stats:
        raise AssertionError("BatchNorm state was not restored")
    model(images)
    if torch.equal(batch_norm.running_mean, mean_before):
        raise AssertionError("Normal train-mode BatchNorm update did not occur")
    if int(batch_norm.num_batches_tracked) != int(count_before) + 1:
        raise AssertionError("Normal BatchNorm update count is incorrect")


def check_ema_teacher_modes():
    code_dir = Path(__file__).resolve().parent
    for filename in ("train_unimatch.py", "train_utilitymatch.py"):
        source = (code_dir / filename).read_text(encoding="utf-8")
        if "ema_model.train()" not in source:
            raise AssertionError(
                f"{filename} does not restore train-mode EMA inference"
            )
        if "ema_model.eval()" in source:
            raise AssertionError(f"{filename} still enables eval-mode EMA inference")


def main():
    check_gradient_ranking()
    check_batchnorm_buffers()
    check_ema_teacher_modes()
    print(
        "UtilityMatch smoke test passed: signed utility ranking, top-2 "
        "selection, no .grad mutation, frozen candidate BN buffers, and "
        "train-mode EMA teachers"
    )


if __name__ == "__main__":
    main()
