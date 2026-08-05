"""Mechanism checks for range-calibrated SafeUtilityMatch."""

from pathlib import Path
from unittest.mock import patch

import torch
import train_utilitymatch_calibrated as calibrated
from test_utilitymatch_safe_smoke import main as check_safe_utilitymatch


class Args:
    strong_aug_prob = 0.8
    blur_prob = 0.5


def check_range_relative_brightness():
    calibrated.safe.legacy.base.args = Args()
    if calibrated._trace_iteration(Args(), 1) != 1001:
        raise AssertionError("minimal augmentation args cannot resolve trace iteration")
    low = torch.tensor([[[0.0, 0.1], [0.2, 0.4]]])
    high = low * 2.0
    images = torch.stack((low, high), dim=0)
    # Per image: enable intensity, contrast=1, brightness fraction=0.25,
    # then disable blur. The same random sequence is used for both images.
    with patch.object(
        calibrated.random, "random", side_effect=[0.0, 1.0, 0.0, 1.0]
    ), patch.object(
        calibrated.random,
        "uniform",
        side_effect=[1.0, 0.25, 1.0, 0.25],
    ):
        output = calibrated.range_calibrated_mri_augmentation(images)
    shift_low = output[0] - low
    shift_high = output[1] - high
    # The exact p01--p99 value depends on quantile interpolation. Scale
    # equivariance is the locked property: doubling intensities must double
    # the applied brightness offset while keeping the sampled fraction fixed.
    torch.testing.assert_close(shift_high, shift_low * 2.0)
    if not torch.all(shift_low > 0):
        raise AssertionError("positive brightness fraction did not raise intensity")


def check_source_invariants():
    code_dir = Path(__file__).resolve().parent
    source = (code_dir / "train_utilitymatch_calibrated.py").read_text(
        encoding="utf-8"
    )
    for fragment in (
        "random.uniform(0.5, 1.5)",
        "random.uniform(-0.25, 0.25)",
        "random.uniform(0.1, 2.0)",
        "torch.quantile(sampled, 0.01",
        "torch.quantile(sampled, 0.99",
        "CALIBRATED-UTILITYMATCH VERIFIED",
        "safe.install_safe_policy()",
        "safe.legacy.main(args)",
    ):
        if fragment not in source:
            raise AssertionError(f"calibrated invariant missing: {fragment}")


def main():
    check_safe_utilitymatch()
    check_range_relative_brightness()
    check_source_invariants()
    print("CalibratedUtilityMatch smoke test passed")


if __name__ == "__main__":
    main()
