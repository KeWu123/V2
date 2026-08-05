"""CalibratedUtilityMatch for the actual non-negative PROMISE12 H5 domain.

The fixed strong-view brightness coefficient is expressed in units of each
slice's sampled p01--p99 range instead of as an absolute 0.25 shift. This
removes the hidden assumption that the input is z-score normalized. The
strict positive-utility abstention from SafeUtilityMatch is retained so that
range-calibrated candidates still require labeled-task gradient agreement.
"""

import csv
import hashlib
import json
import logging
import os
import random
from pathlib import Path

import torch
import train_utilitymatch_safe as safe

logger = logging.getLogger(__name__)
_augmentation_calls = 0
_augmentation_trace_initialized = False


def _trace_iteration(args, augmentation_calls):
    """Map candidate-call count to iteration without requiring trainer args."""
    warmup_iterations = int(getattr(args, "warmup_iterations", 1000))
    num_candidates = int(getattr(args, "num_candidates", 4))
    return warmup_iterations + ((augmentation_calls - 1) // num_candidates) + 1


def _robust_intensity_ranges(images):
    """Estimate per-slice p01--p99 ranges on a fixed, inexpensive grid."""
    # The fixed stride avoids sorting all 65,536 pixels for every one of the
    # four candidate calls while remaining deterministic for a given tensor.
    sampled = images[..., ::4, ::4].float().flatten(1)
    lower = torch.quantile(sampled, 0.01, dim=1)
    upper = torch.quantile(sampled, 0.99, dim=1)
    return (upper - lower).clamp_min(1e-6).to(images.dtype).view(-1, 1, 1)


def range_calibrated_mri_augmentation(images):
    """Apply UniMatch intensity draws with p01--p99-relative brightness."""
    global _augmentation_calls, _augmentation_trace_initialized
    args = safe.legacy.base.args
    robust_ranges = _robust_intensity_ranges(images).detach()
    augmented = []
    applied_offsets = []
    for index, image in enumerate(images):
        view = image.clone()
        if random.random() < args.strong_aug_prob:
            contrast = random.uniform(0.5, 1.5)
            brightness_fraction = random.uniform(-0.25, 0.25)
            mean = view.mean(dim=(-2, -1), keepdim=True)
            intensity_range = robust_ranges[index]
            brightness_offset = brightness_fraction * intensity_range
            view = (
                (view - mean) * contrast
                + mean
                + brightness_offset
            )
            applied_offsets.append(brightness_offset.detach().abs().mean())
        if random.random() < args.blur_prob:
            view = safe.legacy.base.gaussian_blur_2d(
                view, random.uniform(0.1, 2.0)
            )
        augmented.append(view)

    _augmentation_calls += 1
    output = torch.stack(augmented, dim=0)
    # One training iteration creates four candidate sets. Log the first real
    # post-warm-up call and then once per normal 20-iteration log interval.
    if _augmentation_calls == 1 or _augmentation_calls % 80 == 0:
        mean_range = float(robust_ranges.mean().cpu())
        if applied_offsets:
            mean_abs_offset = float(torch.stack(applied_offsets).mean().cpu())
        else:
            mean_abs_offset = 0.0
        # Unit/smoke callers only need the augmentation parameters. Runtime
        # metadata uses locked defaults for optional training-only attributes.
        training_iteration = _trace_iteration(args, _augmentation_calls)
        logger.info(
            "CALIBRATED-AUG active iter=%d candidate_call=%d "
            "p01_p99_range_mean=%.6f abs_brightness_shift_mean=%.6f "
            "brightness_applied=%d/%d",
            training_iteration,
            _augmentation_calls,
            mean_range,
            mean_abs_offset,
            len(applied_offsets),
            len(images),
        )
        output_dir = getattr(args, "output_dir", None)
        if output_dir:
            trace_path = os.path.join(output_dir, "calibrated_augmentation_trace.csv")
            mode = "a" if _augmentation_trace_initialized else "w"
            with open(trace_path, mode, encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                if not _augmentation_trace_initialized:
                    writer.writerow(
                        [
                            "iteration",
                            "candidate_call",
                            "p01_p99_range_mean",
                            "abs_brightness_shift_mean",
                            "brightness_applied",
                            "batch_size",
                        ]
                    )
                writer.writerow(
                    [
                        training_iteration,
                        _augmentation_calls,
                        mean_range,
                        mean_abs_offset,
                        len(applied_offsets),
                        len(images),
                    ]
                )
            _augmentation_trace_initialized = True
    return output


def _install_calibrated_metadata():
    original_safe_save_config = safe._safe_save_config

    def calibrated_save_config(args, split_cases, labeled_counts):
        original_safe_save_config(args, split_cases, labeled_counts)
        path = os.path.join(args.output_dir, "config.json")
        with open(path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        config.update(
            {
                "hypothesis": "H-CALIBRATED-UTILITYMATCH",
                "method": "CalibratedUtilityMatch",
                "augmentation_domain_evidence": (
                    "uploaded H5 is non-negative scaled data, not z-score data; "
                    "940-slice p01-p99 range median=0.211966"
                ),
                "brightness_parameterization": (
                    "Uniform(-0.25,0.25) * per-slice sampled p01-p99 range"
                ),
                "robust_range_sampling": "fixed spatial stride 4",
                "contrast_parameterization": "unchanged Uniform(0.5,1.5)",
                "blur_parameterization": "unchanged sigma Uniform(0.1,2.0)",
                "candidate_policy": (
                    "four calibrated candidates -> Top-2 signed utility -> "
                    "strict positive abstention"
                ),
                "activation_evidence": (
                    "terminal CALIBRATED-AUG and UTILITY-GATE records plus "
                    "calibrated_augmentation_trace.csv and utility_gate_trace.csv"
                ),
                "implementation_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            }
        )
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)

    safe._safe_save_config = calibrated_save_config


def _rewrite_completed_summary(output_dir):
    path = os.path.join(output_dir, "training_summary.json")
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    summary.update(
        {
            "hypothesis": "H-CALIBRATED-UTILITYMATCH",
            "method": "CalibratedUtilityMatch",
            "brightness_parameterization": "fraction_of_sampled_per_slice_p01_p99",
            "calibrated_augmentation_calls": _augmentation_calls,
            "augmentation_trace": "calibrated_augmentation_trace.csv",
        }
    )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def main():
    global _augmentation_calls, _augmentation_trace_initialized
    _augmentation_calls = 0
    _augmentation_trace_initialized = False
    args = safe.legacy.build_parser().parse_args()
    safe.legacy.base.strong_mri_augmentation = range_calibrated_mri_augmentation
    _install_calibrated_metadata()
    safe.install_safe_policy()
    installed_safe_train = safe.legacy.train

    def calibrated_train(train_args, split_cases, labeled_counts):
        if (
            safe.legacy.base.strong_mri_augmentation
            is not range_calibrated_mri_augmentation
        ):
            raise RuntimeError("Calibrated augmentation hook is not active")
        if safe.legacy.rank_candidates is not safe._safe_rank_candidates:
            raise RuntimeError("Positive-utility gate hook is not active")
        logger.info(
            "CALIBRATED-UTILITYMATCH VERIFIED: entry=%s; brightness="
            "U(-0.25,0.25)*sampled_p01_p99; positive-utility gate=active",
            Path(__file__).resolve(),
        )
        logger.info(
            "Iterations 1-%d are supervised warm-up and intentionally match "
            "UtilityMatch; calibrated candidates begin at iteration %d",
            train_args.warmup_iterations,
            train_args.warmup_iterations + 1,
        )
        result = installed_safe_train(train_args, split_cases, labeled_counts)
        if _augmentation_calls == 0:
            raise RuntimeError(
                "Calibrated augmentation was never called; refusing this run"
            )
        return result

    safe.legacy.train = calibrated_train
    if safe.legacy.train is not calibrated_train:
        raise RuntimeError("Calibrated training entry was not installed")
    print(
        "CALIBRATED-UTILITYMATCH ENTRY ACTIVE | first changed iteration="
        f"{args.warmup_iterations + 1} | entry={Path(__file__).resolve()}",
        flush=True,
    )
    safe.legacy.main(args)
    _rewrite_completed_summary(args.output_dir)


if __name__ == "__main__":
    main()
