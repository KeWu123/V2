"""Guarded UtilityMatch: expand the stable candidate pool without losing it.

The four calibrated candidates and strict positive-utility gate from the first
non-collapsing run are retained exactly. Two original-strength UniMatch
candidates are appended as optional exploration. They can displace a stable
rank-2 candidate only when their signed clean-gradient utility ranks higher;
the best stable candidate always keeps one slot. No non-positive selected
strong branch is allowed into the pseudo loss.
"""

import csv
import hashlib
import json
import logging
import os
import random
from pathlib import Path

import torch
import train_utilitymatch as legacy
from utilitymatch_safe import mask_rejected_confidence

CALIBRATED_CANDIDATES = 4
EXPLORATORY_CANDIDATES = 2
TOTAL_CANDIDATES = CALIBRATED_CANDIDATES + EXPLORATORY_CANDIDATES

logger = logging.getLogger(__name__)
_original_save_config = legacy.save_config
_original_train = legacy.train
_rank_calls = 0
_trace_initialized = False
_active_gate_count = 0
_all_rejected_batches = 0
_selected_calibrated = 0
_selected_exploratory = 0
_active_calibrated = 0
_active_exploratory = 0


def _robust_intensity_ranges(images):
    sampled = images[..., ::4, ::4].float().flatten(1)
    lower = torch.quantile(sampled, 0.01, dim=1)
    upper = torch.quantile(sampled, 0.99, dim=1)
    return (upper - lower).clamp_min(1e-6).to(images.dtype).view(-1, 1, 1)


def calibrated_mri_augmentation(images):
    """The exact p01--p99 candidate transform used by the stable run."""
    args = legacy.base.args
    robust_ranges = _robust_intensity_ranges(images).detach()
    augmented = []
    for index, image in enumerate(images):
        view = image.clone()
        if random.random() < args.strong_aug_prob:
            contrast = random.uniform(0.5, 1.5)
            brightness_fraction = random.uniform(-0.25, 0.25)
            mean = view.mean(dim=(-2, -1), keepdim=True)
            view = (
                (view - mean) * contrast
                + mean
                + brightness_fraction * robust_ranges[index]
            )
        if random.random() < args.blur_prob:
            view = legacy.base.gaussian_blur_2d(
                view, random.uniform(0.1, 2.0)
            )
        augmented.append(view)
    return torch.stack(augmented, dim=0)


def make_guarded_candidates(unlabeled_images, args):
    """Return the stable four-candidate pool plus two guarded explorations."""
    count, _, height, width = unlabeled_images.shape
    candidates = []
    sources = ["calibrated"] * CALIBRATED_CANDIDATES + [
        "original"
    ] * EXPLORATORY_CANDIDATES
    for source in sources:
        if source == "calibrated":
            images = calibrated_mri_augmentation(unlabeled_images)
        else:
            images = legacy.base.strong_mri_augmentation(unlabeled_images)
        permutation = torch.randperm(count, device=unlabeled_images.device)
        boxes = legacy.base.obtain_cutmix_boxes(
            count, height, width, unlabeled_images.device
        )
        images = legacy.base.cutmix_tensor(images, images[permutation], boxes)
        candidates.append(
            {
                "images": images,
                "permutation": permutation,
                "boxes": boxes,
                "augmentation_source": source,
            }
        )
    return candidates


def select_guarded_candidates(utilities, keep):
    """Keep one stable view and permit at most one positive exploration."""
    if int(keep) != 2:
        raise ValueError("GuardedUtilityMatch requires exactly two selected views")
    if utilities.ndim != 1 or utilities.numel() != TOTAL_CANDIDATES:
        raise ValueError(
            f"Expected {TOTAL_CANDIDATES} one-dimensional utilities"
        )
    stable_top2 = legacy.select_top_candidates(
        utilities[:CALIBRATED_CANDIDATES], keep=2
    )
    best_exploratory = (
        torch.argmax(utilities[CALIBRATED_CANDIDATES:])
        + CALIBRATED_CANDIDATES
    )
    stable_second = stable_top2[1]
    use_exploratory = (utilities[best_exploratory] > 0) & (
        utilities[best_exploratory] > utilities[stable_second]
    )
    guarded_second = torch.where(
        use_exploratory, best_exploratory, stable_second
    )
    return torch.stack((stable_top2[0], guarded_second))


def rank_guarded_candidates(
    model, candidates, reference_gradient, head_parameters, dice_loss, args
):
    """Rank the superset pool and enforce the stable positive-utility gate."""
    global _rank_calls, _trace_initialized
    global _active_gate_count, _all_rejected_batches
    global _selected_calibrated, _selected_exploratory
    global _active_calibrated, _active_exploratory

    if len(candidates) != TOTAL_CANDIDATES:
        raise RuntimeError(
            f"Guarded pool requires {TOTAL_CANDIDATES} candidates, "
            f"got {len(candidates)}"
        )
    candidate_images = torch.cat(
        [candidate["images"] for candidate in candidates], dim=0
    )
    with torch.no_grad(), legacy.freeze_batchnorm_running_stats(model):
        candidate_output = model(candidate_images)
        if not isinstance(candidate_output, tuple) or len(candidate_output) < 2:
            raise RuntimeError("The fixed U-Net must return logits and features")
        detached_features = candidate_output[1].detach()

    scoring_logits = model.decoder.out_conv(detached_features)
    logits_per_candidate = scoring_logits.chunk(TOTAL_CANDIDATES, dim=0)
    scoring_losses = []
    for logits, candidate in zip(logits_per_candidate, candidates, strict=True):
        loss, _ = legacy.base.confidence_masked_baseline_loss(
            logits,
            candidate["targets"],
            candidate["confidence"],
            dice_loss,
        )
        scoring_losses.append(loss)

    utilities = []
    for index, loss in enumerate(scoring_losses):
        candidate_gradient = legacy.head_gradient(
            loss,
            head_parameters,
            retain_graph=index < len(scoring_losses) - 1,
        )
        utilities.append(
            legacy.gradient_projection_utility(
                candidate_gradient,
                reference_gradient,
                epsilon=args.utility_epsilon,
            )
        )
    utilities = torch.stack(utilities)
    selected_indices = select_guarded_candidates(
        utilities, keep=args.selected_views
    )
    gates = mask_rejected_confidence(
        candidates,
        selected_indices,
        utilities,
        rejected_confidence=-1.0,
    )

    _rank_calls += 1
    selected_ids = selected_indices.detach().cpu().tolist()
    gate_values = [int(value) for value in gates.detach().cpu().tolist()]
    selected_sources = [
        str(candidates[index]["augmentation_source"]) for index in selected_ids
    ]
    active_count = sum(gate_values)
    _active_gate_count += active_count
    _all_rejected_batches += int(active_count == 0)
    for source, gate in zip(selected_sources, gate_values, strict=True):
        if source == "calibrated":
            _selected_calibrated += 1
            _active_calibrated += gate
        else:
            _selected_exploratory += 1
            _active_exploratory += gate

    completed_iteration = int(args.warmup_iterations) + _rank_calls
    if completed_iteration == int(args.warmup_iterations) + 1 or (
        completed_iteration % int(args.log_interval) == 0
    ):
        trace_path = os.path.join(args.output_dir, "guarded_pool_trace.csv")
        mode = "a" if _trace_initialized else "w"
        with open(trace_path, mode, encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            if not _trace_initialized:
                writer.writerow(
                    ["iteration"]
                    + [f"utility_{index}" for index in range(TOTAL_CANDIDATES)]
                    + [
                        "selected_0",
                        "selected_1",
                        "source_0",
                        "source_1",
                        "gate_0",
                        "gate_1",
                        "active_selected_views",
                    ]
                )
            writer.writerow(
                [completed_iteration]
                + utilities.detach().cpu().tolist()
                + selected_ids
                + selected_sources
                + gate_values
                + [active_count]
            )
        _trace_initialized = True
        logger.info(
            "GUARDED-POOL active iter=%d selected=%s sources=%s "
            "utilities=%s gates=%s active=%d/%d",
            completed_iteration,
            selected_ids,
            selected_sources,
            [
                round(float(value), 8)
                for value in utilities[selected_indices].detach().cpu().tolist()
            ],
            gate_values,
            active_count,
            int(args.selected_views),
        )
    return utilities.detach(), selected_indices.detach()


def guarded_save_config(args, split_cases, labeled_counts):
    _original_save_config(args, split_cases, labeled_counts)
    path = os.path.join(args.output_dir, "config.json")
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    config.update(
        {
            "hypothesis": "H-GUARDED-UTILITYMATCH",
            "method": "GuardedUtilityMatch",
            "stability_anchor": "CalibratedUtilityMatch user-reported Dice about 0.816",
            "candidate_pool": {
                "calibrated": CALIBRATED_CANDIDATES,
                "original_strength_exploratory": EXPLORATORY_CANDIDATES,
                "total": TOTAL_CANDIDATES,
                "selected": 2,
            },
            "pool_superset_contract": (
                "the exact four stable candidates are generated first; two "
                "original-strength candidates are appended"
            ),
            "selection_guard": (
                "always retain the best calibrated candidate; at most one "
                "positive exploratory candidate may replace calibrated rank-2"
            ),
            "strong_view_gate": "selected_utility > 0",
            "rejected_branch_policy": "zero strong loss; no weight redistribution",
            "stability_invariant": (
                "no non-positive original-strength or calibrated candidate "
                "can enter a strong pseudo loss"
            ),
            "num_candidates": TOTAL_CANDIDATES,
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }
    )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)


def guarded_train(args, split_cases, labeled_counts):
    global _rank_calls, _trace_initialized
    global _active_gate_count, _all_rejected_batches
    global _selected_calibrated, _selected_exploratory
    global _active_calibrated, _active_exploratory

    if int(args.num_candidates) != 4:
        raise ValueError("Guarded entry expects the locked legacy K=4 argument")
    if legacy.make_candidates is not make_guarded_candidates:
        raise RuntimeError("Guarded candidate-pool hook is not active")
    if legacy.rank_candidates is not rank_guarded_candidates:
        raise RuntimeError("Guarded rank/gate hook is not active")

    _rank_calls = 0
    _trace_initialized = False
    _active_gate_count = 0
    _all_rejected_batches = 0
    _selected_calibrated = 0
    _selected_exploratory = 0
    _active_calibrated = 0
    _active_exploratory = 0
    args.num_candidates = TOTAL_CANDIDATES
    logger.info(
        "GUARDED-UTILITYMATCH VERIFIED: stable_pool=%d exploratory_pool=%d "
        "max_exploratory_selected=1 gate=selected_utility>0 "
        "loss_weights=(0.25,0.25,0.50)",
        CALIBRATED_CANDIDATES,
        EXPLORATORY_CANDIDATES,
    )
    result = _original_train(args, split_cases, labeled_counts)
    if _rank_calls == 0:
        raise RuntimeError("Guarded candidate pool was never ranked")

    summary_path = os.path.join(args.output_dir, "training_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        decisions = _rank_calls * int(args.selected_views)
        summary.update(
            {
                "hypothesis": "H-GUARDED-UTILITYMATCH",
                "method": "GuardedUtilityMatch",
                "candidate_pool_total": TOTAL_CANDIDATES,
                "calibrated_candidates": CALIBRATED_CANDIDATES,
                "exploratory_candidates": EXPLORATORY_CANDIDATES,
                "gate_decisions": decisions,
                "active_gate_decisions": _active_gate_count,
                "active_gate_fraction": _active_gate_count / decisions,
                "all_rejected_batches": _all_rejected_batches,
                "selected_calibrated": _selected_calibrated,
                "selected_exploratory": _selected_exploratory,
                "active_calibrated": _active_calibrated,
                "active_exploratory": _active_exploratory,
                "pool_trace": "guarded_pool_trace.csv",
            }
        )
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    return result


def install_guarded_policy():
    legacy.make_candidates = make_guarded_candidates
    legacy.rank_candidates = rank_guarded_candidates
    legacy.save_config = guarded_save_config
    legacy.train = guarded_train


def main():
    args = legacy.build_parser().parse_args()
    install_guarded_policy()
    print(
        "GUARDED-UTILITYMATCH ENTRY ACTIVE | stable candidates=4 | "
        "guarded original-strength candidates=2 | strict positive gate",
        flush=True,
    )
    legacy.main(args)


if __name__ == "__main__":
    main()
