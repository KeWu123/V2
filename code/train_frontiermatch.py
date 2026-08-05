"""FrontierMatch: joint augmentation-strength and pseudo-coverage selection.

Two independent UniMatch strong-view rays are retained.  Each ray instantiates
three policies along the same sampled augmentation direction: an exact stable
fallback, a coverage-seeking midpoint, and an original-strength high-reliability
view.  The labeled-task gradient utility chooses one policy per ray, and the
strict positive-utility gate prevents a conflicting strong loss.
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

RAYS = 2
POLICIES = (
    {
        "name": "stable",
        "severity": 0.0,
        "threshold": 0.95,
    },
    {
        "name": "coverage",
        "severity": 0.5,
        "threshold": 0.90,
    },
    {
        "name": "reliable",
        "severity": 1.0,
        "threshold": 0.98,
    },
)
POLICIES_PER_RAY = len(POLICIES)
TOTAL_CANDIDATES = RAYS * POLICIES_PER_RAY

logger = logging.getLogger(__name__)
_original_save_config = legacy.save_config
_original_train = legacy.train
_rank_calls = 0
_trace_initialized = False
_active_gate_count = 0
_all_rejected_batches = 0
_selected_counts = {policy["name"]: 0 for policy in POLICIES}
_active_counts = {policy["name"]: 0 for policy in POLICIES}


def _robust_intensity_ranges(images):
    """Estimate deterministic per-slice p01--p99 ranges on a fixed grid."""
    sampled = images[..., ::4, ::4].float().flatten(1)
    lower = torch.quantile(sampled, 0.01, dim=1)
    upper = torch.quantile(sampled, 0.99, dim=1)
    return (upper - lower).clamp_min(1e-6).to(images.dtype).view(-1, 1, 1)


def _sample_frontier_ray(images, args):
    """Instantiate the three policies along one shared stochastic direction."""
    robust_ranges = _robust_intensity_ranges(images).detach()
    policy_views = [[] for _ in POLICIES]
    for index, image in enumerate(images):
        apply_intensity = random.random() < args.strong_aug_prob
        if apply_intensity:
            contrast = random.uniform(0.5, 1.5)
            brightness_fraction = random.uniform(-0.25, 0.25)
            mean = image.mean(dim=(-2, -1), keepdim=True)
        else:
            contrast = 1.0
            brightness_fraction = 0.0
            mean = image.new_zeros((1, 1, 1))

        apply_blur = random.random() < args.blur_prob
        blur_sigma = random.uniform(0.1, 2.0) if apply_blur else 0.0
        for policy_index, policy in enumerate(POLICIES):
            view = image.clone()
            if apply_intensity:
                severity = float(policy["severity"])
                brightness_unit = (
                    (1.0 - severity) * robust_ranges[index]
                    + severity * torch.ones_like(robust_ranges[index])
                )
                view = (
                    (view - mean) * contrast
                    + mean
                    + brightness_fraction * brightness_unit
                )
            if apply_blur:
                view = legacy.base.gaussian_blur_2d(view, blur_sigma)
            policy_views[policy_index].append(view)
    return [torch.stack(views, dim=0) for views in policy_views]


def make_frontier_candidates(unlabeled_images, args):
    """Create two independent rays with three joint policies per ray."""
    count, _, height, width = unlabeled_images.shape
    candidates = []
    for ray_id in range(RAYS):
        policy_images = _sample_frontier_ray(unlabeled_images, args)
        permutation = torch.randperm(count, device=unlabeled_images.device)
        boxes = legacy.base.obtain_cutmix_boxes(
            count, height, width, unlabeled_images.device
        )
        for policy_index, (policy, images) in enumerate(
            zip(POLICIES, policy_images, strict=True)
        ):
            mixed_images = legacy.base.cutmix_tensor(
                images, images[permutation], boxes
            )
            candidates.append(
                {
                    "images": mixed_images,
                    "permutation": permutation,
                    "boxes": boxes,
                    "ray_id": ray_id,
                    "policy_index": policy_index,
                    "policy_name": str(policy["name"]),
                    "severity": float(policy["severity"]),
                    "coverage_threshold": float(policy["threshold"]),
                }
            )
    return candidates


def attach_frontier_targets(candidates, pseudo_labels, pseudo_confidence):
    """Transport targets and encode each candidate's pseudo-label coverage."""
    if len(candidates) != TOTAL_CANDIDATES:
        raise RuntimeError(
            f"FrontierMatch requires {TOTAL_CANDIDATES} candidates, "
            f"got {len(candidates)}"
        )
    for candidate in candidates:
        permutation = candidate["permutation"]
        boxes = candidate["boxes"]
        candidate["targets"] = legacy.base.cutmix_tensor(
            pseudo_labels, pseudo_labels[permutation], boxes
        )
        transported_confidence = legacy.base.cutmix_tensor(
            pseudo_confidence, pseudo_confidence[permutation], boxes
        )
        coverage_mask = transported_confidence >= float(
            candidate["coverage_threshold"]
        )
        candidate["confidence"] = torch.where(
            coverage_mask,
            torch.ones_like(transported_confidence),
            torch.full_like(transported_confidence, -1.0),
        )
        candidate["coverage_ratio"] = coverage_mask.float().mean().detach()


def select_frontier_candidates(utilities, keep):
    """Select the best policy independently inside each strong-view ray."""
    if int(keep) != RAYS:
        raise ValueError(f"FrontierMatch requires exactly {RAYS} selected views")
    if utilities.ndim != 1 or utilities.numel() != TOTAL_CANDIDATES:
        raise ValueError(
            f"Expected {TOTAL_CANDIDATES} one-dimensional utilities"
        )
    by_ray = utilities.reshape(RAYS, POLICIES_PER_RAY)
    local_indices = torch.argmax(by_ray, dim=1)
    offsets = torch.arange(RAYS, device=utilities.device) * POLICIES_PER_RAY
    return local_indices + offsets


def rank_frontier_candidates(
    model, candidates, reference_gradient, head_parameters, dice_loss, args
):
    """Score all joint policies, select one per ray, and apply the sign gate."""
    global _rank_calls, _trace_initialized
    global _active_gate_count, _all_rejected_batches

    if len(candidates) != TOTAL_CANDIDATES:
        raise RuntimeError(
            f"FrontierMatch requires {TOTAL_CANDIDATES} candidates, "
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
    selected_indices = select_frontier_candidates(
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
    selected_policies = [
        str(candidates[index]["policy_name"]) for index in selected_ids
    ]
    selected_severities = [
        float(candidates[index]["severity"]) for index in selected_ids
    ]
    selected_thresholds = [
        float(candidates[index]["coverage_threshold"]) for index in selected_ids
    ]
    selected_coverages = [
        float(candidates[index]["coverage_ratio"].detach().cpu())
        for index in selected_ids
    ]
    active_count = sum(gate_values)
    _active_gate_count += active_count
    _all_rejected_batches += int(active_count == 0)
    for policy_name, gate in zip(selected_policies, gate_values, strict=True):
        _selected_counts[policy_name] += 1
        _active_counts[policy_name] += gate

    completed_iteration = int(args.warmup_iterations) + _rank_calls
    if completed_iteration == int(args.warmup_iterations) + 1 or (
        completed_iteration % int(args.log_interval) == 0
    ):
        trace_path = os.path.join(args.output_dir, "frontier_trace.csv")
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
                        "policy_0",
                        "policy_1",
                        "severity_0",
                        "severity_1",
                        "threshold_0",
                        "threshold_1",
                        "coverage_0",
                        "coverage_1",
                        "gate_0",
                        "gate_1",
                        "active_selected_views",
                    ]
                )
            writer.writerow(
                [completed_iteration]
                + utilities.detach().cpu().tolist()
                + selected_ids
                + selected_policies
                + selected_severities
                + selected_thresholds
                + selected_coverages
                + gate_values
                + [active_count]
            )
        _trace_initialized = True
        logger.info(
            "FRONTIER active iter=%d selected=%s policies=%s "
            "severity=%s threshold=%s coverage=%s utilities=%s "
            "gates=%s active=%d/%d",
            completed_iteration,
            selected_ids,
            selected_policies,
            selected_severities,
            selected_thresholds,
            [round(value, 6) for value in selected_coverages],
            [
                round(float(value), 8)
                for value in utilities[selected_indices].detach().cpu().tolist()
            ],
            gate_values,
            active_count,
            int(args.selected_views),
        )
    return utilities.detach(), selected_indices.detach()


def frontier_save_config(args, split_cases, labeled_counts):
    _original_save_config(args, split_cases, labeled_counts)
    path = os.path.join(args.output_dir, "config.json")
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    config.update(
        {
            "hypothesis": "H-FRONTIERMATCH",
            "method": "FrontierMatch",
            "joint_frontier_policies": [dict(policy) for policy in POLICIES],
            "rays": RAYS,
            "policies_per_ray": POLICIES_PER_RAY,
            "num_candidates": TOTAL_CANDIDATES,
            "selected_views": RAYS,
            "selection": "maximum labeled-gradient utility independently per ray",
            "strong_view_gate": "selected_utility > 0",
            "stable_fallback": (
                "each ray contains p01-p99 brightness with threshold 0.95"
            ),
            "feature_branch_threshold": float(args.confidence_threshold),
            "rejected_branch_policy": (
                "zero strong loss; no missing-weight redistribution"
            ),
            "unchanged_components": (
                "data, split, PreTrain, U-Net, EMA, pseudo-label generator, "
                "feature branch, optimizer, LR, validation, checkpoint and test"
            ),
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }
    )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    # Protocol validation has already locked the legacy CLI value K=4. Expand
    # here so the subsequent Arguments log and the actual trainer both expose
    # the real six-policy runtime instead of reporting a misleading K=4.
    args.num_candidates = TOTAL_CANDIDATES


def frontier_train(args, split_cases, labeled_counts):
    global _rank_calls, _trace_initialized
    global _active_gate_count, _all_rejected_batches

    if int(args.num_candidates) != TOTAL_CANDIDATES:
        raise ValueError(
            "FrontierMatch config hook did not expose the six-policy runtime"
        )
    if legacy.make_candidates is not make_frontier_candidates:
        raise RuntimeError("Frontier candidate hook is not active")
    if legacy.attach_transported_targets is not attach_frontier_targets:
        raise RuntimeError("Frontier pseudo-coverage hook is not active")
    if legacy.rank_candidates is not rank_frontier_candidates:
        raise RuntimeError("Frontier rank/gate hook is not active")

    _rank_calls = 0
    _trace_initialized = False
    _active_gate_count = 0
    _all_rejected_batches = 0
    for policy in POLICIES:
        _selected_counts[policy["name"]] = 0
        _active_counts[policy["name"]] = 0
    logger.info(
        "FRONTIERMATCH VERIFIED: rays=%d policies=%s "
        "selection=best-per-ray gate=selected_utility>0 "
        "loss_weights=(0.25,0.25,0.50)",
        RAYS,
        [
            (
                policy["name"],
                policy["severity"],
                policy["threshold"],
            )
            for policy in POLICIES
        ],
    )
    result = _original_train(args, split_cases, labeled_counts)
    if _rank_calls == 0:
        raise RuntimeError("Frontier policies were never ranked")

    summary_path = os.path.join(args.output_dir, "training_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        decisions = _rank_calls * int(args.selected_views)
        summary.update(
            {
                "hypothesis": "H-FRONTIERMATCH",
                "method": "FrontierMatch",
                "candidate_pool_total": TOTAL_CANDIDATES,
                "rays": RAYS,
                "policies_per_ray": POLICIES_PER_RAY,
                "gate_decisions": decisions,
                "active_gate_decisions": _active_gate_count,
                "active_gate_fraction": _active_gate_count / decisions,
                "all_rejected_batches": _all_rejected_batches,
                "selected_policy_counts": dict(_selected_counts),
                "active_policy_counts": dict(_active_counts),
                "frontier_trace": "frontier_trace.csv",
            }
        )
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    return result


def install_frontier_policy():
    """Install the joint candidate, coverage, ranking and metadata hooks."""
    legacy.make_candidates = make_frontier_candidates
    legacy.attach_transported_targets = attach_frontier_targets
    legacy.rank_candidates = rank_frontier_candidates
    legacy.save_config = frontier_save_config
    legacy.train = frontier_train


def main():
    args = legacy.build_parser().parse_args()
    install_frontier_policy()
    print(
        "FRONTIERMATCH ENTRY ACTIVE | 2 rays x 3 joint policies | "
        "best policy per ray | strict positive utility gate",
        flush=True,
    )
    legacy.main(args)


if __name__ == "__main__":
    main()
