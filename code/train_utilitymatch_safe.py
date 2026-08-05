"""SafeUtilityMatch: strict abstention for conflicting strong-view gradients.

This is an independent entry point layered on the locked UtilityMatch trainer.
No model, data, checkpoint, pseudo-label, candidate augmentation, optimizer,
EMA, validation, or checkpoint behavior is changed. After the original Top-2
ranking, a selected strong branch is allowed into the existing loss only when
its signed clean-gradient utility is strictly positive.
"""

import csv
import json
import logging
import os

import train_utilitymatch as legacy
from utilitymatch_safe import mask_rejected_confidence

_original_rank_candidates = legacy.rank_candidates
_original_save_config = legacy.save_config
_original_train = legacy.train
_rank_calls = 0
_trace_initialized = False
_active_gate_count = 0
_all_rejected_batches = 0
logger = logging.getLogger(__name__)


def _safe_rank_candidates(
    model, candidates, reference_gradient, head_parameters, dice_loss, args
):
    """Run unchanged ranking, then abstain from non-positive selected views."""
    global _rank_calls, _trace_initialized, _active_gate_count, _all_rejected_batches
    utilities, selected_indices = _original_rank_candidates(
        model,
        candidates,
        reference_gradient,
        head_parameters,
        dice_loss,
        args,
    )
    gates = mask_rejected_confidence(
        candidates,
        selected_indices,
        utilities,
        rejected_confidence=-1.0,
    )

    _rank_calls += 1
    active_count = int(gates.sum().item())
    _active_gate_count += active_count
    _all_rejected_batches += int(active_count == 0)
    completed_iteration = int(args.warmup_iterations) + _rank_calls
    if completed_iteration % int(args.log_interval) == 0:
        trace_path = os.path.join(args.output_dir, "utility_gate_trace.csv")
        mode = "a" if _trace_initialized else "w"
        with open(trace_path, mode, encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            if not _trace_initialized:
                writer.writerow(
                    [
                        "iteration",
                        "utility_0",
                        "utility_1",
                        "utility_2",
                        "utility_3",
                        "selected_0",
                        "selected_1",
                        "gate_0",
                        "gate_1",
                        "active_selected_views",
                    ]
                )
            utility_values = utilities.detach().cpu().tolist()
            selected_values = selected_indices.detach().cpu().tolist()
            gate_values = [int(value) for value in gates.detach().cpu().tolist()]
            writer.writerow(
                [completed_iteration]
                + utility_values
                + selected_values
                + gate_values
                + [active_count]
            )
        _trace_initialized = True
        selected_utility_values = utilities[selected_indices].detach().cpu().tolist()
        logger.info(
            "UTILITY-GATE active iter=%d selected_utilities=%s gates=%s "
            "active=%d/%d",
            completed_iteration,
            [round(float(value), 8) for value in selected_utility_values],
            gate_values,
            active_count,
            int(args.selected_views),
        )
    return utilities, selected_indices


def _safe_save_config(args, split_cases, labeled_counts):
    _original_save_config(args, split_cases, labeled_counts)
    path = os.path.join(args.output_dir, "config.json")
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    config.update(
        {
            "hypothesis": "H-SAFE-UTILITYMATCH",
            "method": "SafeUtilityMatch",
            "single_change_from_utilitymatch": (
                "selected strong-view loss is active iff signed utility > 0"
            ),
            "strong_view_gate": "indicator(selected_utility > 0)",
            "rejected_branch_policy": (
                "zero masked pseudo loss; retain forward compute; no weight renormalization"
            ),
            "consistency_formula": (
                "0.25*g1*Lstrong1 + 0.25*g2*Lstrong2 + 0.5*Lfeature"
            ),
            "new_hyperparameters": 0,
        }
    )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)


def _safe_train(args, split_cases, labeled_counts):
    global _rank_calls, _trace_initialized, _active_gate_count, _all_rejected_batches
    _rank_calls = 0
    _trace_initialized = False
    _active_gate_count = 0
    _all_rejected_batches = 0
    logger.info(
        "SafeUtilityMatch enabled: selected strong views require utility > 0; "
        "rejected weights are not redistributed; feature branch is unchanged"
    )
    result = _original_train(args, split_cases, labeled_counts)
    summary_path = os.path.join(args.output_dir, "training_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summary.update(
            {
                "hypothesis": "H-SAFE-UTILITYMATCH",
                "method": "SafeUtilityMatch",
                "strong_view_gate": "selected_utility > 0",
                "gate_decisions": _rank_calls * int(args.selected_views),
                "active_gate_decisions": _active_gate_count,
                "active_gate_fraction": (
                    _active_gate_count / (_rank_calls * int(args.selected_views))
                    if _rank_calls
                    else 0.0
                ),
                "all_rejected_batches": _all_rejected_batches,
                "ranked_batches": _rank_calls,
                "gate_trace": "utility_gate_trace.csv",
            }
        )
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    return result


def install_safe_policy():
    """Install the isolated policy hooks into the imported locked trainer."""
    legacy.rank_candidates = _safe_rank_candidates
    legacy.save_config = _safe_save_config
    legacy.train = _safe_train


def main():
    args = legacy.build_parser().parse_args()
    install_safe_policy()
    legacy.main(args)


if __name__ == "__main__":
    main()
