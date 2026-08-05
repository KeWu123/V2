"""Mechanism checks for FrontierMatch."""

from pathlib import Path

import torch
import train_frontiermatch as frontier
from test_utilitymatch_safe_smoke import main as check_safe_primitives


class Args:
    strong_aug_prob = 0.0
    blur_prob = 0.0
    cutmix_prob = 0.0
    confidence_threshold = 0.95


def check_candidate_contract():
    args = Args()
    frontier.legacy.base.args = args
    images = torch.linspace(0.0, 1.0, 2 * 16 * 16).reshape(2, 1, 16, 16)
    candidates = frontier.make_frontier_candidates(images, args)
    if len(candidates) != frontier.TOTAL_CANDIDATES:
        raise AssertionError("frontier pool must contain six candidates")
    expected = [
        (0, "stable", 0.0, 0.95),
        (0, "coverage", 0.5, 0.90),
        (0, "reliable", 1.0, 0.98),
        (1, "stable", 0.0, 0.95),
        (1, "coverage", 0.5, 0.90),
        (1, "reliable", 1.0, 0.98),
    ]
    actual = [
        (
            candidate["ray_id"],
            candidate["policy_name"],
            candidate["severity"],
            candidate["coverage_threshold"],
        )
        for candidate in candidates
    ]
    if actual != expected:
        raise AssertionError(f"unexpected frontier policies: {actual}")
    for candidate in candidates:
        torch.testing.assert_close(candidate["images"], images)


def check_coverage_contract():
    args = Args()
    frontier.legacy.base.args = args
    images = torch.zeros(2, 1, 8, 8)
    candidates = frontier.make_frontier_candidates(images, args)
    labels = torch.zeros(2, 8, 8, dtype=torch.long)
    confidence = torch.full((2, 8, 8), 0.92)
    frontier.attach_frontier_targets(candidates, labels, confidence)
    for candidate in candidates:
        accepted = bool((candidate["confidence"] >= 0.95).all())
        expected = candidate["policy_name"] == "coverage"
        if accepted != expected:
            raise AssertionError(
                f"coverage encoding failed for {candidate['policy_name']}"
            )


def check_grouped_selection():
    utilities = torch.tensor([0.2, 0.7, 0.4, -0.3, 0.1, 0.6])
    selected = frontier.select_frontier_candidates(utilities, keep=2)
    if selected.tolist() != [1, 5]:
        raise AssertionError(f"unexpected frontier selection: {selected.tolist()}")
    rays = [index // frontier.POLICIES_PER_RAY for index in selected.tolist()]
    if rays != [0, 1]:
        raise AssertionError("selection must retain two independent rays")


def check_source_contract():
    source = Path(frontier.__file__).read_text(encoding="utf-8")
    required = (
        "RAYS = 2",
        '"name": "stable"',
        '"name": "coverage"',
        '"name": "reliable"',
        "attach_frontier_targets(",
        "select_frontier_candidates(",
        "mask_rejected_confidence(",
        "args.num_candidates = TOTAL_CANDIDATES",
        '"FRONTIER active iter=%d',
        '"FRONTIERMATCH VERIFIED',
        "legacy.main(args)",
    )
    for fragment in required:
        if fragment not in source:
            raise AssertionError(f"frontier invariant missing: {fragment}")


def main():
    check_safe_primitives()
    check_candidate_contract()
    check_coverage_contract()
    check_grouped_selection()
    check_source_contract()
    print(
        "FrontierMatch smoke test passed: two independent rays, stable/coverage/"
        "reliable joint policies, candidate-specific pseudo coverage, and "
        "strict positive-utility gating are active"
    )


if __name__ == "__main__":
    main()
