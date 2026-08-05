"""Mechanism-level smoke tests for SafeUtilityMatch."""

from pathlib import Path

import torch
from test_utilitymatch_smoke import check_batchnorm_buffers, check_gradient_ranking
from utilitymatch_safe import mask_rejected_confidence, selected_positive_gates


def _candidates(count=4):
    return [
        {"confidence": torch.tensor([[0.99, 0.96], [0.80, 1.00]])}
        for _ in range(count)
    ]


def check_mixed_gate():
    utilities = torch.tensor([-0.4, 0.3, -0.1, 0.2])
    selected = torch.tensor([1, 2])
    candidates = _candidates()
    accepted_before = candidates[1]["confidence"].clone()
    gates = mask_rejected_confidence(candidates, selected, utilities)
    if gates.tolist() != [True, False]:
        raise AssertionError(f"unexpected mixed gates: {gates.tolist()}")
    torch.testing.assert_close(candidates[1]["confidence"], accepted_before)
    if not torch.all(candidates[2]["confidence"] < 0.95):
        raise AssertionError("rejected view can still enter the confidence mask")


def check_zero_and_all_negative_gate():
    utilities = torch.tensor([-0.4, 0.0, -0.1, -0.2])
    selected = torch.tensor([1, 2])
    gates = selected_positive_gates(utilities, selected)
    if gates.tolist() != [False, False]:
        raise AssertionError("zero utility must abstain under the strict rule")
    candidates = _candidates()
    mask_rejected_confidence(candidates, selected, utilities)
    if any(torch.any(candidates[index]["confidence"] >= 0.95) for index in (1, 2)):
        raise AssertionError("all-negative selected views were not fully masked")


def check_source_invariants():
    code_dir = Path(__file__).resolve().parent
    legacy = (code_dir / "train_utilitymatch.py").read_text(encoding="utf-8")
    safe = (code_dir / "train_utilitymatch_safe.py").read_text(encoding="utf-8")
    required_legacy = (
        "0.25 * loss_u_s1 + 0.25 * loss_u_s2 + 0.5 * loss_u_fp",
        '"seed": 1337',
        '"max_iterations": 30000',
        '"warmup_iterations": 1000',
        '"num_candidates": 4',
        '"selected_views": 2',
    )
    for fragment in required_legacy:
        if fragment not in legacy:
            raise AssertionError(f"locked legacy invariant missing: {fragment}")
    if "mask_rejected_confidence" not in safe or "legacy.main(args)" not in safe:
        raise AssertionError("safe entry no longer wraps the locked trainer")


def main():
    check_gradient_ranking()
    check_batchnorm_buffers()
    check_mixed_gate()
    check_zero_and_all_negative_gate()
    check_source_invariants()
    print(
        "SafeUtilityMatch smoke test passed: signed utility, frozen candidate BN, "
        "strict positive gate, zero/all-negative abstention, accepted-confidence "
        "preservation, and locked legacy formula"
    )


if __name__ == "__main__":
    main()
