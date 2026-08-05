"""Mechanism checks for GuardedUtilityMatch."""

from pathlib import Path

import torch
import train_utilitymatch_guarded as guarded
from test_utilitymatch_safe_smoke import main as check_safe_primitives


class Args:
    strong_aug_prob = 0.0
    blur_prob = 0.0
    cutmix_prob = 0.0


def check_pool_superset_shape():
    args = Args()
    guarded.legacy.base.args = args
    images = torch.linspace(0.0, 1.0, 2 * 16 * 16).reshape(2, 1, 16, 16)
    candidates = guarded.make_guarded_candidates(images, args)
    sources = [candidate["augmentation_source"] for candidate in candidates]
    if sources != ["calibrated"] * 4 + ["original"] * 2:
        raise AssertionError(f"unexpected guarded sources: {sources}")
    if len(candidates) != 6:
        raise AssertionError("guarded pool must contain six candidates")
    for candidate in candidates:
        torch.testing.assert_close(candidate["images"], images)


def check_guarded_selection():
    utilities = torch.tensor([0.3, 0.2, 0.1, -0.1, 0.9, 0.8])
    selected = guarded.select_guarded_candidates(utilities, keep=2)
    if selected.tolist() != [0, 4]:
        raise AssertionError(f"unexpected guarded selection: {selected.tolist()}")
    if int((selected >= guarded.CALIBRATED_CANDIDATES).sum()) > 1:
        raise AssertionError("two exploratory candidates must never be selected")

    non_positive = torch.tensor([0.3, 0.2, 0.1, -0.1, -0.01, -0.02])
    selected = guarded.select_guarded_candidates(non_positive, keep=2)
    if selected.tolist() != [0, 1]:
        raise AssertionError("non-positive exploration displaced a stable view")


def check_source_contract():
    source = Path(guarded.__file__).read_text(encoding="utf-8")
    required = (
        "CALIBRATED_CANDIDATES = 4",
        "EXPLORATORY_CANDIDATES = 2",
        '"augmentation_source": source',
        "select_guarded_candidates(",
        "mask_rejected_confidence(",
        '"GUARDED-POOL active',
        '"GUARDED-UTILITYMATCH VERIFIED',
        "legacy.main(args)",
    )
    for fragment in required:
        if fragment not in source:
            raise AssertionError(f"guarded invariant missing: {fragment}")


def main():
    check_safe_primitives()
    check_pool_superset_shape()
    check_guarded_selection()
    check_source_contract()
    print(
        "GuardedUtilityMatch smoke test passed: four-candidate stable pool "
        "retained, at most one original-strength candidate can be selected, "
        "strict positive gate retained, and all candidates preserve tensor shape"
    )


if __name__ == "__main__":
    main()
