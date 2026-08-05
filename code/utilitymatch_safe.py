"""Conflict-safe primitives for selected UtilityMatch strong views.

The utility sign has a direct first-order interpretation: a positive dot
product means the pseudo-gradient agrees with the labeled clean gradient,
whereas a non-positive value provides no descent-alignment evidence.  This
module turns that interpretation into a strict, parameter-free abstention
rule without changing candidate generation or ranking.
"""

import torch


def selected_positive_gates(utilities, selected_indices):
    """Return one Boolean gate per selected view using the strict ``u > 0`` rule."""
    if utilities.ndim != 1:
        raise ValueError("utilities must be one-dimensional")
    if selected_indices.ndim != 1:
        raise ValueError("selected_indices must be one-dimensional")
    if selected_indices.numel() == 0:
        raise ValueError("at least one selected view is required")
    if not torch.isfinite(utilities).all():
        raise FloatingPointError("candidate utilities contain NaN or Inf")
    if selected_indices.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("selected_indices must contain integer indices")
    if int(selected_indices.min()) < 0 or int(selected_indices.max()) >= utilities.numel():
        raise IndexError("selected candidate index is out of range")
    return utilities[selected_indices.long()] > 0


def mask_rejected_confidence(
    candidates,
    selected_indices,
    utilities,
    rejected_confidence=-1.0,
):
    """Mask rejected selected views so their existing pseudo loss is exactly zero.

    Candidate scoring must happen before this function is called. Accepted
    confidences are left byte-for-byte unchanged. Rejected confidences are set
    below the fixed UniMatch threshold; the existing masked CE and Dice losses
    then both evaluate to zero. The selected views are still forwarded, which
    keeps the compute path and Top-2 candidate structure aligned with the
    original UtilityMatch experiment.
    """
    gates = selected_positive_gates(utilities, selected_indices)
    selected_ids = selected_indices.detach().cpu().tolist()
    if len(candidates) != utilities.numel():
        raise ValueError("candidate and utility counts must match")
    for selected_id, gate in zip(selected_ids, gates.detach().cpu().tolist(), strict=True):
        candidate = candidates[selected_id]
        confidence = candidate.get("confidence")
        if not torch.is_tensor(confidence):
            raise TypeError("each selected candidate must contain tensor confidence")
        candidate["utility_active"] = bool(gate)
        if not gate:
            candidate["confidence"] = torch.full_like(
                confidence, float(rejected_confidence)
            )
    return gates
