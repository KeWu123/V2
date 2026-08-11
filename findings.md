# Findings

## Current understanding

The observed post-warm-up collapse suggests that high aggregate confidence is
not sufficient evidence that foreground pseudo-label supervision is ready.
The current hypothesis is that foreground prediction dynamics should control
trust at both pixel and optimization levels.

## Lessons and constraints

- Do not define the method as historical prediction averaging.
- Do not hard-convert uncertain boundary pixels to foreground.
- Do not mix data splits, label counts, seeds or initial checkpoints.
- Compare per-patient behavior, not only mean Dice.
- Python 3.9 requires removal of `zip(..., strict=True)` and this PyTorch 1.13
  environment requires NumPy below version 2.

## Open questions

- Does temporal reliability predict correctness better than confidence on the
  foreground and boundary subsets?
- Is adaptive global trust sufficient to prevent the post-warm-up Dice drop?
- Does soft boundary guidance improve HD95 without hurting small cases?
