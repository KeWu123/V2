# GuardedUtilityMatch protocol

## Corrected premise

The approximately 0.816 CalibratedUtilityMatch result is below the 0.838397
valid UtilityMatch result, but it is the first recent run reported to train
normally without catastrophic post-warm-up collapse. It is therefore the
stability anchor, not a failed implementation to discard.

The next experiment must preserve two invariants from that run:

1. all four p01--p99 calibrated candidates remain in every candidate pool;
2. no selected candidate with non-positive clean-gradient utility may enter a
   strong pseudo loss.

## Hypothesis

The stable run may underperform because all candidates use conservative
brightness strength and strict gating sometimes leaves insufficient strong
supervision. Appending original-strength candidates can restore useful hard
views, while the existing utility ranking and positive gate prevent an
original-strength candidate from entering when it conflicts with the current
labeled task direction.

## Single mechanism

At each post-warm-up iteration:

```text
stable core:       4 p01--p99-calibrated candidates
guarded expansion: 2 original-UtilityMatch-strength candidates
pool:              6 candidates -> best calibrated + guarded rank-2
selection guard:   at most 1 original-strength candidate
gate:              each selected branch active iff utility > 0
loss:              0.25*g1*Ls1 + 0.25*g2*Ls2 + 0.50*Lfp
```

The four stable candidates are generated first with the same random-call order
as CalibratedUtilityMatch. The two exploratory candidates are appended only
after the stable core exists. The candidate set is therefore a strict superset
of the stable pool rather than a replacement.

The best calibrated candidate always occupies one selected slot. The best
original-strength candidate may replace only calibrated rank-2, and only when
its utility is positive and higher than calibrated rank-2. Two exploratory
views can therefore never take over both strong-view slots.

No data, split, PreTrain, U-Net, EMA mode/update, pseudo labels, confidence
threshold, feature perturbation, optimizer, LR schedule, validation, test or
checkpoint behavior changes. Seed1337, 35/5/10, first seven=191, warm-up1000
and SelfTrain30000 remain locked.

## Stability boundary

This design cannot guarantee a numerical Dice before the server run. It does
guarantee the mechanism-level safety properties that were absent in degrading
runs:

- the stable four-candidate option set is never removed;
- at least one selected slot is reserved for the stable calibrated pool;
- at most one selected slot can come from the exploratory pool;
- an exploratory strong candidate must outrank calibrated rank-2;
- a selected exploratory candidate must also have strictly positive signed
  clean-gradient utility;
- all-negative batches remain feature-branch-only rather than accepting a
  conflicting strong view;
- runtime hooks abort if the six-candidate pool or gate is not active.

## Runtime evidence

The first post-warm-up iteration must print `GUARDED-POOL active iter=1001`.
`guarded_pool_trace.csv` records all six utilities, selected indices, source
(`calibrated` or `original`) and gate decisions. `training_summary.json`
records selection and activation counts by source.

## Locked decision gates

- stability gate: no sustained catastrophic post-warm-up collapse; complete
  30000 iterations without NaN/Inf or hook failure;
- performance floor: do not fall below the stable anchor about 0.816;
- recovery gate: exceed valid UtilityMatch 0.838397;
- advancement gate: exceed temporal-v2 0.841027;
- report all ten paired test-case deltas.

Dual-Timescale UtilityMatch is deferred because immediately removing the
calibrated pool and gate would violate the newly clarified stability premise.
POS+MEO remains a published engineering control after this guarded expansion.
