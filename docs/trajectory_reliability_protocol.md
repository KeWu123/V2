# Trajectory Reliability Adaptive Self-Training

## Research question

Can prediction evolution control when, where, and how strongly unlabeled
supervision is used more reliably than instantaneous confidence and a fixed
training schedule?

The implementation is intentionally not a historical ensemble. The current
train-mode EMA teacher is the only pseudo-label target source. Historical EMA
snapshots evaluate the same current augmented image and are used only to
estimate reliability.

## Locked experimental protocol

- PROMISE12 split: 35 train / 5 validation / 10 test.
- First seven train cases: exactly 191 labeled slices; 940 train slices total.
- Initial state: the same validation-selected UniMatch Pre10000 checkpoint.
- U-Net, SGD state, batch 24/12, seed 1337, augmentation and 30,000 self-train
  iterations are unchanged.
- The online Student selects validation checkpoints and is evaluated on test.
- EMA teacher remains in `train()` during pseudo-label generation.
- Floating EMA state is averaged; integer buffers are copied from Student.

## Temporal reliability

Every 200 iterations, a deterministic frozen copy of the EMA teacher is added
to a queue of the latest four historical states. All queued teachers and the
current EMA teacher predict the same current augmented unlabeled batch,
avoiding coordinate mismatch from random rotation and flipping.

For each pixel, reliability is the mean of:

1. agreement with the latest historical class;
2. normalized inverse probability variance;
3. one minus the class flip rate.

The latest current EMA prediction remains the pseudo-label. Historical
probabilities are never averaged into that target. Pixels whose raw current
class disagrees with the LCC-processed UniMatch target receive zero trajectory
weight, preventing post-processing changes from becoming trusted supervision.

## Controlled ablations

| Mode | Pixel supervision | Global unlabeled strength | Boundary treatment |
|---|---|---|---|
| `baseline` | confidence >= 0.95 | original iteration ramp | none |
| `weighting` | continuous temporal reliability | original iteration ramp | none |
| `adaptive` | confidence >= 0.95 | foreground reliability | none |
| `weighting_adaptive` | continuous temporal reliability | foreground reliability | none |
| `full` | continuous temporal reliability | foreground reliability | stable-core soft boundary |

The `full` mode builds a foreground prototype from stable interior decoder
features. Prototype similarity is blended with the current EMA foreground
probability only inside a boundary band. It produces a soft target and never
hard-converts candidate boundary pixels to foreground.

## Required evidence

Before interpreting Dice, run the frozen validation diagnostic. It reports
confidence and temporal-reliability AUROC/correlation for predicting true
pseudo-label correctness over all pixels, foreground pixels and boundary
pixels. It also counts high-confidence wrong pixels and low-confidence but
temporally stable correct pixels.

The primary ablation order is:

1. fixed UniMatch reference;
2. `weighting` to test Where/How much at pixel level;
3. `adaptive` to test When/How much globally;
4. `weighting_adaptive` to test their interaction;
5. `full` to test whether stable cores recover uncertain boundaries.

Any gain must be reported with Dice, Jaccard, HD95, ASD and all ten case-level
results. A mean gain accompanied by a new severe patient regression is not
treated as successful.

## Entry points

```text
code/trajectory_reliability.py
code/train_trajectory_reliability.py
code/diagnose_trajectory_reliability.py
run_tr_baseline_5090.sh
run_tr_weighting_5090.sh
run_tr_adaptive_5090.sh
run_tr_weighting_adaptive_5090.sh
run_tr_full_5090.sh
test_trajectory_reliability_5090.sh
```
