# FrontierMatch locked protocol

Status: implemented locally; locked before server execution; numerical run pending.

## Scientific hypothesis

A strong-view policy should jointly determine how far the image is perturbed
and how much pseudo-label coverage is trusted. The policy whose induced
pseudo-supervised gradient transfers most positively to the labeled
segmentation task will provide a better reliability--difficulty trade-off than
fixed augmentation strength and a fixed global confidence threshold.

## Fixed experiment

- PROMISE12 source, five lists and H5 contents: unchanged.
- Split: 35/5/10; first seven labeled cases = exactly 191 slices.
- U-Net, Mean Teacher, EMA train-mode protocol and update: unchanged.
- Original fixed PreTrain and reference anchor: unchanged.
- Seed 1337, warm-up 1000, self-training 30000: unchanged.
- Student validation/checkpoint/test behavior: unchanged.
- Feature-perturbation branch keeps the original confidence threshold 0.95.
- Loss remains `0.25*g1*Ls1 + 0.25*g2*Ls2 + 0.50*Lfp`; rejected weights are
  not redistributed.

## Joint frontier candidates

Retain two independent UniMatch strong-view rays. Each ray samples one common
contrast/brightness/blur direction and one CutMix mapping, then instantiates
three policies:

| policy | brightness scale | pseudo-label threshold | role |
|---|---:|---:|---|
| stable | per-slice p01--p99 range | 0.95 | exact stable fallback mechanism |
| coverage | midpoint between p01--p99 and original absolute scale | 0.90 | recover informative lower-confidence pixels |
| reliable | original absolute scale | 0.98 | permit hard perturbation only with stricter targets |

Contrast, blur and CutMix settings remain the locked UniMatch settings. Only
the brightness unit and strong-branch pseudo-label mask vary across policies.
The same stochastic direction within a ray makes the three utilities a direct
comparison of policy rather than unrelated random views.

## Selection and safety

For policy `k`, compute the existing UtilityMatch projection

```text
u_k = <g_k, g_L> / ||g_L||
```

Select the greatest-utility policy independently in each ray, producing two
distinct strong views. A selected strong loss is active iff `u_k > 0`.
Therefore both rays may choose an experimental policy, but neither can train a
negative-transfer branch. An all-negative pair remains feature-branch-only.

This removes GuardedUtilityMatch's global one-exploratory quota while retaining
an exact stable option and the sign-safe fallback inside every ray.

## Runtime proof

- Entry banner: `FRONTIERMATCH ENTRY ACTIVE`.
- Hook proof: `FRONTIERMATCH VERIFIED`.
- First changed iteration: `FRONTIER active iter=1001`.
- `frontier_trace.csv`: all six utilities, selected policies, thresholds,
  severities, coverages and gate decisions.
- `training_summary.json`: selection/activation counts per policy and active
  gate/all-rejected statistics.

The run is invalid if any proof is absent.

## Decision gates

- Stability floor: no sustained post-warm-up collapse and Dice >= 0.816.
- Guarded recovery: Dice > 0.828.
- UtilityMatch recovery: Dice >= 0.838397.
- Advancement: Dice > 0.841027 without worse worst-case patient behavior.
