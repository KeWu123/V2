# H-UTILITYMATCH protocol

## Dataset identity lock for the clean Pre10000 rerun

The prospective from-scratch Pre10000 and its paired UniMatch/UtilityMatch
self-training branches must read the exact same server dataset used by the
existing SAMatch family:

```text
/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source
```

This is an absolute protocol constant, not a discoverable path and not a
user-overridable `DATA_ROOT` for this experiment. In particular, the run must
not fall back to `data/PROMISE12_h5`, a desktop copy, a newly converted H5
tree, or any directory inferred from the current working directory.

The only accepted layout is:

```text
train.list
train_slices.list
val.list
test.list
data/slices/<train_slices.list entry>.h5
data/<val or test case>.h5
```

Before Pre10000 starts, the launcher must resolve the root with `realpath` and
abort unless it equals the absolute path above. It must then enforce 35/5/10
case lists and 940 ordered training slices. The first 191 slice entries must be
exactly the seven labeled-case prefix used by the corrected run:

| Case | Slices |
|---|---:|
| Case48 | 24 |
| Case35 | 23 |
| Case04 | 46 |
| Case25 | 18 |
| Case23 | 20 |
| Case15 | 20 |
| Case08 | 40 |

No entry from these cases may occur after index 190. Every listed H5 file must
exist and contain usable `image` and `label` datasets. Filesystem glob counts
may be printed only as ignored diagnostics and must never determine sampler
indices.

At preflight, record SHA256 for all four list files in a dataset manifest.
Pre10000 and both 30k branches must read the same manifest and abort on any
path, list-hash, order, or H5-presence mismatch. This locks data identity
without relying on unavailable local H5 files.

## Implemented fresh-training entry

The non-overwriting implementation is:

- `code/train_fresh_pretrain.py`: constructs the unchanged U-Net from random
  seed-1337 initialization, loads no checkpoint, validates the exact dataset,
  and runs supervised Pre10000 using the existing UniMatch pretraining loop;
- `run_utilitymatch_fresh_5090.sh`: locks the absolute SAMatch data root,
  records/verifies the four-list SHA256 manifest, runs fresh Pre10000, verifies
  that the resulting checkpoint contains `net` and `opt`, and only then starts
  UtilityMatch Self30000 from that newly created state.

The fresh launcher intentionally accepts no command-line overrides. Run:

```bash
cd /home/aiteam/zhengtaoma/Baseline
bash run_utilitymatch_fresh_5090.sh
```

This implementation runs the requested fresh UtilityMatch branch. The matched
unchanged-UniMatch 30k control should reuse its saved fresh Pre10000 checkpoint
in a separate future run; it is not silently executed as part of this command.

Status: locked before server execution
Date: 2026-08-04

## Question

Can the two UniMatch strong branches be selected by their clean labeled-gradient
utility, rather than sampled blindly or filtered with confidence/stability
proxies, while preserving the original U-Net and pseudo-label path?

## Fixed source and data

- PROMISE12 fixed split: 35 train / 5 validation / 10 test cases.
- The first seven train cases are labeled and must occupy exactly the first 191
  entries in `train_slices.list`; total train slices must be 940.
- The sampler cutoff must be derived from that validated list prefix, never
  from a filesystem glob. With batch 24/12 the locked partition is exactly
  191 labeled, 749 unlabeled, and 15 batches per epoch; any other value aborts.
- Seed: 1337 only.
- Initial state for the already completed original H-UTILITYMATCH run was the
  **original fixed**
  `UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled/pre_train/unet/unet_best_model.pth`;
  a TLP-source checkpoint was not an allowed substitute. The prospective clean
  provenance control defined above instead regenerates one 191-slice
  Pre10000, then supplies that exact same `net` and `opt` state to both its
  unchanged-UniMatch and UtilityMatch branches. These are distinct recorded
  protocols and their checkpoints must not be mixed.
- Self-training: 30,000 iterations, batch 24/12, warm-up 1,000, base LR 0.01
  with the current UniMatch polynomial schedule. The archived original run
  restores SGD state from the original Pre10000; the fresh protocol restores
  it from the newly generated Pre10000 in the same fresh experiment folder.
- Effective local UniMatch constants remain fixed: EMA 0.99 with the
  pseudo-label teacher held in `train()` mode, confidence threshold 0.95,
  feature dropout 0.5,
  consistency 0.1/ramp-up 200, strong/blur/CutMix probabilities 0.8/0.5/0.5,
  and branch weights 0.25/0.25/0.50.

## Intervention

After the unchanged 1,000-iteration warm-up:

1. Generate four candidate strong branches from the exact existing
   brightness/contrast, Gaussian blur, and CutMix distribution. Each candidate
   has its own permutation, CutMix rectangle, transported EMA pseudo-label,
   and transported confidence map.
2. Compute the clean supervised gradient `g_L` using only the existing labeled
   CE+Dice loss and only the parameters of `model.decoder.out_conv`.
3. During a candidate-only forward, prevent BatchNorm running-buffer updates.
   Detach the candidate decoder features and recompute only the output
   convolution so candidate scoring cannot retain a full backbone graph.
4. For candidate `k`, compute the unchanged confidence-masked pseudo loss and
   its output-head gradient `g_k`. Score
   `U_k = <g_k, g_L> / (||g_L|| + eps)`.
5. Select the two candidates with largest `U_k`. Re-run only those two views
   through the normal train-mode student and use the unchanged two strong
   losses. The feature-perturbation branch and all other losses are untouched.

There is no confidence/JS/stability threshold, learned scorer, target
replacement, utility-temperature, or hand-composed module weight.

## Evaluation and interpretation

- Run one complete seed-1337 training as requested; this is an optimization
  experiment, not yet a multi-seed publication claim.
- Test the validation-selected online-student best checkpoint with the existing
  `test_unimatch.py` path and report all ten patient metrics.
- A result warrants continuation only if it exceeds the fixed original
  UniMatch test Dice 0.832233 by at least 0.005 and also exceeds the temporal-v2
  point estimate 0.841027, without introducing a new patient Dice regression
  greater than 0.05 relative to the available original per-case result.
- Because four candidates add scoring compute, a later equal-wall-clock
  UniMatch control is still required for a causal paper claim. It is deferred,
  not silently treated as completed.

## Stop conditions

- Abort before training if the fixed split, exact server data path,
  first-seven/191 ordering, chosen protocol's pretrain path, checkpoint
  structure, dependencies, or CUDA device fails validation.
- Abort on NaN/Inf utility or loss.
- Never overwrite an existing model directory.
- Before creating a fresh run, independently validate the ordered 191-slice
  prefix and require the imported `train_unimatch.patients_to_slices()` to
  return the same 191. Abort on disagreement; do not merely log it.
- Record SHA256 fingerprints of the runtime trainer, dataset loader, U-Net,
  loss, and validation sources. After Pre10000, record and print the saved
  validation-best Dice/iteration before SelfTrain is allowed to start.

## Fresh-run test synchronization

Use `test_utilitymatch_fresh_5090.sh` for the random-init Pre10000 ->
UtilityMatch Self30000 experiment. It resolves only
`server_logs/last_utilitymatch_fresh_run.txt`, requires the fresh PreTrain
configuration and saved four-list manifest, rechecks those list hashes, and
tests the validation-best online Student at
`self_train/unet/unet_best_model.pth`. It never falls back to a retained old
UtilityMatch or historical UniMatch directory. To inspect a particular
periodic online-Student checkpoint from the same fresh self-training folder,
set `CHECKPOINT=/absolute/path/iter_N.pth`.
