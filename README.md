# UtilityMatch / FrontierMatch for PROMISE12

This repository contains the complete training and evaluation code used by the
UtilityMatch family on the fixed PROMISE12 UniMatch protocol. Dataset files,
checkpoints, logs, and predictions are intentionally excluded.

## Included methods

- `UtilityMatch`: ranks four UniMatch strong-view candidates with labeled-task
  gradient utility and trains the best two.
- `SafeUtilityMatch`: adds strict positive-utility abstention.
- `GuardedUtilityMatch`: guarded continuation used in the later experiments.
- `CalibratedUtilityMatch`: scales brightness offsets by the slice p01--p99
  intensity range.
- `FrontierMatch`: selects a joint augmentation-strength / pseudo-label
  coverage policy independently on each of the two UniMatch strong-view rays.

The original UniMatch training entry, test entry, PROMISE12 loader, validation
utilities, model definitions, smoke tests, and server launchers required by
these methods are included under `code/` and the repository root.

## Required local files

Place the unchanged PROMISE12 H5 tree at:

```text
data/PROMISE12_h5_training_source/
  train_slices.list
  train_labeled.list
  train_unlabeled.list
  val.list
  test.list
  data/slices/*.h5
  data/*.h5
```

Place the original fixed UniMatch run below `model/`. The standard launchers
expect its supervised checkpoint at:

```text
model/<original-UniMatch-run>/pre_train/unet/unet_best_model.pth
```

The scripts verify the fixed 35/5/10 split, seven labeled cases, list hashes,
used H5 hashes, seed 1337, checkpoint identity, and GPU type before training.
They do not download or alter the dataset.

## Environment

Install a CUDA-compatible PyTorch build first, then install the remaining
packages:

```bash
python -m pip install -r requirements.txt
chmod +x ./*.sh
```

The launchers use conda environment `BCP` when it exists, otherwise `python3`
or `python` from the active shell.

## Train

Original UtilityMatch:

```bash
bash run_utilitymatch_5090.sh
```

Safe, guarded, calibrated, or FrontierMatch:

```bash
bash run_utilitymatch_safe_5090.sh
bash run_utilitymatch_guarded_5090.sh
bash run_utilitymatch_calibrated_5090.sh
bash run_frontiermatch_5090.sh
```

To select the exact original UniMatch directory explicitly:

```bash
ORIGINAL_UNIMATCH_DIR=/absolute/path/to/original-unimatch-run \
  bash run_frontiermatch_5090.sh
```

Set `DETACH=1` for background execution. Every launcher writes the exact run
directory to `server_logs/last_*_run.txt`.

## Test

```bash
bash test_utilitymatch_5090.sh
bash test_utilitymatch_safe_5090.sh
bash test_utilitymatch_guarded_5090.sh
bash test_utilitymatch_calibrated_5090.sh
bash test_frontiermatch_5090.sh
```

An exact run directory can be supplied when auto-discovery is inappropriate:

```bash
UTILITYMATCH_DIR=/absolute/path/to/run bash test_frontiermatch_5090.sh
```

## Reproducibility boundary

- Fixed split: 35 train / 5 validation / 10 test.
- Fixed labeled cases: first seven cases, 191 labeled slices.
- Fixed seed: 1337.
- Fixed schedule: PreTrain 10,000 iterations and self-training 30,000
  iterations.
- Online Student is used for validation and test, matching the locked
  experimental protocol.

See `docs/` for the implementation protocols and the Chinese method summary.

## Trajectory-reliability experiments

The trajectory branch tests whether prediction dynamics should control
semi-supervised trust. All modes share one trainer and the fixed UniMatch
protocol:

```bash
bash run_tr_baseline_5090.sh
bash run_tr_weighting_5090.sh
bash run_tr_adaptive_5090.sh
bash run_tr_weighting_adaptive_5090.sh
bash run_tr_full_5090.sh
```

Set `DATA_ROOT`, `ORIGINAL_UNIMATCH_DIR`, `GPU`, `REQUIRE_5090`, and `DETACH`
in the same way as UtilityMatch. The complete suite can be run serially with
`run_trajectory_ablation_suite_5090.sh`; it intentionally never launches the
five jobs concurrently.

Run the frozen signal-quality diagnostic before interpreting training gains:

```bash
bash diagnose_trajectory_reliability_5090.sh
```

Evaluate a completed mode with:

```bash
TRAJECTORY_MODE=full bash test_trajectory_reliability_5090.sh
```

The method definition, controlled ablations, and interpretation limits are in
`docs/trajectory_reliability_protocol.md`.
