# Updated PROMISE12 Semi-Supervised Segmentation Code

This repository contains the cleaned PROMISE12 baseline, UniMatch, and
temporal-volume-bank experiments.

## Fixed protocol

- Split: 35 train / 5 validation / 10 test
- Labeled cases: first 7 cases in `train.list`
- Main seed: 1337
- Slice policy: all slices are retained; no `label.sum > 1000` filtering

The exact case lists and protocol are documented in
[`PROMISE12_PROTOCOL.md`](PROMISE12_PROTOCOL.md). Dataset files, model
checkpoints, predictions, and logs are intentionally not stored in Git.

## Prepare and validate data

```bash
python tools/convert_promise12_to_h5.py \
  --raw_root data/PROMISE12/extracted/training_data \
  --out_root data/PROMISE12_h5

python tools/check_promise12_split.py \
  --data_root data/PROMISE12_h5 \
  --require_h5
```

## UniMatch temporal-bank v2

This stage starts from an existing UniMatch self-training checkpoint; it does
not repeat UniMatch pre-training or self-training.

```bash
DATA_ROOT="$HOME/Documents/PROMISE12-Baseline/data/PROMISE12_h5" \
UNIMATCH_DIR="$HOME/Documents/PROMISE12-Baseline/model/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled" \
EXP_NAME="UniMatch_TemporalVolumeBankV2_35_5_10_seed1337" \
GPU=0 LABELNUM=7 SEED=1337 REFINE_ITERATIONS=5000 \
bash run_unimatch_temporal_bank_v2_5090.sh
```

Run the paired UniMatch/refinement test:

```bash
DATA_ROOT="$HOME/Documents/PROMISE12-Baseline/data/PROMISE12_h5" \
UNIMATCH_DIR="$HOME/Documents/PROMISE12-Baseline/model/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled" \
EXP_NAME="UniMatch_TemporalVolumeBankV2_35_5_10_seed1337" \
GPU=0 LABELNUM=7 \
bash test_and_quantify_unimatch_temporal_bank_v2_5090.sh
```

Run protocol tests with:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```
