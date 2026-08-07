# PROMISE12 experiment protocol

This repository uses one canonical split for every reported experiment:

- Train: 35 cases
- Validation: 5 cases
- Test: 10 cases
- Labeled subset: the first 7 cases in `train.list`
- Seed: 1337 unless an experiment explicitly says otherwise
- Slice policy: keep every slice from each training volume

The split is **35/5/10**, not 35/5/5 and not 42/4/4. All 50 cases are used
exactly once, with no overlap. No `label.sum > 1000` filtering is applied.

## Cases

Train:

```text
Case48 Case35 Case04 Case25 Case23 Case15 Case08 Case00 Case17 Case44
Case47 Case11 Case18 Case26 Case42 Case33 Case24 Case14 Case29 Case06
Case27 Case41 Case28 Case13 Case37 Case12 Case40 Case20 Case01 Case32
Case19 Case21 Case39 Case10 Case03
```

Validation:

```text
Case31 Case02 Case07 Case46 Case22
```

Test:

```text
Case09 Case30 Case45 Case34 Case43 Case36 Case38 Case16 Case05 Case49
```

## Validate a prepared dataset

```bash
python tools/check_promise12_split.py \
  --data_root data/PROMISE12_h5 \
  --require_h5
```

## Rebuild from raw PROMISE12 files

```bash
python tools/convert_promise12_to_h5.py \
  --raw_root data/PROMISE12/extracted/training_data \
  --out_root data/PROMISE12_h5
```

The converter writes `train.list`, `train_slices.list`, `val.list`, and
`test.list` from the constants in `tools/promise12_protocol.py`.
