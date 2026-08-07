# UniMatch foreground rescue experiment

This is an isolated short-schedule experiment. It does not modify
`code/train_unimatch.py` or any previous result.

- Dataset: PROMISE12 H5
- Labeled cases: 8
- Seed: 1337
- Pre-training: 1000 iterations
- Self-training: 5000 iterations
- Supplied baseline warm-up: 1000 iterations
- Original UniMatch hard pseudo-label loss: unchanged
- Rescue start: self-training iteration 500
- Rescue ramp: 1000 iterations
- Maximum rescue weight: 0.15
- Rescue candidates: foreground probability at least 0.80, agreement between
  original and horizontally flipped EMA predictions, foreground probability
  disagreement at most 0.10, and within 5 pixels of a high-confidence
  foreground seed
- Rescue supervision: soft KL only

Train:

```bash
bash run_unimatch_fg_rescue_5090.sh
```

Test and create `metric_table.csv`, `metric_table.md`, and
`test_case_metrics.csv`:

```bash
bash test_and_quantify_unimatch_fg_rescue_5090.sh
```
