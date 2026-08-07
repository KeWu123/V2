# PROMISE12 Uni-MedSAM / SAMatch

This implementation follows the complete two-stage SAMatch method and keeps
the existing project UniMatch implementation as the Match branch.

Official references:

- Paper: <https://arxiv.org/abs/2411.16949>
- Code: <https://github.com/apple1986/SAMatch>
- Pinned code commit: `0ab023e643177a8a9dc6f76181c92b52225a71eb`
- SAMatch source license: Apache-2.0

## Schedule terminology

The schedules are not interchangeable:

| Argument | Meaning | Default |
|---|---|---:|
| `MATCH_PRE_ITERATIONS` | Existing project supervised U-Net pretrain | 1000 |
| `MATCH_SELF_ITERATIONS` | UniMatch/Match part of SAMatch warm-up | 30000 |
| `MEDSAM_WARMUP_ITERATIONS` | SAMatch labeled LiteMedSAM adaptation | 30000 |
| `INTERACTIVE_ITERATIONS` | SAMatch joint Match/LiteMedSAM interaction | 30000 |

`MATCH_PRE_ITERATIONS=1000` preserves the project's supervised U-Net
initialization. In the paper's first 30k stage, the Match network and MedSAM
are warmed up separately: `MATCH_SELF_ITERATIONS=30000` trains the complete
UniMatch/Match initialization and `MEDSAM_WARMUP_ITERATIONS=30000` adapts
LiteMedSAM. `INTERACTIVE_ITERATIONS=30000` is the second paper stage and jointly
trains both networks. It is therefore not equivalent to setting the old
`pre/self` pair to `30000/30000`.

Setting `MATCH_SELF_ITERATIONS=5000` keeps every method component but is a
short-schedule experiment rather than the complete paper training schedule.

## Training

```bash
bash run_unimatch_samatch_full_5090.sh
```

Run in the background:

```bash
DETACH=1 bash run_unimatch_samatch_full_5090.sh
```

Reuse completed Match and MedSAM warm-up checkpoints:

```bash
REUSE_WARMUP=1 ALLOW_EXISTING=1 \
bash run_unimatch_samatch_full_5090.sh
```

Run one stage explicitly:

```bash
STAGE=medsam ALLOW_EXISTING=1 \
bash run_unimatch_samatch_full_5090.sh

STAGE=interactive REUSE_WARMUP=1 ALLOW_EXISTING=1 \
bash run_unimatch_samatch_full_5090.sh
```

## Test and quantify

```bash
bash test_and_quantify_unimatch_samatch_full_5090.sh
```

The generated table contains:

- `match_pre`: supervised U-Net initialization;
- `match_self`: UniMatch initialization used by SAMatch;
- `samatch_interactive`: final full Uni-MedSAM model.

The aggregate table is written to `metric_table.csv`, and the per-case table
is written to `test_case_metrics.csv`.

## Complete method components retained

- existing UniMatch Match branch;
- EMA weak-view teacher and largest connected component prompt source;
- two strong MRI views and CutMix;
- feature-dropout UniMatch branch;
- labeled LiteMedSAM warm-up with box prompts;
- Dice, BCE, and IoU LiteMedSAM warm-up losses;
- joint LiteMedSAM update from labeled and unlabeled targets;
- `0.25` unlabeled LiteMedSAM loss weight;
- MedSAM-refined pseudo masks for all three UniMatch unsupervised branches;
- independent polynomial learning-rate decay for Match and LiteMedSAM during
  interactive training.
