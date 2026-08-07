# PROMISE12 UniMatch + S4MC

This experiment is isolated from `train_unimatch.py`. The unchanged reference
experiment remains available through `run_unimatch_5090.sh`.

## What changes

Only pseudo-label reliability after the supplied fixed self-training warm-up
(iteration 1000) changes.
Pre-training, the EMA teacher, 2D largest connected component pseudo masks,
two strong views, CutMix, feature perturbation, loss composition, optimizer,
validation and all comparison settings remain those of UniMatch.

The implementation follows the official S4MC margin/context code:

- Paper: <https://arxiv.org/abs/2308.13900>
- Official repository: <https://github.com/s4mcontext/s4mc>
- Reference loss implementation: <https://github.com/s4mcontext/s4mc/blob/main/s4mc_utils/utils/loss_helper.py>
- Reference schedule: <https://github.com/s4mcontext/s4mc/blob/main/train_semi.py>

For PROMISE12, the original `confidence >= 0.95` mask remains the trusted core.
S4MC uses top1-top2 margin and the strongest four-neighbour probability to add
context-supported foreground pixels. Its keep percentage follows the official
60-to-100 schedule over the active pseudo-label phase. The percentile is
computed only over LCC foreground pixels so the large MRI background cannot
dominate it. Pascal/Cityscapes class statistics are deliberately not copied to
this binary prostate dataset; a class-neutral expectation is used instead.

## Train and evaluate

```bash
bash run_unimatch_s4mc_5090.sh
bash test_and_quantify_unimatch_s4mc_5090.sh
```

The launcher checks every listed H5 signature before training. If an uploaded
folder contains Git LFS pointer text instead of real HDF5 data, it extracts the
included `data/PROMISE12/raw/training_data.zip` and rebuilds the H5 dataset
automatically before starting training.

Optional separate run name:

```bash
EXP_NAME=MT_PROMISE12_UniMatch_S4MC_seed1337_2 bash run_unimatch_s4mc_5090.sh
EXP_NAME=MT_PROMISE12_UniMatch_S4MC_seed1337_2 bash test_and_quantify_unimatch_s4mc_5090.sh
```

The final aggregate table is written to
`model/<EXP_NAME>_7_labeled/metric_table.csv`.
