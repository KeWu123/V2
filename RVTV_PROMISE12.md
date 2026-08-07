# PROMISE12 UniMatch + RVTV pseudo-label routing

This experiment is isolated from the existing baseline and UniMatch files.
It keeps the supplied training protocol (`pretrain=1000`, `self=5000`, fixed
warm-up `1000`) and replaces only UniMatch's fixed confidence mask.

## Pseudo-label source

- The original EMA teacher and 2D LCC hard target are retained.
- A frozen copy of the best pretrain encoder builds a feature bank from the
  eight labeled PROMISE12 cases. Every stored feature has a ground-truth class.
- Reference confidence thresholds are calibrated per class by querying each
  labeled case while excluding that case from the reference bank.
- During the 1000-step supervised warm-up, an online bank records the foreground
  area of each unlabeled slice. Its historical variation gives temporal
  reliability; adjacent slice histories give volume reliability.

After warm-up, pixels are routed as follows:

- `hard`: EMA and labeled reference agree, both are confident, and the slice is
  temporally and volumetrically reliable;
- `soft`: moderately reliable or conflicting pixels use a detached mixture of
  EMA and reference probabilities with low-weight KL supervision;
- `ignore`: the remaining pixels contribute no pseudo-label gradient.

The two strong views, CutMix, feature perturbation branch, branch weights
`(0.25, 0.25, 0.50)`, optimizer, EMA update and supervised CE+Dice are unchanged.

## References

- Reference-guided pseudo labels: <https://arxiv.org/abs/2112.00735>
- Temporal/holistic prediction stability (ST++): <https://arxiv.org/abs/2106.05095>
- Official UniMatch: <https://github.com/LiheYoung/UniMatch>

## RTX 5090

```bash
bash run_unimatch_rvtv_5090.sh
bash test_and_quantify_unimatch_rvtv_5090.sh
```

Training logs report calibrated background/foreground reference thresholds,
hard/soft coverage, foreground hard coverage, reference agreement, temporal
score and adjacent-slice volume score. These values are saved to TensorBoard
under the `rvtv/` prefix as well.
