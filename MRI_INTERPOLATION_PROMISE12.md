# PROMISE12 MRI interpolation experiment

This experiment keeps the existing Baseline + UniMatch method and defaults:

- 7 labeled PROMISE12 cases (222 labeled slices)
- pretrain 1000, self-train 5000, supervised warm-up 1000
- batch size 24, labeled batch size 12
- seed 1337, learning rate 0.01
- EMA teacher, 2D-LCC, UniMatch strong views, CutMix and feature dropout unchanged

The only change is interpolation:

- MRI image rotation and resize: linear interpolation (`order=1`)
- segmentation labels and output masks: nearest-neighbour (`order=0`)

Train on the RTX 5090 server:

```bash
bash run_unimatch_mri_interp_5090.sh
```

If the uploaded `PROMISE12_h5` files are Git LFS pointer files, the launcher
automatically extracts `data/PROMISE12/raw/training_data.zip` and rebuilds the
real H5 dataset before training.

Evaluate and print/save the metric table:

```bash
bash test_and_quantify_unimatch_mri_interp_5090.sh
```

Use a different output name when repeating without overwriting:

```bash
EXP_NAME=MT_PROMISE12_UniMatch_MRIInterp_repeat \
bash run_unimatch_mri_interp_5090.sh
```
