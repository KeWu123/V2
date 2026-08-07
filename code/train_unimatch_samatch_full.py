"""Full PROMISE12 Uni-MedSAM/SAMatch entry point.

The existing UniMatch Match branch remains in ``train_unimatch.py``. SAMatch
adds a labeled LiteMedSAM warm-up and a joint interactive stage.
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from samatch_full_training import (
    baseline,
    configure_logging,
    run_interactive,
    run_match_warmup,
    run_medsam_warmup,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

parser = argparse.ArgumentParser(
    description="Full Uni-MedSAM/SAMatch training on PROMISE12")
parser.add_argument(
    "--root_path", type=str,
    default=str(PROJECT_ROOT / "data" / "PROMISE12_h5"))
parser.add_argument(
    "--output_root", type=str, default=str(PROJECT_ROOT / "model"))
parser.add_argument(
    "--exp", type=str, default="MT_PROMISE12_UniMatch_SAMatchFull")
parser.add_argument("--model", type=str, default="unet")
parser.add_argument(
    "--stage", choices=("all", "match", "medsam", "interactive"),
    default="all")

# Project supervised initialization plus the paper's Match warm-up.
parser.add_argument("--match_pre_iterations", type=int, default=1000)
parser.add_argument("--match_self_iterations", type=int, default=30000)

# Method-specific complete SAMatch schedules.
parser.add_argument("--medsam_warmup_iterations", type=int, default=30000)
parser.add_argument("--interactive_iterations", type=int, default=30000)

parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--labeled_bs", type=int, default=12)
parser.add_argument("--labelnum", type=int, default=7)
parser.add_argument("--patch_size", type=int, nargs=2, default=[256, 256])
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--deterministic", type=int, default=1)

# Unchanged current UniMatch configuration.
parser.add_argument("--base_lr", type=float, default=0.01)
parser.add_argument("--ema_decay", type=float, default=0.99)
parser.add_argument("--consistency_type", type=str, default="mse")
parser.add_argument("--consistency", type=float, default=0.1)
parser.add_argument("--consistency_rampup", type=float, default=200.0)
parser.add_argument("--confidence_threshold", type=float, default=0.95)
parser.add_argument("--feature_dropout", type=float, default=0.5)
parser.add_argument("--strong_aug_prob", type=float, default=0.8)
parser.add_argument("--blur_prob", type=float, default=0.5)
parser.add_argument("--cutmix_prob", type=float, default=0.5)

# SAMatch-specific configuration.
parser.add_argument("--medsam_lr", type=float, default=5e-5)
parser.add_argument("--interactive_sam_lr", type=float, default=5e-5)
parser.add_argument("--sam_unlabeled_weight", type=float, default=0.25)
parser.add_argument("--bbox_shift", type=int, default=5)
parser.add_argument("--validation_interval", type=int, default=200)
parser.add_argument("--save_interval", type=int, default=3000)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--samatch_source_dir", type=str, required=True)
parser.add_argument("--medsam_pretrained", type=str, required=True)
parser.add_argument("--reuse_warmup", action="store_true")


def configure_reproducibility(config):
    if config.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)


def main(config):
    if config.num_classes != 2:
        raise ValueError("PROMISE12 SAMatch is a binary implementation")
    if config.patch_size != [256, 256]:
        raise ValueError("Official LiteMedSAM requires 256x256 input")
    if not 0 < config.labeled_bs < config.batch_size:
        raise ValueError("labeled_bs must be between 0 and batch_size")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_reproducibility(config)

    experiment_dir = (
        Path(config.output_root).resolve() /
        "{}_{}_labeled".format(config.exp, config.labelnum))
    match_pre_dir = (
        experiment_dir / "match_warmup" / "pre_train" / config.model)
    match_self_dir = (
        experiment_dir / "match_warmup" / "self_train" / config.model)
    medsam_dir = experiment_dir / "medsam_warmup"
    interactive_dir = experiment_dir / "interactive" / config.model
    for path in (
            match_pre_dir, match_self_dir, medsam_dir, interactive_dir):
        path.mkdir(parents=True, exist_ok=True)

    # train_unimatch.py expects these two names internally.
    config.pre_iterations = config.match_pre_iterations
    config.max_iterations = config.match_self_iterations
    baseline.args = config

    if config.stage in ("all", "match"):
        configure_logging(experiment_dir / "samatch_full.log")
        run_match_warmup(config, match_pre_dir, match_self_dir)
    if config.stage in ("all", "medsam"):
        configure_logging(experiment_dir / "samatch_full.log")
        run_medsam_warmup(config, medsam_dir)

    match_checkpoint = (
        match_self_dir / "{}_best_model.pth".format(config.model))
    medsam_checkpoint = medsam_dir / "medsam_lite_best.pth"
    if config.stage in ("all", "interactive"):
        missing = [
            str(path) for path in (match_checkpoint, medsam_checkpoint)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Interactive stage needs both warm-up checkpoints: " +
                ", ".join(missing))
        run_interactive(
            config, match_checkpoint, medsam_checkpoint, interactive_dir)
    logging.info("Requested full SAMatch stages completed")


if __name__ == "__main__":
    main(parser.parse_args())
