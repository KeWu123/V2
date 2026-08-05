"""Fresh supervised Pre10000 for the locked PROMISE12 UtilityMatch run.

This entry intentionally loads no model checkpoint.  It validates the exact
SAMatch PROMISE12 server tree, initializes the unchanged U-Net after seeding,
and delegates the supervised optimization loop to the current UniMatch code.
"""

import argparse
import glob
import hashlib
import json
import logging
import os
import random
import re
import sys
from collections import OrderedDict

import h5py
import numpy as np
import torch
from torch.backends import cudnn


_entry_argv = sys.argv[:]
try:
    sys.argv = [sys.argv[0]]
    import train_unimatch as base
finally:
    sys.argv = _entry_argv


EXPECTED_DATA_ROOT = (
    "/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source"
)
EXPECTED_LABELED_COUNTS = OrderedDict(
    (
        ("Case48", 24),
        ("Case35", 23),
        ("Case04", 46),
        ("Case25", 18),
        ("Case23", 20),
        ("Case15", 20),
        ("Case08", 40),
    )
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fresh locked PROMISE12 supervised Pre10000"
    )
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="unet")
    parser.add_argument("--pre_iterations", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--labeled_bs", type=int, default=12)
    parser.add_argument("--labelnum", type=int, default=7)
    parser.add_argument("--patch_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--num_classes", type=int, default=2)
    return parser


def read_nonempty(path, strip_extension=False):
    with open(path, "r", encoding="utf-8") as handle:
        values = [line.strip() for line in handle if line.strip()]
    if strip_extension:
        values = [value.split(".")[0] for value in values]
    return values


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_h5(path):
    if not os.path.isfile(path) or os.path.getsize(path) < 1024:
        raise FileNotFoundError("Missing or truncated locked H5: {}".format(path))
    with open(path, "rb") as handle:
        if handle.read(8) != b"\x89HDF\r\n\x1a\n":
            raise ValueError("Invalid HDF5 signature: {}".format(path))
    with h5py.File(path, "r") as handle:
        if not {"image", "label"}.issubset(handle.keys()):
            raise ValueError("Missing image/label datasets: {}".format(path))
        if handle["image"].shape != handle["label"].shape:
            raise ValueError("Image/label shape mismatch: {}".format(path))


def validate_protocol(args):
    locked_ints = {
        "pre_iterations": 10000,
        "batch_size": 24,
        "labeled_bs": 12,
        "labelnum": 7,
        "seed": 1337,
        "deterministic": 1,
        "num_classes": 2,
    }
    for name, expected in locked_ints.items():
        actual = int(getattr(args, name))
        if actual != expected:
            raise ValueError(
                "Fresh Pre10000 requires {}={}, got {}".format(
                    name, expected, actual
                )
            )
    if args.model != "unet":
        raise ValueError("Fresh Pre10000 requires the unchanged U-Net")
    if list(args.patch_size) != [256, 256]:
        raise ValueError("Fresh Pre10000 requires patch_size 256 256")
    if abs(float(args.base_lr) - 0.01) > 1e-12:
        raise ValueError("Fresh Pre10000 requires base_lr=0.01")

    requested_root = os.path.abspath(args.root_path)
    resolved_root = os.path.realpath(requested_root)
    if requested_root != EXPECTED_DATA_ROOT or resolved_root != EXPECTED_DATA_ROOT:
        raise ValueError(
            "Dataset root must be the exact SAMatch path {} (got abspath={}, "
            "realpath={})".format(EXPECTED_DATA_ROOT, requested_root, resolved_root)
        )

    list_paths = {
        name: os.path.join(requested_root, name)
        for name in ("train.list", "train_slices.list", "val.list", "test.list")
    }
    for path in list_paths.values():
        if not os.path.isfile(path):
            raise FileNotFoundError("Missing locked dataset list: {}".format(path))

    train = read_nonempty(list_paths["train.list"], strip_extension=True)
    val = read_nonempty(list_paths["val.list"], strip_extension=True)
    test = read_nonempty(list_paths["test.list"], strip_extension=True)
    slices = read_nonempty(list_paths["train_slices.list"])
    if (len(train), len(val), len(test), len(slices)) != (35, 5, 10, 940):
        raise ValueError(
            "Expected train/val/test/slices=35/5/10/940, got {}/{}/{}/{}".format(
                len(train), len(val), len(test), len(slices)
            )
        )
    if len(set(train + val + test)) != 50:
        raise ValueError("Locked train/val/test case lists overlap")

    expected_cases = list(EXPECTED_LABELED_COUNTS)
    if train[:7] != expected_cases:
        raise ValueError(
            "First seven train cases must be {}, got {}".format(
                expected_cases, train[:7]
            )
        )
    prefixes = tuple(case + "_slice" for case in expected_cases)
    counts = OrderedDict(
        (
            case,
            sum(name.startswith(case + "_slice") for name in slices),
        )
        for case in expected_cases
    )
    if counts != EXPECTED_LABELED_COUNTS:
        raise ValueError(
            "Locked labeled slice counts differ: expected {}, got {}".format(
                dict(EXPECTED_LABELED_COUNTS), dict(counts)
            )
        )
    if not all(name.startswith(prefixes) for name in slices[:191]):
        raise ValueError("The first 191 train_slices entries are not all labeled")
    if any(name.startswith(prefixes) for name in slices[191:]):
        raise ValueError("A labeled-case slice occurs after train_slices index 190")

    for name in slices:
        check_h5(os.path.join(requested_root, "data", "slices", name + ".h5"))
    for case in val + test:
        check_h5(os.path.join(requested_root, "data", case + ".h5"))

    manifest = {
        "absolute_root": requested_root,
        "resolved_root": resolved_root,
        "split_counts": {"train": 35, "val": 5, "test": 10, "slices": 940},
        "labeled_cases": expected_cases,
        "labeled_slice_counts": dict(counts),
        "labeled_slices": 191,
        "unlabeled_slices": 749,
        "batches_per_epoch": 15,
        "list_sha256": {
            name: sha256(path) for name, path in list_paths.items()
        },
    }
    return manifest


def configure_logging(output_dir):
    formatter = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S"
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)
    file_handler = logging.FileHandler(
        os.path.join(output_dir, "train.log"), mode="a", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def main(args):
    args.root_path = os.path.abspath(args.root_path)
    args.output_dir = os.path.abspath(args.output_dir)
    if os.path.exists(args.output_dir):
        raise FileExistsError(
            "Refusing to overwrite fresh Pre10000 output: {}".format(
                args.output_dir
            )
        )
    manifest = validate_protocol(args)
    runtime_labeled_slices = int(
        base.patients_to_slices(args.root_path, args.labelnum)
    )
    if runtime_labeled_slices != 191:
        raise RuntimeError(
            "The imported train_unimatch.py is not synchronized: its "
            "patients_to_slices() reports {} instead of the locked 191. "
            "Update code/train_unimatch.py before training.".format(
                runtime_labeled_slices
            )
        )

    code_dir = os.path.dirname(os.path.abspath(base.__file__))
    runtime_files = OrderedDict(
        (
            ("train_unimatch.py", os.path.abspath(base.__file__)),
            (
                "dataloaders/dataset.py",
                os.path.join(code_dir, "dataloaders", "dataset.py"),
            ),
            ("networks/unet.py", os.path.join(code_dir, "networks", "unet.py")),
            ("utils/losses.py", os.path.join(code_dir, "utils", "losses.py")),
            ("utils/val_2d.py", os.path.join(code_dir, "utils", "val_2d.py")),
        )
    )
    for path in runtime_files.values():
        if not os.path.isfile(path):
            raise FileNotFoundError("Missing runtime training source: {}".format(path))
    runtime_code_sha256 = OrderedDict(
        (name, sha256(path)) for name, path in runtime_files.items()
    )

    os.makedirs(args.output_dir)
    configure_logging(args.output_dir)
    seed_everything(args.seed)

    config = vars(args).copy()
    config.update(
        {
            "initialization": "random_seeded_no_checkpoint_loaded",
            "protocol": "PROMISE12_exact_SAMatch_root_first7_191_Pre10000",
            "dataset_manifest": manifest,
            "base_training_entry": os.path.abspath(base.__file__),
            "runtime_labeled_slices": runtime_labeled_slices,
            "runtime_code_sha256": runtime_code_sha256,
            "pretrain_recipe": {
                "model": "UNet(in_chns=1, class_num=2)",
                "initialization": "random_seed_1337_no_checkpoint",
                "optimizer": "SGD(lr=0.01,momentum=0.9,weight_decay=0.0001)",
                "lr_schedule": "fixed_0.01",
                "loss": "0.5*(CrossEntropy+Dice)",
                "labeled_batch": 12,
                "total_batch": 24,
                "validation": "online_student_every_200_iterations",
            },
        }
    )
    with open(
        os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    logging.info("Arguments: %s", args)
    logging.info("Dataset manifest: %s", manifest)
    logging.info(
        "FRESH PRE10000 START: random initialization; no checkpoint is loaded"
    )
    logging.info(
        "Protocol-locked sampler: labeled=191 unlabeled=749 batches=15 "
        "source=train_slices.list prefix"
    )
    logging.info(
        "Runtime cross-check: imported patients_to_slices()=%d", 
        runtime_labeled_slices,
    )
    logging.info("Runtime training-code SHA256: %s", runtime_code_sha256)
    logging.info(
        "PreTrain recipe: random U-Net; SGD lr=0.01 fixed, momentum=0.9, "
        "weight_decay=1e-4; CE+Dice; batch=24/12; val=Student"
    )
    base.args = args
    base.pre_train(args, args.output_dir)

    checkpoint_path = os.path.join(args.output_dir, "unet_best_model.pth")
    if not os.path.isfile(checkpoint_path):
        raise RuntimeError(
            "Fresh Pre10000 finished without best checkpoint: {}".format(
                checkpoint_path
            )
        )
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not {"net", "opt"}.issubset(checkpoint):
        raise RuntimeError("Fresh checkpoint must contain both net and opt states")

    best_pattern = re.compile(r"^iter_(\d+)_dice_([0-9.]+)\.pth$")
    validation_records = []
    for path in glob.glob(os.path.join(args.output_dir, "iter_*_dice_*.pth")):
        match = best_pattern.match(os.path.basename(path))
        if match:
            validation_records.append(
                (float(match.group(2)), int(match.group(1)), os.path.abspath(path))
            )
    if not validation_records:
        raise RuntimeError("Fresh Pre10000 produced no validation checkpoint")
    best_validation_dice, best_validation_iteration, best_iteration_path = max(
        validation_records
    )

    summary = {
        "completed_iterations": 10000,
        "initialization": "random_seeded_no_checkpoint_loaded",
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": sha256(checkpoint_path),
        "dataset_list_sha256": manifest["list_sha256"],
        "labeled_slices": 191,
        "unlabeled_slices": 749,
        "batches_per_epoch": 15,
        "runtime_labeled_slices": runtime_labeled_slices,
        "runtime_code_sha256": runtime_code_sha256,
        "best_validation_dice_rounded": best_validation_dice,
        "best_validation_iteration": best_validation_iteration,
        "best_iteration_checkpoint": best_iteration_path,
    }
    with open(
        os.path.join(args.output_dir, "training_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    logging.info("FRESH PRE10000 COMPLETED: %s", summary)


if __name__ == "__main__":
    main(build_parser().parse_args())
