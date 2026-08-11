"""Frozen validation diagnostic for confidence versus trajectory reliability."""

import argparse
import csv
import json
import os
import re

import h5py
import numpy as np
import torch
from scipy.ndimage import zoom

import train_utilitymatch as common
from trajectory_reliability import boundary_band, trajectory_statistics


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--history_dir", type=str, default="")
    parser.add_argument("--checkpoint_paths", nargs="*", default=[])
    parser.add_argument("--history_count", type=int, default=4)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--patch_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--boundary_radius", type=int, default=2)
    return parser


def numbered_checkpoints(history_dir):
    pattern = re.compile(r"^iter_(\d+)\.pth$")
    values = []
    if history_dir and os.path.isdir(history_dir):
        for name in os.listdir(history_dir):
            match = pattern.match(name)
            if match:
                values.append((int(match.group(1)), os.path.join(history_dir, name)))
    return [path for _, path in sorted(values)]


def resolve_checkpoints(args):
    paths = [os.path.abspath(path) for path in args.checkpoint_paths]
    if not paths:
        paths = numbered_checkpoints(os.path.abspath(args.history_dir))
        paths = paths[-int(args.history_count) :]
    if len(paths) < 2:
        raise ValueError("At least two ordered checkpoints are required")
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Missing checkpoints: {}".format(missing))
    return paths


def extract_state(raw):
    if isinstance(raw, dict) and "net" in raw:
        raw = raw["net"]
    if not isinstance(raw, dict):
        raise TypeError("Checkpoint is not a state dictionary")
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in raw.items()
    }


def load_models(paths):
    models = []
    for path in paths:
        model = common.base.UNet(in_chns=1, class_num=2).cuda()
        try:
            raw = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            raw = torch.load(path, map_location="cpu")
        model.load_state_dict(extract_state(raw), strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models.append(model)
    return models


def binary_auroc(correctness, scores):
    labels = np.asarray(correctness, dtype=np.uint8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    scores = scores[order]
    distinct_end = np.r_[np.flatnonzero(np.diff(scores)), labels.size - 1]
    true_positive = np.cumsum(labels)[distinct_end]
    false_positive = (distinct_end + 1) - true_positive
    tpr = np.r_[0.0, true_positive / positives, 1.0]
    fpr = np.r_[0.0, false_positive / negatives, 1.0]
    return float(np.trapz(tpr, fpr))


def point_biserial(correctness, scores):
    labels = np.asarray(correctness, dtype=np.float64).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size < 2 or labels.std() == 0 or values.std() == 0:
        return float("nan")
    return float(np.corrcoef(labels, values)[0, 1])


@torch.no_grad()
def evaluate_case(models, image, label, patch_size, boundary_radius):
    depth, height, width = image.shape
    confidence_values = []
    reliability_values = []
    correctness_values = []
    foreground_values = []
    boundary_values = []

    for index in range(depth):
        resized_image = zoom(
            image[index],
            (patch_size[0] / height, patch_size[1] / width),
            order=0,
        )
        resized_label = zoom(
            label[index],
            (patch_size[0] / height, patch_size[1] / width),
            order=0,
        )
        tensor = torch.from_numpy(resized_image).float().unsqueeze(0).unsqueeze(0).cuda()
        probabilities = []
        current_logits = None
        for model in models:
            output = model(tensor)
            logits = output[0] if isinstance(output, tuple) else output
            current_logits = logits
            probabilities.append(torch.softmax(logits, dim=1)[:, 1])
        history = torch.stack(probabilities, dim=0)
        reliability = trajectory_statistics(history)["reliability"]
        current_probability = history[-1]
        raw_prediction = current_probability >= 0.5
        prediction = common.base.get_masks(current_logits, nms=1).bool()
        reliability = reliability * (raw_prediction == prediction).float()
        confidence = torch.where(
            prediction, current_probability, 1.0 - current_probability
        )
        target = torch.from_numpy(resized_label).long().unsqueeze(0).cuda()
        correctness = prediction == (target == 1)
        foreground = target == 1
        boundary = boundary_band(target, radius=boundary_radius)

        confidence_values.append(confidence.cpu().numpy().reshape(-1))
        reliability_values.append(reliability.cpu().numpy().reshape(-1))
        correctness_values.append(correctness.cpu().numpy().reshape(-1))
        foreground_values.append(foreground.cpu().numpy().reshape(-1))
        boundary_values.append(boundary.cpu().numpy().reshape(-1))

    return {
        "confidence": np.concatenate(confidence_values),
        "reliability": np.concatenate(reliability_values),
        "correct": np.concatenate(correctness_values).astype(bool),
        "foreground": np.concatenate(foreground_values).astype(bool),
        "boundary": np.concatenate(boundary_values).astype(bool),
    }


def summarize_region(case_name, region_name, mask, values):
    rows = []
    correct = values["correct"][mask]
    for signal in ("confidence", "reliability"):
        score = values[signal][mask]
        rows.append(
            {
                "case": case_name,
                "region": region_name,
                "signal": signal,
                "pixels": int(mask.sum()),
                "accuracy": float(correct.mean()) if correct.size else float("nan"),
                "auroc_correctness": binary_auroc(correct, score),
                "correlation_correctness": point_biserial(correct, score),
            }
        )
    return rows


def main(args):
    args.root_path = os.path.abspath(args.root_path)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_paths = resolve_checkpoints(args)
    models = load_models(checkpoint_paths)
    with open(
        os.path.join(args.root_path, args.split + ".list"), "r", encoding="utf-8"
    ) as handle:
        cases = [line.strip().split(".")[0] for line in handle if line.strip()]

    rows = []
    event_rows = []
    pooled = {key: [] for key in ("confidence", "reliability", "correct", "foreground", "boundary")}
    for case in cases:
        with h5py.File(
            os.path.join(args.root_path, "data", case + ".h5"), "r"
        ) as handle:
            values = evaluate_case(
                models,
                handle["image"][:],
                handle["label"][:],
                args.patch_size,
                args.boundary_radius,
            )
        for key in pooled:
            pooled[key].append(values[key])
        regions = {
            "all": np.ones_like(values["correct"], dtype=bool),
            "gt_foreground": values["foreground"],
            "gt_boundary": values["boundary"],
        }
        for region_name, mask in regions.items():
            rows.extend(summarize_region(case, region_name, mask, values))
        high_confidence_wrong = (values["confidence"] >= 0.95) & (~values["correct"])
        low_confidence_stable_correct = (
            (values["confidence"] < 0.95)
            & (values["reliability"] >= 0.75)
            & values["correct"]
        )
        event_rows.append(
            {
                "case": case,
                "pixels": int(values["correct"].size),
                "high_confidence_wrong": int(high_confidence_wrong.sum()),
                "high_confidence_wrong_ratio": float(high_confidence_wrong.mean()),
                "low_confidence_stable_correct": int(low_confidence_stable_correct.sum()),
                "low_confidence_stable_correct_ratio": float(
                    low_confidence_stable_correct.mean()
                ),
            }
        )

    pooled = {key: np.concatenate(value) for key, value in pooled.items()}
    regions = {
        "all": np.ones_like(pooled["correct"], dtype=bool),
        "gt_foreground": pooled["foreground"],
        "gt_boundary": pooled["boundary"],
    }
    for region_name, mask in regions.items():
        rows.extend(summarize_region("ALL", region_name, mask, pooled))

    metric_path = os.path.join(args.output_dir, "trajectory_signal_metrics.csv")
    with open(metric_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    event_path = os.path.join(args.output_dir, "trajectory_error_events.csv")
    with open(event_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=event_rows[0].keys())
        writer.writeheader()
        writer.writerows(event_rows)
    with open(
        os.path.join(args.output_dir, "diagnostic_config.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {"checkpoints": checkpoint_paths, "split": args.split},
            handle,
            indent=2,
        )
    print("Saved {}".format(metric_path))
    print("Saved {}".format(event_path))
    for row in rows:
        if row["case"] == "ALL":
            print(row)


if __name__ == "__main__":
    main(build_parser().parse_args())
