#!/usr/bin/env bash
set -Eeuo pipefail

# Test the three U-Net checkpoints in a full SAMatch run and create one table:
# Match supervised pretrain, Match UniMatch self-train, SAMatch interactive.

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${PROJECT_ROOT}/code"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/PROMISE12_h5}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/model}"
EXP_NAME="${EXP_NAME:-MT_PROMISE12_UniMatch_SAMatchFull}"
LABELNUM="${LABELNUM:-7}"
GPU="${GPU:-0}"
SAVE_RESULT="${SAVE_RESULT:-True}"
NMS="${NMS:-0}"
EXPERIMENT_DIR="${MODEL_ROOT}/${EXP_NAME}_${LABELNUM}_labeled"

MATCH_PRE_DIR="${EXPERIMENT_DIR}/match_warmup/pre_train/unet"
MATCH_SELF_DIR="${EXPERIMENT_DIR}/match_warmup/self_train/unet"
INTERACTIVE_DIR="${EXPERIMENT_DIR}/interactive/unet"

PYTHON_LAUNCHER=()
if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_LAUNCHER=("${PYTHON_BIN}")
elif [[ -n "${CONDA_ENV:-}" ]]; then
    PYTHON_LAUNCHER=(conda run --no-capture-output -n "${CONDA_ENV}" python)
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_LAUNCHER=(python3)
else
    PYTHON_LAUNCHER=(python)
fi

run_stage() {
    local stage_label="$1"
    local snapshot_dir="$2"
    local checkpoint="${snapshot_dir}/unet_best_model.pth"
    [[ -f "${checkpoint}" ]] || {
        echo "Missing ${stage_label} checkpoint: ${checkpoint}" >&2
        exit 2
    }
    echo "Testing ${stage_label}: ${checkpoint}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_LAUNCHER[@]}" \
        "${CODE_DIR}/test_unimatch.py" \
        --root_path "${DATA_ROOT}" \
        --exp "${EXP_NAME}" \
        --labelnum "${LABELNUM}" \
        --stage_name self_train \
        --snapshot_path "${snapshot_dir}" \
        --checkpoint_path "${checkpoint}" \
        --auto_find_checkpoint False \
        --gpu 0 \
        --save_result "${SAVE_RESULT}" \
        --nms "${NMS}"
}

run_stage match_pre "${MATCH_PRE_DIR}"
run_stage match_self "${MATCH_SELF_DIR}"
run_stage samatch_interactive "${INTERACTIVE_DIR}"

MATCH_PRE_PERFORMANCE="${MATCH_PRE_DIR}/performance.txt" \
MATCH_SELF_PERFORMANCE="${MATCH_SELF_DIR}/performance.txt" \
INTERACTIVE_PERFORMANCE="${INTERACTIVE_DIR}/performance.txt" \
OUTPUT_DIR="${EXPERIMENT_DIR}" \
"${PYTHON_LAUNCHER[@]}" - <<'PY'
import csv
import os
import re
from pathlib import Path

sources = (
    ("match_pre", Path(os.environ["MATCH_PRE_PERFORMANCE"])),
    ("match_self", Path(os.environ["MATCH_SELF_PERFORMANCE"])),
    ("samatch_interactive", Path(os.environ["INTERACTIVE_PERFORMANCE"])),
)
output_dir = Path(os.environ["OUTPUT_DIR"])
case_pattern = re.compile(
    r"^(\S+) -> Dice: ([\d.eE+-]+), Jaccard: ([\d.eE+-]+), "
    r"HD95: ([\d.eE+-]+), ASD: ([\d.eE+-]+)$"
)
metric_pattern = re.compile(
    r"^(Dice|Jaccard|HD95|ASD):\s+([\d.eE+-]+)$")

case_rows = []
summary_rows = []
for stage, path in sources:
    averages = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        case_match = case_pattern.match(line.strip())
        if case_match:
            case_rows.append({
                "stage": stage,
                "case": case_match.group(1),
                "dice": float(case_match.group(2)),
                "jaccard": float(case_match.group(3)),
                "hd95": float(case_match.group(4)),
                "asd": float(case_match.group(5)),
            })
            continue
        metric_match = metric_pattern.match(line.strip())
        if metric_match:
            averages[metric_match.group(1).lower()] = float(
                metric_match.group(2))
    required = {"dice", "jaccard", "hd95", "asd"}
    missing = required.difference(averages)
    if missing:
        raise SystemExit(
            f"Could not parse {sorted(missing)} from {path}")
    summary_rows.append({"stage": stage, **averages})

with (output_dir / "test_case_metrics.csv").open(
        "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=("stage", "case", "dice", "jaccard", "hd95", "asd"))
    writer.writeheader()
    writer.writerows(case_rows)

with (output_dir / "metric_table.csv").open(
        "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=("stage", "dice", "jaccard", "hd95", "asd"))
    writer.writeheader()
    writer.writerows(summary_rows)

lines = [
    "| Stage | Dice | Jaccard | HD95 | ASD |",
    "|---|---:|---:|---:|---:|",
]
for row in summary_rows:
    lines.append(
        "| {stage} | {dice:.6f} | {jaccard:.6f} | "
        "{hd95:.6f} | {asd:.6f} |".format(**row))
markdown = "\n".join(lines) + "\n"
(output_dir / "metric_table.md").write_text(markdown, encoding="utf-8")
print("\n================ Full SAMatch Metric Table ================\n")
print(markdown, end="")
print(f"\nAggregate CSV: {output_dir / 'metric_table.csv'}")
print(f"Case-level CSV: {output_dir / 'test_case_metrics.csv'}")
PY
