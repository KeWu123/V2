#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${ROOT}/code"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/PROMISE12_h5}"
GPU="${GPU:-0}"
EXP_NAME="${EXP_NAME:-MT_PROMISE12_UniMatch_MRIInterp}"
LABELNUM="${LABELNUM:-7}"
EXPERIMENT_DIR="${ROOT}/model/${EXP_NAME}_${LABELNUM}_labeled"
PRETRAIN_DIR="${EXPERIMENT_DIR}/pre_train/unet"
SELF_DIR="${EXPERIMENT_DIR}/self_train/unet"
SAVE_RESULT="${SAVE_RESULT:-False}"
NMS="${NMS:-0}"

PYTHON=()
if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON=("${PYTHON_BIN}")
elif [[ -n "${CONDA_ENV:-}" ]]; then
    PYTHON=(conda run --no-capture-output -n "${CONDA_ENV}" python)
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=(python3)
else
    PYTHON=(python)
fi

[[ -f "${PRETRAIN_DIR}/unet_best_model.pth" ]] || {
    echo "Missing pretrain checkpoint: ${PRETRAIN_DIR}/unet_best_model.pth" >&2
    exit 2
}
[[ -f "${SELF_DIR}/unet_best_model.pth" ]] || {
    echo "Missing self-train checkpoint: ${SELF_DIR}/unet_best_model.pth" >&2
    exit 2
}

cd "${CODE_DIR}"
for STAGE in pre_train self_train; do
    echo "Testing ${STAGE}..."
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON[@]}" \
        test_unimatch_mri_interp.py \
        --root_path "${DATA_ROOT}" \
        --exp "${EXP_NAME}" \
        --labelnum "${LABELNUM}" \
        --stage_name "${STAGE}" \
        --model_root "${ROOT}/model" \
        --gpu 0 \
        --save_result "${SAVE_RESULT}" \
        --nms "${NMS}" \
        --auto_find_checkpoint False
done

PRETRAIN_PERFORMANCE="${PRETRAIN_DIR}/performance.txt" \
SELF_PERFORMANCE="${SELF_DIR}/performance.txt" \
OUTPUT_DIR="${EXPERIMENT_DIR}" \
"${PYTHON[@]}" - <<'PY'
import csv
import os
import re
from pathlib import Path

sources = (
    ("pre_train", Path(os.environ["PRETRAIN_PERFORMANCE"])),
    ("self_train", Path(os.environ["SELF_PERFORMANCE"])),
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
        match = case_pattern.match(line.strip())
        if match:
            case_rows.append({
                "stage": stage,
                "case": match.group(1),
                "dice": float(match.group(2)),
                "jaccard": float(match.group(3)),
                "hd95": float(match.group(4)),
                "asd": float(match.group(5)),
            })
            continue
        match = metric_pattern.match(line.strip())
        if match:
            averages[match.group(1).lower()] = float(match.group(2))
    missing = {"dice", "jaccard", "hd95", "asd"} - averages.keys()
    if missing:
        raise SystemExit(f"Could not parse {sorted(missing)} from {path}")
    summary_rows.append({"stage": stage, **averages})

with (output_dir / "test_case_metrics.csv").open(
        "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=("stage", "case", "dice", "jaccard", "hd95", "asd"))
    writer.writeheader()
    writer.writerows(case_rows)

summary_csv = output_dir / "metric_table.csv"
with summary_csv.open("w", encoding="utf-8", newline="") as handle:
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
table = "\n".join(lines) + "\n"
(output_dir / "metric_table.md").write_text(table, encoding="utf-8")
print("\n================ Metric Table ================\n")
print(table, end="")
print(f"\nAggregate CSV: {summary_csv}")
print(f"Case-level CSV: {output_dir / 'test_case_metrics.csv'}")
PY
