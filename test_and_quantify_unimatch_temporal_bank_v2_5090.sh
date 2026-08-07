#!/usr/bin/env bash
set -Eeuo pipefail

BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${BASELINE_ROOT}/code"
DATA_ROOT="${DATA_ROOT:-${BASELINE_ROOT}/data/PROMISE12_h5}"
SPLIT_CHECKER="${BASELINE_ROOT}/tools/check_promise12_split.py"
GPU="${GPU:-0}"
LABELNUM="${LABELNUM:-7}"
EXP_NAME="${EXP_NAME:-UniMatch_TemporalVolumeBankV2_35_5_10_seed1337}"
MODEL_ROOT="${BASELINE_ROOT}/model"
OUTPUT_ROOT="${MODEL_ROOT}/${EXP_NAME}_${LABELNUM}_labeled"
REFINE_DIR="${REFINE_DIR:-${OUTPUT_ROOT}/temporal_bank/unet}"
UNIMATCH_FOLDER_NAME="UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled"
UNIMATCH_DIR="${UNIMATCH_DIR:-}"
SAVE_RESULT="${SAVE_RESULT:-True}"
NMS="${NMS:-0}"
REFINE_STAGE="${REFINE_STAGE:-temporal_bank_v2}"
REFINE_RESULT_SUBDIR="${REFINE_RESULT_SUBDIR:-temporal_bank}"

if [[ -z "${UNIMATCH_DIR}" ]]; then
    for candidate in \
        "${BASELINE_ROOT}/../${UNIMATCH_FOLDER_NAME}" \
        "${BASELINE_ROOT}/${UNIMATCH_FOLDER_NAME}" \
        "${MODEL_ROOT}/${UNIMATCH_FOLDER_NAME}"; do
        if [[ -f "${candidate}/self_train/unet/unet_best_model.pth" ]]; then
            UNIMATCH_DIR="${candidate}"
            break
        fi
    done
fi
[[ -n "${UNIMATCH_DIR}" ]] || {
    echo "Existing UniMatch folder not found. Set UNIMATCH_DIR." >&2
    exit 2
}

UNIMATCH_CHECKPOINT="${UNIMATCH_CHECKPOINT:-${UNIMATCH_DIR}/self_train/unet/unet_best_model.pth}"
REFINE_CHECKPOINT="${REFINE_CHECKPOINT:-${REFINE_DIR}/unet_best_model.pth}"
[[ -f "${UNIMATCH_CHECKPOINT}" ]] || {
    echo "Missing UniMatch checkpoint: ${UNIMATCH_CHECKPOINT}" >&2
    exit 2
}
[[ -f "${REFINE_CHECKPOINT}" ]] || {
    echo "Missing refinement checkpoint: ${REFINE_CHECKPOINT}" >&2
    exit 2
}

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

"${PYTHON_LAUNCHER[@]}" "${SPLIT_CHECKER}" \
    --data_root "${DATA_ROOT}" --require_h5

mkdir -p \
    "${OUTPUT_ROOT}/comparison/unimatch" \
    "${OUTPUT_ROOT}/comparison/${REFINE_RESULT_SUBDIR}"
cd "${CODE_DIR}"

run_test() {
    local name="$1"
    local checkpoint="$2"
    local result_dir="$3"
    echo "Testing ${name}: ${checkpoint}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_LAUNCHER[@]}" test_unimatch.py \
        --root_path "${DATA_ROOT}" \
        --exp "${EXP_NAME}" \
        --labelnum "${LABELNUM}" \
        --stage_name self_train \
        --snapshot_path "${result_dir}" \
        --checkpoint_path "${checkpoint}" \
        --model_root "${MODEL_ROOT}" \
        --gpu 0 \
        --save_result "${SAVE_RESULT}" \
        --test_save_path "${result_dir}/test_predictions" \
        --nms "${NMS}" \
        --auto_find_checkpoint False
}

run_test "fixed_unimatch" "${UNIMATCH_CHECKPOINT}" \
    "${OUTPUT_ROOT}/comparison/unimatch"
run_test "${REFINE_STAGE}" "${REFINE_CHECKPOINT}" \
    "${OUTPUT_ROOT}/comparison/${REFINE_RESULT_SUBDIR}"

UNIMATCH_PERFORMANCE="${OUTPUT_ROOT}/comparison/unimatch/performance.txt" \
REFINE_PERFORMANCE="${OUTPUT_ROOT}/comparison/${REFINE_RESULT_SUBDIR}/performance.txt" \
REFINE_STAGE="${REFINE_STAGE}" \
REFINE_RESULT_SUBDIR="${REFINE_RESULT_SUBDIR}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
"${PYTHON_LAUNCHER[@]}" - <<'PY'
import csv
import os
import re
from pathlib import Path

sources = (
    ("fixed_unimatch", Path(os.environ["UNIMATCH_PERFORMANCE"])),
    (os.environ["REFINE_STAGE"], Path(os.environ["REFINE_PERFORMANCE"])),
)
case_pattern = re.compile(
    r"^(\S+) -> Dice: ([\d.eE+-]+), Jaccard: ([\d.eE+-]+), "
    r"HD95: ([\d.eE+-]+), ASD: ([\d.eE+-]+)$"
)
metric_pattern = re.compile(r"^(Dice|Jaccard|HD95|ASD):\s+([\d.eE+-]+)$")
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
    missing = {"dice", "jaccard", "hd95", "asd"}.difference(averages)
    if missing:
        raise SystemExit(f"Could not parse {path}: missing {sorted(missing)}")
    summary_rows.append({"stage": stage, **averages})

output = Path(os.environ["OUTPUT_ROOT"])
with (output / "test_case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle, fieldnames=("stage", "case", "dice", "jaccard", "hd95", "asd")
    )
    writer.writeheader()
    writer.writerows(case_rows)
with (output / "metric_table.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle, fieldnames=("stage", "dice", "jaccard", "hd95", "asd")
    )
    writer.writeheader()
    writer.writerows(summary_rows)

lines = [
    "| Stage | Dice | Jaccard | HD95 | ASD |",
    "|---|---:|---:|---:|---:|",
]
for row in summary_rows:
    lines.append(
        "| {stage} | {dice:.6f} | {jaccard:.6f} | {hd95:.6f} | {asd:.6f} |".format(**row)
    )
table = "\n".join(lines) + "\n"
(output / "metric_table.md").write_text(table, encoding="utf-8")
print("\n================ Metric Table ================\n")
print(table)
print("Aggregate CSV:", output / "metric_table.csv")
print("Case-level CSV:", output / "test_case_metrics.csv")
print(
    "Predictions:",
    output
    / "comparison"
    / os.environ.get("REFINE_RESULT_SUBDIR", "temporal_bank")
    / "test_predictions",
)
PY
