#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate the current validation-best UtilityMatch online student. This can be
# used during training after the first best checkpoint has been written.

BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${BASELINE_ROOT}/code"
DATA_ROOT="${DATA_ROOT:-${BASELINE_ROOT}/data/PROMISE12_h5_training_source}"
MODEL_ROOT="${BASELINE_ROOT}/model"
SERVER_LOG_DIR="${BASELINE_ROOT}/server_logs"
LAST_RUN_FILE="${SERVER_LOG_DIR}/last_utilitymatch_run.txt"
GPU="${GPU:-0}"
REQUIRE_5090="${REQUIRE_5090:-1}"
UTILITYMATCH_DIR="${UTILITYMATCH_DIR:-}"
SAVE_RESULT="${SAVE_RESULT:-False}"
NMS="${NMS:-0}"

if [[ -z "${UTILITYMATCH_DIR}" && -f "${LAST_RUN_FILE}" ]]; then
    IFS= read -r UTILITYMATCH_DIR <"${LAST_RUN_FILE}"
fi
if [[ -n "${UTILITYMATCH_DIR}" && ! -d "${UTILITYMATCH_DIR}" ]]; then
    UTILITYMATCH_DIR=""
fi
if [[ -z "${UTILITYMATCH_DIR}" ]]; then
    UTILITYMATCH_DIR="$(
        find "${MODEL_ROOT}" -maxdepth 1 -type d \
            -name 'UtilityMatch_*_7_labeled' -printf '%T@ %p\n' 2>/dev/null |
            sort -nr | head -n 1 | cut -d' ' -f2-
    )"
fi
if [[ -z "${UTILITYMATCH_DIR}" || ! -d "${UTILITYMATCH_DIR}" ]]; then
    echo "No UtilityMatch run found." >&2
    echo "Set UTILITYMATCH_DIR to model/UtilityMatch_<timestamp>_7_labeled." >&2
    exit 2
fi

SELF_DIR="${UTILITYMATCH_DIR}/self_train/unet"
CHECKPOINT="${CHECKPOINT:-${SELF_DIR}/unet_best_model.pth}"
EXPERIMENT_BASENAME="$(basename -- "${UTILITYMATCH_DIR}")"
EXP_NAME="${EXPERIMENT_BASENAME%_7_labeled}"

PYTHON_LAUNCHER=()
if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_LAUNCHER=("${PYTHON_BIN}")
elif [[ -n "${CONDA_ENV:-}" ]]; then
    command -v conda >/dev/null 2>&1 || {
        echo "CONDA_ENV is set but conda is unavailable." >&2
        exit 2
    }
    PYTHON_LAUNCHER=(conda run --no-capture-output -n "${CONDA_ENV}" python)
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_LAUNCHER=(python3)
elif command -v python >/dev/null 2>&1; then
    PYTHON_LAUNCHER=(python)
else
    echo "No Python found. Activate the original UniMatch environment." >&2
    exit 2
fi

[[ -f "${CHECKPOINT}" ]] || {
    echo "UtilityMatch best checkpoint not found yet: ${CHECKPOINT}" >&2
    echo "Wait for the first validation checkpoint, or set CHECKPOINT explicitly." >&2
    exit 2
}
[[ -f "${DATA_ROOT}/test.list" ]] || {
    echo "Missing fixed test list: ${DATA_ROOT}/test.list" >&2
    exit 2
}
[[ "$(grep -cve '^[[:space:]]*$' "${DATA_ROOT}/test.list")" -eq 10 ]] || {
    echo "The fixed test list must contain 10 cases." >&2
    exit 2
}

cd "${CODE_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" REQUIRE_5090="${REQUIRE_5090}" \
    "${PYTHON_LAUNCHER[@]}" - <<'PY'
import os
import torch
for name in (
    "h5py", "medpy", "numpy", "scipy", "SimpleITK", "skimage", "torch", "tqdm",
):
    __import__(name)
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
gpu_name = torch.cuda.get_device_name(0)
print(f"PyTorch={torch.__version__}, CUDA={torch.version.cuda}, GPU={gpu_name}")
if os.environ.get("REQUIRE_5090", "1") == "1" and "5090" not in gpu_name:
    raise SystemExit(f"Expected RTX 5090, got {gpu_name}")
PY

echo "Testing UtilityMatch checkpoint: ${CHECKPOINT}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_LAUNCHER[@]}" test_unimatch.py \
    --root_path "${DATA_ROOT}" \
    --exp "${EXP_NAME}" \
    --labelnum 7 \
    --stage_name self_train \
    --snapshot_path "${SELF_DIR}" \
    --checkpoint_path "${CHECKPOINT}" \
    --model_root "${MODEL_ROOT}" \
    --gpu 0 \
    --save_result "${SAVE_RESULT}" \
    --nms "${NMS}" \
    --auto_find_checkpoint False

PERFORMANCE="${SELF_DIR}/performance.txt" OUTPUT_DIR="${UTILITYMATCH_DIR}" \
    "${PYTHON_LAUNCHER[@]}" - <<'PY'
import csv
import os
import re
from pathlib import Path

performance = Path(os.environ["PERFORMANCE"])
output_dir = Path(os.environ["OUTPUT_DIR"])
case_pattern = re.compile(
    r"^(\S+) -> Dice: ([\d.eE+-]+), Jaccard: ([\d.eE+-]+), "
    r"HD95: ([\d.eE+-]+), ASD: ([\d.eE+-]+)$")
metric_pattern = re.compile(r"^(Dice|Jaccard|HD95|ASD):\s+([\d.eE+-]+)$")
if not performance.is_file():
    raise SystemExit(f"Missing performance file after testing: {performance}")

case_rows = []
averages = {}
for line in performance.read_text(encoding="utf-8").splitlines():
    match = case_pattern.match(line.strip())
    if match:
        case_rows.append({
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
if len(case_rows) != 10:
    raise SystemExit(f"Expected 10 case metrics, parsed {len(case_rows)}")
if {"dice", "jaccard", "hd95", "asd"}.difference(averages):
    raise SystemExit(f"Incomplete aggregate metrics: {averages}")

output_dir.mkdir(parents=True, exist_ok=True)
with (output_dir / "test_case_metrics.csv").open(
        "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle, fieldnames=("case", "dice", "jaccard", "hd95", "asd"))
    writer.writeheader()
    writer.writerows(case_rows)
with (output_dir / "metric_table.csv").open(
        "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle, fieldnames=("dice", "jaccard", "hd95", "asd"))
    writer.writeheader()
    writer.writerow(averages)
markdown = (
    "| Dice | Jaccard | HD95 | ASD |\n"
    "|---:|---:|---:|---:|\n"
    "| {dice:.6f} | {jaccard:.6f} | {hd95:.6f} | {asd:.6f} |\n"
).format(**averages)
(output_dir / "metric_table.md").write_text(markdown, encoding="utf-8")
print("\n================ UtilityMatch Metric Table ================\n")
print(markdown, end="")
print(f"Case metrics: {output_dir / 'test_case_metrics.csv'}")
PY

echo "UtilityMatch test completed"
echo "Raw performance: ${SELF_DIR}/performance.txt"
echo "Metric table:    ${UTILITYMATCH_DIR}/metric_table.md"
