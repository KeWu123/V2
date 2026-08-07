#!/usr/bin/env bash
set -Eeuo pipefail

# Test Baseline + UniMatch + Embedding Matching checkpoints, then create
# case-level and aggregate CSV/Markdown metric tables.

BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${BASELINE_ROOT}/code"
DATA_ROOT="${DATA_ROOT:-${BASELINE_ROOT}/data/PROMISE12_h5}"
GPU="${GPU:-0}"
REQUIRE_5090="${REQUIRE_5090:-1}"
MODEL_ROOT="${BASELINE_ROOT}/model"
EXP_NAME="${EXP_NAME:-MT_PROMISE12_UniMatch_EmbeddingMatching_v2}"
LABELNUM="${LABELNUM:-7}"
EXPERIMENT_DIR="${MODEL_ROOT}/${EXP_NAME}_${LABELNUM}_labeled"
PRETRAIN_DIR="${EXPERIMENT_DIR}/pre_train/unet"
SELF_DIR="${EXPERIMENT_DIR}/self_train/unet"
SAVE_RESULT="${SAVE_RESULT:-False}"
NMS="${NMS:-0}"

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
    echo "No Python found. Activate the environment or set CONDA_ENV/PYTHON_BIN." >&2
    exit 2
fi

[[ -f "${CODE_DIR}/test_embedding_matching.py" ]] || {
    echo "Missing test entry: ${CODE_DIR}/test_embedding_matching.py" >&2
    exit 2
}
[[ -f "${PRETRAIN_DIR}/unet_best_model.pth" ]] || {
    echo "Missing pretrain checkpoint: ${PRETRAIN_DIR}/unet_best_model.pth" >&2
    exit 2
}
[[ -f "${SELF_DIR}/unet_best_model.pth" ]] || {
    echo "Missing self-train checkpoint: ${SELF_DIR}/unet_best_model.pth" >&2
    exit 2
}
[[ -f "${DATA_ROOT}/test.list" ]] || {
    echo "Missing test list: ${DATA_ROOT}/test.list" >&2
    exit 2
}

cd "${CODE_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" REQUIRE_5090="${REQUIRE_5090}" \
    "${PYTHON_LAUNCHER[@]}" - <<'PY'
import os

import torch

for name in ("h5py", "medpy", "numpy", "scipy", "SimpleITK", "skimage", "torch", "tqdm"):
    __import__(name)
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
gpu_name = torch.cuda.get_device_name(0)
print(f"PyTorch={torch.__version__}, CUDA={torch.version.cuda}")
print(f"Visible GPU={gpu_name}, compute_capability={torch.cuda.get_device_capability(0)}")
if os.environ.get("REQUIRE_5090", "1") == "1" and "5090" not in gpu_name:
    raise SystemExit(f"Expected RTX 5090, got {gpu_name}")
PY

run_stage() {
    local stage="$1"
    local -a test_args=(
        --root_path "${DATA_ROOT}"
        --exp "${EXP_NAME}"
        --labelnum "${LABELNUM}"
        --stage_name "${stage}"
        --model_root "${MODEL_ROOT}"
        --gpu 0
        --save_result "${SAVE_RESULT}"
        --nms "${NMS}"
        --auto_find_checkpoint False
    )

    echo "Testing ${stage} checkpoint..."
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_LAUNCHER[@]}" \
        test_embedding_matching.py "${test_args[@]}"
}

run_stage pre_train
run_stage self_train

PRETRAIN_PERFORMANCE="${PRETRAIN_DIR}/performance.txt" \
SELF_PERFORMANCE="${SELF_DIR}/performance.txt" \
OUTPUT_DIR="${EXPERIMENT_DIR}" \
"${PYTHON_LAUNCHER[@]}" - <<'PY'
import csv
import os
import re
from pathlib import Path

sources = (
    ("pre_train", Path(os.environ["PRETRAIN_PERFORMANCE"])),
    ("self_train", Path(os.environ["SELF_PERFORMANCE"])),
)
output_dir = Path(os.environ["OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)
case_pattern = re.compile(
    r"^(\S+) -> Dice: ([\d.eE+-]+), Jaccard: ([\d.eE+-]+), "
    r"HD95: ([\d.eE+-]+), ASD: ([\d.eE+-]+)$"
)
metric_pattern = re.compile(r"^(Dice|Jaccard|HD95|ASD):\s+([\d.eE+-]+)$")

case_rows = []
summary_rows = []
for stage, path in sources:
    if not path.is_file():
        raise SystemExit(f"Missing performance file after testing: {path}")
    averages = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        case_match = case_pattern.match(line.strip())
        if case_match:
            case_rows.append(
                {
                    "stage": stage,
                    "case": case_match.group(1),
                    "dice": float(case_match.group(2)),
                    "jaccard": float(case_match.group(3)),
                    "hd95": float(case_match.group(4)),
                    "asd": float(case_match.group(5)),
                }
            )
            continue
        metric_match = metric_pattern.match(line.strip())
        if metric_match:
            averages[metric_match.group(1).lower()] = float(metric_match.group(2))
    required = {"dice", "jaccard", "hd95", "asd"}
    if required.difference(averages):
        raise SystemExit(f"Could not parse average metrics from {path}")
    summary_rows.append({"stage": stage, **averages})

case_csv = output_dir / "test_case_metrics.csv"
with case_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=("stage", "case", "dice", "jaccard", "hd95", "asd"))
    writer.writeheader()
    writer.writerows(case_rows)

summary_csv = output_dir / "metric_table.csv"
with summary_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=("stage", "dice", "jaccard", "hd95", "asd"))
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
markdown = "\n".join(lines) + "\n"
(output_dir / "metric_table.md").write_text(markdown, encoding="utf-8")
print("\n================ Metric Table ================\n")
print(markdown, end="")
print(f"\nAggregate CSV: {summary_csv}")
print(f"Case-level CSV: {case_csv}")
PY
