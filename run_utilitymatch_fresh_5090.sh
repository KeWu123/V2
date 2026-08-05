#!/usr/bin/env bash
set -Eeuo pipefail

# Random initialization -> locked supervised Pre10000 -> UtilityMatch Self30000.
# Existing scripts, checkpoints, and experiment directories are never overwritten.

BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_BASELINE_ROOT="/home/aiteam/zhengtaoma/Baseline"
DATA_ROOT="/home/aiteam/zhengtaoma/Baseline/data/PROMISE12_h5_training_source"
CODE_DIR="${BASELINE_ROOT}/code"
MODEL_ROOT="${BASELINE_ROOT}/model"
SERVER_LOG_DIR="${BASELINE_ROOT}/server_logs"
SCRIPT_PATH="${BASELINE_ROOT}/$(basename -- "${BASH_SOURCE[0]}")"
GPU="${GPU:-0}"
DETACH="${DETACH:-0}"
REQUIRE_5090="${REQUIRE_5090:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-UtilityFresh_${RUN_TAG}}"
EXPERIMENT_DIR="${MODEL_ROOT}/${EXP_NAME}_7_labeled"
PRETRAIN_DIR="${EXPERIMENT_DIR}/pre_train/unet"
SELF_DIR="${EXPERIMENT_DIR}/self_train/unet"
PRETRAIN_CHECKPOINT="${PRETRAIN_DIR}/unet_best_model.pth"
DATASET_MANIFEST="${EXPERIMENT_DIR}/dataset_lists.sha256"
CODE_MANIFEST="${EXPERIMENT_DIR}/training_code.sha256"
LAST_RUN_FILE="${SERVER_LOG_DIR}/last_utilitymatch_fresh_run.txt"
LAST_PRETRAIN_LOG_FILE="${SERVER_LOG_DIR}/last_utilitymatch_fresh_pretrain_log.txt"

mkdir -p "${SERVER_LOG_DIR}"

if (( $# != 0 )); then
    echo "This locked launcher accepts no command-line overrides." >&2
    echo "Use only: bash ${SCRIPT_PATH}" >&2
    exit 2
fi

if [[ "$(realpath -m -- "${BASELINE_ROOT}")" != "${EXPECTED_BASELINE_ROOT}" ]]; then
    echo "This experiment is locked to ${EXPECTED_BASELINE_ROOT}" >&2
    echo "Current script root: $(realpath -m -- "${BASELINE_ROOT}")" >&2
    exit 2
fi
if [[ "$(realpath -e -- "${DATA_ROOT}")" != "${DATA_ROOT}" ]]; then
    echo "Exact SAMatch PROMISE12 root is missing or redirected: ${DATA_ROOT}" >&2
    exit 2
fi
if [[ -e "${EXPERIMENT_DIR}" ]]; then
    echo "Refusing to overwrite existing output: ${EXPERIMENT_DIR}" >&2
    exit 3
fi

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

if [[ "${DETACH}" == "1" && "${_UTILITY_FRESH_DETACHED:-0}" != "1" ]]; then
    DETACHED_LOG="${SERVER_LOG_DIR}/utility_fresh_${RUN_TAG}_nohup.log"
    nohup env _UTILITY_FRESH_DETACHED=1 DETACH=0 RUN_TAG="${RUN_TAG}" \
        EXP_NAME="${EXP_NAME}" GPU="${GPU}" REQUIRE_5090="${REQUIRE_5090}" \
        bash "${SCRIPT_PATH}" >"${DETACHED_LOG}" 2>&1 </dev/null &
    printf 'Started fresh UtilityMatch in background: PID=%s\nLog: %s\n' \
        "$!" "${DETACHED_LOG}"
    exit 0
fi

if command -v flock >/dev/null 2>&1; then
    exec 9>"${SERVER_LOG_DIR}/.unimatch_training.lock"
    flock -n 9 || {
        echo "Another Baseline/UniMatch training process is already running." >&2
        exit 3
    }
fi

for path in \
    "${CODE_DIR}/train_fresh_pretrain.py" \
    "${CODE_DIR}/train_utilitymatch.py" \
    "${CODE_DIR}/utilitymatch.py" \
    "${CODE_DIR}/test_utilitymatch_smoke.py" \
    "${CODE_DIR}/train_unimatch.py" \
    "${BASELINE_ROOT}/test_utilitymatch_fresh_5090.sh" \
    "${BASELINE_ROOT}/test_utilitymatch_5090.sh"; do
    [[ -f "${path}" ]] || {
        echo "Missing required code: ${path}" >&2
        exit 2
    }
done

cd "${CODE_DIR}"
echo "Checking code, exact SAMatch PROMISE12 path, and RTX 5090..."
CUDA_VISIBLE_DEVICES="${GPU}" REQUIRE_5090="${REQUIRE_5090}" \
    "${PYTHON_LAUNCHER[@]}" - <<'PY'
import importlib
import os
from pathlib import Path

for name in (
    "h5py", "medpy", "numpy", "scipy", "skimage", "tensorboardX",
    "torch", "torchvision", "tqdm",
):
    importlib.import_module(name)

import torch

for name in (
    "train_fresh_pretrain.py", "train_utilitymatch.py", "utilitymatch.py",
    "test_utilitymatch_smoke.py", "train_unimatch.py",
):
    path = Path(name)
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
print("Python syntax/dependency check passed")

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
gpu_name = torch.cuda.get_device_name(0)
print(f"PyTorch={torch.__version__}, CUDA={torch.version.cuda}, GPU={gpu_name}")
if os.environ.get("REQUIRE_5090", "1") == "1" and "5090" not in gpu_name:
    raise SystemExit(f"Expected RTX 5090, got {gpu_name}")
PY

echo "Running deterministic UtilityMatch smoke test..."
"${PYTHON_LAUNCHER[@]}" test_utilitymatch_smoke.py

mkdir -p "${EXPERIMENT_DIR}"
(
    cd "${DATA_ROOT}"
    sha256sum train.list train_slices.list val.list test.list
) >"${DATASET_MANIFEST}"
(
    cd "${CODE_DIR}"
    sha256sum \
        train_fresh_pretrain.py \
        train_unimatch.py \
        train_utilitymatch.py \
        utilitymatch.py \
        test_utilitymatch_smoke.py \
        dataloaders/dataset.py \
        networks/unet.py \
        utils/losses.py \
        utils/val_2d.py
) >"${CODE_MANIFEST}"
printf '%s\n' "${EXPERIMENT_DIR}" >"${LAST_RUN_FILE}"

echo "======================================================================"
echo "Fresh UtilityMatch: complete two-stage training"
echo "Data:           ${DATA_ROOT}"
echo "Initialization: RANDOM seed 1337; no old checkpoint"
echo "Stage 1:        supervised Pre10000, first7=191"
echo "Stage 2:        UtilityMatch Self30000 from the new Pre10000 net+opt"
echo "Output:         ${EXPERIMENT_DIR}"
echo "Terminal:       tqdm and validation remain visible"
echo "======================================================================"

PRETRAIN_LOG="${SERVER_LOG_DIR}/utility_fresh_pre_${RUN_TAG}.log"
printf '%s\n' "${PRETRAIN_LOG}" >"${LAST_PRETRAIN_LOG_FILE}"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PYTHON_LAUNCHER[@]}" train_fresh_pretrain.py \
        --root_path "${DATA_ROOT}" \
        --output_dir "${PRETRAIN_DIR}" \
        --pre_iterations 10000 \
        --seed 1337 2>&1 | tee "${PRETRAIN_LOG}"

[[ -f "${PRETRAIN_CHECKPOINT}" ]] || {
    echo "Fresh Pre10000 checkpoint not found: ${PRETRAIN_CHECKPOINT}" >&2
    exit 4
}
(
    cd "${DATA_ROOT}"
    sha256sum -c "${DATASET_MANIFEST}"
)

PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT}" \
    "${PYTHON_LAUNCHER[@]}" - <<'PY'
import os
import torch

path = os.environ["PRETRAIN_CHECKPOINT"]
try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location="cpu")
if not isinstance(checkpoint, dict) or not {"net", "opt"}.issubset(checkpoint):
    raise SystemExit("Fresh Pre10000 must contain both net and opt states")
print(f"Fresh Pre10000 verified: {path}")
PY

PRETRAIN_SUMMARY="${PRETRAIN_DIR}/training_summary.json"
[[ -f "${PRETRAIN_SUMMARY}" ]] || {
    echo "Fresh Pre10000 summary not found: ${PRETRAIN_SUMMARY}" >&2
    exit 4
}
PRETRAIN_SUMMARY="${PRETRAIN_SUMMARY}" "${PYTHON_LAUNCHER[@]}" - <<'PY'
import json
import os

path = os.environ["PRETRAIN_SUMMARY"]
with open(path, "r", encoding="utf-8") as handle:
    summary = json.load(handle)
if summary.get("runtime_labeled_slices") != 191:
    raise SystemExit(
        "Fresh PreTrain runtime sampler was not 191: {}".format(
            summary.get("runtime_labeled_slices")))
print(
    "Fresh Pre10000 validation best: dice={:.4f}, iteration={}".format(
        float(summary["best_validation_dice_rounded"]),
        int(summary["best_validation_iteration"])))
print("Fresh runtime code hashes:")
for name, digest in summary["runtime_code_sha256"].items():
    print(f"  {digest}  {name}")
PY

echo "Starting UtilityMatch only after fresh Pre10000 verification..."
echo "Reference-anchor metadata uses the same fresh file; no old self-training weight is loaded."
SELF_LOG="${SERVER_LOG_DIR}/utility_fresh_self_${RUN_TAG}.log"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PYTHON_LAUNCHER[@]}" train_utilitymatch.py \
        --root_path "${DATA_ROOT}" \
        --pretrained_model_path "${PRETRAIN_CHECKPOINT}" \
        --anchor_checkpoint "${PRETRAIN_CHECKPOINT}" \
        --output_dir "${SELF_DIR}" \
        --max_iterations 30000 \
        --seed 1337 2>&1 | tee "${SELF_LOG}"

(
    cd "${DATA_ROOT}"
    sha256sum -c "${DATASET_MANIFEST}"
)

echo "Fresh UtilityMatch training completed"
echo "Pretrain log: ${PRETRAIN_LOG}"
echo "Self log:     ${SELF_LOG}"
echo "Fresh weight: ${PRETRAIN_CHECKPOINT}"
echo "Best model:   ${SELF_DIR}/unet_best_model.pth"
echo "Test command: bash '${BASELINE_ROOT}/test_utilitymatch_fresh_5090.sh'"
