#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${PROJECT_ROOT}/code"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/PROMISE12_h5_training_source}"
MODEL_ROOT="${PROJECT_ROOT}/model"
SERVER_LOG_DIR="${PROJECT_ROOT}/server_logs"
GPU="${GPU:-0}"
SEED="${SEED:-1337}"
REQUIRE_5090="${REQUIRE_5090:-1}"
DETACH="${DETACH:-0}"
TRAJECTORY_MODE="${TRAJECTORY_MODE:-full}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-Trajectory_${TRAJECTORY_MODE}_${RUN_TAG}}"
EXPERIMENT_DIR="${MODEL_ROOT}/${EXP_NAME}_7_labeled"
OUTPUT_DIR="${EXPERIMENT_DIR}/self_train/unet"
LAST_RUN_FILE="${SERVER_LOG_DIR}/last_trajectory_${TRAJECTORY_MODE}_run.txt"
ORIGINAL_NAME="UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled"
ORIGINAL_UNIMATCH_DIR="${ORIGINAL_UNIMATCH_DIR:-}"

case "${TRAJECTORY_MODE}" in
    baseline|weighting|adaptive|weighting_adaptive|full) ;;
    *) echo "Unknown TRAJECTORY_MODE=${TRAJECTORY_MODE}" >&2; exit 2 ;;
esac
[[ "${SEED}" == "1337" ]] || {
    echo "Locked protocol requires seed 1337." >&2
    exit 2
}

mkdir -p "${SERVER_LOG_DIR}"
if [[ -z "${ORIGINAL_UNIMATCH_DIR}" ]]; then
    for candidate in \
        "${MODEL_ROOT}/${ORIGINAL_NAME}" \
        "${PROJECT_ROOT}/../Updated_code/model/${ORIGINAL_NAME}" \
        "${PROJECT_ROOT}/../${ORIGINAL_NAME}"; do
        if [[ -f "${candidate}/pre_train/unet/unet_best_model.pth" &&
              -f "${candidate}/self_train/unet/unet_best_model.pth" ]]; then
            ORIGINAL_UNIMATCH_DIR="${candidate}"
            break
        fi
    done
fi
[[ -n "${ORIGINAL_UNIMATCH_DIR}" ]] || {
    echo "Set ORIGINAL_UNIMATCH_DIR to ${ORIGINAL_NAME}." >&2
    exit 2
}
[[ "$(basename -- "${ORIGINAL_UNIMATCH_DIR%/}")" == "${ORIGINAL_NAME}" ]] || {
    echo "Wrong original UniMatch folder: ${ORIGINAL_UNIMATCH_DIR}" >&2
    exit 2
}

PRETRAIN_CHECKPOINT="${ORIGINAL_UNIMATCH_DIR}/pre_train/unet/unet_best_model.pth"
ANCHOR_CHECKPOINT="${ORIGINAL_UNIMATCH_DIR}/self_train/unet/unet_best_model.pth"
[[ -f "${PRETRAIN_CHECKPOINT}" ]] || { echo "Missing ${PRETRAIN_CHECKPOINT}" >&2; exit 2; }
[[ -f "${ANCHOR_CHECKPOINT}" ]] || { echo "Missing ${ANCHOR_CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATA_ROOT}/train_slices.list" ]] || { echo "Missing data: ${DATA_ROOT}" >&2; exit 2; }
[[ ! -e "${EXPERIMENT_DIR}" ]] || {
    echo "Refusing to overwrite ${EXPERIMENT_DIR}" >&2
    exit 3
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

if [[ "${DETACH}" == "1" && "${_TRAJECTORY_DETACHED:-0}" != "1" ]]; then
    DETACHED_LOG="${SERVER_LOG_DIR}/trajectory_${TRAJECTORY_MODE}_${RUN_TAG}_nohup.log"
    nohup env _TRAJECTORY_DETACHED=1 DETACH=0 \
        TRAJECTORY_MODE="${TRAJECTORY_MODE}" RUN_TAG="${RUN_TAG}" \
        EXP_NAME="${EXP_NAME}" DATA_ROOT="${DATA_ROOT}" GPU="${GPU}" \
        SEED="${SEED}" REQUIRE_5090="${REQUIRE_5090}" \
        ORIGINAL_UNIMATCH_DIR="${ORIGINAL_UNIMATCH_DIR}" \
        bash "${BASH_SOURCE[0]}" "$@" >"${DETACHED_LOG}" 2>&1 </dev/null &
    echo "Started trajectory ${TRAJECTORY_MODE}: PID=$!"
    echo "Log: ${DETACHED_LOG}"
    exit 0
fi

if command -v flock >/dev/null 2>&1; then
    exec 9>"${SERVER_LOG_DIR}/.trajectory_training.lock"
    flock -n 9 || { echo "Another trajectory run is active." >&2; exit 3; }
fi

cd "${CODE_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" REQUIRE_5090="${REQUIRE_5090}" \
DATA_ROOT="${DATA_ROOT}" "${PYTHON_LAUNCHER[@]}" - <<'PY'
import os
from pathlib import Path
import numpy as np
import torch

root = Path(os.environ["DATA_ROOT"])
expected = {"train.list": 35, "val.list": 5, "test.list": 10,
            "train_slices.list": 940}
for name, count in expected.items():
    path = root / name
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if len(values) != count:
        raise SystemExit(f"{name}: expected {count}, got {len(values)}")
if int(np.__version__.split(".")[0]) >= 2 and torch.__version__.startswith("1."):
    raise SystemExit("PyTorch 1.x requires numpy<2 in this environment")
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
name = torch.cuda.get_device_name(0)
print(f"Python preflight: numpy={np.__version__} torch={torch.__version__} GPU={name}")
if os.environ.get("REQUIRE_5090", "1") == "1" and "5090" not in name:
    raise SystemExit(f"Expected RTX 5090, got {name}")
PY

"${PYTHON_LAUNCHER[@]}" test_trajectory_reliability_smoke.py
printf '%s\n' "${EXPERIMENT_DIR}" >"${LAST_RUN_FILE}"
RUN_LOG="${SERVER_LOG_DIR}/trajectory_${TRAJECTORY_MODE}_${RUN_TAG}.log"
echo "Trajectory mode: ${TRAJECTORY_MODE}"
echo "Data: ${DATA_ROOT}"
echo "Pretrain: ${PRETRAIN_CHECKPOINT}"
echo "Output: ${OUTPUT_DIR}"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PYTHON_LAUNCHER[@]}" train_trajectory_reliability.py \
        --root_path "${DATA_ROOT}" \
        --pretrained_model_path "${PRETRAIN_CHECKPOINT}" \
        --anchor_checkpoint "${ANCHOR_CHECKPOINT}" \
        --output_dir "${OUTPUT_DIR}" \
        --mode "${TRAJECTORY_MODE}" \
        "$@" 2>&1 | tee "${RUN_LOG}"

echo "Training completed: ${EXPERIMENT_DIR}"
echo "Test: TRAJECTORY_MODE=${TRAJECTORY_MODE} TRAJECTORY_DIR='${EXPERIMENT_DIR}' bash test_trajectory_reliability_5090.sh"

