#!/usr/bin/env bash
set -Eeuo pipefail

# PROMISE12 short-schedule UniMatch + non-destructive foreground rescue.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${ROOT}/code"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/PROMISE12_h5}"
RAW_ARCHIVE="${RAW_ARCHIVE:-${ROOT}/data/PROMISE12/raw/training_data.zip}"
RAW_ROOT="${RAW_ROOT:-${ROOT}/data/PROMISE12/extracted/training_data}"
CONVERTER="${ROOT}/tools/convert_promise12_to_h5.py"
H5_PREPARER="${ROOT}/tools/ensure_promise12_h5.py"
MODEL_ROOT="${ROOT}/model"
GPU="${GPU:-0}"
SEED="${SEED:-1337}"
EXP_NAME="${EXP_NAME:-MT_PROMISE12_UniMatch_FGRescue}"
EXPERIMENT_DIR="${MODEL_ROOT}/${EXP_NAME}_7_labeled"
LOG_DIR="${ROOT}/server_logs"
DETACH="${DETACH:-0}"
SCRIPT_PATH="${ROOT}/$(basename -- "${BASH_SOURCE[0]}")"
mkdir -p "${LOG_DIR}"

if [[ "${DETACH}" == "1" && "${_FG_RESCUE_DETACHED:-0}" != "1" ]]; then
    LOG_FILE="${LOG_DIR}/fg_rescue_$(date +%Y%m%d_%H%M%S).log"
    nohup env _FG_RESCUE_DETACHED=1 DETACH=0 \
        bash "${SCRIPT_PATH}" >"${LOG_FILE}" 2>&1 </dev/null &
    echo "Started in background: PID=$!"
    echo "Log: ${LOG_FILE}"
    exit 0
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON=("${PYTHON_BIN}")
elif [[ -n "${CONDA_ENV:-}" ]]; then
    PYTHON=(conda run --no-capture-output -n "${CONDA_ENV}" python)
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=(python3)
else
    PYTHON=(python)
fi

[[ -f "${CODE_DIR}/train_unimatch_fg_rescue.py" ]] || {
    echo "Missing ${CODE_DIR}/train_unimatch_fg_rescue.py" >&2
    exit 2
}
[[ -f "${H5_PREPARER}" && -f "${CONVERTER}" ]] || {
    echo "Missing PROMISE12 H5 preparation tools below ${ROOT}/tools" >&2
    exit 2
}

# Generated H5 files in the uploaded repository can be 131-byte Git-LFS
# pointers. This is a required preflight: validate every file used by the
# train/val/test lists and rebuild all H5 files from the real raw ZIP if needed.
if ! "${PYTHON[@]}" -c 'import h5py, numpy, SimpleITK' >/dev/null 2>&1; then
    echo "Installing PROMISE12 conversion dependencies..."
    "${PYTHON[@]}" -m pip install --disable-pip-version-check \
        h5py numpy SimpleITK
fi
"${PYTHON[@]}" "${H5_PREPARER}" \
    --data_root "${DATA_ROOT}" \
    --archive "${RAW_ARCHIVE}" \
    --raw_root "${RAW_ROOT}" \
    --converter "${CONVERTER}"

[[ -f "${DATA_ROOT}/train.list" && -f "${DATA_ROOT}/val.list" ]] || {
    echo "PROMISE12 H5 data is missing below ${DATA_ROOT}" >&2
    exit 2
}
if find "${EXPERIMENT_DIR}" -type f -name '*.pth' -print -quit \
    2>/dev/null | grep -q .; then
    if [[ "${ALLOW_EXISTING:-0}" != "1" ]]; then
        echo "Existing checkpoints found below ${EXPERIMENT_DIR}." >&2
        echo "Use a new EXP_NAME, or set ALLOW_EXISTING=1 explicitly." >&2
        exit 3
    fi
fi

echo "Data:   ${DATA_ROOT}"
echo "GPU:    ${GPU}"
echo "Seed:   ${SEED}"
echo "Output: ${EXPERIMENT_DIR}"
echo "Config: pre=1000 self=5000 warmup=1000 labelnum=7 batch=24/12"
echo "Change: original UniMatch retained + foreground-only TTA soft rescue"

cd "${CODE_DIR}"
RUN_LOG="${LOG_DIR}/fg_rescue_console_$(date +%Y%m%d_%H%M%S).log"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON[@]}" \
    train_unimatch_fg_rescue.py \
    --root_path "${DATA_ROOT}" \
    --exp "${EXP_NAME}" \
    --seed "${SEED}" \
    --pre_iterations 1000 \
    --max_iterations 5000 \
    --labelnum 7 \
    --batch_size 24 \
    --labeled_bs 12 2>&1 | tee "${RUN_LOG}"

echo "Training completed"
echo "Console log: ${RUN_LOG}"
echo "Evaluate: bash test_and_quantify_unimatch_fg_rescue_5090.sh"
