#!/usr/bin/env bash
set -Eeuo pipefail

# Temporal Bank v2 refinement with full POS/MEO gradient composition.
# Existing UniMatch training is skipped and its checkpoint is kept fixed as
# the initialization and temporal-bank source.

BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${BASELINE_ROOT}/code"
DATA_ROOT="${DATA_ROOT:-${BASELINE_ROOT}/data/PROMISE12_h5}"
SPLIT_CHECKER="${BASELINE_ROOT}/tools/check_promise12_split.py"
GPU="${GPU:-0}"
SEED="${SEED:-1337}"
LABELNUM="${LABELNUM:-7}"
REFINE_ITERATIONS="${REFINE_ITERATIONS:-5000}"
EXP_NAME="${EXP_NAME:-UniMatch_TemporalBank_POS_MEO_35_5_10_seed1337}"
MODEL_ROOT="${BASELINE_ROOT}/model"
OUTPUT_DIR="${MODEL_ROOT}/${EXP_NAME}_${LABELNUM}_labeled/temporal_bank_pos_meo/unet"
UNIMATCH_FOLDER_NAME="UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled"
UNIMATCH_DIR="${UNIMATCH_DIR:-}"
DETACH="${DETACH:-0}"
SERVER_LOG_DIR="${BASELINE_ROOT}/server_logs"
SCRIPT_PATH="${BASELINE_ROOT}/$(basename -- "${BASH_SOURCE[0]}")"

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
HISTORY_DIR="${HISTORY_DIR:-${UNIMATCH_DIR}/self_train/unet}"

[[ -f "${UNIMATCH_CHECKPOINT}" ]] || {
    echo "Missing UniMatch checkpoint: ${UNIMATCH_CHECKPOINT}" >&2
    exit 2
}
[[ -f "${DATA_ROOT}/train_slices.list" ]] || {
    echo "Missing PROMISE12 data: ${DATA_ROOT}" >&2
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

mkdir -p "${SERVER_LOG_DIR}"
if [[ "${DETACH}" == "1" && "${_POS_MEO_DETACHED:-0}" != "1" ]]; then
    SERVER_LOG="${SERVER_LOG_DIR}/temporal_bank_pos_meo_$(date +%Y%m%d_%H%M%S).log"
    nohup env _POS_MEO_DETACHED=1 DETACH=0 \
        bash "${SCRIPT_PATH}" "$@" >"${SERVER_LOG}" 2>&1 </dev/null &
    printf 'Started in background: PID=%s\nLog: %s\n' "$!" "${SERVER_LOG}"
    exit 0
fi

if [[ -f "${OUTPUT_DIR}/last_checkpoint.pth" && "${ALLOW_EXISTING:-0}" != "1" ]]; then
    echo "Existing POS/MEO refinement found: ${OUTPUT_DIR}" >&2
    echo "Use a new EXP_NAME, or set ALLOW_EXISTING=1 to reuse that directory." >&2
    exit 3
fi

echo "Data:              ${DATA_ROOT}"
echo "Fixed UniMatch:    ${UNIMATCH_CHECKPOINT}"
echo "History evidence: ${HISTORY_DIR}"
echo "Output:            ${OUTPUT_DIR}"
echo "GPU / seed:        ${GPU} / ${SEED}"
echo "Protocol:          UniMatch training skipped; refinement=${REFINE_ITERATIONS}"
echo "Temporal Bank:     unchanged v2 (MC8, routing, curriculum, refresh)"
echo "Optimization:      full POS + MEO on supervised and weighted bank gradients"

cd "${CODE_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_LAUNCHER[@]}" \
    train_unimatch_temporal_bank_pos_meo.py \
        --root_path "${DATA_ROOT}" \
        --unimatch_checkpoint "${UNIMATCH_CHECKPOINT}" \
        --history_dir "${HISTORY_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --refine_iterations "${REFINE_ITERATIONS}" \
        --labelnum "${LABELNUM}" \
        --seed "${SEED}" \
        "$@"

echo "Training completed"
echo "Best checkpoint: ${OUTPUT_DIR}/unet_best_model.pth"
echo "Test: EXP_NAME=${EXP_NAME} bash test_and_quantify_unimatch_temporal_bank_pos_meo_5090.sh"
