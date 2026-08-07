#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/PROMISE12_h5}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXP_NAME="${EXP_NAME:-MT_PROMISE12_UniMatch_ProtoCal_191slices}"
PRE_ITERATIONS="${PRE_ITERATIONS:-1000}"
SELF_ITERATIONS="${SELF_ITERATIONS:-5000}"
LABELNUM="${LABELNUM:-7}"
BATCH_SIZE="${BATCH_SIZE:-24}"
LABELED_BS="${LABELED_BS:-12}"
SEED="${SEED:-1337}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-${PROJECT_ROOT}/server_logs}"

# Module-only settings.  The original UniMatch arguments remain unchanged.
export PROTOTYPE_MOMENTUM="${PROTOTYPE_MOMENTUM:-0.9}"
export PROTOTYPE_TEMPERATURE="${PROTOTYPE_TEMPERATURE:-0.2}"
export PROTOTYPE_MIN_WEIGHT="${PROTOTYPE_MIN_WEIGHT:-0.5}"

mkdir -p "${SERVER_LOG_DIR}"

echo "Data:   ${DATA_ROOT}"
echo "GPU:    ${GPU}"
echo "Seed:   ${SEED}"
echo "Output: ${PROJECT_ROOT}/model/${EXP_NAME}_${LABELNUM}_labeled"
echo "Config: pre=${PRE_ITERATIONS} self=${SELF_ITERATIONS} labelnum=${LABELNUM} batch=${BATCH_SIZE}/${LABELED_BS}"
echo "Change: GT-only class prototypes softly weight accepted UniMatch pseudo labels"
echo "Proto:  momentum=${PROTOTYPE_MOMENTUM} temperature=${PROTOTYPE_TEMPERATURE} min_weight=${PROTOTYPE_MIN_WEIGHT}"

RUN_LOG="${SERVER_LOG_DIR}/proto_calibrated_$(date +%Y%m%d_%H%M%S).log"
cd "${PROJECT_ROOT}/code"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u \
    train_unimatch_proto_calibrated.py \
    --root_path "${DATA_ROOT}" \
    --exp "${EXP_NAME}" \
    --pre_iterations "${PRE_ITERATIONS}" \
    --max_iterations "${SELF_ITERATIONS}" \
    --labelnum "${LABELNUM}" \
    --batch_size "${BATCH_SIZE}" \
    --labeled_bs "${LABELED_BS}" \
    --seed "${SEED}" \
    "$@" 2>&1 | tee "${RUN_LOG}"

echo "Training completed"
echo "Console log: ${RUN_LOG}"
echo "Run: EXP_NAME=${EXP_NAME} bash test_and_quantify_unimatch_proto_calibrated_5090.sh"
