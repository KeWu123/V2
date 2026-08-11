#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/PROMISE12_h5_training_source}"
ORIGINAL_NAME="UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled"
ORIGINAL_UNIMATCH_DIR="${ORIGINAL_UNIMATCH_DIR:-${ROOT}/model/${ORIGINAL_NAME}}"
HISTORY_DIR="${ORIGINAL_UNIMATCH_DIR}/self_train/unet"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/model/trajectory_reliability_diagnostic}"
GPU="${GPU:-0}"
cd "${ROOT}/code"
CUDA_VISIBLE_DEVICES="${GPU}" python diagnose_trajectory_reliability.py \
    --root_path "${DATA_ROOT}" \
    --history_dir "${HISTORY_DIR}" \
    --history_count "${HISTORY_COUNT:-4}" \
    --split "${SPLIT:-val}" \
    --output_dir "${OUTPUT_DIR}"
