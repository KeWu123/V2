#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAJECTORY_MODE="${TRAJECTORY_MODE:-full}"
LAST_RUN_FILE="${ROOT}/server_logs/last_trajectory_${TRAJECTORY_MODE}_run.txt"
TRAJECTORY_DIR="${TRAJECTORY_DIR:-}"
if [[ -z "${TRAJECTORY_DIR}" && -f "${LAST_RUN_FILE}" ]]; then
    IFS= read -r TRAJECTORY_DIR <"${LAST_RUN_FILE}"
fi
[[ -n "${TRAJECTORY_DIR}" && -d "${TRAJECTORY_DIR}" ]] || {
    echo "Set TRAJECTORY_DIR or run trajectory mode ${TRAJECTORY_MODE} first." >&2
    exit 2
}
UTILITYMATCH_DIR="${TRAJECTORY_DIR}" \
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/PROMISE12_h5_training_source}" \
GPU="${GPU:-0}" REQUIRE_5090="${REQUIRE_5090:-1}" \
SAVE_RESULT="${SAVE_RESULT:-False}" NMS="${NMS:-0}" \
    exec bash "${ROOT}/test_utilitymatch_5090.sh"
