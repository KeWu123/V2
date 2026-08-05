#!/usr/bin/env bash
set -Eeuo pipefail

# Test the validation-best online Student from the latest fresh
# random-init Pre10000 -> UtilityMatch Self30000 run.

BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_BASELINE_ROOT="/home/aiteam/zhengtaoma/Baseline"
DATA_ROOT="${EXPECTED_BASELINE_ROOT}/data/PROMISE12_h5_training_source"
MODEL_ROOT="${EXPECTED_BASELINE_ROOT}/model"
SERVER_LOG_DIR="${EXPECTED_BASELINE_ROOT}/server_logs"
LAST_RUN_FILE="${SERVER_LOG_DIR}/last_utilitymatch_fresh_run.txt"
BASE_TEST="${BASELINE_ROOT}/test_utilitymatch_5090.sh"
UTILITYMATCH_DIR="${UTILITYMATCH_DIR:-}"

if [[ "$(realpath -m -- "${BASELINE_ROOT}")" != "${EXPECTED_BASELINE_ROOT}" ]]; then
    echo "This fresh test is locked to ${EXPECTED_BASELINE_ROOT}" >&2
    echo "Current script root: $(realpath -m -- "${BASELINE_ROOT}")" >&2
    exit 2
fi
[[ -f "${BASE_TEST}" ]] || {
    echo "Missing shared UtilityMatch evaluator: ${BASE_TEST}" >&2
    exit 2
}
[[ "$(realpath -e -- "${DATA_ROOT}")" == "${DATA_ROOT}" ]] || {
    echo "Exact SAMatch PROMISE12 root is missing or redirected: ${DATA_ROOT}" >&2
    exit 2
}

if [[ -z "${UTILITYMATCH_DIR}" ]]; then
    [[ -f "${LAST_RUN_FILE}" ]] || {
        echo "Fresh-run marker not found: ${LAST_RUN_FILE}" >&2
        echo "Start the fresh run first, or set UTILITYMATCH_DIR explicitly." >&2
        exit 2
    }
    IFS= read -r UTILITYMATCH_DIR <"${LAST_RUN_FILE}"
fi

UTILITYMATCH_DIR="$(realpath -m -- "${UTILITYMATCH_DIR}")"
case "${UTILITYMATCH_DIR}/" in
    "${MODEL_ROOT}/"*) ;;
    *)
        echo "Fresh run must be below ${MODEL_ROOT}: ${UTILITYMATCH_DIR}" >&2
        exit 2
        ;;
esac
[[ -d "${UTILITYMATCH_DIR}" ]] || {
    echo "Fresh UtilityMatch directory not found: ${UTILITYMATCH_DIR}" >&2
    exit 2
}

PRETRAIN_CONFIG="${UTILITYMATCH_DIR}/pre_train/unet/config.json"
DATASET_MANIFEST="${UTILITYMATCH_DIR}/dataset_lists.sha256"
SELF_DIR="${UTILITYMATCH_DIR}/self_train/unet"
DEFAULT_CHECKPOINT="${SELF_DIR}/unet_best_model.pth"
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT}}"
CHECKPOINT="$(realpath -m -- "${CHECKPOINT}")"

[[ -f "${PRETRAIN_CONFIG}" ]] || {
    echo "This is not a recorded fresh run; missing ${PRETRAIN_CONFIG}" >&2
    exit 2
}
[[ -f "${DATASET_MANIFEST}" ]] || {
    echo "Fresh dataset manifest is missing: ${DATASET_MANIFEST}" >&2
    exit 2
}
case "${CHECKPOINT}" in
    "${SELF_DIR}/"*.pth) ;;
    *)
        echo "Refusing a checkpoint outside this fresh self-training directory:" >&2
        echo "  ${CHECKPOINT}" >&2
        exit 2
        ;;
esac
[[ -f "${CHECKPOINT}" ]] || {
    echo "Fresh self-training checkpoint not found yet: ${CHECKPOINT}" >&2
    echo "Wait for the first self-training validation, or set CHECKPOINT to an existing iter_*.pth in ${SELF_DIR}." >&2
    exit 2
}

(
    cd "${DATA_ROOT}"
    sha256sum -c "${DATASET_MANIFEST}"
)

echo "Testing fresh UtilityMatch online Student"
echo "Run:        ${UTILITYMATCH_DIR}"
echo "Data:       ${DATA_ROOT}"
echo "Checkpoint: ${CHECKPOINT}"

exec env \
    DATA_ROOT="${DATA_ROOT}" \
    UTILITYMATCH_DIR="${UTILITYMATCH_DIR}" \
    CHECKPOINT="${CHECKPOINT}" \
    GPU="${GPU:-0}" \
    REQUIRE_5090="${REQUIRE_5090:-1}" \
    SAVE_RESULT="${SAVE_RESULT:-False}" \
    NMS="${NMS:-0}" \
    bash "${BASE_TEST}"
