#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate the latest SafeUtilityMatch validation-best online Student.

BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="${BASELINE_ROOT}/model"
SERVER_LOG_DIR="${BASELINE_ROOT}/server_logs"
RUN_PREFIX="${RUN_PREFIX:-SafeUtilityMatch}"
EXPECTED_HYPOTHESIS="${EXPECTED_HYPOTHESIS:-H-SAFE-UTILITYMATCH}"
LAST_RUN_FILE="${LAST_RUN_FILE:-${SERVER_LOG_DIR}/last_utilitymatch_safe_run.txt}"
SHARED_TEST="${BASELINE_ROOT}/test_utilitymatch_5090.sh"
UTILITYMATCH_DIR="${UTILITYMATCH_DIR:-}"

[[ -f "${SHARED_TEST}" ]] || { echo "Missing shared evaluator: ${SHARED_TEST}" >&2; exit 2; }
if [[ -z "${UTILITYMATCH_DIR}" && -f "${LAST_RUN_FILE}" ]]; then
    IFS= read -r UTILITYMATCH_DIR <"${LAST_RUN_FILE}"
fi
if [[ -z "${UTILITYMATCH_DIR}" ]]; then
    UTILITYMATCH_DIR="$(
        find "${MODEL_ROOT}" -maxdepth 1 -type d \
            -name "${RUN_PREFIX}_*_7_labeled" -printf '%T@ %p\n' 2>/dev/null |
            sort -nr | head -n 1 | cut -d' ' -f2-
    )"
fi
[[ -n "${UTILITYMATCH_DIR}" && -d "${UTILITYMATCH_DIR}" ]] || {
    echo "No ${RUN_PREFIX} run found." >&2
    echo "Set UTILITYMATCH_DIR explicitly if needed." >&2
    exit 2
}
[[ -f "${UTILITYMATCH_DIR}/self_train/unet/config.json" ]] || {
    echo "Missing SafeUtilityMatch config: ${UTILITYMATCH_DIR}" >&2
    exit 2
}
grep -q "${EXPECTED_HYPOTHESIS}" "${UTILITYMATCH_DIR}/self_train/unet/config.json" || {
    echo "Refusing a directory without ${EXPECTED_HYPOTHESIS}: ${UTILITYMATCH_DIR}" >&2
    exit 2
}

UTILITYMATCH_DIR="${UTILITYMATCH_DIR}" bash "${SHARED_TEST}" "$@"
