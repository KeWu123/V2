#!/usr/bin/env bash
set -Eeuo pipefail

BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
METHOD_NAME=CalibratedUtilityMatch \
TRAIN_ENTRY=train_utilitymatch_calibrated.py \
SMOKE_ENTRY=test_utilitymatch_calibrated_smoke.py \
RUN_LOG_PREFIX=utilitymatch_calibrated \
TEST_SCRIPT=test_utilitymatch_calibrated_5090.sh \
METHOD_DESCRIPTION="sampled-p01-p99-relative brightness plus strict positive-utility abstention with runtime activation traces" \
LAST_RUN_FILE="${BASELINE_ROOT}/server_logs/last_utilitymatch_calibrated_run.txt" \
  bash "${BASELINE_ROOT}/run_utilitymatch_safe_5090.sh" "$@"
