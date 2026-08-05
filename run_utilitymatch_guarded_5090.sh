#!/usr/bin/env bash
set -Eeuo pipefail

BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
METHOD_NAME=GuardedUtilityMatch \
TRAIN_ENTRY=train_utilitymatch_guarded.py \
SMOKE_ENTRY=test_utilitymatch_guarded_smoke.py \
RUN_LOG_PREFIX=utilitymatch_guarded \
TEST_SCRIPT=test_utilitymatch_guarded_5090.sh \
METHOD_DESCRIPTION="retain four stable calibrated candidates; append two explorations; reserve one stable selected slot; select at most one positive exploration" \
TRACE_NAME=guarded_pool_trace.csv \
LAST_RUN_FILE="${BASELINE_ROOT}/server_logs/last_utilitymatch_guarded_run.txt" \
  bash "${BASELINE_ROOT}/run_utilitymatch_safe_5090.sh" "$@"
