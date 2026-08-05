#!/usr/bin/env bash
set -Eeuo pipefail

BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
METHOD_NAME=FrontierMatch \
TRAIN_ENTRY=train_frontiermatch.py \
SMOKE_ENTRY=test_frontiermatch_smoke.py \
RUN_LOG_PREFIX=frontiermatch \
TEST_SCRIPT=test_frontiermatch_5090.sh \
METHOD_DESCRIPTION="two UniMatch rays; utility selects stable, coverage, or reliable joint augmentation-pseudo-label policy per ray; strict positive gate" \
TRACE_NAME=frontier_trace.csv \
LAST_RUN_FILE="${BASELINE_ROOT}/server_logs/last_frontiermatch_run.txt" \
  bash "${BASELINE_ROOT}/run_utilitymatch_safe_5090.sh" "$@"

