#!/usr/bin/env bash
set -Eeuo pipefail

BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_PREFIX=CalibratedUtilityMatch \
EXPECTED_HYPOTHESIS=H-CALIBRATED-UTILITYMATCH \
LAST_RUN_FILE="${BASELINE_ROOT}/server_logs/last_utilitymatch_calibrated_run.txt" \
  bash "${BASELINE_ROOT}/test_utilitymatch_safe_5090.sh" "$@"
