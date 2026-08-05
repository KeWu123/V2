#!/usr/bin/env bash
set -Eeuo pipefail

BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_PREFIX=FrontierMatch \
EXPECTED_HYPOTHESIS=H-FRONTIERMATCH \
LAST_RUN_FILE="${BASELINE_ROOT}/server_logs/last_frontiermatch_run.txt" \
  bash "${BASELINE_ROOT}/test_utilitymatch_safe_5090.sh" "$@"

