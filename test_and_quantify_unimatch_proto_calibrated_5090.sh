#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export EXP_NAME="${EXP_NAME:-MT_PROMISE12_UniMatch_ProtoCal_191slices}"
export LABELNUM="${LABELNUM:-7}"

exec bash "${PROJECT_ROOT}/test_and_quantify_unimatch_5090.sh" "$@"
