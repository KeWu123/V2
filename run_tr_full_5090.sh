#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TRAJECTORY_MODE=full
export EXP_NAME="${EXP_NAME:-Trajectory_FullSoftBoundary_seed1337}"
exec bash "${ROOT}/run_trajectory_reliability_5090.sh" "$@"

