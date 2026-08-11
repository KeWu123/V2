#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TRAJECTORY_MODE=weighting
export EXP_NAME="${EXP_NAME:-Trajectory_Weighting_seed1337}"
exec bash "${ROOT}/run_trajectory_reliability_5090.sh" "$@"

