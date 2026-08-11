#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SUITE_TAG="${SUITE_TAG:-$(date +%Y%m%d_%H%M%S)}"
MODES=(baseline weighting adaptive weighting_adaptive full)

for mode in "${MODES[@]}"; do
    echo "Starting trajectory ablation: ${mode}"
    TRAJECTORY_MODE="${mode}" \
    EXP_NAME="Trajectory_${mode}_${SUITE_TAG}_seed1337" \
    DETACH=0 \
        bash "${ROOT}/run_trajectory_reliability_5090.sh"
done

echo "Trajectory ablation suite completed: ${SUITE_TAG}"

