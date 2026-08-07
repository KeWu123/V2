#!/usr/bin/env bash
set -Eeuo pipefail

BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LABELNUM="${LABELNUM:-7}"
EXP_NAME="${EXP_NAME:-UniMatch_TemporalBank_POS_MEO_35_5_10_seed1337}"
OUTPUT_ROOT="${BASELINE_ROOT}/model/${EXP_NAME}_${LABELNUM}_labeled"

export EXP_NAME
export LABELNUM
export REFINE_STAGE="temporal_bank_pos_meo"
export REFINE_RESULT_SUBDIR="temporal_bank_pos_meo"
export REFINE_DIR="${REFINE_DIR:-${OUTPUT_ROOT}/temporal_bank_pos_meo/unet}"

exec bash "${BASELINE_ROOT}/test_and_quantify_unimatch_temporal_bank_v2_5090.sh" "$@"
