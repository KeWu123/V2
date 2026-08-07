#!/usr/bin/env bash
set -Eeuo pipefail

# Architecture/checkpoint format is unchanged, so the existing UniMatch
# evaluator produces the same case-level and aggregate tables for this run.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXP_NAME="${EXP_NAME:-MT_PROMISE12_UniMatch_FGRescue}"
exec bash "${ROOT}/test_and_quantify_unimatch_5090.sh"
