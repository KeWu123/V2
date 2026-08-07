#!/usr/bin/env bash
set -Eeuo pipefail

# The S4MC experiment has the same U-Net/checkpoint format as UniMatch, so the
# established evaluator is reused with an S4MC-specific default experiment.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXP_NAME="${EXP_NAME:-MT_PROMISE12_UniMatch_S4MC}"
exec bash "${ROOT}/test_and_quantify_unimatch_5090.sh"
