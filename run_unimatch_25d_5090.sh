#!/usr/bin/env bash
set -Eeuo pipefail

# PROMISE12 original UniMatch baseline + adjacent-slice 2.5D input.
# Training defaults: pre=1000, self=5000, fixed warmup=1000, labelnum=7,
# batch=24/12, lr=0.01, seed=1337 and the original UniMatch parameters.

BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${BASELINE_ROOT}/code"
DATA_ROOT="${DATA_ROOT:-${BASELINE_ROOT}/data/PROMISE12_h5}"
RAW_ARCHIVE="${RAW_ARCHIVE:-${BASELINE_ROOT}/data/PROMISE12/raw/training_data.zip}"
RAW_ROOT="${RAW_ROOT:-${BASELINE_ROOT}/data/PROMISE12/extracted/training_data}"
CONVERTER="${BASELINE_ROOT}/tools/convert_promise12_to_h5.py"
GPU="${GPU:-0}"
SEED="${SEED:-1337}"
EXP_NAME="${EXP_NAME:-MT_PROMISE12_UniMatch_25D}"
DETACH="${DETACH:-0}"
EXPERIMENT_DIR="${BASELINE_ROOT}/model/${EXP_NAME}_7_labeled"
LOG_DIR="${BASELINE_ROOT}/server_logs"
SCRIPT_PATH="${BASELINE_ROOT}/$(basename -- "${BASH_SOURCE[0]}")"

mkdir -p "${LOG_DIR}"

if [[ "${DETACH}" == "1" && "${_UNIMATCH_25D_DETACHED:-0}" != "1" ]]; then
    LOG_PATH="${LOG_DIR}/train_25d_$(date +%Y%m%d_%H%M%S).log"
    nohup env _UNIMATCH_25D_DETACHED=1 DETACH=0 \
        bash "${SCRIPT_PATH}" >"${LOG_PATH}" 2>&1 </dev/null &
    echo "Started in background: PID=$!"
    echo "Log: ${LOG_PATH}"
    exit 0
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON=("${PYTHON_BIN}")
elif [[ -n "${CONDA_ENV:-}" ]]; then
    PYTHON=(conda run --no-capture-output -n "${CONDA_ENV}" python)
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=(python3)
else
    PYTHON=(python)
fi

[[ -f "${CODE_DIR}/train_unimatch_25d.py" ]] || {
    echo "Missing ${CODE_DIR}/train_unimatch_25d.py" >&2
    exit 2
}
h5_ready() {
    local first_slice
    first_slice="$(find "${DATA_ROOT}/data/slices" -maxdepth 1 -type f -name '*.h5' -print -quit 2>/dev/null || true)"
    [[ -n "${first_slice}" ]] || return 1
    (( $(stat -c '%s' "${first_slice}" 2>/dev/null || echo 0) >= 1024 )) || return 1
    ! head -c 128 "${first_slice}" | grep -aq '^version https://git-lfs.github.com/spec/v1'
}

# Uploaded Git repositories may contain H5 LFS pointer files. Rebuild only in
# that case; a prepared server dataset takes this fast path without conversion.
if ! h5_ready; then
    [[ -f "${RAW_ARCHIVE}" && -f "${CONVERTER}" ]] || {
        echo "H5 data is invalid and the local PROMISE12 archive/converter is missing." >&2
        exit 2
    }
    if head -c 128 "${RAW_ARCHIVE}" | grep -aq '^version https://git-lfs.github.com/spec/v1'; then
        echo "PROMISE12 archive is also a Git LFS pointer: ${RAW_ARCHIVE}" >&2
        exit 2
    fi
    echo "H5 files are absent or LFS pointers; rebuilding PROMISE12 once..."
    "${PYTHON[@]}" - "${RAW_ARCHIVE}" "${RAW_ROOT}" <<'PY'
import sys
import zipfile
from pathlib import Path

archive = Path(sys.argv[1])
raw_root = Path(sys.argv[2])
raw_root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive) as bundle:
    bundle.extractall(raw_root)
PY
    "${PYTHON[@]}" "${CONVERTER}" --raw_root "${RAW_ROOT}" --out_root "${DATA_ROOT}"
fi

h5_ready || {
    echo "PROMISE12 conversion did not produce usable H5 data below ${DATA_ROOT}" >&2
    exit 2
}
[[ -f "${DATA_ROOT}/train_slices.list" && -f "${DATA_ROOT}/val.list" ]] || {
    echo "PROMISE12 lists are missing below ${DATA_ROOT}" >&2
    exit 2
}

if find "${EXPERIMENT_DIR}" -type f -name '*.pth' -print -quit 2>/dev/null | grep -q .; then
    if [[ "${ALLOW_EXISTING:-0}" != "1" ]]; then
        echo "Existing checkpoints found below ${EXPERIMENT_DIR}." >&2
        echo "Use a new EXP_NAME, or set ALLOW_EXISTING=1 deliberately." >&2
        exit 3
    fi
fi

echo "Data:   ${DATA_ROOT}"
echo "GPU:    ${GPU}"
echo "Seed:   ${SEED}"
echo "Output: ${EXPERIMENT_DIR}"
echo "Config: original UniMatch + [z-1,z,z+1], pre=1000 self=5000 warmup=1000 labelnum=7 batch=24/12"

cd "${CODE_DIR}"
RUN_LOG="${LOG_DIR}/train_25d_console_$(date +%Y%m%d_%H%M%S).log"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON[@]}" \
    train_unimatch_25d.py \
    --root_path "${DATA_ROOT}" \
    --exp "${EXP_NAME}" \
    --seed "${SEED}" 2>&1 | tee "${RUN_LOG}"

echo "Training completed"
echo "Console log: ${RUN_LOG}"
echo "Run: bash test_and_quantify_unimatch_25d_5090.sh"
