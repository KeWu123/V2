#!/usr/bin/env bash
set -Eeuo pipefail

# PROMISE12 Baseline + UniMatch with only the MRI interpolation corrected.
# Defaults are unchanged: pretrain=1000, self-train=5000, labelnum=7,
# seed=1337, batch=24/12, lr=0.01.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${ROOT}/code"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/PROMISE12_h5}"
RAW_ARCHIVE="${RAW_ARCHIVE:-${ROOT}/data/PROMISE12/raw/training_data.zip}"
RAW_ROOT="${RAW_ROOT:-${ROOT}/data/PROMISE12/extracted/training_data}"
CONVERTER="${ROOT}/tools/convert_promise12_to_h5.py"
GPU="${GPU:-0}"
SEED="${SEED:-1337}"
EXP_NAME="${EXP_NAME:-MT_PROMISE12_UniMatch_MRIInterp}"
DETACH="${DETACH:-0}"
EXPERIMENT_DIR="${ROOT}/model/${EXP_NAME}_7_labeled"
LOG_DIR="${ROOT}/server_logs"
SCRIPT_PATH="${ROOT}/$(basename -- "${BASH_SOURCE[0]}")"

mkdir -p "${LOG_DIR}"

if [[ "${DETACH}" == "1" && "${_MRI_INTERP_DETACHED:-0}" != "1" ]]; then
    LOG_FILE="${LOG_DIR}/mri_interp_$(date +%Y%m%d_%H%M%S).log"
    nohup env _MRI_INTERP_DETACHED=1 DETACH=0 \
        bash "${SCRIPT_PATH}" >"${LOG_FILE}" 2>&1 </dev/null &
    echo "Started: PID=$!"
    echo "Log: ${LOG_FILE}"
    exit 0
fi

PYTHON=()
if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON=("${PYTHON_BIN}")
elif [[ -n "${CONDA_ENV:-}" ]]; then
    PYTHON=(conda run --no-capture-output -n "${CONDA_ENV}" python)
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=(python3)
else
    PYTHON=(python)
fi

[[ -f "${CODE_DIR}/train_unimatch_mri_interp.py" ]] || {
    echo "Missing ${CODE_DIR}/train_unimatch_mri_interp.py" >&2
    exit 2
}

prepare_promise12_h5() {
    local first_slice=""
    first_slice="$(find "${DATA_ROOT}/data/slices" -maxdepth 1 -type f \
        -name '*.h5' -print -quit 2>/dev/null || true)"
    if [[ -n "${first_slice}" ]] && \
       [[ "$(head -c 8 "${first_slice}" | od -An -tx1 | tr -d ' \n')" == "894844460d0a1a0a" ]]; then
        return
    fi

    echo "PROMISE12 H5 files are missing or are Git LFS pointers; rebuilding from raw data..."
    [[ -f "${RAW_ARCHIVE}" ]] || {
        echo "Missing raw PROMISE12 archive: ${RAW_ARCHIVE}" >&2
        exit 2
    }
    [[ "$(head -c 2 "${RAW_ARCHIVE}")" == "PK" ]] || {
        echo "Raw archive is not a real ZIP file: ${RAW_ARCHIVE}" >&2
        exit 2
    }
    [[ -f "${CONVERTER}" ]] || {
        echo "Missing converter: ${CONVERTER}" >&2
        exit 2
    }

    if [[ ! -f "${RAW_ROOT}/Case00.raw" ]]; then
        "${PYTHON[@]}" - "${RAW_ARCHIVE}" "${RAW_ROOT}" <<'PY'
import os
import sys
import zipfile
from pathlib import Path

archive = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
destination.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive) as bundle:
    for member in bundle.infolist():
        target = (destination / member.filename).resolve()
        if os.path.commonpath((str(destination), str(target))) != str(destination):
            raise SystemExit(f"Unsafe ZIP member: {member.filename}")
    bundle.extractall(destination)
print(f"Extracted PROMISE12 to {destination}")
PY
    fi

    "${PYTHON[@]}" "${CONVERTER}" \
        --raw_root "${RAW_ROOT}" --out_root "${DATA_ROOT}"
}

prepare_promise12_h5

# This is the one necessary dataset check: validate every slice that the
# DataLoader will open, so a remaining LFS pointer is reported before training.
"${PYTHON[@]}" - "${DATA_ROOT}" <<'PY'
import sys
from pathlib import Path

import h5py

root = Path(sys.argv[1])
names = [line.strip() for line in (root / "train_slices.list").read_text().splitlines()
         if line.strip()]
for name in names:
    path = root / "data" / "slices" / f"{name}.h5"
    try:
        with h5py.File(path, "r") as handle:
            if "image" not in handle or "label" not in handle:
                raise RuntimeError("missing image/label datasets")
    except Exception as error:
        raise SystemExit(f"Invalid H5 file {path}: {error}") from error
print(f"PROMISE12 H5 ready: {len(names)} training slices")
PY

if find "${EXPERIMENT_DIR}" -type f -name '*.pth' -print -quit 2>/dev/null | grep -q .; then
    if [[ "${ALLOW_EXISTING:-0}" != "1" ]]; then
        echo "Existing checkpoints found: ${EXPERIMENT_DIR}" >&2
        echo "Use another EXP_NAME, or set ALLOW_EXISTING=1 deliberately." >&2
        exit 3
    fi
fi

echo "Data:   ${DATA_ROOT}"
echo "GPU:    ${GPU}"
echo "Seed:   ${SEED}"
echo "Output: ${EXPERIMENT_DIR}"
echo "Config: pre=1000 self=5000 warmup=1000 labelnum=7 batch=24/12"
echo "Change: MRI rotate/resize=linear; label/prediction resize=nearest"

cd "${CODE_DIR}"
RUN_LOG="${LOG_DIR}/mri_interp_console_$(date +%Y%m%d_%H%M%S).log"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON[@]}" \
    train_unimatch_mri_interp.py \
    --root_path "${DATA_ROOT}" \
    --exp "${EXP_NAME}" \
    --seed "${SEED}" 2>&1 | tee "${RUN_LOG}"

echo "Training completed"
echo "Log: ${RUN_LOG}"
echo "Evaluate: EXP_NAME=${EXP_NAME} bash test_and_quantify_unimatch_mri_interp_5090.sh"
