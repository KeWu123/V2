#!/usr/bin/env bash
set -Eeuo pipefail

# PROMISE12 UniMatch + S4MC launcher. All baseline/UniMatch training defaults
# live in train_unimatch_s4mc.py; this script only selects paths and the GPU.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/PROMISE12_h5}"
RAW_ARCHIVE="${RAW_ARCHIVE:-${ROOT}/data/PROMISE12/raw/training_data.zip}"
RAW_ROOT="${RAW_ROOT:-${ROOT}/data/PROMISE12/extracted/training_data}"
CONVERTER="${ROOT}/tools/convert_promise12_to_h5.py"
GPU="${GPU:-0}"
SEED="${SEED:-1337}"
EXP_NAME="${EXP_NAME:-MT_PROMISE12_UniMatch_S4MC}"
DETACH="${DETACH:-0}"
OUT_DIR="${ROOT}/model/${EXP_NAME}_7_labeled"
LOG_DIR="${ROOT}/server_logs"
mkdir -p "${LOG_DIR}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON=("${PYTHON_BIN}")
elif [[ -n "${CONDA_ENV:-}" ]]; then
    PYTHON=(conda run --no-capture-output -n "${CONDA_ENV}" python)
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=(python3)
else
    PYTHON=(python)
fi

validate_h5_dataset() {
    DATA_ROOT="${DATA_ROOT}" "${PYTHON[@]}" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["DATA_ROOT"])
magic = b"\x89HDF\r\n\x1a\n"

def read_list(name):
    path = root / name
    if not path.is_file():
        raise SystemExit(f"missing list: {path}")
    values = [line.strip().split(".")[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise SystemExit(f"empty list: {path}")
    return values

paths = [root / "data" / "slices" / f"{name}.h5" for name in read_list("train_slices.list")]
for split in ("val.list", "test.list"):
    paths.extend(root / "data" / f"{name}.h5" for name in read_list(split))

for path in paths:
    if not path.is_file():
        raise SystemExit(f"missing H5: {path}")
    with path.open("rb") as handle:
        header = handle.read(128)
    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise SystemExit(f"Git LFS pointer instead of H5: {path}")
    if not header.startswith(magic):
        raise SystemExit(f"invalid H5 signature: {path}")

print(f"PROMISE12 H5 signatures valid: {len(paths)} files")
PY
}

prepare_h5_dataset() {
    [[ -f "${CONVERTER}" ]] || {
        echo "Missing converter: ${CONVERTER}" >&2
        exit 2
    }

    if [[ ! -f "${RAW_ROOT}/Case00.mhd" || ! -f "${RAW_ROOT}/Case00.raw" ]]; then
        [[ -f "${RAW_ARCHIVE}" ]] || {
            echo "Missing PROMISE12 archive: ${RAW_ARCHIVE}" >&2
            exit 2
        }
        echo "Extracting the real PROMISE12 data from ${RAW_ARCHIVE}..."
        "${PYTHON[@]}" - "${RAW_ARCHIVE}" "${RAW_ROOT}" <<'PY'
import hashlib
import os
import sys
import zipfile
from pathlib import Path

archive = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
expected_md5 = "f6d23994117c989daf07e5291edd0aea"

with archive.open("rb") as handle:
    first = handle.read(128)
if first.startswith(b"version https://git-lfs.github.com/spec/v1"):
    raise SystemExit(f"PROMISE12 ZIP is only a Git LFS pointer: {archive}")

digest = hashlib.md5()
with archive.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != expected_md5:
    raise SystemExit(
        f"PROMISE12 ZIP is damaged: expected MD5 {expected_md5}, got {digest.hexdigest()}"
    )

destination.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive) as bundle:
    for member in bundle.infolist():
        target = (destination / member.filename).resolve()
        if os.path.commonpath((str(destination), str(target))) != str(destination):
            raise SystemExit(f"unsafe ZIP member: {member.filename}")
    bundle.extractall(destination)

if not (destination / "Case00.mhd").is_file() or not (destination / "Case00.raw").is_file():
    raise SystemExit(f"PROMISE12 extraction is incomplete: {destination}")
print(f"PROMISE12 extracted: {destination}")
PY
    fi

    if ! "${PYTHON[@]}" -c 'import h5py, numpy, SimpleITK' >/dev/null 2>&1; then
        echo "Installing the three conversion dependencies..."
        "${PYTHON[@]}" -m pip install --disable-pip-version-check h5py numpy SimpleITK
    fi

    echo "Rebuilding PROMISE12 H5 files from the original volumes..."
    "${PYTHON[@]}" "${CONVERTER}" \
        --raw_root "${RAW_ROOT}" --out_root "${DATA_ROOT}"
}

if [[ "${DETACH}" == "1" && "${_S4MC_DETACHED:-0}" != "1" ]]; then
    LOG="${LOG_DIR}/s4mc_$(date +%Y%m%d_%H%M%S).log"
    nohup env _S4MC_DETACHED=1 DETACH=0 DATA_ROOT="${DATA_ROOT}" \
        GPU="${GPU}" SEED="${SEED}" EXP_NAME="${EXP_NAME}" \
        bash "$0" >"${LOG}" 2>&1 </dev/null &
    echo "Started: PID=$!"
    echo "Log: ${LOG}"
    exit 0
fi

[[ -f "${ROOT}/code/train_unimatch_s4mc.py" ]] || {
    echo "Missing code/train_unimatch_s4mc.py" >&2
    exit 2
}
if ! validate_h5_dataset; then
    echo "The uploaded H5 folder is invalid; rebuilding it automatically."
    prepare_h5_dataset
    validate_h5_dataset || {
        echo "PROMISE12 H5 rebuild failed: ${DATA_ROOT}" >&2
        exit 2
    }
fi

if find "${OUT_DIR}" -type f -name '*.pth' -print -quit 2>/dev/null | grep -q . \
    && [[ "${ALLOW_EXISTING:-0}" != "1" ]]; then
    echo "Existing checkpoints found: ${OUT_DIR}" >&2
    echo "Use a new EXP_NAME, or set ALLOW_EXISTING=1 intentionally." >&2
    exit 3
fi

echo "Data:   ${DATA_ROOT}"
echo "GPU:    ${GPU}"
echo "Seed:   ${SEED}"
echo "Output: ${OUT_DIR}"
echo "Config: pre=1000 self=5000 warmup=1000 labelnum=7 batch=24/12"
echo "Change: UniMatch pseudo-label selection + foreground-aware S4MC context"

cd "${ROOT}/code"
"${PYTHON[@]}" -m py_compile train_unimatch_s4mc.py
LOG="${LOG_DIR}/s4mc_console_$(date +%Y%m%d_%H%M%S).log"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON[@]}" train_unimatch_s4mc.py \
    --root_path "${DATA_ROOT}" --exp "${EXP_NAME}" --seed "${SEED}" \
    2>&1 | tee "${LOG}"

echo "Training completed"
echo "Log: ${LOG}"
echo "Evaluate: EXP_NAME=${EXP_NAME} bash test_and_quantify_unimatch_s4mc_5090.sh"
