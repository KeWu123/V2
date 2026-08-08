#!/usr/bin/env bash
set -Eeuo pipefail

# RTX 5090 launcher for the PROMISE12 UniMatch V2 PLQ experiment.
#
# Inherits the UniMatch V2 A0 protocol (35/5/10, 7 labeled cases, seed 1337,
# pretrain 10000 / self-train 30000, batch 24/12, SGD lr 0.01, EMA 0.99,
# CutMix, strong MRI augmentation, comp_drop dual-view) and replaces ONLY the
# pseudo-label selection with class-aware thresholds + an entropy gate
# (see code/train_unimatch_v2_plq.py).

BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${BASELINE_ROOT}/code"
SCRIPT_PATH="${BASELINE_ROOT}/$(basename -- "${BASH_SOURCE[0]}")"
DATA_ROOT="${DATA_ROOT:-${BASELINE_ROOT}/data/PROMISE12_h5}"
RAW_ARCHIVE="${RAW_ARCHIVE:-${BASELINE_ROOT}/data/PROMISE12/raw/training_data.zip}"
RAW_ROOT="${RAW_ROOT:-${BASELINE_ROOT}/data/PROMISE12/extracted/training_data}"
CONVERTER="${BASELINE_ROOT}/tools/convert_promise12_to_h5.py"
SPLIT_CHECKER="${BASELINE_ROOT}/tools/check_promise12_split.py"
AUTO_PREPARE="${AUTO_PREPARE:-1}"
AUTO_INSTALL_DATA_DEPS="${AUTO_INSTALL_DATA_DEPS:-1}"
GPU="${GPU:-0}"
SEED="${SEED:-1337}"
EXP_NAME="${EXP_NAME:-UniMatchV2_PLQ_label7_seed1337}"
PRE_ITERATIONS="${PRE_ITERATIONS:-10000}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
THRESHOLD_BG="${THRESHOLD_BG:-0.95}"
THRESHOLD_FG="${THRESHOLD_FG:-0.80}"
ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD:-0.5}"
DETACH="${DETACH:-0}"
REQUIRE_5090="${REQUIRE_5090:-1}"
MODEL_ROOT="${BASELINE_ROOT}/model"
EXPERIMENT_DIR="${MODEL_ROOT}/${EXP_NAME}_7_labeled"
SERVER_LOG_DIR="${BASELINE_ROOT}/server_logs"

mkdir -p "${SERVER_LOG_DIR}"

if [[ "${DETACH}" == "1" && "${_DESKTOP_UNIMATCH_V2_PLQ_DETACHED:-0}" != "1" ]]; then
    SERVER_LOG="${SERVER_LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
    nohup env _DESKTOP_UNIMATCH_V2_PLQ_DETACHED=1 DETACH=0 \
        bash "${SCRIPT_PATH}" >"${SERVER_LOG}" 2>&1 </dev/null &
    printf 'Started in background: PID=%s\nLog: %s\n' "$!" "${SERVER_LOG}"
    exit 0
fi

PYTHON_LAUNCHER=()
if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_LAUNCHER=("${PYTHON_BIN}")
elif [[ -n "${CONDA_ENV:-}" ]]; then
    command -v conda >/dev/null 2>&1 || {
        echo "CONDA_ENV is set but conda is unavailable." >&2
        exit 2
    }
    PYTHON_LAUNCHER=(conda run --no-capture-output -n "${CONDA_ENV}" python)
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_LAUNCHER=(python3)
elif command -v python >/dev/null 2>&1; then
    PYTHON_LAUNCHER=(python)
else
    echo "No Python found. Activate the training environment or set CONDA_ENV/PYTHON_BIN." >&2
    exit 2
fi

[[ -f "${CODE_DIR}/train_unimatch_v2_plq.py" ]] || {
    echo "Missing training entry: ${CODE_DIR}/train_unimatch_v2_plq.py" >&2
    exit 2
}
[[ -f "${CONVERTER}" ]] || {
    echo "Missing PROMISE12 converter: ${CONVERTER}" >&2
    exit 2
}

h5_needs_prepare() {
    local first_slice
    if ! "${PYTHON_LAUNCHER[@]}" "${SPLIT_CHECKER}" \
        --data_root "${DATA_ROOT}" --require_h5 >/dev/null 2>&1; then
        return 0
    fi
    first_slice="$(find "${DATA_ROOT}/data/slices" -maxdepth 1 -type f -name '*.h5' \
        -print -quit 2>/dev/null || true)"
    if [[ -z "${first_slice}" ]]; then
        return 0
    fi
    if head -c 128 "${first_slice}" | grep -aq \
        '^version https://git-lfs.github.com/spec/v1'; then
        return 0
    fi
    if (( $(stat -c '%s' "${first_slice}") < 1024 )); then
        return 0
    fi
    return 1
}

prepare_dataset() {
    if ! h5_needs_prepare; then
        echo "PROMISE12 H5 data already prepared: ${DATA_ROOT}"
        return
    fi
    if [[ "${AUTO_PREPARE}" != "1" ]]; then
        echo "PROMISE12 H5 data is missing or contains Git LFS pointers, and AUTO_PREPARE=0." >&2
        exit 2
    fi
    [[ -f "${RAW_ARCHIVE}" ]] || {
        echo "Missing PROMISE12 archive: ${RAW_ARCHIVE}" >&2
        exit 2
    }
    if head -c 128 "${RAW_ARCHIVE}" | grep -aq \
        '^version https://git-lfs.github.com/spec/v1'; then
        echo "The PROMISE12 archive is a Git LFS pointer: ${RAW_ARCHIVE}" >&2
        exit 2
    fi

    echo "Verifying and extracting PROMISE12 archive..."
    "${PYTHON_LAUNCHER[@]}" - "${RAW_ARCHIVE}" "${RAW_ROOT}" <<'PY'
import hashlib
import os
import sys
import zipfile
from pathlib import Path

archive = Path(sys.argv[1]).resolve()
raw_root = Path(sys.argv[2]).resolve()
expected_md5 = "f6d23994117c989daf07e5291edd0aea"
digest = hashlib.md5()
with archive.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
actual_md5 = digest.hexdigest()
if actual_md5 != expected_md5:
    raise SystemExit(
        f"PROMISE12 archive MD5 mismatch: expected {expected_md5}, got {actual_md5}"
    )
raw_root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive) as bundle:
    corrupt = bundle.testzip()
    if corrupt is not None:
        raise SystemExit(f"Corrupt ZIP member: {corrupt}")
    for member in bundle.infolist():
        destination = (raw_root / member.filename).resolve()
        if os.path.commonpath((str(raw_root), str(destination))) != str(raw_root):
            raise SystemExit(f"Unsafe ZIP member: {member.filename}")
    bundle.extractall(raw_root)
case00 = raw_root / "Case00.raw"
if not case00.is_file() or case00.stat().st_size < 1_000_000:
    raise SystemExit(f"Extraction did not produce real raw data: {case00}")
print(f"Archive verified and extracted to {raw_root}")
PY

    if ! "${PYTHON_LAUNCHER[@]}" - <<'PY'
import h5py
import numpy
import SimpleITK
PY
    then
        if [[ "${AUTO_INSTALL_DATA_DEPS}" != "1" ]]; then
            echo "Install conversion dependencies: pip install h5py numpy SimpleITK" >&2
            exit 2
        fi
        echo "Installing missing PROMISE12 conversion dependencies..."
        "${PYTHON_LAUNCHER[@]}" -m pip install --disable-pip-version-check \
            h5py numpy SimpleITK
    fi

    "${PYTHON_LAUNCHER[@]}" "${CONVERTER}" \
        --raw_root "${RAW_ROOT}" --out_root "${DATA_ROOT}"
    if h5_needs_prepare; then
        echo "PROMISE12 conversion did not produce usable H5 data." >&2
        exit 2
    fi
    echo "PROMISE12 H5 conversion completed"
}

if command -v flock >/dev/null 2>&1; then
    exec 9>"${SERVER_LOG_DIR}/.unimatch_v2_plq_training.lock"
    flock -n 9 || {
        echo "Another UniMatch V2 PLQ training process is already running." >&2
        exit 3
    }
fi

prepare_dataset

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
    echo "PROMISE12 preparation/signature scan completed."
    exit 0
fi

cd "${CODE_DIR}"

echo "Checking code, PROMISE12 H5 data, dependencies, and RTX 5090..."
CUDA_VISIBLE_DEVICES="${GPU}" CODE_DIR="${CODE_DIR}" DATA_ROOT="${DATA_ROOT}" \
    REQUIRE_5090="${REQUIRE_5090}" "${PYTHON_LAUNCHER[@]}" - <<'PY'
import importlib
import os
from pathlib import Path

required = (
    "h5py",
    "medpy",
    "numpy",
    "scipy",
    "skimage",
    "tensorboardX",
    "torch",
    "torchvision",
    "tqdm",
)
for name in required:
    importlib.import_module(name)

import h5py
import torch

code_dir = Path(os.environ["CODE_DIR"])
data_root = Path(os.environ["DATA_ROOT"])
syntax_targets = [
    code_dir / "train_unimatch_v2_plq.py",
    code_dir / "dataloaders" / "dataset.py",
    code_dir / "utils" / "losses.py",
    code_dir / "utils" / "ramps.py",
    code_dir / "utils" / "val_2d.py",
]
for path in syntax_targets:
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
print("Python syntax check passed")

list_names = ("train.list", "train_slices.list", "val.list", "test.list")
for name in list_names:
    path = data_root / name
    if not path.is_file():
        raise SystemExit(f"Missing dataset list: {path}")

train_slices = [
    value.strip()
    for value in (data_root / "train_slices.list").read_text(encoding="utf-8").splitlines()
    if value.strip()
]
train_cases = [
    value.strip()
    for value in (data_root / "train.list").read_text(encoding="utf-8").splitlines()
    if value.strip()
]
labeled_case_set = set(train_cases[:7])
labeled_names = [
    name for name in train_slices
    if name.rsplit("_slice", 1)[0] in labeled_case_set
]
if train_slices[:len(labeled_names)] != labeled_names:
    raise SystemExit(
        "train_slices.list must group the first 7 train.list cases first"
    )
labeled_slice_count = len(labeled_names)
if len(train_slices) <= labeled_slice_count:
    raise SystemExit(
        f"PROMISE12 requires more than {labeled_slice_count} training slices"
    )

def check_h5(path):
    if not path.is_file():
        raise SystemExit(f"Missing H5 file: {path}")
    with path.open("rb") as handle:
        header = handle.read(128)
    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise SystemExit(f"Git LFS pointer found instead of HDF5 data: {path}")
    if not header.startswith(b"\x89HDF\r\n\x1a\n"):
        raise SystemExit(f"Invalid HDF5 signature: {path}")
    with h5py.File(path, "r") as handle:
        missing = {"image", "label"}.difference(handle.keys())
        if missing:
            raise SystemExit(f"Missing datasets {sorted(missing)} in {path}")

first_slice = data_root / "data" / "slices" / f"{train_slices[0]}.h5"
check_h5(first_slice)
for split in ("val", "test"):
    cases = [
        value.strip().split(".")[0]
        for value in (data_root / f"{split}.list").read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    if not cases:
        raise SystemExit(f"No cases in {data_root / f'{split}.list'}")
    for case in cases:
        check_h5(data_root / "data" / f"{case}.h5")

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
gpu_name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
print(f"PyTorch={torch.__version__}, CUDA={torch.version.cuda}")
print(f"Visible GPU={gpu_name}, compute_capability={capability}")
if os.environ.get("REQUIRE_5090", "1") == "1" and "5090" not in gpu_name:
    raise SystemExit(f"Expected RTX 5090, got {gpu_name}")
layer = torch.nn.Conv2d(1, 4, 3, padding=1).cuda().eval()
with torch.no_grad():
    result = layer(torch.randn(2, 1, 64, 64, device="cuda"))
torch.cuda.synchronize()
print(f"CUDA convolution check passed: {tuple(result.shape)}")
print(
    f"Dataset check passed: train_slices={len(train_slices)}, "
    f"labelnum=7, labeled_slices={labeled_slice_count}"
)
PY

if find "${EXPERIMENT_DIR}" -type f -name '*.pth' -print -quit 2>/dev/null | grep -q .; then
    if [[ "${ALLOW_EXISTING:-0}" != "1" ]]; then
        echo "Existing checkpoints found below ${EXPERIMENT_DIR}." >&2
        echo "The original code has no exact-resume support and starts from scratch." >&2
        echo "Move the old experiment, or explicitly set ALLOW_EXISTING=1 to overwrite/reuse that directory." >&2
        exit 3
    fi
fi

echo "Project: ${BASELINE_ROOT}"
echo "Data:    ${DATA_ROOT}"
echo "GPU:     ${GPU}"
echo "Seed:    ${SEED}"
echo "Output:  ${EXPERIMENT_DIR}"
echo "Configured defaults: PROMISE12 35/5/10, labelnum=7 (191 labeled slices), pre=${PRE_ITERATIONS}, self=${MAX_ITERATIONS}, warmup=1000, batch=24, labeled_bs=12, lr=0.01, seed=${SEED}"
echo "PLQ defaults: threshold_bg=${THRESHOLD_BG}, threshold_fg=${THRESHOLD_FG}, entropy_threshold=${ENTROPY_THRESHOLD}"
printf 'Launching:'
printf ' %q' env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_LAUNCHER[@]}" \
    train_unimatch_v2_plq.py --root_path "${DATA_ROOT}" --exp "${EXP_NAME}" --seed "${SEED}" \
    --pre_iterations "${PRE_ITERATIONS}" --max_iterations "${MAX_ITERATIONS}" \
    --threshold_bg "${THRESHOLD_BG}" --threshold_fg "${THRESHOLD_FG}" \
    --entropy_threshold "${ENTROPY_THRESHOLD}" "$@"
printf '\n'

RUN_LOG="${SERVER_LOG_DIR}/train_console_$(date +%Y%m%d_%H%M%S).log"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_LAUNCHER[@]}" \
    train_unimatch_v2_plq.py --root_path "${DATA_ROOT}" \
        --exp "${EXP_NAME}" --seed "${SEED}" \
        --pre_iterations "${PRE_ITERATIONS}" --max_iterations "${MAX_ITERATIONS}" \
        --threshold_bg "${THRESHOLD_BG}" --threshold_fg "${THRESHOLD_FG}" \
        --entropy_threshold "${ENTROPY_THRESHOLD}" \
        "$@" 2>&1 | tee "${RUN_LOG}"

echo "Training completed"
echo "Console log: ${RUN_LOG}"
echo "Pretrain: ${EXPERIMENT_DIR}/pre_train/unet"
echo "Self-train: ${EXPERIMENT_DIR}/self_train/unet"
echo "Run test_and_quantify_unimatch_v2_5090.sh (EXP_NAME=<same>) to evaluate both checkpoints."
