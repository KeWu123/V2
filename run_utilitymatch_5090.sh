#!/usr/bin/env bash
set -Eeuo pipefail

# Full H-UTILITYMATCH run from the original fixed UniMatch pretrain checkpoint.

BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${BASELINE_ROOT}/code"
SCRIPT_PATH="${BASELINE_ROOT}/$(basename -- "${BASH_SOURCE[0]}")"
DATA_ROOT="${DATA_ROOT:-${BASELINE_ROOT}/data/PROMISE12_h5_training_source}"
MODEL_ROOT="${BASELINE_ROOT}/model"
SERVER_LOG_DIR="${BASELINE_ROOT}/server_logs"
GPU="${GPU:-0}"
SEED="${SEED:-1337}"
REQUIRE_5090="${REQUIRE_5090:-1}"
DETACH="${DETACH:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-UtilityMatch_${RUN_TAG}}"
EXPERIMENT_DIR="${MODEL_ROOT}/${EXP_NAME}_7_labeled"
OUTPUT_DIR="${EXPERIMENT_DIR}/self_train/unet"
LAST_RUN_FILE="${SERVER_LOG_DIR}/last_utilitymatch_run.txt"
UNIMATCH_FOLDER_NAME="UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled"
ORIGINAL_UNIMATCH_DIR="${ORIGINAL_UNIMATCH_DIR:-}"

mkdir -p "${SERVER_LOG_DIR}"

if [[ "${SEED}" != "1337" ]]; then
    echo "The locked UtilityMatch experiment uses seed 1337, got ${SEED}." >&2
    exit 2
fi

if [[ -z "${ORIGINAL_UNIMATCH_DIR}" ]]; then
    for candidate in \
        "${BASELINE_ROOT}/../${UNIMATCH_FOLDER_NAME}" \
        "${BASELINE_ROOT}/${UNIMATCH_FOLDER_NAME}" \
        "${MODEL_ROOT}/${UNIMATCH_FOLDER_NAME}"; do
        if [[ -f "${candidate}/pre_train/unet/unet_best_model.pth" &&
              -f "${candidate}/self_train/unet/unet_best_model.pth" ]]; then
            ORIGINAL_UNIMATCH_DIR="${candidate}"
            break
        fi
    done
fi
if [[ -z "${ORIGINAL_UNIMATCH_DIR}" ]]; then
    echo "Original fixed UniMatch folder not found: ${UNIMATCH_FOLDER_NAME}" >&2
    echo "Set ORIGINAL_UNIMATCH_DIR to that exact original folder." >&2
    echo "Do not point it to TLPSource or another newly trained run." >&2
    exit 2
fi
if [[ "$(basename -- "${ORIGINAL_UNIMATCH_DIR%/}")" != "${UNIMATCH_FOLDER_NAME}" ]]; then
    echo "UtilityMatch must use the original fixed UniMatch folder name." >&2
    echo "Expected: ${UNIMATCH_FOLDER_NAME}" >&2
    echo "Got:      $(basename -- "${ORIGINAL_UNIMATCH_DIR%/}")" >&2
    exit 2
fi

PRETRAIN_CHECKPOINT="${ORIGINAL_UNIMATCH_DIR}/pre_train/unet/unet_best_model.pth"
ANCHOR_CHECKPOINT="${ORIGINAL_UNIMATCH_DIR}/self_train/unet/unet_best_model.pth"
[[ -f "${PRETRAIN_CHECKPOINT}" ]] || {
    echo "Missing original pretrain checkpoint: ${PRETRAIN_CHECKPOINT}" >&2
    exit 2
}
[[ -f "${ANCHOR_CHECKPOINT}" ]] || {
    echo "Missing original UniMatch anchor: ${ANCHOR_CHECKPOINT}" >&2
    exit 2
}

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
    echo "No Python found. Activate the original UniMatch environment." >&2
    exit 2
fi

if [[ "${DETACH}" == "1" && "${_UTILITYMATCH_DETACHED:-0}" != "1" ]]; then
    DETACHED_LOG="${SERVER_LOG_DIR}/utilitymatch_${RUN_TAG}_nohup.log"
    nohup env _UTILITYMATCH_DETACHED=1 DETACH=0 RUN_TAG="${RUN_TAG}" \
        EXP_NAME="${EXP_NAME}" DATA_ROOT="${DATA_ROOT}" GPU="${GPU}" \
        SEED="${SEED}" REQUIRE_5090="${REQUIRE_5090}" \
        ORIGINAL_UNIMATCH_DIR="${ORIGINAL_UNIMATCH_DIR}" \
        bash "${SCRIPT_PATH}" "$@" >"${DETACHED_LOG}" 2>&1 </dev/null &
    printf 'Started UtilityMatch in background: PID=%s\nLog: %s\n' \
        "$!" "${DETACHED_LOG}"
    exit 0
fi

if command -v flock >/dev/null 2>&1; then
    exec 9>"${SERVER_LOG_DIR}/.unimatch_training.lock"
    flock -n 9 || {
        echo "Another Baseline/UniMatch training process is already running." >&2
        exit 3
    }
fi

[[ -f "${CODE_DIR}/train_utilitymatch.py" ]] || {
    echo "Missing training entry: ${CODE_DIR}/train_utilitymatch.py" >&2
    exit 2
}
[[ -f "${CODE_DIR}/utilitymatch.py" ]] || {
    echo "Missing utility module: ${CODE_DIR}/utilitymatch.py" >&2
    exit 2
}
if [[ -e "${EXPERIMENT_DIR}" ]]; then
    echo "Refusing to overwrite existing output: ${EXPERIMENT_DIR}" >&2
    echo "Omit RUN_TAG/EXP_NAME for a new timestamped directory." >&2
    exit 3
fi

cd "${CODE_DIR}"
echo "Checking code, original weights, complete 35/5/10 data, and GPU..."
CUDA_VISIBLE_DEVICES="${GPU}" CODE_DIR="${CODE_DIR}" DATA_ROOT="${DATA_ROOT}" \
    PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT}" \
    ANCHOR_CHECKPOINT="${ANCHOR_CHECKPOINT}" REQUIRE_5090="${REQUIRE_5090}" \
    "${PYTHON_LAUNCHER[@]}" - <<'PY'
import hashlib
import importlib
import os
from collections import Counter
from pathlib import Path

for name in (
    "h5py", "medpy", "numpy", "scipy", "skimage", "tensorboardX",
    "torch", "torchvision", "tqdm",
):
    importlib.import_module(name)

import h5py
import torch

code_dir = Path(os.environ["CODE_DIR"])
data_root = Path(os.environ["DATA_ROOT"])
for name in (
    "train_utilitymatch.py", "utilitymatch.py", "test_utilitymatch_smoke.py",
    "train_unimatch.py", "test_unimatch.py",
):
    path = code_dir / name
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
print("Python syntax check passed")

def read_list(name):
    path = data_root / name
    if not path.is_file():
        raise SystemExit(f"Missing fixed dataset list: {path}")
    return [
        value.strip().split(".")[0]
        for value in path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]

train = read_list("train.list")
val = read_list("val.list")
test = read_list("test.list")
slices = [
    value.strip()
    for value in (data_root / "train_slices.list").read_text(
        encoding="utf-8").splitlines()
    if value.strip()
]
if (len(train), len(val), len(test), len(slices)) != (35, 5, 10, 940):
    raise SystemExit(
        "Expected train/val/test/slices=35/5/10/940, got "
        f"{len(train)}/{len(val)}/{len(test)}/{len(slices)}")
if len(set(train + val + test)) != 50:
    raise SystemExit("Fixed split lists overlap")

labeled = train[:7]
counts = Counter()
for index, slice_name in enumerate(slices):
    slice_path = data_root / "data" / "slices" / f"{slice_name}.h5"
    if not slice_path.is_file() or slice_path.stat().st_size < 1024:
        raise SystemExit(f"Missing or truncated training slice: {slice_path}")
    with slice_path.open("rb") as handle:
        if handle.read(8) != b"\x89HDF\r\n\x1a\n":
            raise SystemExit(f"Invalid training-slice HDF5 signature: {slice_path}")
    with h5py.File(slice_path, "r") as handle:
        if not {"image", "label"}.issubset(handle.keys()):
            raise SystemExit(f"Missing image/label datasets: {slice_path}")
        if handle["image"].shape != handle["label"].shape:
            raise SystemExit(f"Image/label shape mismatch: {slice_path}")
    for case in labeled:
        if slice_name.startswith(case + "_slice"):
            counts[case] += 1
            if index >= 191:
                raise SystemExit(
                    f"Labeled slice occurs after first 191 entries: {slice_name}")
            break
if sum(counts.values()) != 191 or any(counts[case] == 0 for case in labeled):
    raise SystemExit(f"Expected first7=191 nonempty slices, got {dict(counts)}")

filesystem_glob_count = sum(
    len(list((data_root / "data" / "slices").glob(f"{case}_slice*.h5")))
    for case in labeled
)
if filesystem_glob_count != 191:
    print(
        "Filesystem glob finds "
        f"{filesystem_glob_count} labeled-case paths; this count is ignored. "
        "The fixed trainer uses exactly the validated 191 list entries."
    )

def check_volume(case):
    path = data_root / "data" / f"{case}.h5"
    if not path.is_file() or path.stat().st_size < 1024:
        raise SystemExit(f"Missing or truncated volume H5: {path}")
    with path.open("rb") as handle:
        if handle.read(8) != b"\x89HDF\r\n\x1a\n":
            raise SystemExit(f"Invalid HDF5 signature: {path}")
    with h5py.File(path, "r") as handle:
        if not {"image", "label"}.issubset(handle.keys()):
            raise SystemExit(f"Missing image/label datasets: {path}")
        if handle["image"].shape != handle["label"].shape:
            raise SystemExit(f"Image/label shape mismatch: {path}")

for case in val + test:
    check_volume(case)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

checkpoint_path = os.environ["PRETRAIN_CHECKPOINT"]
try:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
if not isinstance(checkpoint, dict) or not {"net", "opt"}.issubset(checkpoint):
    raise SystemExit(
        "Original pretrain checkpoint must contain both 'net' and 'opt' states")
print(f"Fixed labeled cases/slices: {dict(counts)}")
print("Fixed sampler partition: labeled=191 unlabeled=749 batches=15")
print(f"Original pretrain SHA256: {sha256(checkpoint_path)}")
print(f"Original anchor SHA256:   {sha256(os.environ['ANCHOR_CHECKPOINT'])}")

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
gpu_name = torch.cuda.get_device_name(0)
print(f"PyTorch={torch.__version__}, CUDA={torch.version.cuda}")
print(f"Visible GPU={gpu_name}, capability={torch.cuda.get_device_capability(0)}")
if os.environ.get("REQUIRE_5090", "1") == "1" and "5090" not in gpu_name:
    raise SystemExit(f"Expected RTX 5090, got {gpu_name}")
print("UtilityMatch preflight passed")
PY

echo "Running deterministic utility/BN smoke test..."
"${PYTHON_LAUNCHER[@]}" test_utilitymatch_smoke.py

printf '%s\n' "${EXPERIMENT_DIR}" >"${LAST_RUN_FILE}"
echo "======================================================================"
echo "UtilityMatch full training"
echo "Data:             ${DATA_ROOT}"
echo "Original folder:  ${ORIGINAL_UNIMATCH_DIR}"
echo "Initial weight:   ${PRETRAIN_CHECKPOINT}"
echo "Reference anchor: ${ANCHOR_CHECKPOINT}"
echo "Output:           ${OUTPUT_DIR}"
echo "Protocol:         35/5/10, first7=191, seed1337, self30000"
echo "UniMatch fixed:   batch24/12, warmup1000, lr0.01-poly, tau0.95"
echo "UtilityMatch:     four exact candidates -> top two clean-gradient utility"
echo "Terminal:         tqdm progress and validation remain visible"
echo "======================================================================"
printf 'Launching:'
printf ' %q' env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PYTHON_LAUNCHER[@]}" train_utilitymatch.py \
    --root_path "${DATA_ROOT}" \
    --pretrained_model_path "${PRETRAIN_CHECKPOINT}" \
    --anchor_checkpoint "${ANCHOR_CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" "$@"
printf '\n'

RUN_LOG="${SERVER_LOG_DIR}/utilitymatch_${RUN_TAG}.log"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PYTHON_LAUNCHER[@]}" train_utilitymatch.py \
        --root_path "${DATA_ROOT}" \
        --pretrained_model_path "${PRETRAIN_CHECKPOINT}" \
        --anchor_checkpoint "${ANCHOR_CHECKPOINT}" \
        --output_dir "${OUTPUT_DIR}" \
        "$@" 2>&1 | tee "${RUN_LOG}"

echo "UtilityMatch training completed"
echo "Terminal log: ${RUN_LOG}"
echo "Best model:   ${OUTPUT_DIR}/unet_best_model.pth"
echo "Summary:      ${OUTPUT_DIR}/training_summary.json"
echo "Test command: UTILITYMATCH_DIR='${EXPERIMENT_DIR}' bash '${BASELINE_ROOT}/test_utilitymatch_5090.sh'"
