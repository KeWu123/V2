#!/usr/bin/env bash
set -Eeuo pipefail

# SafeUtilityMatch from the same fixed UniMatch PreTrain and PROMISE12 source.

BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${BASELINE_ROOT}/code"
DATA_ROOT="${DATA_ROOT:-${BASELINE_ROOT}/data/PROMISE12_h5_training_source}"
MODEL_ROOT="${BASELINE_ROOT}/model"
SERVER_LOG_DIR="${BASELINE_ROOT}/server_logs"
SCRIPT_PATH="${BASELINE_ROOT}/$(basename -- "${BASH_SOURCE[0]}")"
GPU="${GPU:-0}"
SEED="${SEED:-1337}"
REQUIRE_5090="${REQUIRE_5090:-1}"
DETACH="${DETACH:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
METHOD_NAME="${METHOD_NAME:-SafeUtilityMatch}"
TRAIN_ENTRY="${TRAIN_ENTRY:-train_utilitymatch_safe.py}"
SMOKE_ENTRY="${SMOKE_ENTRY:-test_utilitymatch_safe_smoke.py}"
RUN_LOG_PREFIX="${RUN_LOG_PREFIX:-utilitymatch_safe}"
TEST_SCRIPT="${TEST_SCRIPT:-test_utilitymatch_safe_5090.sh}"
METHOD_DESCRIPTION="${METHOD_DESCRIPTION:-selected strong loss active iff signed utility > 0}"
TRACE_NAME="${TRACE_NAME:-utility_gate_trace.csv}"
EXP_NAME="${EXP_NAME:-${METHOD_NAME}_${RUN_TAG}}"
EXPERIMENT_DIR="${MODEL_ROOT}/${EXP_NAME}_7_labeled"
OUTPUT_DIR="${EXPERIMENT_DIR}/self_train/unet"
LAST_RUN_FILE="${LAST_RUN_FILE:-${SERVER_LOG_DIR}/last_utilitymatch_safe_run.txt}"
UNIMATCH_FOLDER_NAME="UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled"
ORIGINAL_UNIMATCH_DIR="${ORIGINAL_UNIMATCH_DIR:-}"

mkdir -p "${SERVER_LOG_DIR}"
[[ "${SEED}" == "1337" ]] || {
    echo "Locked SafeUtilityMatch requires seed 1337, got ${SEED}." >&2
    exit 2
}

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
if [[ -z "${ORIGINAL_UNIMATCH_DIR}" ||
      "$(basename -- "${ORIGINAL_UNIMATCH_DIR%/}")" != "${UNIMATCH_FOLDER_NAME}" ]]; then
    echo "Set ORIGINAL_UNIMATCH_DIR to the exact original folder:" >&2
    echo "  ${UNIMATCH_FOLDER_NAME}" >&2
    echo "A new UtilityMatch/PreTrain/TLP weight is not accepted." >&2
    exit 2
fi

PRETRAIN_CHECKPOINT="${ORIGINAL_UNIMATCH_DIR}/pre_train/unet/unet_best_model.pth"
ANCHOR_CHECKPOINT="${ORIGINAL_UNIMATCH_DIR}/self_train/unet/unet_best_model.pth"
[[ -f "${PRETRAIN_CHECKPOINT}" ]] || {
    echo "Missing original PreTrain weight: ${PRETRAIN_CHECKPOINT}" >&2
    exit 2
}
[[ -f "${ANCHOR_CHECKPOINT}" ]] || {
    echo "Missing original UniMatch reference weight: ${ANCHOR_CHECKPOINT}" >&2
    exit 2
}
[[ ! -e "${EXPERIMENT_DIR}" ]] || {
    echo "Refusing to overwrite: ${EXPERIMENT_DIR}" >&2
    exit 3
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

if [[ "${DETACH}" == "1" && "${_UTILITY_SAFE_DETACHED:-0}" != "1" ]]; then
    DETACHED_LOG="${SERVER_LOG_DIR}/${RUN_LOG_PREFIX}_${RUN_TAG}_nohup.log"
    nohup env _UTILITY_SAFE_DETACHED=1 DETACH=0 RUN_TAG="${RUN_TAG}" \
        EXP_NAME="${EXP_NAME}" DATA_ROOT="${DATA_ROOT}" GPU="${GPU}" \
        SEED="${SEED}" REQUIRE_5090="${REQUIRE_5090}" \
        ORIGINAL_UNIMATCH_DIR="${ORIGINAL_UNIMATCH_DIR}" \
        METHOD_NAME="${METHOD_NAME}" TRAIN_ENTRY="${TRAIN_ENTRY}" \
        SMOKE_ENTRY="${SMOKE_ENTRY}" RUN_LOG_PREFIX="${RUN_LOG_PREFIX}" \
        TEST_SCRIPT="${TEST_SCRIPT}" METHOD_DESCRIPTION="${METHOD_DESCRIPTION}" \
        TRACE_NAME="${TRACE_NAME}" \
        LAST_RUN_FILE="${LAST_RUN_FILE}" \
        bash "${SCRIPT_PATH}" "$@" >"${DETACHED_LOG}" 2>&1 </dev/null &
    printf 'Started %s: PID=%s\nLog: %s\n' "${METHOD_NAME}" "$!" "${DETACHED_LOG}"
    exit 0
fi

if command -v flock >/dev/null 2>&1; then
    exec 9>"${SERVER_LOG_DIR}/.unimatch_training.lock"
    flock -n 9 || {
        echo "Another Baseline/UniMatch training process is already running." >&2
        exit 3
    }
fi

for path in \
    "${CODE_DIR}/${TRAIN_ENTRY}" \
    "${CODE_DIR}/utilitymatch_safe.py" \
    "${CODE_DIR}/${SMOKE_ENTRY}" \
    "${CODE_DIR}/train_utilitymatch_safe.py" \
    "${CODE_DIR}/test_utilitymatch_safe_smoke.py" \
    "${CODE_DIR}/train_utilitymatch.py" \
    "${CODE_DIR}/utilitymatch.py" \
    "${CODE_DIR}/train_unimatch.py"; do
    [[ -f "${path}" ]] || { echo "Missing required code: ${path}" >&2; exit 2; }
done

cd "${CODE_DIR}"
echo "Checking exact 35/5/10 PROMISE12 source, original weights, code, and GPU..."
CUDA_VISIBLE_DEVICES="${GPU}" CODE_DIR="${CODE_DIR}" DATA_ROOT="${DATA_ROOT}" \
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT}" ANCHOR_CHECKPOINT="${ANCHOR_CHECKPOINT}" \
REQUIRE_5090="${REQUIRE_5090}" TRAIN_ENTRY="${TRAIN_ENTRY}" \
SMOKE_ENTRY="${SMOKE_ENTRY}" "${PYTHON_LAUNCHER[@]}" - <<'PY'
import hashlib
import importlib
import os
from collections import Counter
from pathlib import Path

for name in ("h5py", "medpy", "numpy", "scipy", "skimage", "tensorboardX",
             "torch", "torchvision", "tqdm"):
    importlib.import_module(name)
import h5py
import torch

code = Path(os.environ["CODE_DIR"])
data = Path(os.environ["DATA_ROOT"])
names = {"train_utilitymatch_safe.py", "utilitymatch_safe.py",
         "test_utilitymatch_safe_smoke.py", "train_utilitymatch.py",
         "utilitymatch.py", "train_unimatch.py",
         os.environ["TRAIN_ENTRY"], os.environ["SMOKE_ENTRY"]}
for name in sorted(names):
    path = code / name
    compile(path.read_text(encoding="utf-8"), str(path), "exec")

def lines(name):
    path = data / name
    if not path.is_file():
        raise SystemExit(f"Missing fixed list: {path}")
    return [x.strip().split(".")[0] for x in path.read_text().splitlines() if x.strip()]

train, val, test, slices = (lines("train.list"), lines("val.list"),
                            lines("test.list"), lines("train_slices.list"))
all_slices = lines("all_slices.list")
if (len(train), len(val), len(test), len(slices)) != (35, 5, 10, 940):
    raise SystemExit("Expected train/val/test/slices=35/5/10/940, got "
                     f"{len(train)}/{len(val)}/{len(test)}/{len(slices)}")
if len(set(train + val + test)) != 50:
    raise SystemExit("Fixed train/val/test lists overlap")
if len(all_slices) != 1377:
    raise SystemExit(f"Expected 1377 all-slice entries, got {len(all_slices)}")

expected_list_bundle = "e0bd27c2d40977ab97b3059fecff965f1f270b7139ad01ad679b85e86ccf41e3"
list_bundle = hashlib.sha256()
for name in ("train.list", "val.list", "test.list", "train_slices.list", "all_slices.list"):
    content = (data / name).read_bytes().replace(b"\r\n", b"\n").rstrip(b"\n")
    list_bundle.update(name.encode("utf-8"))
    list_bundle.update(b"\0")
    list_bundle.update(hashlib.sha256(content).digest())
    list_bundle.update(b"\n")
if list_bundle.hexdigest() != expected_list_bundle:
    raise SystemExit(
        "Dataset list bundle differs from KeWu123/data@"
        "e58bb4db80006862a92e977b8525f513478c631a")
labeled = train[:7]
counts = Counter(case for item in slices for case in labeled
                 if item.startswith(case + "_slice"))
if sum(counts.values()) != 191 or any(counts[case] == 0 for case in labeled):
    raise SystemExit(f"Expected first7=191 slices, got {dict(counts)}")
prefixes = tuple(case + "_slice" for case in labeled)
if not all(item.startswith(prefixes) for item in slices[:191]):
    raise SystemExit("The first 191 list entries are not exactly the first 7 cases")
if any(item.startswith(prefixes) for item in slices[191:]):
    raise SystemExit("A labeled-case slice occurs after list index 190")

verified_h5 = []

def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()

for item in slices:
    path = data / "data" / "slices" / f"{item}.h5"
    if not path.is_file():
        raise SystemExit(f"Missing listed slice: {path}")
    with h5py.File(path, "r") as handle:
        if not {"image", "label"}.issubset(handle.keys()):
            raise SystemExit(f"Invalid slice datasets: {path}")
    verified_h5.append((f"data/slices/{item}.h5", file_digest(path)))
for case in val + test:
    path = data / "data" / f"{case}.h5"
    if not path.is_file():
        raise SystemExit(f"Missing validation/test volume: {path}")
    with h5py.File(path, "r") as handle:
        if not {"image", "label"}.issubset(handle.keys()):
            raise SystemExit(f"Invalid volume datasets: {path}")
    verified_h5.append((f"data/{case}.h5", file_digest(path)))

expected_h5_bundle = "332e491c9022a4542be148c770c56f00f5817a2141cce5d5795ebc91cbb6fe73"
h5_bundle = hashlib.sha256()
for relative, digest in sorted(verified_h5):
    h5_bundle.update(relative.encode("utf-8"))
    h5_bundle.update(b"\0")
    h5_bundle.update(digest)
    h5_bundle.update(b"\n")
if h5_bundle.hexdigest() != expected_h5_bundle:
    raise SystemExit(
        "The 940 training slices plus 15 validation/test volumes differ from "
        "KeWu123/data@e58bb4db80006862a92e977b8525f513478c631a")

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

pretrain = os.environ["PRETRAIN_CHECKPOINT"]
try:
    state = torch.load(pretrain, map_location="cpu", weights_only=False)
except TypeError:
    state = torch.load(pretrain, map_location="cpu")
if not isinstance(state, dict) or not {"net", "opt"}.issubset(state):
    raise SystemExit("Original PreTrain checkpoint must contain net and opt")
print(f"PROMISE12 root: {data.resolve()}")
print("Dataset identity: KeWu123/data@e58bb4db80006862a92e977b8525f513478c631a")
print(f"List bundle SHA256: {list_bundle.hexdigest()}")
print(f"Used-H5 bundle SHA256: {h5_bundle.hexdigest()} ({len(verified_h5)} files)")
print(f"Fixed labeled cases/slices: {dict(counts)}")
print("Fixed sampler: labeled=191, unlabeled=749, batches=15")
print(f"Original PreTrain SHA256: {sha256(pretrain)}")
print(f"Original UniMatch SHA256: {sha256(os.environ['ANCHOR_CHECKPOINT'])}")
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
gpu = torch.cuda.get_device_name(0)
print(f"PyTorch={torch.__version__}, CUDA={torch.version.cuda}, GPU={gpu}")
if os.environ.get("REQUIRE_5090", "1") == "1" and "5090" not in gpu:
    raise SystemExit(f"Expected RTX 5090, got {gpu}")
PY

"${PYTHON_LAUNCHER[@]}" "${SMOKE_ENTRY}"
printf '%s\n' "${EXPERIMENT_DIR}" >"${LAST_RUN_FILE}"

echo "======================================================================"
echo "${METHOD_NAME} full training"
echo "Data:             ${DATA_ROOT}"
echo "Initial weight:   ${PRETRAIN_CHECKPOINT}"
echo "Reference anchor: ${ANCHOR_CHECKPOINT}"
echo "Output:           ${OUTPUT_DIR}"
echo "Fixed protocol:   35/5/10, first7=191, seed1337, warmup1000, self30000"
echo "Method:           ${METHOD_DESCRIPTION}"
echo "Loss:             .25*g1*Ls1 + .25*g2*Ls2 + .50*Lfp (no renormalization)"
echo "Training entry:   ${CODE_DIR}/${TRAIN_ENTRY}"
echo "Training SHA256:  $(sha256sum "${CODE_DIR}/${TRAIN_ENTRY}" | awk '{print $1}')"
echo "Terminal:         tqdm and validation remain visible"
echo "======================================================================"

RUN_LOG="${SERVER_LOG_DIR}/${RUN_LOG_PREFIX}_${RUN_TAG}.log"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PYTHON_LAUNCHER[@]}" "${TRAIN_ENTRY}" \
        --root_path "${DATA_ROOT}" \
        --pretrained_model_path "${PRETRAIN_CHECKPOINT}" \
        --anchor_checkpoint "${ANCHOR_CHECKPOINT}" \
        --output_dir "${OUTPUT_DIR}" \
        "$@" 2>&1 | tee "${RUN_LOG}"

echo "${METHOD_NAME} training completed"
echo "Log:        ${RUN_LOG}"
echo "Best model: ${OUTPUT_DIR}/unet_best_model.pth"
echo "Method trace: ${OUTPUT_DIR}/${TRACE_NAME}"
echo "Test:       bash '${BASELINE_ROOT}/${TEST_SCRIPT}'"
