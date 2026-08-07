#!/usr/bin/env bash
set -Eeuo pipefail

# Full PROMISE12 Uni-MedSAM/SAMatch experiment.
#
# Match initialization keeps this project's UniMatch protocol:
#   supervised match_pre=1000, then paper Match warm-up=30000,
#   labelnum=7 (191 slices), seed=1337.
# The complete SAMatch method then adds:
#   LiteMedSAM labeled warm-up=30000, interactive training=30000.

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${PROJECT_ROOT}/code"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/PROMISE12_h5}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/model}"
SERVER_LOG_DIR="${PROJECT_ROOT}/server_logs"
EXP_NAME="${EXP_NAME:-MT_PROMISE12_UniMatch_SAMatchFull}"
GPU="${GPU:-0}"
SEED="${SEED:-1337}"
LABELNUM="${LABELNUM:-7}"
MATCH_PRE_ITERATIONS="${MATCH_PRE_ITERATIONS:-1000}"
MATCH_SELF_ITERATIONS="${MATCH_SELF_ITERATIONS:-30000}"
MEDSAM_WARMUP_ITERATIONS="${MEDSAM_WARMUP_ITERATIONS:-30000}"
INTERACTIVE_ITERATIONS="${INTERACTIVE_ITERATIONS:-30000}"
STAGE="${STAGE:-all}"
REUSE_WARMUP="${REUSE_WARMUP:-0}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
DETACH="${DETACH:-0}"
AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-1}"

SAMATCH_COMMIT="${SAMATCH_COMMIT:-0ab023e643177a8a9dc6f76181c92b52225a71eb}"
SAMATCH_SOURCE_DIR="${SAMATCH_SOURCE_DIR:-${PROJECT_ROOT}/third_party/SAMatch}"
MEDSAM_PRETRAINED="${MEDSAM_PRETRAINED:-${PROJECT_ROOT}/pretrained/lite_medsam.pth}"
MEDSAM_URL="${MEDSAM_URL:-https://huggingface.co/GleghornLab/medsam-vit-b/resolve/main/lite_medsam.pth?download=true}"
MEDSAM_MIRROR_URL="${MEDSAM_MIRROR_URL:-https://hf-mirror.com/GleghornLab/medsam-vit-b/resolve/main/lite_medsam.pth}"
MEDSAM_SHA256="${MEDSAM_SHA256:-79d8c9dca6db4d69d3f905579e5250af05e859fff9c1f543e89a513c3028ce76}"

EXPERIMENT_DIR="${MODEL_ROOT}/${EXP_NAME}_${LABELNUM}_labeled"
SCRIPT_PATH="${PROJECT_ROOT}/$(basename -- "${BASH_SOURCE[0]}")"
mkdir -p "${SERVER_LOG_DIR}"

if [[ "${DETACH}" == "1" && "${_SAMATCH_DETACHED:-0}" != "1" ]]; then
    RUN_LOG="${SERVER_LOG_DIR}/samatch_full_$(date +%Y%m%d_%H%M%S).log"
    nohup env _SAMATCH_DETACHED=1 DETACH=0 \
        bash "${SCRIPT_PATH}" >"${RUN_LOG}" 2>&1 </dev/null &
    printf 'Started full SAMatch in background: PID=%s\nLog: %s\n' \
        "$!" "${RUN_LOG}"
    exit 0
fi

PYTHON_LAUNCHER=()
if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_LAUNCHER=("${PYTHON_BIN}")
elif [[ -n "${CONDA_ENV:-}" ]]; then
    PYTHON_LAUNCHER=(conda run --no-capture-output -n "${CONDA_ENV}" python)
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_LAUNCHER=(python3)
else
    PYTHON_LAUNCHER=(python)
fi

for required in \
    "${CODE_DIR}/train_unimatch.py" \
    "${CODE_DIR}/train_unimatch_samatch_full.py" \
    "${DATA_ROOT}/train_slices.list" \
    "${DATA_ROOT}/val.list" \
    "${DATA_ROOT}/test.list"; do
    [[ -f "${required}" ]] || {
        echo "Missing required file: ${required}" >&2
        exit 2
    }
done

if [[ ! -d "${SAMATCH_SOURCE_DIR}/.git" ]]; then
    command -v git >/dev/null 2>&1 || {
        echo "git is required to obtain the official SAMatch source." >&2
        exit 2
    }
    mkdir -p "$(dirname -- "${SAMATCH_SOURCE_DIR}")"
    git clone https://github.com/apple1986/SAMatch.git "${SAMATCH_SOURCE_DIR}"
fi
git -C "${SAMATCH_SOURCE_DIR}" fetch --quiet origin "${SAMATCH_COMMIT}"
git -C "${SAMATCH_SOURCE_DIR}" checkout --quiet "${SAMATCH_COMMIT}"

checkpoint_is_valid() {
    local checkpoint="$1"
    local actual_sha256
    [[ -s "${checkpoint}" ]] || return 1
    actual_sha256="$(sha256sum "${checkpoint}" | awk '{print $1}')"
    [[ "${actual_sha256}" == "${MEDSAM_SHA256}" ]]
}

download_checkpoint() {
    local url="$1"
    local destination="$2"
    echo "Trying LiteMedSAM source: ${url}"
    if command -v curl >/dev/null 2>&1; then
        curl -fL \
            --retry 10 \
            --retry-all-errors \
            --retry-delay 3 \
            --connect-timeout 30 \
            --max-time 1800 \
            --continue-at - \
            -o "${destination}" \
            "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget \
            --continue \
            --tries=10 \
            --timeout=30 \
            -O "${destination}" \
            "${url}"
    else
        echo "curl or wget is required to download LiteMedSAM." >&2
        return 2
    fi
}

if ! checkpoint_is_valid "${MEDSAM_PRETRAINED}"; then
    mkdir -p "$(dirname -- "${MEDSAM_PRETRAINED}")"
    checkpoint_part="${MEDSAM_PRETRAINED}.part"

    # A previous reset can leave either the final file or .part incomplete.
    # Only a file with the pinned SHA256 is accepted as the real checkpoint.
    if [[ -f "${MEDSAM_PRETRAINED}" ]]; then
        echo "Removing an incomplete/invalid LiteMedSAM checkpoint."
        rm -f -- "${MEDSAM_PRETRAINED}"
    fi

    download_urls=("${MEDSAM_MIRROR_URL}")
    if [[ "${MEDSAM_URL}" != "${MEDSAM_MIRROR_URL}" ]]; then
        download_urls+=("${MEDSAM_URL}")
    fi

    download_succeeded=0
    for download_url in "${download_urls[@]}"; do
        if download_checkpoint "${download_url}" "${checkpoint_part}"; then
            if checkpoint_is_valid "${checkpoint_part}"; then
                mv -f -- "${checkpoint_part}" "${MEDSAM_PRETRAINED}"
                download_succeeded=1
                break
            fi
            echo "Downloaded file failed SHA256 verification; trying the next source." >&2
        else
            echo "Download failed; trying the next source." >&2
        fi
        rm -f -- "${checkpoint_part}"
    done

    if [[ "${download_succeeded}" != "1" ]]; then
        echo "Unable to download a valid LiteMedSAM checkpoint." >&2
        echo "You may copy lite_medsam.pth to: ${MEDSAM_PRETRAINED}" >&2
        echo "Expected SHA256: ${MEDSAM_SHA256}" >&2
        exit 2
    fi
fi

actual_sha256="$(sha256sum "${MEDSAM_PRETRAINED}" | awk '{print $1}')"
echo "LiteMedSAM checkpoint verified: ${actual_sha256}"

if ! "${PYTHON_LAUNCHER[@]}" - <<'PY'
import einops
import timm
PY
then
    if [[ "${AUTO_INSTALL_DEPS}" != "1" ]]; then
        echo "Install SAMatch dependencies: pip install timm einops" >&2
        exit 2
    fi
    "${PYTHON_LAUNCHER[@]}" -m pip install --disable-pip-version-check \
        timm einops
fi

if find "${EXPERIMENT_DIR}" -type f -name '*.pth' -print -quit \
        2>/dev/null | grep -q .; then
    if [[ "${ALLOW_EXISTING}" != "1" && "${REUSE_WARMUP}" != "1" ]]; then
        echo "Existing checkpoints found: ${EXPERIMENT_DIR}" >&2
        echo "Choose another EXP_NAME, or set REUSE_WARMUP=1/ALLOW_EXISTING=1." >&2
        exit 3
    fi
fi

reuse_args=()
if [[ "${REUSE_WARMUP}" == "1" ]]; then
    reuse_args+=(--reuse_warmup)
fi

echo "Project:             ${PROJECT_ROOT}"
echo "Data:                ${DATA_ROOT}"
echo "Output:              ${EXPERIMENT_DIR}"
echo "GPU / seed:          ${GPU} / ${SEED}"
echo "Match protocol:      pre=${MATCH_PRE_ITERATIONS}, self=${MATCH_SELF_ITERATIONS}"
echo "SAMatch protocol:    MedSAM warm-up=${MEDSAM_WARMUP_ITERATIONS}, interactive=${INTERACTIVE_ITERATIONS}"
echo "Official source:     ${SAMATCH_SOURCE_DIR}@${SAMATCH_COMMIT}"
echo "LiteMedSAM weights:  ${MEDSAM_PRETRAINED}"

RUN_LOG="${SERVER_LOG_DIR}/samatch_full_console_$(date +%Y%m%d_%H%M%S).log"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_LAUNCHER[@]}" \
    "${CODE_DIR}/train_unimatch_samatch_full.py" \
    --root_path "${DATA_ROOT}" \
    --output_root "${MODEL_ROOT}" \
    --exp "${EXP_NAME}" \
    --stage "${STAGE}" \
    --seed "${SEED}" \
    --labelnum "${LABELNUM}" \
    --match_pre_iterations "${MATCH_PRE_ITERATIONS}" \
    --match_self_iterations "${MATCH_SELF_ITERATIONS}" \
    --medsam_warmup_iterations "${MEDSAM_WARMUP_ITERATIONS}" \
    --interactive_iterations "${INTERACTIVE_ITERATIONS}" \
    --samatch_source_dir "${SAMATCH_SOURCE_DIR}" \
    --medsam_pretrained "${MEDSAM_PRETRAINED}" \
    "${reuse_args[@]}" "$@" 2>&1 | tee "${RUN_LOG}"

echo "Full SAMatch training completed"
echo "Console log: ${RUN_LOG}"
echo "Run: EXP_NAME=${EXP_NAME} bash test_and_quantify_unimatch_samatch_full_5090.sh"
