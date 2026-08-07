#!/usr/bin/env bash
set -Eeuo pipefail

# Sequential PROMISE12 baseline repetitions on one RTX 5090.
# Each seed has an independent experiment folder; completed seeds are reusable.

BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${BASELINE_ROOT}/$(basename -- "${BASH_SOURCE[0]}")"
TRAIN_SCRIPT="${BASELINE_ROOT}/run_baseline_5090.sh"
TEST_SCRIPT="${BASELINE_ROOT}/test_and_quantify_baseline_5090.sh"
MODEL_ROOT="${BASELINE_ROOT}/model"
SERVER_LOG_DIR="${BASELINE_ROOT}/server_logs"

SEEDS="${SEEDS:-1337 2024 3407}"
EXP_PREFIX="${EXP_PREFIX:-MT_PROMISE12_baseline}"
LABELNUM="${LABELNUM:-7}"
DETACH="${DETACH:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

mkdir -p "${SERVER_LOG_DIR}"

if [[ "${DETACH}" == "1" && "${_BASELINE_MULTISEED_DETACHED:-0}" != "1" ]]; then
    SERVER_LOG="${SERVER_LOG_DIR}/multiseed_$(date +%Y%m%d_%H%M%S).log"
    nohup env _BASELINE_MULTISEED_DETACHED=1 DETACH=0 \
        bash "${SCRIPT_PATH}" >"${SERVER_LOG}" 2>&1 </dev/null &
    printf 'Started multi-seed run in background: PID=%s\nLog: %s\n' "$!" "${SERVER_LOG}"
    exit 0
fi

[[ -f "${TRAIN_SCRIPT}" ]] || { echo "Missing ${TRAIN_SCRIPT}" >&2; exit 2; }
[[ -f "${TEST_SCRIPT}" ]] || { echo "Missing ${TEST_SCRIPT}" >&2; exit 2; }
[[ "${LABELNUM}" == "7" ]] || {
    echo "This PROMISE12 baseline is configured for LABELNUM=7, got ${LABELNUM}." >&2
    exit 2
}

read -r -a SEED_ARRAY <<< "${SEEDS}"
if (( ${#SEED_ARRAY[@]} < 2 )); then
    echo "Provide at least two seeds, for example: SEEDS='1337 2024 3407'" >&2
    exit 2
fi
for seed in "${SEED_ARRAY[@]}"; do
    [[ "${seed}" =~ ^[0-9]+$ ]] || {
        echo "Invalid integer seed: ${seed}" >&2
        exit 2
    }
done

echo "============================================================"
echo "PROMISE12 baseline multi-seed experiment"
echo "Seeds: ${SEED_ARRAY[*]}"
echo "Each run: 7 labeled cases (dynamic slice count), pre=1000, self=5000, warmup=1000"
echo "============================================================"

for seed in "${SEED_ARRAY[@]}"; do
    exp_name="${EXP_PREFIX}_seed${seed}"
    exp_dir="${MODEL_ROOT}/${exp_name}_${LABELNUM}_labeled"
    pre_checkpoint="${exp_dir}/pre_train/unet/unet_best_model.pth"
    self_checkpoint="${exp_dir}/self_train/unet/unet_best_model.pth"
    metric_table="${exp_dir}/metric_table.csv"

    echo
    echo "========== seed=${seed}, exp=${exp_name} =========="
    if [[ "${SKIP_COMPLETED}" == "1" && -f "${metric_table}" ]]; then
        echo "Completed result exists; skipping training and testing: ${metric_table}"
        continue
    fi

    if [[ -f "${pre_checkpoint}" && -f "${self_checkpoint}" ]]; then
        echo "Both checkpoints exist; skipping training and running evaluation."
    else
        if find "${exp_dir}" -type f -name '*.pth' -print -quit 2>/dev/null | grep -q .; then
            echo "Incomplete checkpoint set exists in ${exp_dir}." >&2
            echo "The original baseline cannot resume exactly; move this seed directory and rerun." >&2
            exit 3
        fi
        SEED="${seed}" EXP_NAME="${exp_name}" DETACH=0 \
            bash "${TRAIN_SCRIPT}"
    fi

    EXP_NAME="${exp_name}" LABELNUM="${LABELNUM}" \
        bash "${TEST_SCRIPT}"
done

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
    echo "No Python found for metric aggregation." >&2
    exit 2
fi

MODEL_ROOT="${MODEL_ROOT}" EXP_PREFIX="${EXP_PREFIX}" LABELNUM="${LABELNUM}" \
SEEDS="${SEED_ARRAY[*]}" "${PYTHON_LAUNCHER[@]}" - <<'PY'
import csv
import os
import statistics
from pathlib import Path

model_root = Path(os.environ["MODEL_ROOT"])
prefix = os.environ["EXP_PREFIX"]
labelnum = os.environ["LABELNUM"]
seeds = os.environ["SEEDS"].split()
output_dir = model_root / f"{prefix}_multiseed_{labelnum}_labeled"
output_dir.mkdir(parents=True, exist_ok=True)

rows = []
for seed in seeds:
    source = model_root / f"{prefix}_seed{seed}_{labelnum}_labeled" / "metric_table.csv"
    if not source.is_file():
        raise SystemExit(f"Missing seed metric table: {source}")
    with source.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "seed": seed,
                    "stage": row["stage"],
                    "dice": float(row["dice"]),
                    "jaccard": float(row["jaccard"]),
                    "hd95": float(row["hd95"]),
                    "asd": float(row["asd"]),
                }
            )

per_seed_path = output_dir / "metrics_by_seed.csv"
with per_seed_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=("seed", "stage", "dice", "jaccard", "hd95", "asd"),
    )
    writer.writeheader()
    writer.writerows(rows)

metrics = ("dice", "jaccard", "hd95", "asd")
stages = ("pre_train", "self_train")
summary = []
for stage in stages:
    stage_rows = [row for row in rows if row["stage"] == stage]
    if len(stage_rows) != len(seeds):
        raise SystemExit(
            f"Expected {len(seeds)} rows for {stage}, got {len(stage_rows)}"
        )
    item = {"stage": stage, "runs": len(stage_rows)}
    for metric in metrics:
        values = [row[metric] for row in stage_rows]
        item[f"{metric}_mean"] = statistics.mean(values)
        item[f"{metric}_std"] = statistics.stdev(values)
    summary.append(item)

fields = ["stage", "runs"]
for metric in metrics:
    fields.extend((f"{metric}_mean", f"{metric}_std"))
summary_path = output_dir / "metric_mean_std.csv"
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary)

lines = [
    "| Stage | Runs | Dice | Jaccard | HD95 | ASD |",
    "|---|---:|---:|---:|---:|---:|",
]
for row in summary:
    lines.append(
        "| {stage} | {runs} | {dice_mean:.6f} +/- {dice_std:.6f} | "
        "{jaccard_mean:.6f} +/- {jaccard_std:.6f} | "
        "{hd95_mean:.6f} +/- {hd95_std:.6f} | "
        "{asd_mean:.6f} +/- {asd_std:.6f} |".format(**row)
    )
markdown = "\n".join(lines) + "\n"
markdown_path = output_dir / "metric_mean_std.md"
markdown_path.write_text(markdown, encoding="utf-8")

print("\n========== Multi-seed mean +/- std ==========\n")
print(markdown, end="")
print(f"\nPer-seed CSV: {per_seed_path}")
print(f"Mean/std CSV: {summary_path}")
print(f"Markdown: {markdown_path}")
PY
