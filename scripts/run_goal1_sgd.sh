#!/usr/bin/env bash
set -euo pipefail

# Goal 1 (SGD): For k=2, study med growth with number of points (n),
# up to m=1024 as requested. SGD scales to larger n.
# Results will be stored in results/sgd/<metric>/goal1_k2/<timestamp>/

METRICS=(inner_product l2 cosine)
# n up to 1024
N_LIST=(8 16 32 64 128 256 512 1024)

# SGD defaults are defined in code, but we expose knobs here if needed
SGD_BATCH_SIZE=${SGD_BATCH_SIZE:-256}
SGD_NUM_SAMPLES=${SGD_NUM_SAMPLES:-50000}
SGD_WORKERS=${SGD_WORKERS:-0}
SGD_PATIENCE=${SGD_PATIENCE:-500}
SGD_LR=${SGD_LR:-1e-2}
SGD_MAX_STEPS=${SGD_MAX_STEPS:-}
SGD_NEGATIVE_SAMPLING=${SGD_NEGATIVE_SAMPLING:-}

NUM_EPOCHS=${NUM_EPOCHS:-1000}

RUN_ID="$(date +%Y%m%d_%H%M%S)"

for metric in "${METRICS[@]}"; do
  OUT_BASE="/workspace/results/sgd/${metric}/goal1_k2"
  OUT_DIR="${OUT_BASE}/${RUN_ID}"
  mkdir -p "${OUT_DIR}"

  pushd "${OUT_DIR}" >/dev/null
  export PYTHONPATH="/workspace:${PYTHONPATH:-}"

  echo "[INFO] Starting SGD Goal 1 run" | tee -a run.log
  echo "metric=${metric}" | tee -a run.log
  echo "k=2" | tee -a run.log
  echo "n_values=${N_LIST[*]}" | tee -a run.log
  echo "timestamp=${RUN_ID}" | tee -a run.log

  ARGS=(
    --trainer sgd
    --k 2
    --scoring_function "${metric}"
    --n_values ${N_LIST[@]}
    --num_epochs ${NUM_EPOCHS}
    --sgd_batch_size ${SGD_BATCH_SIZE}
    --sgd_num_samples ${SGD_NUM_SAMPLES}
    --sgd_workers ${SGD_WORKERS}
    --sgd_patience ${SGD_PATIENCE}
    --sgd_lr ${SGD_LR}
  )
  if [[ -n "${SGD_MAX_STEPS}" ]]; then ARGS+=(--sgd_max_steps "${SGD_MAX_STEPS}"); fi
  if [[ -n "${SGD_NEGATIVE_SAMPLING}" ]]; then ARGS+=(--sgd_negative_sampling "${SGD_NEGATIVE_SAMPLING}"); fi

  python -u /workspace/main.py "${ARGS[@]}" |& tee -a run.log

  popd >/dev/null

echo "[DONE] SGD Goal 1 for metric=${metric} saved to ${OUT_DIR}" | tee -a "${OUT_DIR}/run.log"
done
