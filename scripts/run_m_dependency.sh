#!/usr/bin/env bash
set -euo pipefail

# m_dependency: med growth vs number of points (n)
# Usage: run_m_dependency.sh [gd|sgd]

TRAINER="${1:-${TRAINER:-gd}}"
if [[ "${TRAINER}" != "gd" && "${TRAINER}" != "sgd" ]]; then
  echo "Usage: $0 [gd|sgd]" >&2
  exit 1
fi

METRICS=(inner_product l2 cosine)
K="${K:-2}"

# Default n-lists differ by trainer
if [[ "${TRAINER}" == "gd" ]]; then
  DEFAULT_N_LIST=(8 16 32 64 128 256)
  NUM_EPOCHS=${NUM_EPOCHS:-2000}
  PATIENCE=${PATIENCE:-200}
  LEARNING_RATE=${LEARNING_RATE:-1e-2}
else
  DEFAULT_N_LIST=(8 16 32 64 128 256 512 1024)
  NUM_EPOCHS=${NUM_EPOCHS:-1000}
  PATIENCE=${PATIENCE:-500}
  LEARNING_RATE=${LEARNING_RATE:-1e-2}
fi

# Allow override via env var N_LIST
if [[ -n "${N_LIST:-}" ]]; then
  # shellcheck disable=SC2206
  N_LIST_ARR=(${N_LIST})
else
  N_LIST_ARR=(${DEFAULT_N_LIST[@]})
fi

# SGD knobs (no-op for GD but accepted by entry script)
SGD_BATCH_SIZE=${SGD_BATCH_SIZE:-256}
SGD_NUM_SAMPLES=${SGD_NUM_SAMPLES:-50000}
SGD_WORKERS=${SGD_WORKERS:-0}
SGD_PATIENCE=${SGD_PATIENCE:-500}
SGD_LR=${SGD_LR:-1e-2}
SGD_MAX_STEPS=${SGD_MAX_STEPS:-}
SGD_NEGATIVE_SAMPLING=${SGD_NEGATIVE_SAMPLING:-}

RUN_ID="$(date +%Y%m%d_%H%M%S)"
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

for metric in "${METRICS[@]}"; do
  OUT_DIR="/workspace/results/${TRAINER}/${metric}/m_dependency/k_${K}/${RUN_ID}"
  mkdir -p "${OUT_DIR}"
  pushd "${OUT_DIR}" >/dev/null

  echo "[INFO] m_dependency run" | tee -a run.log
  echo "trainer=${TRAINER}" | tee -a run.log
  echo "metric=${metric}" | tee -a run.log
  echo "k=${K}" | tee -a run.log
  echo "n_values=${N_LIST_ARR[*]}" | tee -a run.log
  echo "timestamp=${RUN_ID}" | tee -a run.log

  ARGS=(
    --trainer "${TRAINER}"
    --k "${K}"
    --scoring_function "${metric}"
    --n_values ${N_LIST_ARR[@]}
    --num_epochs ${NUM_EPOCHS}
    --patience ${PATIENCE}
    --learning_rate ${LEARNING_RATE}
    --sgd_batch_size ${SGD_BATCH_SIZE}
    --sgd_num_samples ${SGD_NUM_SAMPLES}
    --sgd_workers ${SGD_WORKERS}
    --sgd_patience ${SGD_PATIENCE}
    --sgd_lr ${SGD_LR}
  )
  if [[ -n "${SGD_MAX_STEPS}" ]]; then ARGS+=(--sgd_max_steps "${SGD_MAX_STEPS}"); fi
  if [[ -n "${SGD_NEGATIVE_SAMPLING}" ]]; then ARGS+=(--sgd_negative_sampling "${SGD_NEGATIVE_SAMPLING}"); fi

  python -u /workspace/main.py "${ARGS[@]}" |& tee -a run.log
  echo "[DONE] m_dependency ${TRAINER} ${metric} saved to ${OUT_DIR}" | tee -a run.log
  popd >/dev/null

done
