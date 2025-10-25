#!/usr/bin/env bash
set -euo pipefail

# k_dependency: med growth vs k at moderate n
# Usage: run_k_dependency.sh [gd|sgd]

TRAINER="${1:-${TRAINER:-gd}}"
if [[ "${TRAINER}" != "gd" && "${TRAINER}" != "sgd" ]]; then
  echo "Usage: $0 [gd|sgd]" >&2
  exit 1
fi

METRICS=(inner_product l2 cosine)

# Defaults differ by trainer; override via env as needed
if [[ "${TRAINER}" == "gd" ]]; then
  N=${N:-128}
  DEFAULT_K_LIST=(2 3 4 5 6 7 8 9 10)
  NUM_EPOCHS=${NUM_EPOCHS:-2000}
  PATIENCE=${PATIENCE:-200}
  LEARNING_RATE=${LEARNING_RATE:-1e-2}
else
  N=${N:-512}
  DEFAULT_K_LIST=(2 3 4 5 6 7 8 9 10 12 16)
  NUM_EPOCHS=${NUM_EPOCHS:-1000}
  PATIENCE=${PATIENCE:-500}
  LEARNING_RATE=${LEARNING_RATE:-1e-2}
fi

# Allow override via env var K_LIST
if [[ -n "${K_LIST:-}" ]]; then
  # shellcheck disable=SC2206
  K_LIST_ARR=(${K_LIST})
else
  K_LIST_ARR=(${DEFAULT_K_LIST[@]})
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
  OUT_DIR="/workspace/results/${TRAINER}/${metric}/k_dependency/n_${N}/${RUN_ID}"
  mkdir -p "${OUT_DIR}"
  pushd "${OUT_DIR}" >/dev/null

  echo "[INFO] k_dependency run" | tee -a run.log
  echo "trainer=${TRAINER}" | tee -a run.log
  echo "metric=${metric}" | tee -a run.log
  echo "n=${N}" | tee -a run.log
  echo "k_values=${K_LIST_ARR[*]}" | tee -a run.log
  echo "timestamp=${RUN_ID}" | tee -a run.log
  echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo none)" | tee -a run.log
  echo "hostname=$(hostname)" | tee -a run.log
  echo "python=$(python -V 2>&1)" | tee -a run.log

  for k in "${K_LIST_ARR[@]}"; do
    if (( k > N )); then
      echo "[WARN] Skipping k=${k} > n=${N}" | tee -a run.log
      continue
    fi

    echo "--- Running k=${k} ---" | tee -a run.log
    ARGS=(
      --trainer "${TRAINER}"
      --k "${k}"
      --scoring_function "${metric}"
      --n_values "${N}"
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

    echo "[CMD] python -u /workspace/main.py ${ARGS[*]}" | tee -a "k_${k}.log"
    /usr/bin/env time -f "[TIME] elapsed=%E user=%U sys=%S maxrss=%MKB" \
      python -u /workspace/main.py "${ARGS[@]}" |& tee -a "k_${k}.log"
  done

  echo "[DONE] k_dependency ${TRAINER} ${metric} saved to ${OUT_DIR}" | tee -a run.log
  popd >/dev/null

done
