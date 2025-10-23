#!/usr/bin/env bash
set -euo pipefail

# Goal 2 (SGD): Increase k at moderate n to study med vs k.
# Results stored in results/sgd/<metric>/goal2_k_sweep/<timestamp>/

METRICS=(inner_product l2 cosine)
N=${N:-512}
K_LIST=(${K_LIST:-2 3 4 5 6 7 8 9 10 12 16})

# SGD knobs (overridable via env)
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
  OUT_BASE="/workspace/results/sgd/${metric}/goal2_k_sweep"
  OUT_DIR="${OUT_BASE}/${RUN_ID}"
  mkdir -p "${OUT_DIR}"

  pushd "${OUT_DIR}" >/dev/null
  export PYTHONPATH="/workspace:${PYTHONPATH:-}"

  echo "[INFO] Starting SGD Goal 2 run" | tee -a run.log
  echo "metric=${metric}" | tee -a run.log
  echo "n=${N}" | tee -a run.log
  echo "k_values=${K_LIST[*]}" | tee -a run.log
  echo "timestamp=${RUN_ID}" | tee -a run.log

  for k in "${K_LIST[@]}"; do
    if (( k > N )); then
      echo "[WARN] Skipping k=${k} > n=${N}" | tee -a run.log
      continue
    fi

    echo "--- Running k=${k} ---" | tee -a run.log

    ARGS=(
      --trainer sgd
      --k "${k}"
      --scoring_function "${metric}"
      --n_values "${N}"
      --num_epochs ${NUM_EPOCHS}
      --sgd_batch_size ${SGD_BATCH_SIZE}
      --sgd_num_samples ${SGD_NUM_SAMPLES}
      --sgd_workers ${SGD_WORKERS}
      --sgd_patience ${SGD_PATIENCE}
      --sgd_lr ${SGD_LR}
    )
    if [[ -n "${SGD_MAX_STEPS}" ]]; then ARGS+=(--sgd_max_steps "${SGD_MAX_STEPS}"); fi
    if [[ -n "${SGD_NEGATIVE_SAMPLING}" ]]; then ARGS+=(--sgd_negative_sampling "${SGD_NEGATIVE_SAMPLING}"); fi

    python -u /workspace/main.py "${ARGS[@]}" |& tee -a "k_${k}.log"
  done

  popd >/dev/null

echo "[DONE] SGD Goal 2 for metric=${metric} saved to ${OUT_DIR}" | tee -a "${OUT_DIR}/run.log"
done
