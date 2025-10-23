#!/usr/bin/env bash
set -euo pipefail

# Goal 2 (GD): Increase k at moderate n to study med vs k.
# Results stored in results/gd/<metric>/goal2_k_sweep/<timestamp>/

METRICS=(inner_product l2 cosine)
# Choose moderate n where full-batch GD is feasible
N=${N:-128}
# Example k sweep: from 2 up to, say, 10 (and <= n)
K_LIST=(${K_LIST:-2 3 4 5 6 7 8 9 10})

NUM_EPOCHS=${NUM_EPOCHS:-2000}
PATIENCE=${PATIENCE:-200}
LEARNING_RATE=${LEARNING_RATE:-1e-2}

RUN_ID="$(date +%Y%m%d_%H%M%S)"

for metric in "${METRICS[@]}"; do
  OUT_BASE="/workspace/results/gd/${metric}/goal2_k_sweep"
  OUT_DIR="${OUT_BASE}/${RUN_ID}"
  mkdir -p "${OUT_DIR}"

  pushd "${OUT_DIR}" >/dev/null
  export PYTHONPATH="/workspace:${PYTHONPATH:-}"

  echo "[INFO] Starting GD Goal 2 run" | tee -a run.log
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
    python -u /workspace/main.py \
      --trainer gd \
      --k "${k}" \
      --scoring_function "${metric}" \
      --n_values "${N}" \
      --num_epochs ${NUM_EPOCHS} \
      --patience ${PATIENCE} \
      --learning_rate ${LEARNING_RATE} \
      |& tee -a "k_${k}.log"
  done

  popd >/dev/null

echo "[DONE] GD Goal 2 for metric=${metric} saved to ${OUT_DIR}" | tee -a "${OUT_DIR}/run.log"
done
