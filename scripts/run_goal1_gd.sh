#!/usr/bin/env bash
set -euo pipefail

# Goal 1 (GD): For k=2, study med growth with number of points (n),
# up to a safe upper bound for full-batch GD.
# Results will be stored in results/gd/<metric>/goal1_k2/<timestamp>/

METRICS=(inner_product l2 cosine)
# Conservative n values for GD (avoid OOM for large n)
N_LIST=(8 16 32 64 128 256 512)

RUN_ID="$(date +%Y%m%d_%H%M%S)"

for metric in "${METRICS[@]}"; do
  OUT_BASE="/workspace/results/gd/${metric}/goal1_k2"
  OUT_DIR="${OUT_BASE}/${RUN_ID}"
  mkdir -p "${OUT_DIR}"

  pushd "${OUT_DIR}" >/dev/null
  export PYTHONPATH="/workspace:${PYTHONPATH:-}"

  echo "[INFO] Starting GD Goal 1 run" | tee -a run.log
  echo "metric=${metric}" | tee -a run.log
  echo "k=2" | tee -a run.log
  echo "n_values=${N_LIST[*]}" | tee -a run.log
  echo "timestamp=${RUN_ID}" | tee -a run.log

  python -u /workspace/main.py \
    --trainer gd \
    --k 2 \
    --scoring_function "${metric}" \
    --n_values ${N_LIST[@]} \
    --num_epochs 2000 \
    --patience 200 \
    --learning_rate 1e-2 \
    |& tee -a run.log

  popd >/dev/null

echo "[DONE] GD Goal 1 for metric=${metric} saved to ${OUT_DIR}" | tee -a "${OUT_DIR}/run.log"
done
