#!/usr/bin/env bash
set -euo pipefail

# Run centroid-embedding MED experiment: find minimal embedding dimension
# via GD/SGD optimization on random point configurations.
#
# Usage: ./run_medc.sh
#        METRIC=l2 K=3 N_LIST="8 16 32 64" ./run_medc.sh
#        TRAINER=sgd SGD_LR=1e-3 ./run_medc.sh
#
# Results written to results/<trainer>/<metric>/<rundir>/<timestamp>/
#   - results.json        : unified format (see CLAUDE.md Result protocol)
#   - config.json         : run configuration + environment info

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_ROOT}/results"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

K_LIST="${K_LIST:-2}"
METRIC="${METRIC:-inner_product}"
TRAINER="${TRAINER:-gd}"
N_LIST="${N_LIST:-8 16 32 64 128 256 512 1024}"

NUM_EPOCHS="${NUM_EPOCHS:-1000}"
PATIENCE="${PATIENCE:-1000}"
LEARNING_RATE="${LEARNING_RATE:-1}"

# SGD-specific
SGD_BATCH_SIZE="${SGD_BATCH_SIZE:-256}"
SGD_NUM_SAMPLES="${SGD_NUM_SAMPLES:-50000}"
SGD_LR="${SGD_LR:-1e-2}"
SGD_PATIENCE="${SGD_PATIENCE:-500}"
SGD_WORKERS="${SGD_WORKERS:-0}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${RESULTS_DIR}/${TRAINER}/${METRIC}/grid/${RUN_ID}"
mkdir -p "${OUT_DIR}"
pushd "${OUT_DIR}" >/dev/null

echo "[INFO] MED-C (centroid embedding) run" | tee run.log
echo "  trainer    = ${TRAINER}" | tee -a run.log
echo "  metric     = ${METRIC}" | tee -a run.log
echo "  k_values   = ${K_LIST}" | tee -a run.log
echo "  n_values   = ${N_LIST}" | tee -a run.log
echo "  num_epochs = ${NUM_EPOCHS}" | tee -a run.log
echo "  patience   = ${PATIENCE}" | tee -a run.log
echo "  lr         = ${LEARNING_RATE}" | tee -a run.log
echo "  timestamp  = ${RUN_ID}" | tee -a run.log
echo "  git_commit = $(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo none)" | tee -a run.log

ARGS=(
  --trainer "${TRAINER}"
  --k_values ${K_LIST}
  --scoring_function "${METRIC}"
  --n_values ${N_LIST}
  --num_epochs ${NUM_EPOCHS}
  --patience ${PATIENCE}
  --learning_rate ${LEARNING_RATE}
)

if [ "${TRAINER}" = "sgd" ]; then
  ARGS+=(
    --sgd_batch_size ${SGD_BATCH_SIZE}
    --sgd_num_samples ${SGD_NUM_SAMPLES}
    --sgd_lr ${SGD_LR}
    --sgd_patience ${SGD_PATIENCE}
    --sgd_workers ${SGD_WORKERS}
  )
fi

echo "[CMD] python -m med.mean_embedding.cli ${ARGS[*]}" | tee -a run.log
python -u -m med.mean_embedding.cli "${ARGS[@]}" 2>&1 | tee -a run.log

echo "[DONE] Results saved to ${OUT_DIR}" | tee -a run.log
popd >/dev/null
