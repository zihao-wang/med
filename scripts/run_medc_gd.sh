#!/usr/bin/env bash
set -euo pipefail

# Run MED-C experiment: find minimal embedding dimension for k=2
# across varying numbers of objects (n).
#
# Hyperparameters match the paper: Adam, lr=1, 1000 steps.
#
# Usage: ./run_m_dependency.sh
#        METRIC=l2 N_LIST="32 64 128" ./run_m_dependency.sh
#
# Results written to results/gd/<metric>/m_dependency/k_<k>/<timestamp>/
#   - results.json        : unified format (see CLAUDE.md Result protocol)
#   - config.json         : run configuration + environment info
#   - minimal_dem_log.txt : human-readable results log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_ROOT}/results"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

K="${K:-2}"
METRIC="${METRIC:-inner_product}"
N_LIST="${N_LIST:-8 16 32 64 128 256 512 1024}"

# Paper hyperparameters: Adam, lr=1, 1000 steps
NUM_EPOCHS="${NUM_EPOCHS:-1000}"
PATIENCE="${PATIENCE:-1000}"
LEARNING_RATE="${LEARNING_RATE:-1}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${RESULTS_DIR}/gd/${METRIC}/m_dependency/k_${K}/${RUN_ID}"
mkdir -p "${OUT_DIR}"
pushd "${OUT_DIR}" >/dev/null

echo "[INFO] MED-C m_dependency run" | tee run.log
echo "  k         = ${K}" | tee -a run.log
echo "  metric    = ${METRIC}" | tee -a run.log
echo "  n_values  = ${N_LIST}" | tee -a run.log
echo "  num_epochs= ${NUM_EPOCHS}" | tee -a run.log
echo "  patience  = ${PATIENCE}" | tee -a run.log
echo "  lr        = ${LEARNING_RATE}" | tee -a run.log
echo "  timestamp = ${RUN_ID}" | tee -a run.log
echo "  git_commit = $(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo none)" | tee -a run.log

ARGS=(
  --trainer gd
  --k_values "${K}"
  --scoring_function "${METRIC}"
  --n_values ${N_LIST}
  --num_epochs ${NUM_EPOCHS}
  --patience ${PATIENCE}
  --learning_rate ${LEARNING_RATE}
)

echo "[CMD] python -m med.mean_embedding.cli ${ARGS[*]}" | tee -a run.log
python -u -m med.mean_embedding.cli "${ARGS[@]}" 2>&1 | tee -a run.log

echo "[DONE] Results saved to ${OUT_DIR}" | tee -a run.log
popd >/dev/null
