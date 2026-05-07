#!/usr/bin/env bash
set -euo pipefail

# Run cyclic-polytope MED experiment: find minimal embedding dimension
# via polynomial construction on the moment-curve point set.
#
# Usage: ./run_med.sh
#        M=20 K=3 MAX_CHECKS=5000 ./run_med.sh
#        M_LIST="8 16 32 64" K=2 ./run_med.sh
#        K_LIST="2 3 4 5" M_LIST="8 16 32 64" ./run_med.sh   # grid mode
#
# Results written to results/cyclic_polytope/<label>/<timestamp>/
#   - results.json        : unified format (see CLAUDE.md Result protocol)
#   - config.json         : run configuration + environment info
#   - run.log             : full output log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_ROOT}/results"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

M="${M:-}"
M_LIST="${M_LIST:-8 16 32 64 128 256 512 1024}"
K="${K:-2}"
K_LIST="${K_LIST:-}"
MAX_CHECKS="${MAX_CHECKS:-2000}"
SEED="${SEED:-42}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"

# Build run label
if [ -n "${M}" ]; then
  if [ -n "${K_LIST}" ]; then
    LABEL="m${M}_k_list"
  else
    LABEL="m${M}_k${K}"
  fi
elif [ -n "${K_LIST}" ]; then
  LABEL="k_list"
else
  LABEL="m_list_k${K}"
fi

OUT_DIR="${RESULTS_DIR}/cyclic_polytope/${LABEL}/${RUN_ID}"
mkdir -p "${OUT_DIR}"
pushd "${OUT_DIR}" >/dev/null

echo "[INFO] MED (cyclic polytope) run" | tee run.log
echo "  m          = ${M:-<from m_values>}" | tee -a run.log
echo "  m_values   = ${M_LIST}" | tee -a run.log
echo "  k          = ${K}" | tee -a run.log
echo "  k_values   = ${K_LIST}" | tee -a run.log
echo "  max_checks = ${MAX_CHECKS}" | tee -a run.log
echo "  seed       = ${SEED}" | tee -a run.log
echo "  timestamp  = ${RUN_ID}" | tee -a run.log
echo "  git_commit = $(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo none)" | tee -a run.log

ARGS=(
  --max_checks "${MAX_CHECKS}"
  --seed "${SEED}"
)

# Build m_values: single M overrides M_LIST
if [ -n "${M}" ]; then
  ARGS+=(--m_values "${M}")
else
  ARGS+=(--m_values ${M_LIST})
fi

# Build k arguments
if [ -n "${K_LIST}" ]; then
  ARGS+=(--k_values ${K_LIST})
else
  ARGS+=(--k "${K}")
fi

echo "[CMD] python -m med.cyclic_polytope.cli ${ARGS[*]}" | tee -a run.log
python -u -m med.cyclic_polytope.cli "${ARGS[@]}" 2>&1 | tee -a run.log

echo "" | tee -a run.log
echo "[DONE] Results saved to ${OUT_DIR}" | tee -a run.log
popd >/dev/null
