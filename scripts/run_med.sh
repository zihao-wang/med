#!/usr/bin/env bash
set -euo pipefail

# Run cyclic-polytope MED experiment: find minimal embedding dimension
# via polynomial construction on the moment-curve point set.
#
# Usage: ./run_med.sh
#        K_LIST=3 M=20 ./run_med.sh
#        M_LIST="8 16 32 64" K_LIST=2 ./run_med.sh
#        K_LIST="2 3 4 5" M_LIST="8 16 32 64" ./run_med.sh
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
K_LIST="${K_LIST:-2}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

# Build run label
if [ -n "${M}" ]; then
  LABEL="m${M}_k_list"
else
  LABEL="m_list_k_list"
fi

OUT_DIR="${RESULTS_DIR}/cyclic_polytope/${LABEL}/${RUN_ID}"
mkdir -p "${OUT_DIR}"
pushd "${OUT_DIR}" >/dev/null

echo "[INFO] MED (cyclic polytope) run" | tee run.log
echo "  m          = ${M:-<from m_values>}" | tee -a run.log
echo "  m_values   = ${M_LIST}" | tee -a run.log
echo "  k_values   = ${K_LIST}" | tee -a run.log
echo "  timestamp  = ${RUN_ID}" | tee -a run.log
echo "  git_commit = $(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo none)" | tee -a run.log

ARGS=(--k_values ${K_LIST})

# Build m_values: single M overrides M_LIST
if [ -n "${M}" ]; then
  ARGS+=(--m_values "${M}")
else
  ARGS+=(--m_values ${M_LIST})
fi

echo "[CMD] python -m med.cyclic_polytope.cli ${ARGS[*]}" | tee -a run.log
python -u -m med.cyclic_polytope.cli "${ARGS[@]}" 2>&1 | tee -a run.log

echo "" | tee -a run.log
echo "[DONE] Results saved to ${OUT_DIR}" | tee -a run.log
popd >/dev/null
