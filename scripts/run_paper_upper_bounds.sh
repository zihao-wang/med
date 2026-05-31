#!/usr/bin/env bash
set -euo pipefail

# Run the paper upper-bound witness pipeline for k=2 and m in 10 20 ... 640.
#
# Outputs:
#   results/upper_bound_witness/results.json
#   results/upper_bound_witness/upper_bound_witness_table.{csv,tex}
#   results/upper_bound_witness/top2_dimension_fit.pdf
#   paper/{figures,tables}/ exported copies unless NO_PAPER_COPY=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/med-matplotlib-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/med-xdg-cache}"

OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/results/upper_bound_witness}"
PAPER_TABLE_DIR="${PAPER_TABLE_DIR:-${PROJECT_ROOT}/paper/tables}"
PAPER_FIGURE_DIR="${PAPER_FIGURE_DIR:-${PROJECT_ROOT}/paper/figures}"
NO_PAPER_COPY="${NO_PAPER_COPY:-0}"

mkdir -p "${OUT_DIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

if [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_CMD=("${PYTHON_BIN}")
elif [ -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
  PYTHON_CMD=("${PROJECT_ROOT}/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=(uv run python)
else
  PYTHON_CMD=(python)
fi

PAPER_ARGS=()
if [ "${NO_PAPER_COPY}" = "1" ]; then
  PAPER_ARGS+=(--no-paper-copy)
else
  mkdir -p "${PAPER_TABLE_DIR}" "${PAPER_FIGURE_DIR}"
  PAPER_ARGS+=(
    --paper-table-dir "${PAPER_TABLE_DIR}"
    --paper-figure-dir "${PAPER_FIGURE_DIR}"
  )
fi

"${PYTHON_CMD[@]}" -u "${PROJECT_ROOT}/scripts/paper_upper_bounds.py" \
  --output-root "${OUT_DIR}" \
  "${PAPER_ARGS[@]}" \
  "$@" 2>&1 | tee "${OUT_DIR}/run.log"
