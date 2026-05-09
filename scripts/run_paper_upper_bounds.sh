#!/usr/bin/env bash
set -euo pipefail

# Run the paper upper-bound witness pipeline for k=2 and m in 10 20 ... 640.
#
# Outputs:
#   results/upper_bound_witness/results.json
#   results/upper_bound_witness/upper_bound_witness_table.{csv,tex}
#   results/upper_bound_witness/compare_plot{1,2}.pdf
#   paper/table/upper_bound_witness_table.{csv,tex}
#   paper/figure/compare_plot{1,2}.pdf

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/results/upper_bound_witness}"
PAPER_TABLE_DIR="${PAPER_TABLE_DIR:-${PROJECT_ROOT}/paper/table}"
PAPER_FIGURE_DIR="${PAPER_FIGURE_DIR:-${PROJECT_ROOT}/paper/figure}"

mkdir -p "${OUT_DIR}"

python -u -m med.paper_pipeline \
  --output-root "${OUT_DIR}" \
  --paper-table-dir "${PAPER_TABLE_DIR}" \
  --paper-figure-dir "${PAPER_FIGURE_DIR}" \
  "$@" 2>&1 | tee "${OUT_DIR}/run.log"
