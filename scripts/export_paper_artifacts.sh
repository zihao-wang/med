#!/usr/bin/env bash
set -euo pipefail

# Full paper artifact reproduction.
#
# This recomputes the experiments depicted in paper/draft_main.tex and exports
# the resulting figures and tables to paper/{figures,tables}/. It does not build
# paper artifacts by reading finished paper outputs.
#
# Useful overrides:
#   RESUME=1                     reuse completed LIMIT rows while filling gaps
#   QWEN_LOCAL_FILES_ONLY=1      require the Qwen tokenizer to be in local cache
#   DEVICE=mps|cuda|cpu          torch device for LIMIT retrieval

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_ROOT}/results"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/med-matplotlib-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/med-xdg-cache}"

if [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_CMD=("${PYTHON_BIN}")
elif [ -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
  PYTHON_CMD=("${PROJECT_ROOT}/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=(uv run python)
else
  PYTHON_CMD=(python)
fi

UPPER_OUT_DIR="${UPPER_OUT_DIR:-${RESULTS_DIR}/upper_bound_witness}"
LIMIT_OUT_DIR="${LIMIT_OUT_DIR:-${RESULTS_DIR}/unlimit/random_embeddings}"
CYCLIC_OUT_DIR="${CYCLIC_OUT_DIR:-${RESULTS_DIR}/unlimit/cyclic_overfit}"
PAPER_TABLE_DIR="${PAPER_TABLE_DIR:-${PROJECT_ROOT}/paper/tables}"
PAPER_FIGURE_DIR="${PAPER_FIGURE_DIR:-${PROJECT_ROOT}/paper/figures}"

DIMS="${DIMS:-32 64 128 256 512 1024 2048 4096}"
SPLITS="${SPLITS:-limit-small limit}"
TOKENIZERS="${TOKENIZERS:-handmade qwen vanilla}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-0.6B}"
QWEN_LOCAL_FILES_ONLY="${QWEN_LOCAL_FILES_ONLY:-0}"
RESUME="${RESUME:-0}"
BASE_SEED="${BASE_SEED:-42}"
DEVICE="${DEVICE:-cpu}"
SCORE_CHUNK_SIZE="${SCORE_CHUNK_SIZE:-2048}"

mkdir -p \
  "${UPPER_OUT_DIR}" \
  "${LIMIT_OUT_DIR}" \
  "${CYCLIC_OUT_DIR}" \
  "${PAPER_TABLE_DIR}" \
  "${PAPER_FIGURE_DIR}" \
  "${MPLCONFIGDIR}" \
  "${XDG_CACHE_HOME}"

echo "[1/3] Recomputing synthetic top-2 upper-bound witnesses"
"${PYTHON_CMD[@]}" -u "${PROJECT_ROOT}/scripts/paper_upper_bounds.py" \
  --mode run \
  --output-root "${UPPER_OUT_DIR}" \
  --paper-table-dir "${PAPER_TABLE_DIR}" \
  --paper-figure-dir "${PAPER_FIGURE_DIR}"

LIMIT_ARGS=(
  --mode run
  --output-dir "${LIMIT_OUT_DIR}"
  --base-seed "${BASE_SEED}"
  --device "${DEVICE}"
  --score-chunk-size "${SCORE_CHUNK_SIZE}"
  --qwen-model "${QWEN_MODEL}"
  --paper-table-dir "${PAPER_TABLE_DIR}"
  --paper-figure-dir "${PAPER_FIGURE_DIR}"
  --dims ${DIMS}
  --splits ${SPLITS}
  --tokenizers ${TOKENIZERS}
)
if [ "${QWEN_LOCAL_FILES_ONLY}" = "1" ]; then
  LIMIT_ARGS+=(--qwen-local-files-only)
fi
if [ "${RESUME}" = "1" ]; then
  LIMIT_ARGS+=(--resume)
fi

echo "[2/3] Recomputing LIMIT random-token retrieval"
"${PYTHON_CMD[@]}" -u "${PROJECT_ROOT}/scripts/limit_random_embeddings.py" "${LIMIT_ARGS[@]}"

echo "[3/3] Recomputing cyclic-polytope LIMIT overfit tables"
"${PYTHON_CMD[@]}" -u "${PROJECT_ROOT}/scripts/limit_cyclic_overfit.py" \
  --output-dir "${CYCLIC_OUT_DIR}" \
  --paper-table-dir "${PAPER_TABLE_DIR}"

echo "[DONE] Paper figures exported to ${PAPER_FIGURE_DIR}"
echo "[DONE] Paper tables exported to ${PAPER_TABLE_DIR}"
