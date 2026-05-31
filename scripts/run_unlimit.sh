#!/usr/bin/env bash
set -euo pipefail

# Run the unlimit random-token embedding sweep on LiMIT-small and LiMIT.
#
# Usage:
#   ./run_unlimit.sh
#   MODE=plot ./run_unlimit.sh
#   DIMS="32 64 128" ./run_unlimit.sh
#   SPLITS="limit-small" ./run_unlimit.sh
#   TOKENIZERS="handmade qwen" ./run_unlimit.sh
#   TOKENIZERS="vanilla" ./run_unlimit.sh
#
# Results written to results/unlimit/random_embeddings/
#   - config.json  : run configuration
#   - results.json : per-(dataset, dim) metrics
#   - summary.csv  : compact table with recall_at_2
#   - limit_retrieval_table.tex
#   - limit_retrieval_limit.pdf and limit_retrieval_limit_small.pdf
#   - run.log      : full stdout/stderr log
#
# Paper-facing figures/tables are exported to paper/{figures,tables}/ unless NO_PAPER_COPY=1.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_ROOT}/results"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/med-matplotlib-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/med-xdg-cache}"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

DIMS="${DIMS:-32 64 128 256 512 1024 2048 4096}"
SPLITS="${SPLITS:-limit-small limit}"
TOKENIZERS="${TOKENIZERS:-handmade qwen vanilla}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-0.6B}"
QWEN_LOCAL_FILES_ONLY="${QWEN_LOCAL_FILES_ONLY:-0}"
RESUME="${RESUME:-1}"
MODE="${MODE:-run}"
BASE_SEED="${BASE_SEED:-42}"
DEVICE="${DEVICE:-cpu}"
SCORE_CHUNK_SIZE="${SCORE_CHUNK_SIZE:-2048}"
OUT_DIR="${OUT_DIR:-${RESULTS_DIR}/unlimit/random_embeddings}"
PAPER_TABLE_DIR="${PAPER_TABLE_DIR:-${PROJECT_ROOT}/paper/tables}"
PAPER_FIGURE_DIR="${PAPER_FIGURE_DIR:-${PROJECT_ROOT}/paper/figures}"
NO_PAPER_COPY="${NO_PAPER_COPY:-0}"

mkdir -p "${OUT_DIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"
pushd "${OUT_DIR}" >/dev/null

if [ "${RESUME}" = "1" ] && [ -f run.log ]; then
  printf "\n[INFO] resuming unlimit random-token embedding run\n" | tee -a run.log
else
  : > run.log
  echo "[INFO] unlimit random-token embedding run" | tee -a run.log
fi
echo "  dims      = ${DIMS}" | tee -a run.log
echo "  datasets  = ${SPLITS}" | tee -a run.log
echo "  tokenizers= ${TOKENIZERS}" | tee -a run.log
echo "  qwen_model= ${QWEN_MODEL}" | tee -a run.log
echo "  mode      = ${MODE}" | tee -a run.log
echo "  resume    = ${RESUME}" | tee -a run.log
echo "  base_seed = ${BASE_SEED}" | tee -a run.log
echo "  device    = ${DEVICE}" | tee -a run.log
echo "  score_chunk_size = ${SCORE_CHUNK_SIZE}" | tee -a run.log
echo "  python    = ${PYTHON_BIN}" | tee -a run.log
echo "  output_dir= ${OUT_DIR}" | tee -a run.log
echo "  git_commit = $(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo none)" | tee -a run.log

ARGS=(
  --mode "${MODE}"
  --output-dir "${OUT_DIR}"
  --base-seed "${BASE_SEED}"
  --device "${DEVICE}"
  --score-chunk-size "${SCORE_CHUNK_SIZE}"
  --qwen-model "${QWEN_MODEL}"
  --dims ${DIMS}
  --splits ${SPLITS}
  --tokenizers ${TOKENIZERS}
)

if [ "${NO_PAPER_COPY}" = "1" ]; then
  ARGS+=(--no-paper-copy)
else
  mkdir -p "${PAPER_TABLE_DIR}" "${PAPER_FIGURE_DIR}"
  ARGS+=(
    --paper-table-dir "${PAPER_TABLE_DIR}"
    --paper-figure-dir "${PAPER_FIGURE_DIR}"
  )
fi

if [ "${QWEN_LOCAL_FILES_ONLY}" = "1" ]; then
  ARGS+=(--qwen-local-files-only)
fi
if [ "${RESUME}" = "1" ]; then
  ARGS+=(--resume)
fi

echo "[CMD] ${PYTHON_BIN} ${PROJECT_ROOT}/scripts/limit_random_embeddings.py ${ARGS[*]}" | tee -a run.log
"${PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/limit_random_embeddings.py" "${ARGS[@]}" 2>&1 | tee -a run.log

echo "[DONE] Results saved to ${OUT_DIR}" | tee -a run.log
popd >/dev/null
