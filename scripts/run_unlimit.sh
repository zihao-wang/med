#!/usr/bin/env bash
set -euo pipefail

# Run the unlimit random-token embedding sweep on LIMIT-small and LIMIT.
#
# Usage:
#   ./run_unlimit.sh
#   MODE=plot ./run_unlimit.sh
#   DIMS="32 64 128" ./run_unlimit.sh
#   SPLITS="limit-small" ./run_unlimit.sh
#   TOKENIZERS="handmade qwen" ./run_unlimit.sh
#   SCORE_CHUNK_SIZE=1024 ./run_unlimit.sh
#   NO_PHRASE_COVER_FREE=1 ./run_unlimit.sh
#
# Results written to results/unlimit/random_embeddings/
#   - config.json  : run configuration
#   - results.json : per-(dataset, tokenizer, dim) metrics
#   - summary.csv  : compact table with recall_at_2
#   - limit_retrieval_table.tex
#   - limit_retrieval.pdf
#   - run.log      : full stdout/stderr log
#
# Final paper artifacts are copied to paper/table/ and paper/figure/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_ROOT}/results"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

DIMS="${DIMS:-32 64 128 256 512 1024 2048 4096}"
SPLITS="${SPLITS:-limit-small limit}"
TOKENIZERS="${TOKENIZERS:-handmade qwen}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-0.6B}"
QWEN_LOCAL_FILES_ONLY="${QWEN_LOCAL_FILES_ONLY:-0}"
RESUME="${RESUME:-1}"
MODE="${MODE:-run}"
BASE_SEED="${BASE_SEED:-42}"
DEVICE="${DEVICE:-cpu}"
SCORE_CHUNK_SIZE="${SCORE_CHUNK_SIZE:-2048}"
NO_PHRASE_COVER_FREE="${NO_PHRASE_COVER_FREE:-${NO_COVER_FREE:-0}}"
OUT_DIR="${OUT_DIR:-${RESULTS_DIR}/unlimit/random_embeddings}"
PAPER_TABLE_DIR="${PAPER_TABLE_DIR:-${PROJECT_ROOT}/paper/table}"
PAPER_FIGURE_DIR="${PAPER_FIGURE_DIR:-${PROJECT_ROOT}/paper/figure}"

mkdir -p "${OUT_DIR}"
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
echo "  score_chk = ${SCORE_CHUNK_SIZE}" | tee -a run.log
echo "  phrasecf  = $([ "${NO_PHRASE_COVER_FREE}" = "1" ] && echo disabled || echo enabled)" | tee -a run.log
echo "  python    = ${PYTHON_BIN}" | tee -a run.log
echo "  output_dir= ${OUT_DIR}" | tee -a run.log
echo "  paper_tbl = ${PAPER_TABLE_DIR}" | tee -a run.log
echo "  paper_fig = ${PAPER_FIGURE_DIR}" | tee -a run.log
echo "  git_commit = $(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo none)" | tee -a run.log

ARGS=(
  --mode "${MODE}"
  --output-dir "${OUT_DIR}"
  --paper-table-dir "${PAPER_TABLE_DIR}"
  --paper-figure-dir "${PAPER_FIGURE_DIR}"
  --base-seed "${BASE_SEED}"
  --device "${DEVICE}"
  --score-chunk-size "${SCORE_CHUNK_SIZE}"
  --qwen-model "${QWEN_MODEL}"
  --dims ${DIMS}
  --splits ${SPLITS}
  --tokenizers ${TOKENIZERS}
)

if [ "${QWEN_LOCAL_FILES_ONLY}" = "1" ]; then
  ARGS+=(--qwen-local-files-only)
fi
if [ "${RESUME}" = "1" ]; then
  ARGS+=(--resume)
fi
if [ "${NO_PHRASE_COVER_FREE}" = "1" ]; then
  ARGS+=(--no-phrase-cover-free)
fi

echo "[CMD] ${PYTHON_BIN} -m unlimit.random_embedding_sweep ${ARGS[*]}" | tee -a run.log
"${PYTHON_BIN}" -u -m unlimit.random_embedding_sweep "${ARGS[@]}" 2>&1 | tee -a run.log

echo "[DONE] Results saved to ${OUT_DIR}" | tee -a run.log
popd >/dev/null
