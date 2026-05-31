# Minimal Embeddable Dimension (MED)

Companion code for the ICML 2026 paper: *"$\mathbb{R}^{2k}$ is Theoretically Large Enough for Embedding-based Top-$k$ Retrieval."*

This repository reproduces the paper-scope experiments for inner-product top-2
retrieval. The released scripts focus on:

- synthetic top-2 upper-bound witnesses for all singleton and pair queries;
- LIMIT and LIMIT-small random token-sum retrieval with Recall@2;
- cyclic-polytope LIMIT overfit tables.

Cosine and Euclidean experiment variants are not part of this reproduction
pipeline.

## Quick start

```bash
git clone https://github.com/zihao-wang/med.git
cd med
uv sync                            # or: pip install -r requirements.txt
uv run pytest tests -q
```

Requires Python 3.10+.

## Reproducing Paper Artifacts

The release code is scoped to the figures and tables consumed directly by
`paper/draft_main.tex`. Generated experiment caches live under `results/`;
paper-ready exports are copied to `paper/figures/` and `paper/tables/`.

| Goal | Results cache | Paper export |
|---|---|
| Synthetic top-2 witness dimensions | `results/upper_bound_witness/results.json`, `upper_bound_witness_table.{csv,tex}`, `top2_dimension_fit.pdf` | `paper/figures/top2_dimension_fit.pdf`, `paper/tables/upper_bound_witness_table.{csv,tex}` |
| LIMIT random token-sum Recall@2 | `results/unlimit/random_embeddings/results.json`, `summary.csv`, `limit_retrieval_table.tex`, `limit_promptriever_crossing.tex`, `limit_retrieval_limit.pdf`, `limit_retrieval_limit_small.pdf` | `paper/figures/limit_retrieval_limit.pdf`, `paper/figures/limit_retrieval_limit_small.pdf`, `paper/tables/limit_retrieval_table.tex`, `paper/tables/limit_promptriever_crossing.tex` |
| LIMIT cyclic-polytope overfit witness | `results/unlimit/cyclic_overfit/*` | `paper/tables/limit_cyclic_overfit_summary.tex`, `paper/tables/limit_small_cyclic_*_table.tex` |

Run the full reproduction pipeline:

```bash
bash scripts/export_paper_artifacts.sh
```

This recomputes the depicted experiments and writes intermediate results under
`results/`, then exports the paper-facing files to `paper/figures/` and
`paper/tables/`. It does not build the paper artifacts by reading the finished
paper files.

For a clean rerun, remove the cached experiment outputs first:

```bash
rm -rf results/upper_bound_witness \
       results/unlimit/random_embeddings \
       results/unlimit/cyclic_overfit
bash scripts/export_paper_artifacts.sh
```

The full run is expensive, especially the synthetic centroid GD sweep and the
full LIMIT retrieval grid. Intermediate outputs are written during the run:
`paper_upper_bounds.py` checkpoints after every checked candidate dimension, and
`limit_random_embeddings.py` checkpoints after every completed
dataset/tokenizer/dimension row.

For interrupted LIMIT runs, use `RESUME=1` to reuse completed rows and fill only
missing rows:

```bash
RESUME=1 bash scripts/export_paper_artifacts.sh
```

If the Qwen tokenizer is already cached locally, use:

```bash
QWEN_LOCAL_FILES_ONLY=1 RESUME=1 bash scripts/export_paper_artifacts.sh
```

Memory-sensitive knobs are exposed for the two large parts of the pipeline:

- `--constraint-chunk-size` in `scripts/paper_upper_bounds.py` controls how many
  subset constraints are materialized per centroid-GD chunk; the default is
  `8192`.
- `SCORE_CHUNK_SIZE` or `--score-chunk-size` in the LIMIT scripts controls how
  many documents are scored at once; the default is `2048`.

Individual paper-scope producers are also available:

```bash
bash scripts/run_paper_upper_bounds.sh
bash scripts/run_unlimit.sh
uv run python scripts/limit_cyclic_overfit.py --paper-table-dir paper/tables
```

To regenerate only plots and tables from saved rows:

```bash
uv run python scripts/paper_upper_bounds.py --mode plot \
  --output-root results/upper_bound_witness \
  --paper-table-dir paper/tables \
  --paper-figure-dir paper/figures

MODE=plot RESUME=1 bash scripts/run_unlimit.sh
```

`scripts/paper_upper_bounds.py` recomputes the $k=2$ synthetic witness grid
used for the top-2 dimension figure. The cyclic-polytope checker verifies every
singleton and pair query exactly; the centroid-GD checker records the binary
search path and violation counts as an upper-bound witness.
`scripts/limit_random_embeddings.py` recomputes random token-sum LIMIT retrieval
for the handmade, Qwen, and vanilla tokenizers.
`scripts/limit_cyclic_overfit.py` recomputes the dimension-4 cyclic-polytope
LIMIT overfit tables from the packaged LIMIT JSONL assets.

Only the three paper-used PDFs are generated:

- `top2_dimension_fit.pdf`
- `limit_retrieval_limit.pdf`
- `limit_retrieval_limit_small.pdf`

## Project structure

```
├── paper/                  # Camera-ready paper artifacts
│   ├── draft_main.tex      #   canonical LaTeX entry point; sections are inlined
│   ├── figures/            #   figures used by draft_main.tex
│   ├── tables/             #   table inputs used by draft_main.tex
│   └── overleaf_final.zip  #   final Overleaf upload snapshot
├── src/med/                # editable package root
│   ├── scoring.py          #   inner-product/cosine scoring helpers
│   ├── plotting.py         #   WBNL curve reference and plot styling
│   ├── checker.py          #   FeasibilityChecker ABC + CheckResult
│   ├── experiment.py       #   shared experiment orchestration (binary search, grid)
│   ├── search.py           #   shared binary search over dimensions
│   ├── mean_embedding/     #   centroid GD witness implementation
│   ├── cyclic_polytope/    #   cyclic-polytope witness implementation
│   └── unlimit/            #   LIMIT retrieval subpackage
│       ├── datasets/       #     packaged LIMIT/LIMIT-small JSONL loader
│       ├── tokenizers/     #     handmade phrase, Qwen subword, and vanilla tokenizers
│       └── retrieval/      #     random-token scoring + metrics
├── scripts/                # Paper reproduction entrypoints
│   ├── export_paper_artifacts.sh # full paper-scope reproduction
│   ├── paper_upper_bounds.py     # synthetic top-2 figure producer
│   ├── limit_random_embeddings.py # LIMIT random-token figure/table producer
│   ├── limit_cyclic_overfit.py   # cyclic-polytope LIMIT table producer
│   ├── run_paper_upper_bounds.sh
│   └── run_unlimit.sh
├── results/                # cached experiment outputs, safe to delete and rerun
├── tests/                  # pytest suite
└── requirements.txt
```

## Tests

```bash
uv run pytest tests -q
```

## Build the paper

```bash
cd paper
SOURCE_DATE_EPOCH=1780164721 FORCE_SOURCE_DATE=1 latexmk -pdf draft_main.tex
```

## License

MIT License - see [LICENSE](LICENSE).
