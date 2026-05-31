# Minimal Embeddable Dimension (MED)

Companion code for the ICML 2026 paper: *"$\mathbb{R}^{2k}$ is Theoretically Large Enough for Embedding-based Top-$k$ Retrieval."*

Studies the minimal dimension required to embed subset memberships into vector spaces under inner product, cosine similarity, and Euclidean distance scoring.

## Quick start

```bash
git clone https://github.com/zihao-wang/med.git
cd med
uv sync                            # or: pip install -r requirements.txt
```

Requires Python 3.10+.

## Reproducing Paper Artifacts

The release code is scoped to the figures and tables consumed by
`paper/draft_main.tex`.

| Paper artifact | Generated file |
|---|---|
| Synthetic top-2 dimension figure | `paper/figures/top2_dimension_fit.pdf` |
| LIMIT Recall@2 panels | `paper/figures/limit_retrieval_limit.pdf`, `paper/figures/limit_retrieval_limit_small.pdf` |
| Promptriever crossing table | `paper/tables/limit_promptriever_crossing.tex` |
| LIMIT Recall@2 grid | `paper/tables/limit_retrieval_table.tex` |
| Cyclic-polytope overfit summary | `paper/tables/limit_cyclic_overfit_summary.tex` |
| LIMIT-small cyclic document/query embeddings | `paper/tables/limit_small_cyclic_*_table.tex` |

Run the full reproduction pipeline:

```bash
bash scripts/export_paper_artifacts.sh
```

This recomputes the depicted experiments and writes intermediate results under
`results/`, then exports the paper-facing files to `paper/figures/` and
`paper/tables/`. It does not build the paper artifacts by reading the finished
paper files.

The full run is expensive, especially the synthetic centroid GD sweep and full
LIMIT retrieval grid. For interrupted LIMIT runs, use `RESUME=1` to reuse
completed rows and fill missing rows:

```bash
RESUME=1 bash scripts/export_paper_artifacts.sh
```

Individual paper-scope producers are also available:

```bash
bash scripts/run_paper_upper_bounds.sh
bash scripts/run_unlimit.sh
python scripts/limit_cyclic_overfit.py --paper-table-dir paper/tables
```

`scripts/paper_upper_bounds.py` recomputes the $k=2$ synthetic witness grid
used for the top-2 dimension figure. `scripts/limit_random_embeddings.py`
recomputes random token-sum LIMIT retrieval for the handmade, Qwen, and vanilla
tokenizers. `scripts/limit_cyclic_overfit.py` recomputes the dimension-4
cyclic-polytope LIMIT overfit tables from the packaged LIMIT JSONL assets.

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
└── requirements.txt
```

## Build the paper

```bash
cd paper
SOURCE_DATE_EPOCH=1780164721 FORCE_SOURCE_DATE=1 latexmk -pdf draft_main.tex
```

## License

MIT License — see [LICENSE](LICENSE).
