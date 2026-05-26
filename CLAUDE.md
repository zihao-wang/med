# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

ICML 2026 paper: "$\mathbb{R}^{2k}$ is Theoretically Large Enough for Embedding-based Top-$k$ Retrieval." Studies the minimal embeddable dimension (MED) required for perfect top-$k$ subset retrieval. The runnable experiments currently use inner product scoring. Proves $d = \Theta(k)$ tight bounds independent of universe size $m$, reframing retrieval limits from approximability to learnability.

## Build

```bash
cd paper && latexmk -pdf icml2026.tex
```

Or for a single pass (e.g., to resolve references):

```bash
cd paper && pdflatex icml2026.tex && bibtex icml2026 && pdflatex icml2026.tex && pdflatex icml2026.tex
```

## Paper structure

- `paper/icml2026.tex` — main entry point. Sets up document class, authors, abstract, includes `content.tex` and `appendix.tex`.
- `paper/content.tex` — the main body (introduction, problem definition, theoretical bounds, simulations, related work, conclusion).
- `paper/appendix.tex` — supplementary proofs and experimental details.
- `paper/custom_command.tex` — custom math macros (`\real`, `\med`, `\fln`, `\fcos`, `\flp`, `\medc`, etc.) and theorem environments. Also configures Python code listings styling.
- `paper/preprint.bib` — bibliography.
- `paper/icml2026.sty`, `paper/icml2026.bst` — ICML 2026 style files (provided by conference).
- `paper/math_commands.tex` — standard math command definitions (provided by ICML template).

## LaTeX macros to know

| Macro | Meaning |
|-------|---------|
| `\real` | $\mathbb{R}$ |
| `\med(m,k;s)` | Minimal Embeddable Dimension given size $m$, subset size $k$, scoring function $s$ |
| `\medc(m,k;s)` | MED in centroid setting |
| `\fln` / `\fcos` / `\flp` | Functional classes: linear / cosine / $\ell_p$ scoring |
| `\draft{text}` | Blue draft annotation |
| `\hh{text}` / `\yhc{text}` | Red/green author comments |

## Code structure

```
med/
├── scoring.py              # shared inner product scoring
├── plotting.py             # WBNL fitted curve reference and plot styling
├── paper_pipeline.py       # combined paper pipeline: run, table, figures
├── mean_embedding/
│   ├── cli.py              # package CLI for custom sweeps
│   ├── trainer_gd.py       # full-batch GD trainer with early stopping (O(C(n,k)) memory)
│   └── experiment.py       # binary search for minimal d, grid search over (k, n)
└── cyclic_polytope/
    ├── construct.py        # moment curve points + polynomial construction of separating queries
    ├── experiment.py       # factory wrapping shared Experiment
    └── cli.py              # package CLI for custom sweeps
unlimit/
├── datasets/limit.py       # packaged LiMIT/LiMIT-small JSONL loader
├── limit_figure.py         # reproduce paper's limit_retrieval.pdf
├── random_embedding_sweep.py # run random-token embedding sweeps
├── tokenizers/             # HandmadeTokenizer (vocab.txt)
└── retrieval/              # random-token scoring + metrics
scripts/
├── run_paper_upper_bounds.sh  # combined paper pipeline: run, table, figures
└── run_unlimit.sh             # LiMIT random-token embedding sweep
```

## Run experiments

The MED paper pipeline scans m (number of total objects) over `10 20 40 80 160 320 640` by default.
Override with `--m_values 10 20 40`.

**Paper upper-bound witness pipeline:**
```bash
bash scripts/run_paper_upper_bounds.sh
```
Runs k=2, m in `10 20 40 80 160 320 640` for both cyclic polytope and centroid GD, writes a combined
`results.json`, `upper_bound_witness_table.{csv,tex}`, and writes final
paper artifacts into `paper/table/` and `paper/figure/`.

**LiMIT random-token retrieval sweep:**
```bash
bash scripts/run_unlimit.sh
DIMS="32 64 128" SPLITS="limit-small" bash scripts/run_unlimit.sh
python -m unlimit.limit_figure --mode run
```

**Direct CLI (for custom sweeps):**
```bash
python -m med.mean_embedding.cli --k_values 2 --n_values 10 20 40 80 160 --scoring_function inner_product
python -m med.mean_embedding.cli --k_values 2 3 4 5 --n_values 10 20 40 80 160 320
```

## Result protocol

### Directory hierarchy

```
results/
  upper_bound_witness/                            # run_paper_upper_bounds.sh
  unlimit/random_embeddings/                      # run_unlimit.sh
```

Shell scripts write deterministic output directories and overwrite derived JSON/CSV/TeX/PDF files on rerun.

### Unified `results.json` format

Both MED-C (centroid embedding) and MED (cyclic polytope) experiments write a single
`results.json` with the same top-level schema:

```json
{
  "experiment": "medc",
  "k": 2,
  "scoring_function": "inner_product",
  "trainer": "gd",
  "results": [
    {"m": 8, "med": 6, "search_path": [...], "time": 1.2},
    {"m": 16, "med": 7, "search_path": [...], "time": 2.3}
  ]
}
```

Top-level fields:

| Field | MED-C | MED | Description |
|-------|-------|-----|-------------|
| `experiment` | `"medc"` | `"med"` | Experiment type |
| `k` | ✓ | ✓ (if uniform) | Subset size (omitted in grid mode) |
| `scoring_function` | ✓ | — | `inner_product` |
| `trainer` | ✓ | — | Always `gd` |
| `results` | ✓ | ✓ | Array or nested object (see below) |

**Single-k `results`** — array of per-m objects:
```json
{"m": 8, "med": 6, "search_path": [...], "time": 1.2}
```

**Grid mode `results`** (MED-C only) — nested `{k: [per-m array]}`:
```json
{"2": [{"m": 8, "med": 6, ...}, ...], "3": [{"m": 8, "med": 8, ...}, ...]}
```

Per-result fields:

| Field | Type | Description |
|-------|------|-------------|
| `m` | int | Number of objects |
| `med` | int or null | Minimal embedding dimension (`-1` or `null` if infeasible) |
| `search_path` | array | Binary search trace |
| `time` | float | Wall-clock seconds for this `m` |

**search_path entries:** `{"dimension": d, "feasible": bool, "time": float}` — dimension tested, whether the checker passed, and elapsed seconds.

### Other output files

| File | Created by | Content |
|------|-----------|---------|
| `config.json` | `cli.py` | Full run config + env info (Python version, torch, CUDA) |
| `minimal_dem_log.txt` | `experiment.py` | Lines: `[RESULT] minimal dimension @ k=X & n=Y is Z` |
| `run.log` | shell scripts | Full stdout/stderr via `tee` |

### Comparison / figure outputs

The paper pipeline writes final figures to `paper/figure/compare_plot1.pdf` (m* vs d) and `paper/figure/compare_plot2.pdf` (d* vs m), and the final LaTeX table to `paper/table/upper_bound_witness_table.tex`.

**`unlimit/limit_figure.py`** writes `results/unlimit/limit_figure/results.json` by default as a JSON array of per-(split, dim) metrics:
```json
[{"split": "small", "dim": 8, "seed": 50, "top2_exact_match": 0.12, "recall_at_1": 0.45, "recall_at_2": 0.50, "mean_rank": 3.5, ...}, ...]
```
`scripts/run_unlimit.sh` writes final artifacts to `paper/figure/limit_retrieval.pdf` and `paper/table/limit_retrieval_table.tex`.

Both support `--mode run` (run + save) and `--mode plot` (load saved JSON, replot).

Python listing style for the paper appendix is configured in `custom_command.tex`.
