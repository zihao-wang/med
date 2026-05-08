# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

ICML 2026 paper: "$\mathbb{R}^{2k}$ is Theoretically Large Enough for Embedding-based Top-$k$ Retrieval." Studies the minimal embeddable dimension (MED) required for perfect top-$k$ subset retrieval under inner product, cosine similarity, and Euclidean distance scoring functions. Proves $d = \Theta(k)$ tight bounds independent of universe size $m$, reframing retrieval limits from approximability to learnability.

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
├── scoring.py             # shared scoring functions (inner_product, L2, cosine, L1)
├── plotting.py            # WBNL fitted curve reference and plot styling
├── mean_embedding/
│   ├── cli.py             # package CLI for custom sweeps
│   ├── compare_plots.py   # reproduce paper's compare_plot{1,2}.pdf
│   ├── trainer_gd.py      # full-batch GD trainer with early stopping (O(C(n,k)) memory)
│   ├── trainer_sgd.py     # stochastic trainer using random k-subsets
│   └── experiment.py      # binary search for minimal d, grid search over (k, n)
├── cyclic_polytope/
│   ├── generator.py        # moment curve point set generation
│   ├── construct.py        # polynomial construction of separating queries
│   ├── checker.py          # CyclicPolytopeChecker (FeasibilityChecker impl)
│   ├── experiment.py       # factory wrapping shared Experiment
│   └── cli.py              # package CLI for custom sweeps
└── unlimit/               # LiMIT retrieval library (RP+OMP)
    ├── datasets/limit.py   #   LiMIT/LiMIT-small JSONL loader from DeepMind GitHub
    ├── limit_figure.py     #   reproduce paper's limit_retrieval.pdf
    ├── tokenizers/         #   HandmadeTokenizer (vocab.txt) + QwenSubwordTokenizer
    └── retrieval/          #   RP+OMP scoring (NumPy/PyTorch backends) + metrics
scripts/
├── run_med.sh          # MED via cyclic polytope LP (moment-curve point sets)
├── run_medc_gd.sh      # MED-C via centroid embedding with full-batch GD
└── run_medc_sgd.sh     # MED-C via centroid embedding with stochastic SGD
```

## Run experiments

All scripts scan m (number of total objects) from 8 to 1024 in powers of 2 by default.
Override with e.g. `N_LIST="8 16 32"` or `M_LIST="8 16 32"` depending on the script.

**Centroid embedding (GD):**
```bash
bash scripts/run_medc_gd.sh                          # k=2, inner_product
METRIC=l2 K=3 bash scripts/run_medc_gd.sh             # k=3, Euclidean
```

**Centroid embedding (SGD):**
```bash
bash scripts/run_medc_sgd.sh                          # k=2, inner_product
TRAINER=sgd bash scripts/run_medc_sgd.sh               # explicit SGD
```

**Cyclic polytope (LP-based):**
```bash
bash scripts/run_med.sh                                # M_LIST defaults to 8..1024, k=2
M=16 K=3 bash scripts/run_med.sh                       # single (m=16, k=3)
M_LIST="8 16 32" K=2 bash scripts/run_med.sh           # custom m sweep
```

**Direct CLI (for custom sweeps):**
```bash
python -m med.mean_embedding.cli --k 2 --n_values 8 16 32 64 128 --scoring_function inner_product
python -m med.mean_embedding.cli --k_values 2 3 4 5 --n_values 8 16 32 64 128 256
python -m med.mean_embedding.cli --trainer sgd --k 2 --n_values 8 16 32 64 128 256
```

## Result protocol

### Directory hierarchy

```
results/
  gd/{metric}/m_dependency/k_{k}/{timestamp}/     # run_medc_gd.sh (single k)
  gd/{metric}/k_{k}/{timestamp}/                  # run_medc_gd.sh (single k, newer)
  sgd/{metric}/m_dependency/k_{k}/{timestamp}/    # run_medc_sgd.sh (single k)
  sgd/{metric}/k_{k}/{timestamp}/                 # run_medc_sgd.sh (single k, newer)
  sgd/{metric}/grid/{timestamp}/                  # run_medc_sgd.sh (grid mode, K_LIST set)
  cyclic_polytope/{label}/{timestamp}/            # run_med.sh
```

Timestamps are `YYYYMMDD_HHMMSS`. The `{label}` for cyclic polytope runs is `m{M}_k{K}` (e.g. `m5_k2`), `m_list_k{K}`, or `k_list`.

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
| `scoring_function` | ✓ | — | `inner_product`, `l2`, `cosine`, or `l1` |
| `trainer` | ✓ | — | `gd` or `sgd` |
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
| `search_path` | array | Binary search trace (entries differ by experiment type) |
| `time` | float | Wall-clock seconds for this `m` |

**MED-C search_path entries:** `{"dimension": d, "violations": v}` — dimension tested and number of constraint violations found by the trainer.

**MED search_path entries:** `{"dimension": n, "feasible": bool, "checks": int, "time": float}` — dimension tested, construction-based verification result (squared-polynomial query), number of subset checks, and timing.

### Other output files

| File | Created by | Content |
|------|-----------|---------|
| `config.json` | `cli.py` | Full run config + env info (Python version, torch, CUDA) |
| `minimal_dem_log.txt` | `experiment.py` | Lines: `[RESULT] minimal dimension @ k=X & n=Y is Z` |
| `run.log` | shell scripts | Full stdout/stderr via `tee` |

### Comparison / figure outputs

**`med/mean_embedding/compare_plots.py`** writes `compare_plot_results.json` (default: repo root):
```json
{"d_star": {"8": 6, "16": 7, ...}, "m_star": {"6": 8, "7": 16, ...}, "k": 2, "scoring": "inner_product"}
```
Generates `paper/compare_plot1.pdf` (m* vs d) and `paper/compare_plot2.pdf` (d* vs m).

**`med/unlimit/limit_figure.py`** writes `limit_results.json` (default: repo root) as a JSON array of per-(split, dim) metrics:
```json
[{"split": "small", "dim": 8, "omp_steps": 0, "top2_exact_match": 0.12, "recall_at_1": 0.45, "mean_rank": 3.5, ...}, ...]
```
Generates `paper/limit_retrieval.pdf`. A sentinel row with `"dim": -1` records the membership baseline.

Both support `--mode run` (run + save) and `--mode plot` (load saved JSON, replot).

Python listing style for the paper appendix is configured in `custom_command.tex`.
