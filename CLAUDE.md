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
src/
├── scoring.py             # shared scoring functions (inner_product, L2, cosine, L1)
├── plotting.py            # WBNL fitted curve reference and plot styling
├── mean_embedding/
│   ├── train_gd.py        # full-batch GD trainer with early stopping
│   ├── trainer_sgd.py     # stochastic trainer using random k-subsets
│   └── experiment.py      # binary search for minimal d, grid search over (k, n)
└── cyclic_polytope/
    ├── generator.py        # moment curve point set generation
    └── verification.py     # LP feasibility check for face separability
scripts/
├── generate_compare_plots.py  # reproduce paper's compare_plot{1,2}.pdf
├── run_all_experiments.sh     # master runner for all experiments
├── run_joint_dependency.sh    # joint (k, n) grid sweep
├── run_k_dependency.sh        # MED vs k at fixed n
└── run_m_dependency.sh        # MED vs n at fixed k
main.py                    # CLI for single-k and grid experiments
verify_cyclic_polytope.py  # CLI for cyclic polytope verification
```

## Reproduce paper plots

```bash
python scripts/generate_compare_plots.py --mode run
```

Generates `paper/compare_plot1.pdf` (critical m* vs d) and `paper/compare_plot2.pdf` (critical d* vs m, log-scale) by running GD centroid embedding experiments with k=2 and comparing against the WBNL fitted curve.

To regenerate only plots from saved results: `python scripts/generate_compare_plots.py --mode plot`.

## Run experiments

Single k: `python main.py --k 2 --n_values 8 16 32 64 128 --scoring_function inner_product`
Grid sweep: `python main.py --k_values 2 3 4 5 --n_values 8 16 32 64 128 256`
With SGD: `python main.py --trainer sgd --k 2 --n_values 8 16 32 64 128 256`

Python listing style for the paper appendix is configured in `custom_command.tex`.
