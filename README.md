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

## Reproducing the paper's figures

The paper contains one figure (Figure 1) with two panels, both comparing centroid embedding experiments against the fitted curve from Weller et al. (2025):

| Panel | File | Content |
|-------|------|---------|
| Fig 1a | `paper/compare_plot1.pdf` | Critical number of points $m^*(d)$ vs dimension $d$ |
| Fig 1b | `paper/compare_plot2.pdf` | Critical dimension $d^*(m)$ vs number of points $m$ (log-scale x) |

### Step-by-step

**1. Generate both plots from scratch (runs the full experiment):**

```bash
uv run python scripts/generate_compare_plots.py --mode run
```

This runs centroid embedding experiments with $k=2$ and inner product scoring. It searches for feasible embeddings across a grid of $n$ (number of points) and $d$ (dimension) values, then saves the results to `compare_plot_results.json` and writes the two PDFs to `paper/`.

Expected runtime: ~10–30 minutes on a modern GPU; longer on CPU. Progress is printed to stdout.

**2. Regenerate plots from saved results (no re-computation):**

```bash
uv run python scripts/generate_compare_plots.py --mode plot
```

Reads `compare_plot_results.json` and redraws the PDFs. Useful for tweaking plot styling without rerunning the experiment.

**3. Reproduce with other scoring functions:**

```bash
uv run python scripts/generate_compare_plots.py --mode run --scoring l2
uv run python scripts/generate_compare_plots.py --mode run --scoring cosine
```

Supported scoring functions: `inner_product` (default), `l2`, `cosine`, `l1`.

### What the experiment does

For each $(n, d)$ pair, the script trains a set of $n$ learnable vectors in $\mathbb{R}^d$ using gradient descent. The loss function enforces that for every $k$-subset, the centroid embedding has a higher score with its member vectors than with non-members. If the loss reaches zero (no violations), a feasible embedding exists for that $(n, d)$ pair.

- **Phase 1** — for each $n$, binary search for the minimal $d$ where violations = 0 → $d^*(n)$
- **Phase 2** — for each $d$, binary search for the maximal $n$ where violations = 0 → $m^*(d)$

The results are plotted against the WBNL fitted curve $m(d) = -10.53 + 4.03d + 0.052d^2 + 0.0037d^3$ from Weller et al. (2025).

### Faster reproduction (reduced grid)

To get a quick sense of the results with fewer $(n, d)$ points:

```bash
uv run python scripts/generate_compare_plots.py --mode run \
    --n-values 8 16 32 64 \
    --d-values 1 2 3 4 5 6 7 8 9 10
```

Reduce training epochs for a rougher but faster sweep:

```bash
uv run python scripts/generate_compare_plots.py --mode run \
    --num-epochs 500 --patience 50
```

## Project structure

```
├── paper/                  # LaTeX source for the ICML 2026 paper
│   ├── icml2026.tex        #   main entry point
│   ├── content.tex         #   main body
│   ├── appendix.tex        #   supplementary proofs + experiment code listing
│   ├── custom_command.tex  #   custom math macros
│   └── preprint.bib        #   bibliography
├── src/                    # Python library code
│   ├── scoring.py          #   shared scoring functions (inner product, L2, cosine, L1)
│   ├── plotting.py         #   WBNL curve reference and plot styling
│   ├── mean_embedding/     #   centroid embedding experiments
│   │   ├── train_gd.py     #     full-batch GD trainer
│   │   ├── trainer_sgd.py  #     stochastic trainer (random k-subsets)
│   │   └── experiment.py   #     experiment orchestration (binary search, grid search)
│   └── cyclic_polytope/    #   cyclic polytope face verification
│       ├── generator.py    #     moment curve point generation
│       └── verification.py #     LP-based face check
├── scripts/                # Experiment scripts and plot generation
│   ├── generate_compare_plots.py  # reproduce paper's compare_plot{1,2}.pdf
│   ├── run_all_experiments.sh     # master: joint grid for GD + SGD
│   ├── run_joint_dependency.sh    # joint (k, n) grid sweep
│   ├── run_k_dependency.sh        # MED vs k at fixed n
│   └── run_m_dependency.sh        # MED vs n at fixed k
├── main.py                 # CLI entry point for single-k and grid experiments
├── verify_cyclic_polytope.py  # CLI for cyclic polytope face checks
└── requirements.txt
```

## Running custom experiments

Single $k$, single scoring function:

```bash
uv run python main.py --k 2 --n_values 8 16 32 64 128 --scoring_function inner_product
```

Grid sweep over $k$ and $n$:

```bash
uv run python main.py --k_values 2 3 4 5 --n_values 8 16 32 64 128 256
```

Using SGD instead of GD:

```bash
uv run python main.py --trainer sgd --k 2 --n_values 8 16 32 64 128 256
```

Results are saved as JSON in the current working directory.

Run a full sweep from shell scripts (saves to `results/`):

```bash
bash scripts/run_joint_dependency.sh gd
bash scripts/run_k_dependency.sh gd
bash scripts/run_m_dependency.sh gd
```

## Cyclic polytope verification

```bash
uv run python verify_cyclic_polytope.py --m 64 --n 5
uv run python verify_cyclic_polytope.py --m 64 --n 5 --pairs_only
```

## Build the paper

```bash
cd paper && latexmk -pdf icml2026.tex
```

## License

MIT License — see [LICENSE](LICENSE).
