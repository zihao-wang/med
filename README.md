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
bash scripts/generate_compare_plots.sh --mode run
```

This runs centroid embedding experiments with $k=2$ and inner product scoring. It searches for feasible embeddings across a grid of $n$ (number of points) and $d$ (dimension) values, then saves the results to `compare_plot_results.json` and writes the two PDFs to `paper/`.

Expected runtime: ~10–30 minutes on a modern GPU; longer on CPU. Progress is printed to stdout.

**2. Regenerate plots from saved results (no re-computation):**

```bash
bash scripts/generate_compare_plots.sh --mode plot
```

Reads `compare_plot_results.json` and redraws the PDFs. Useful for tweaking plot styling without rerunning the experiment.

**3. Reproduce with other scoring functions:**

```bash
bash scripts/generate_compare_plots.sh --mode run --scoring l2
bash scripts/generate_compare_plots.sh --mode run --scoring cosine
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
bash scripts/generate_compare_plots.sh --mode run \
    --n-values 8 16 32 64 \
    --d-values 1 2 3 4 5 6 7 8 9 10
```

Reduce training epochs for a rougher but faster sweep:

```bash
bash scripts/generate_compare_plots.sh --mode run \
    --num-epochs 500 --patience 50
```

### LiMIT retrieval figure

The paper also includes a figure (Figure 2) showing **training-free** retrieval quality on the real-world LiMIT benchmark (RP+OMP with random Gaussian token vectors). This demonstrates that sufficient embedding dimension enables perfect retrieval even without any training — directly supporting the paper's thesis that limitations stem from **learnability**, not geometric capacity.

| Panel | File | Content |
|-------|------|---------|
| Fig 2a | `paper/limit_retrieval.pdf` (left) | Top-2 exact match vs embedding dimension $d$ |
| Fig 2b | `paper/limit_retrieval.pdf` (right) | Mean rank vs embedding dimension $d$ (log-log) |

**1. Generate the figure (LiMIT-small, ~1 minute):**

```bash
bash scripts/generate_limit_figure.sh --mode run
```

Loads the LiMIT-small dataset (46 docs, 1000 queries), runs RP+OMP retrieval across embedding dimensions $d \in \{8, 16, \ldots, 1024\}$ and OMP step counts, then saves `paper/limit_retrieval.pdf`.

**2. Include LiMIT-full (~50k docs, slower):**

```bash
bash scripts/generate_limit_figure.sh --mode run --full
```

**3. Regenerate from cached results:**

```bash
bash scripts/generate_limit_figure.sh --mode plot
```

### What the LiMIT experiment shows

LiMIT (LIkes Memory Identification Test) is a retrieval benchmark where each query asks "Who likes X?" and the corpus contains person profiles with comma-separated likes. Every query has exactly 2 relevant documents. RP+OMP (Random Projection + Orthogonal Matching Pursuit) is a **training-free** method: token vectors are random Gaussian, document/query embeddings are sums of token vectors, and scoring uses document-local OMP residuals. As dimension $d$ increases, retrieval quality (top-2 exact match) climbs from near-zero at $d=16$ to perfect at $d \ge 64$ (with sufficient OMP steps).

## Project structure

```
├── paper/                  # LaTeX source for the ICML 2026 paper
│   ├── icml2026.tex        #   main entry point
│   ├── content.tex         #   main body
│   ├── appendix.tex        #   supplementary proofs + experiment code listing
│   ├── custom_command.tex  #   custom math macros
│   └── preprint.bib        #   bibliography
├── med/                    # Python package code
│   ├── scoring.py          #   shared scoring functions (inner product, L2, cosine, L1)
│   ├── plotting.py         #   WBNL curve reference and plot styling
│   ├── mean_embedding/     #   centroid embedding experiments
│   │   ├── cli.py          #     package CLI for custom sweeps
│   │   ├── compare_plots.py #     paper compare-plot runner
│   │   ├── trainer_gd.py   #     full-batch GD trainer
│   │   ├── trainer_sgd.py  #     stochastic trainer (random k-subsets)
│   │   └── experiment.py   #     experiment orchestration (binary search, grid search)
│   ├── cyclic_polytope/    #   cyclic polytope face verification
│   │   ├── generator.py    #     moment curve point generation
│   │   └── verification.py #     LP-based face check
│   └── unlimit/            #   LiMIT retrieval library (RP+OMP)
│       ├── datasets/       #     LiMIT/LiMIT-small JSONL loader
│       ├── limit_figure.py #     paper LiMIT figure runner
│       ├── tokenizers/     #     handmade phrase + Qwen subword tokenizers
│       └── retrieval/      #     RP+OMP scoring (NumPy/PyTorch) + metrics
├── scripts/                # Bash launchers only
│   ├── generate_compare_plots.sh  # launch med.mean_embedding.compare_plots
│   ├── generate_limit_figure.sh   # launch med.unlimit.limit_figure
│   ├── run_all_experiments.sh     # master: joint grid for GD + SGD
│   ├── run_joint_dependency.sh    # joint (k, n) grid sweep
│   ├── run_k_dependency.sh        # MED vs k at fixed n
│   └── run_m_dependency.sh        # MED vs n at fixed k
└── requirements.txt
```

## Running custom experiments

Single $k$, single scoring function:

```bash
uv run python -m med.mean_embedding.cli --k 2 --n_values 8 16 32 64 128 --scoring_function inner_product
```

Grid sweep over $k$ and $n$:

```bash
uv run python -m med.mean_embedding.cli --k_values 2 3 4 5 --n_values 8 16 32 64 128 256
```

Using SGD instead of GD:

```bash
uv run python -m med.mean_embedding.cli --trainer sgd --k 2 --n_values 8 16 32 64 128 256
```

Results are saved as JSON in the current working directory.

Run a full sweep from shell scripts (saves to `results/`):

```bash
bash scripts/run_joint_dependency.sh gd
bash scripts/run_k_dependency.sh gd
bash scripts/run_m_dependency.sh gd
```

## Build the paper

```bash
cd paper && latexmk -pdf icml2026.tex
```

## License

MIT License — see [LICENSE](LICENSE).
