# Minimal Embeddable Dimension (MED)

Companion code for the ICML 2026 paper: *"$\mathbb{R}^{2k}$ is Theoretically Large Enough for Embedding-based Top-$k$ Retrieval."*

Studies the minimal dimension required to embed subset memberships into vector spaces, with the runnable experiments using inner product scoring.

## Quick start

```bash
git clone https://github.com/zihao-wang/med.git
cd med
uv sync                            # or: pip install -r requirements.txt
```

Requires Python 3.10+.

## Reproducing the paper's figures

The paper contains one upper-bound witness figure with two panels, comparing cyclic-polytope and centroid GD witnesses against the fitted curve from Weller et al. (2025):

| Panel | File | Content |
|-------|------|---------|
| Fig 1a | `paper/figure/compare_plot1.pdf` | Critical number of points $m^*(d)$ vs dimension $d$ |
| Fig 1b | `paper/figure/compare_plot2.pdf` | Critical dimension $d^*(m)$ vs number of points $m$ (log-scale x) |

### Step-by-step

**1. Generate the paper tables and figures from scratch:**

```bash
bash scripts/run_paper_upper_bounds.sh
```

This runs the default $k=2$ upper-bound witness grid with cyclic-polytope and centroid GD witnesses, then writes `results/upper_bound_witness/results.json`, `results/upper_bound_witness/upper_bound_witness_table.{csv,tex}`, and the two PDFs to `paper/figure/`.

Expected runtime: ~10–30 minutes on a modern GPU; longer on CPU. Progress is printed to stdout.

**2. Regenerate plots and tables from saved results (no re-computation):**

```bash
python -m med.paper_pipeline --mode plot
```

Reads `results/upper_bound_witness/results.json` or falls back to `results/upper_bound_witness/upper_bound_witness_table.csv`, then redraws the PDFs and table files.

### What the experiment does

For each $m$ value, the pipeline searches for the minimal witness dimension for two constructions:

- **Cyclic polytope witness** — verifies the moment-curve squared-polynomial construction.
- **Centroid GD witness** — trains $m$ learnable vectors in $\mathbb{R}^d$ and checks whether all top-$k$ centroid constraints have zero violations.

The results are plotted against the WBNL fitted curve $m(d) = -10.53 + 4.03d + 0.052d^2 + 0.0037d^3$ from Weller et al. (2025).

### Faster reproduction (reduced grid)

To get a quick sense of the results with fewer $(n, d)$ points:

```bash
bash scripts/run_paper_upper_bounds.sh --m_values 10 20 40 80
```

Reduce training epochs for a rougher but faster sweep:

```bash
bash scripts/run_paper_upper_bounds.sh --num_epochs 500 --patience 50
```

### LiMIT retrieval figure

The paper also includes a figure (Figure 2) showing **training-free** retrieval quality on the real-world LiMIT benchmark with random Gaussian token vectors. This demonstrates that sufficient embedding dimension enables strong retrieval even without any training — directly supporting the paper's thesis that limitations stem from **learnability**, not geometric capacity.

| Panel | File | Content |
|-------|------|---------|
| Fig 2a | `paper/figure/limit_retrieval.pdf` (left) | Top-2 exact match vs embedding dimension $d$ |
| Fig 2b | `paper/figure/limit_retrieval.pdf` (right) | Mean rank vs embedding dimension $d$ (log-log) |

**1. Generate the figure (LiMIT-small, ~1 minute):**

```bash
python -m unlimit.limit_figure --mode run
```

Loads the packaged LiMIT-small dataset (46 docs, 1000 queries), runs random-token retrieval across embedding dimensions $d \in \{8, 16, \ldots, 1024\}$, then saves `paper/figure/limit_retrieval.pdf`.

**2. Include LiMIT-full (~50k docs, slower):**

```bash
python -m unlimit.limit_figure --mode run --full
```

**3. Regenerate from cached results:**

```bash
python -m unlimit.limit_figure --mode plot
```

### What the LiMIT experiment shows

LiMIT (LIkes Memory Identification Test) is a retrieval benchmark where each query asks "Who likes X?" and the corpus contains person profiles with comma-separated likes. Every query has exactly 2 relevant documents. The included training-free baseline assigns each vocabulary item a random Gaussian vector, embeds documents and queries by summing token vectors, and scores with inner product. As dimension $d$ increases, retrieval quality improves without learned parameters.

## Project structure

```
├── paper/                  # LaTeX source for the ICML 2026 paper
│   ├── icml2026.tex        #   main entry point
│   ├── content.tex         #   main body
│   ├── appendix.tex        #   supplementary proofs + experiment code listing
│   ├── custom_command.tex  #   custom math macros
│   └── preprint.bib        #   bibliography
├── med/                    # Python package code
│   ├── scoring.py          #   shared inner product scoring
│   ├── plotting.py         #   WBNL curve reference and plot styling
│   ├── experiment.py       #   shared experiment orchestration (binary search, grid)
│   ├── search.py           #   shared binary search over dimensions
│   ├── mean_embedding/     #   centroid embedding experiments
│   │   ├── cli.py          #     package CLI for custom sweeps
│   │   ├── trainer_gd.py   #     full-batch GD trainer
│   │   └── experiment.py   #     factory wrapping shared Experiment
│   ├── cyclic_polytope/    #   cyclic polytope face verification
│   │   ├── construct.py    #     moment curve points + squared-polynomial query construction
│   │   ├── cli.py          #     package CLI for custom sweeps
│   │   └── experiment.py   #     factory wrapping shared Experiment
├── unlimit/                # LiMIT retrieval library
│   ├── datasets/           #   packaged LiMIT/LiMIT-small JSONL loader
│   ├── limit_figure.py     #   paper LiMIT figure runner
│   ├── tokenizers/         #   handmade phrase tokenizer
│   └── retrieval/          #   random-token scoring + metrics
├── scripts/                # Bash launchers for result-directory runs
│   ├── run_paper_upper_bounds.sh # paper upper-bound witness pipeline
│   └── run_unlimit.sh      #   LiMIT random-token embedding sweep
└── requirements.txt
```

## Running custom experiments

Single $k$, single scoring function:

```bash
uv run python -m med.mean_embedding.cli --k_values 2 --n_values 10 20 40 80 160 --scoring_function inner_product
```

Grid sweep over $k$ and $n$:

```bash
uv run python -m med.mean_embedding.cli --k_values 2 3 4 5 --n_values 10 20 40 80 160 320
```

Results are saved as JSON in the current working directory.

Run a full sweep from shell scripts (saves to deterministic `results/` locations with logs):

```bash
bash scripts/run_paper_upper_bounds.sh
bash scripts/run_unlimit.sh
```

The paper pipeline runs the default $k=2$, $m \in \{10,20,40,80,160,320,640\}$ inner-product
upper-bound witness grid for both cyclic polytope and centroid GD, writes a combined
JSON/CSV/LaTeX table under `results/upper_bound_witness/`, and writes final paper
tables to `paper/table/` and figures to `paper/figure/`.

## Build the paper

```bash
cd paper && latexmk -pdf icml2026.tex
```

## License

MIT License — see [LICENSE](LICENSE).
