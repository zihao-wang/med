"""Generate compare_plot1.pdf and compare_plot2.pdf for the ICML 2026 paper.

Reproduces the comparison figures between centroid embedding results and the
WBNL fitted curve (Weller et al., 2025). Meant to be run from the repo root.

Plot 1: critical number of points m*(d) vs dimension d.
Plot 2: critical dimension d*(m) vs number of points m (log-scale x).

Usage:
    python -m med.mean_embedding.compare_plots --mode run   # run experiments then plot
    python -m med.mean_embedding.compare_plots --mode plot  # plot from saved results
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]

from med._device import resolve_device
from med.defaults import DEFAULT_M_VALUES
from med.mean_embedding.checker import MeanEmbeddingChecker
from med.plotting import (
    invert_wbnl_curve,
    set_paper_style,
    wbnl_critical_m,
)


# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
DEFAULT_N_VALUES = DEFAULT_M_VALUES
DEFAULT_D_VALUES = list(range(1, 31))
DEFAULT_K = 2
DEFAULT_SCORING = "inner_product"
DEFAULT_NUM_EPOCHS = 1000
DEFAULT_PATIENCE = 1000
DEFAULT_LR = 2.0
RESULTS_FILE = _REPO_ROOT / "results" / "mean_embedding" / "compare_plots" / "results.json"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "paper" / "figure"


# ---------------------------------------------------------------------------
# Core search routines
# ---------------------------------------------------------------------------


def find_minimal_d(
    n: int,
    k: int,
    scoring: str,
    d_min: int = 1,
    d_max: int = 50,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    learning_rate: float = DEFAULT_LR,
) -> Optional[int]:
    """Binary search for minimal dimension d where a feasible embedding exists."""
    left, right = d_min, d_max
    best_d: Optional[int] = None
    checker = MeanEmbeddingChecker(
        m=n,
        k=k,
        scoring_function=scoring,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        patience=patience,
    )

    while left <= right:
        mid = (left + right) // 2
        result = checker.check(mid)
        print(f"  [d*(n)] n={n}, d={mid} -> feasible={result.feasible}, violations={result.details.get('violations', -1)}")

        if result.feasible:
            best_d = mid
            right = mid - 1
        else:
            left = mid + 1

    return best_d


def find_maximal_n(
    d: int,
    k: int,
    scoring: str,
    n_min: int = 4,
    n_max: int = 512,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    learning_rate: float = DEFAULT_LR,
) -> Optional[int]:
    """Binary search for maximum number of points n with a feasible embedding in dimension d."""
    left, right = n_min, n_max
    best_n: Optional[int] = None

    while left <= right:
        mid = (left + right) // 2
        checker = MeanEmbeddingChecker(
            m=mid,
            k=k,
            scoring_function=scoring,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            patience=patience,
        )
        result = checker.check(d)
        print(f"  [m*(d)] d={d}, n={mid} -> feasible={result.feasible}, violations={result.details.get('violations', -1)}")

        if result.feasible:
            best_n = mid
            left = mid + 1
        else:
            right = mid - 1

    return best_n


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


def _inverted_med_frontier(d_star_data: dict) -> tuple[list[int], list[float]]:
    pairs = [
        (int(d), int(m))
        for m, d in d_star_data.items()
        if d is not None and int(d) > 0
    ]
    if not pairs:
        return [], []

    best_m_by_d: dict[int, int] = {}
    for d, m in pairs:
        best_m_by_d[d] = max(m, best_m_by_d.get(d, 0))

    frontier: list[tuple[int, int]] = []
    best_m = 0
    for d, m in sorted(best_m_by_d.items()):
        if m > best_m:
            frontier.append((d, m))
            best_m = m

    return [d for d, _ in frontier], [float(m) for _, m in frontier]


def generate_plots(results: dict, output_dir: str) -> None:
    """Generate compare_plot1.pdf and compare_plot2.pdf from experiment results."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_paper_style()

    d_star_data = results["d_star"]  # {n: d} pairs

    # ---- Plot 1: m*(d) vs d (critical points vs dimension) ----
    fig1, ax1 = plt.subplots(figsize=(8, 5))

    # WBNL curve
    d_vals, m_vals = _inverted_med_frontier(d_star_data)
    max_d = max(d_vals) if d_vals else 10
    d_range = np.linspace(1, max_d, 200)
    ax1.plot(
        d_range,
        wbnl_critical_m(d_range),
        "k--",
        linewidth=2,
        label="WBNL (2025) fitted",
    )

    # Our centroid results: inverted d*(m) frontier with redundant points removed.
    if d_vals:
        ax1.plot(
            d_vals,
            m_vals,
            "bo-",
            linewidth=2,
            markersize=7,
            label="Centroid (ours)",
        )

    ax1.set_xlabel("Dimension $d$")
    ax1.set_ylabel("Critical number of points $m^*$")
    ax1.set_title("Critical points vs dimension ($k=2$)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    path1 = os.path.join(output_dir, "compare_plot1.pdf")
    fig1.savefig(path1)
    plt.close(fig1)
    print(f"Saved {path1}")

    # ---- Plot 2: d*(m) vs m (critical dimension vs points, log-scale) ----
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    # WBNL curve (inverted)
    m_range = np.logspace(
        np.log10(5), np.log10(max(int(k) for k in d_star_data.keys()) * 1.5), 200
    )
    ax2.plot(
        m_range,
        invert_wbnl_curve(m_range),
        "k--",
        linewidth=2,
        label="WBNL (2025) fitted",
    )

    # Our centroid results: d*(m)
    n_vals = sorted(int(k) for k in d_star_data.keys() if d_star_data[k] is not None)
    d_min_vals = [d_star_data[str(n)] for n in n_vals]
    ax2.plot(
        n_vals, d_min_vals, "bo-", linewidth=2, markersize=7, label="Centroid (ours)"
    )

    ax2.set_xlabel("Number of points $m$")
    ax2.set_ylabel("Critical dimension $d^*$")
    ax2.set_xscale("log")
    ax2.set_title("Critical dimension vs number of points ($k=2$)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    path2 = os.path.join(output_dir, "compare_plot2.pdf")
    fig2.savefig(path2)
    plt.close(fig2)
    print(f"Saved {path2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate compare plots for MED paper (centroid vs WBNL)"
    )
    p.add_argument(
        "--mode",
        choices=["run", "plot"],
        default="run",
        help="'run' = run experiments + plot; 'plot' = plot from saved JSON",
    )
    p.add_argument(
        "--scoring",
        default=DEFAULT_SCORING,
        choices=["inner_product", "l2", "cosine", "l1"],
    )
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument(
        "--n-values",
        type=int,
        nargs="*",
        default=DEFAULT_N_VALUES,
        help="n values for d*(n) search",
    )
    p.add_argument(
        "--d-values",
        type=int,
        nargs="*",
        default=DEFAULT_D_VALUES,
        help="d values for m*(d) search",
    )
    p.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for output PDFs",
    )
    p.add_argument(
        "--results-file",
        default=str(RESULTS_FILE),
        help="Path to save/load experiment results JSON",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "run":
        dev = resolve_device()
        print("=== Running centroid embedding experiments (k=2) ===")
        print(
            f"Device: {dev}, scoring: {args.scoring}, epochs: {args.num_epochs}, patience: {args.patience}"
        )

        # Search d*(n) for each n
        print("\n--- Phase 1: d*(n) — minimal dimension for each n ---")
        d_star: Dict[str, Optional[int]] = {}
        for n in args.n_values:
            d = find_minimal_d(
                n=n,
                k=args.k,
                scoring=args.scoring,
                num_epochs=args.num_epochs,
                patience=args.patience,
                learning_rate=args.lr,
            )
            d_star[str(n)] = d
            print(f"  d*({n}) = {d}")

        # Search m*(d) for each d
        print("\n--- Phase 2: m*(d) — maximal points for each dimension ---")
        m_star: Dict[str, Optional[int]] = {}
        for d in args.d_values:
            n = find_maximal_n(
                d=d,
                k=args.k,
                scoring=args.scoring,
                num_epochs=args.num_epochs,
                patience=args.patience,
                learning_rate=args.lr,
            )
            m_star[str(d)] = n
            print(f"  m*({d}) = {n}")

        results = {
            "d_star": d_star,
            "m_star": m_star,
            "k": args.k,
            "scoring": args.scoring,
            "learning_rate": args.lr,
        }
        os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
        with open(args.results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.results_file}")

    else:
        print(f"Loading results from {args.results_file}")
        with open(args.results_file) as f:
            results = json.load(f)

    generate_plots(results, args.output_dir)


if __name__ == "__main__":
    main()
