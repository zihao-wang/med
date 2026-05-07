"""Generate limit_retrieval.pdf for the ICML 2026 paper.

Reproduces the LiMIT retrieval figure comparing training-free RP+OMP retrieval
quality against embedding dimension on real-world data (LiMIT-small and
optionally LiMIT-full). Demonstrates that sufficient dimension yields perfect
retrieval even without any training — supporting the paper's thesis that
limitations stem from learnability, not geometric capacity.

Usage:
    python -m med.unlimit.limit_figure --mode run         # run + plot (small)
    python -m med.unlimit.limit_figure --mode run --full  # include LiMIT-full
    python -m med.unlimit.limit_figure --mode plot        # plot from cached JSON
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]

from med.unlimit.datasets import load_limit
from med.unlimit.tokenizers.handmade import (
    UNK_TOKEN_ID,
    tokenize_corpus_records,
    tokenize_query_records,
)
from med.unlimit.retrieval.rp_omp_numpy import (
    build_token_matrix_np,
    omp_pair_doclocal_np,
    row_normalize_np,
    sum_token_rows_np,
)
from med.unlimit.retrieval.metrics import build_qrels_tensor, retrieval_metrics_from_logits

import torch

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DIMS = [8, 16, 32, 64, 128, 256, 512, 1024]
DEFAULT_OMP_STEPS = [0, 1, 4, 16, 64]
BASE_SEED = 42
RESULTS_FILE = "limit_results.json"


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------


def _membership_scores(
    corpus_tokens: List[List[int]],
    query_tokens: List[List[int]],
) -> np.ndarray:
    """Baseline: docs containing all query tokens score 1/len(doc)."""
    nq = len(query_tokens)
    nc = len(corpus_tokens)
    out = np.zeros((nq, nc), dtype=np.float32)
    for i, qt in enumerate(query_tokens):
        need = set(qt)
        if not need:
            continue
        for j, dt in enumerate(corpus_tokens):
            if need.issubset(set(dt)):
                out[i, j] = 1.0 / max(len(dt), 1)
    return out


def run_sweep(
    split: str,
    dims: List[int],
    omp_steps_list: List[int],
    base_seed: int = BASE_SEED,
) -> List[dict]:
    """Run RP+OMP sweep over (dim, omp_steps) on a LiMIT split.

    Returns a list of result dicts with keys: split, dim, omp_steps,
    top2_exact_match, recall_at_1, mean_rank.
    """
    print(f"\n{'='*60}")
    print(f"Loading LiMIT-{split} …")
    corpus, queries, qrels = load_limit(split)  # type: ignore[arg-type]
    tc = tokenize_corpus_records(corpus)
    tq = tokenize_query_records(queries)

    corpus_ids = [r["_id"] for r in tc]
    query_ids = [r["_id"] for r in tq]
    corpus_tokens = [r["token_ids"] for r in tc]
    query_tokens = [r["token_ids"] for r in tq]

    nc = len(corpus_tokens)
    nq = len(query_tokens)
    print(f"  corpus docs: {nc}  |  queries: {nq}  |  qrels: {len(qrels)}")

    # Build binary relevance matrix for metric computation
    dev = torch.device("cpu")
    y = build_qrels_tensor(qrels, query_ids, corpus_ids, dev)

    # Membership baseline (training-free upper bound)
    print("Computing membership baseline …")
    scores_mem = _membership_scores(corpus_tokens, query_tokens)
    mem_metrics = _evaluate_np(scores_mem, y)

    results: List[dict] = []
    for dim in dims:
        seed = base_seed + dim
        X = build_token_matrix_np(UNK_TOKEN_ID + 1, dim, seed)
        D = sum_token_rows_np(corpus_tokens, X)
        Q = sum_token_rows_np(query_tokens, X)

        for K in omp_steps_list:
            label = f"d={dim}, omp={K}"
            print(f"  {label} …", end=" ", flush=True)
            scores = _scores_local_omp_np(D, Q, corpus_tokens, query_tokens, X, K)
            m = _evaluate_np(scores, y)
            m["dim"] = dim
            m["omp_steps"] = K
            m["split"] = split
            results.append(m)
            print(
                f"top2_EM={m['top2_exact_match']:.3f}  "
                f"R@1={m['recall_at_1']:.3f}  "
                f"mean_rank={m['mean_rank']:.2f}"
            )

    # Add membership baseline row
    results.append(
        {
            "split": split,
            "dim": -1,  # sentinel: membership
            "omp_steps": -1,
            "top2_exact_match": mem_metrics["top2_exact_match"],
            "recall_at_1": mem_metrics["recall_at_1"],
            "mean_rank": mem_metrics["mean_rank"],
        }
    )
    return results


# ---------------------------------------------------------------------------
# NumPy scoring helpers (adapted from rp_omp_numpy)
# ---------------------------------------------------------------------------


def _scores_local_omp_np(
    D: np.ndarray,
    Q: np.ndarray,
    corpus_tokens: List[List[int]],
    query_tokens: List[List[int]],
    X_raw: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Compute query-local OMP scores (NumPy backend)."""
    Xn = row_normalize_np(X_raw)
    nq, nc = len(query_tokens), len(corpus_tokens)

    # Precompute per-document residuals
    r_empty: List[np.ndarray] = []
    r_forbid: List[Dict[int, np.ndarray]] = []
    for j in range(nc):
        dt = corpus_tokens[j]
        y = D[j].astype(np.float32)
        r_empty.append(omp_pair_doclocal_np(y, Xn, dt, set(), n_steps))
        dct: Dict[int, np.ndarray] = {}
        for t in set(dt):
            dct[t] = omp_pair_doclocal_np(y, Xn, dt, {t}, n_steps)
        r_forbid.append(dct)

    out = np.zeros((nq, nc), dtype=np.float32)
    for i in range(nq):
        forb = set(query_tokens[i])
        qi = Q[i].astype(np.float32)
        for j in range(nc):
            fb = forb & set(corpus_tokens[j])
            if not fb:
                r = r_empty[j]
            elif len(fb) == 1:
                r = r_forbid[j][next(iter(fb))]
            else:
                r = omp_pair_doclocal_np(D[j].astype(np.float32), Xn, corpus_tokens[j], fb, n_steps)
            out[i, j] = float(qi @ r)
    return out


def _evaluate_np(scores: np.ndarray, y: torch.Tensor) -> dict:
    """Bridge NumPy scores → PyTorch metrics."""
    t = torch.from_numpy(scores.astype(np.float32))
    return retrieval_metrics_from_logits(t, y)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _set_style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    return plt


# Distinct markers/colors for OMP steps
_OMP_STYLE = {
    0:  ("s", "C0", "-"),
    1:  ("o", "C1", "-"),
    4:  ("^", "C2", "-"),
    16: ("D", "C3", "-"),
    64: ("v", "C4", "-"),
}


def generate_plots(results: List[dict], output_dir: str, full_included: bool) -> None:
    """Generate limit_retrieval.pdf from sweep results."""
    plt = _set_style()

    # Separate membership baselines from sweep rows
    sweep = [r for r in results if r["dim"] >= 0]
    mem_rows = [r for r in results if r["dim"] == -1]

    splits = sorted(set(r["split"] for r in sweep))
    omp_values = sorted(set(r["omp_steps"] for r in sweep))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ---- Panel (a): Top-2 exact match vs dimension ----
    ax = axes[0]
    for split in splits:
        split_rows = [r for r in sweep if r["split"] == split]
        for K in omp_values:
            pts = sorted(
                [r for r in split_rows if r["omp_steps"] == K],
                key=lambda r: r["dim"],
            )
            if not pts:
                continue
            xs = [p["dim"] for p in pts]
            ys = [p["top2_exact_match"] for p in pts]
            marker, color, ls = _OMP_STYLE.get(K, ("o", "gray", "-"))
            label = f"{split}, OMP={K}"
            ax.plot(xs, ys, marker=marker, color=color, linestyle=ls,
                    linewidth=1.5, markersize=6, label=label, alpha=0.85)

    # Membership baseline (horizontal line at y=1.0)
    ax.axhline(y=1.0, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(x=min(DEFAULT_DIMS) * 1.2, y=1.01, s="membership", fontsize=9,
            color="black", alpha=0.6)

    ax.set_xlabel("Embedding dimension $d$")
    ax.set_ylabel("Top-2 exact match")
    ax.set_title("Top-2 exact match vs dimension (LiMIT)")
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.05, 1.08)
    ax.grid(True, alpha=0.25)
    if len(splits) <= 1:
        ax.legend(fontsize=8, loc="lower right")

    # ---- Panel (b): Mean rank vs dimension ----
    ax = axes[1]
    for split in splits:
        split_rows = [r for r in sweep if r["split"] == split]
        for K in omp_values:
            pts = sorted(
                [r for r in split_rows if r["omp_steps"] == K],
                key=lambda r: r["dim"],
            )
            if not pts:
                continue
            xs = [p["dim"] for p in pts]
            ys = [p["mean_rank"] for p in pts]
            marker, color, ls = _OMP_STYLE.get(K, ("o", "gray", "-"))
            label = f"{split}, OMP={K}"
            ax.plot(xs, ys, marker=marker, color=color, linestyle=ls,
                    linewidth=1.5, markersize=6, label=label, alpha=0.85)

    # Membership baseline (mean_rank = 1.0)
    ax.axhline(y=1.0, color="black", linestyle=":", linewidth=1, alpha=0.6)

    ax.set_xlabel("Embedding dimension $d$")
    ax.set_ylabel("Mean rank")
    ax.set_title("Mean rank vs dimension (LiMIT)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    if len(splits) <= 1:
        ax.legend(fontsize=8, loc="upper right")

    # Shared legend for multi-split case
    if len(splits) > 1:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=8, loc="lower center",
                   ncol=min(6, len(handles)), bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(
        "LiMIT retrieval: RP+OMP (training-free) quality vs embedding dimension",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()

    path = os.path.join(output_dir, "limit_retrieval.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate LiMIT retrieval figure for MED paper"
    )
    p.add_argument(
        "--mode", choices=["run", "plot"], default="run",
        help="'run' = run sweep + plot; 'plot' = plot from saved JSON",
    )
    p.add_argument(
        "--full", action="store_true",
        help="Include LiMIT-full dataset (~50k docs; slower)",
    )
    p.add_argument(
        "--dims", type=int, nargs="*", default=DEFAULT_DIMS,
        help="Embedding dimensions to sweep",
    )
    p.add_argument(
        "--omp-steps", type=int, nargs="*", default=DEFAULT_OMP_STEPS,
        help="OMP step counts to sweep",
    )
    p.add_argument(
        "--output-dir", default=str(_REPO_ROOT / "paper"),
        help="Directory for output PDF",
    )
    p.add_argument(
        "--results-file", default=str(_REPO_ROOT / RESULTS_FILE),
        help="Path to save/load sweep results JSON",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "run":
        all_results: List[dict] = []

        # LiMIT-small
        all_results.extend(
            run_sweep("small", args.dims, args.omp_steps, BASE_SEED)
        )

        # LiMIT-full (optional)
        if args.full:
            # Use a sparser grid for full to keep runtime manageable
            full_dims = [d for d in args.dims if d >= 64]
            full_omp = [k for k in args.omp_steps if k in (0, 1, 16, 64)]
            all_results.extend(
                run_sweep("full", full_dims, full_omp, BASE_SEED)
            )

        os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
        with open(args.results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.results_file}")

    else:
        print(f"Loading results from {args.results_file}")
        with open(args.results_file) as f:
            all_results = json.load(f)

    full_included = any(r["split"] == "full" for r in all_results)
    generate_plots(all_results, args.output_dir, full_included)


if __name__ == "__main__":
    main()
