"""Generate limit_retrieval.pdf for the random-token LiMIT experiment.

The experiment assigns each vocabulary item a fixed random Gaussian vector,
embeds each document/query by summing its token vectors, and ranks documents by
query/document inner product. It sweeps embedding dimension on LiMIT-small and,
optionally, LiMIT-full.

Usage:
    python -m unlimit.limit_figure --mode run
    python -m unlimit.limit_figure --mode run --full
    python -m unlimit.limit_figure --mode plot
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]

from unlimit.datasets import load_limit
from unlimit.retrieval.metrics import build_qrels_tensor, retrieval_metrics_from_logits
from unlimit.retrieval.random_embeddings import (
    build_random_token_matrix,
    score_random_embeddings,
)
from unlimit.tokenizers.handmade import (
    UNK_TOKEN_ID,
    tokenize_corpus_records,
    tokenize_query_records,
)

DEFAULT_DIMS = [32, 64, 128, 256, 512, 1024, 2048, 4096]
BASE_SEED = 42
RESULTS_FILE = _REPO_ROOT / "results" / "unlimit" / "limit_figure" / "results.json"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "paper" / "figure"


def _membership_scores(
    corpus_tokens: list[list[int]],
    query_tokens: list[list[int]],
) -> torch.Tensor:
    """Baseline: documents containing all query tokens score ``1 / len(doc)``."""
    scores = torch.zeros(len(query_tokens), len(corpus_tokens), dtype=torch.float32)
    doc_sets = [set(tokens) for tokens in corpus_tokens]
    for query_idx, tokens in enumerate(query_tokens):
        required = set(tokens)
        if not required:
            continue
        for doc_idx, doc_set in enumerate(doc_sets):
            if required.issubset(doc_set):
                scores[query_idx, doc_idx] = 1.0 / max(len(corpus_tokens[doc_idx]), 1)
    return scores


def run_sweep(
    split: str,
    dims: list[int],
    base_seed: int = BASE_SEED,
) -> list[dict]:
    """Run the random-token embedding dimension sweep on one LiMIT split."""
    print(f"\n{'=' * 60}")
    print(f"Loading LiMIT-{split} ...")
    corpus, queries, qrels = load_limit(split)  # type: ignore[arg-type]
    tokenized_corpus = tokenize_corpus_records(corpus)
    tokenized_queries = tokenize_query_records(queries)

    corpus_ids = [record["_id"] for record in tokenized_corpus]
    query_ids = [record["_id"] for record in tokenized_queries]
    corpus_tokens = [record["token_ids"] for record in tokenized_corpus]
    query_tokens = [record["token_ids"] for record in tokenized_queries]

    print(
        f"  corpus docs: {len(corpus_tokens)}  |  "
        f"queries: {len(query_tokens)}  |  qrels: {len(qrels)}"
    )

    device = torch.device("cpu")
    labels = build_qrels_tensor(qrels, query_ids, corpus_ids, device)

    print("Computing membership baseline ...")
    membership_metrics = retrieval_metrics_from_logits(
        _membership_scores(corpus_tokens, query_tokens),
        labels,
    )

    results: list[dict] = []
    num_token_types = UNK_TOKEN_ID + 1
    for dim in dims:
        seed = base_seed + dim
        print(f"  d={dim} ...", end=" ", flush=True)
        token_matrix = build_random_token_matrix(num_token_types, dim, seed)
        scores = score_random_embeddings(corpus_tokens, query_tokens, token_matrix)
        metrics = retrieval_metrics_from_logits(scores, labels)
        metrics["dim"] = dim
        metrics["seed"] = seed
        metrics["split"] = split
        results.append(metrics)
        print(
            f"top2_EM={metrics['top2_exact_match']:.3f}  "
            f"R@1={metrics['recall_at_1']:.3f}  "
            f"R@2={metrics['recall_at_2']:.3f}  "
            f"mean_rank={metrics['mean_rank']:.2f}"
        )

    results.append(
        {
            "split": split,
            "dim": -1,
            "seed": -1,
            "top2_exact_match": membership_metrics["top2_exact_match"],
            "recall_at_1": membership_metrics["recall_at_1"],
            "recall_at_2": membership_metrics["recall_at_2"],
            "mean_rank": membership_metrics["mean_rank"],
            "num_queries_eval": membership_metrics["num_queries_eval"],
            "num_queries_top2_eval": membership_metrics["num_queries_top2_eval"],
        }
    )
    return results


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


_SPLIT_STYLE = {
    "small": ("o", "C0", "-"),
    "full": ("s", "C1", "--"),
}


def generate_plots(results: list[dict], output_dir: str) -> None:
    """Generate ``limit_retrieval.pdf`` from sweep results."""
    plt = _set_style()
    os.makedirs(output_dir, exist_ok=True)

    sweep = [row for row in results if row["dim"] >= 0]
    splits = sorted({row["split"] for row in sweep})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    ax = axes[0]
    for split in splits:
        points = sorted(
            [row for row in sweep if row["split"] == split],
            key=lambda row: row["dim"],
        )
        marker, color, linestyle = _SPLIT_STYLE.get(split, ("o", "gray", "-"))
        ax.plot(
            [point["dim"] for point in points],
            [point["top2_exact_match"] for point in points],
            marker=marker,
            color=color,
            linestyle=linestyle,
            linewidth=1.7,
            markersize=6,
            label=f"LiMIT-{split}",
        )
    ax.axhline(y=1.0, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(
        x=min(DEFAULT_DIMS) * 1.2,
        y=1.01,
        s="membership",
        fontsize=9,
        color="black",
        alpha=0.6,
    )
    ax.set_xlabel("Embedding dimension $d$")
    ax.set_ylabel("Top-2 exact match")
    ax.set_title("Top-2 exact match vs dimension")
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.05, 1.08)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    for split in splits:
        points = sorted(
            [row for row in sweep if row["split"] == split],
            key=lambda row: row["dim"],
        )
        marker, color, linestyle = _SPLIT_STYLE.get(split, ("o", "gray", "-"))
        ax.plot(
            [point["dim"] for point in points],
            [point["mean_rank"] for point in points],
            marker=marker,
            color=color,
            linestyle=linestyle,
            linewidth=1.7,
            markersize=6,
            label=f"LiMIT-{split}",
        )
    ax.axhline(y=1.0, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Embedding dimension $d$")
    ax.set_ylabel("Mean rank")
    ax.set_title("Mean rank vs dimension")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "LiMIT retrieval with random token embeddings",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()

    path = os.path.join(output_dir, "limit_retrieval.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LiMIT random-token retrieval figure"
    )
    parser.add_argument(
        "--mode",
        choices=["run", "plot"],
        default="run",
        help="'run' = run sweep + plot; 'plot' = plot from saved JSON",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include LiMIT-full dataset (~50k docs; slower)",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="*",
        default=DEFAULT_DIMS,
        help="Embedding dimensions to sweep",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for output PDF",
    )
    parser.add_argument(
        "--results-file",
        default=str(RESULTS_FILE),
        help="Path to save/load sweep results JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "run":
        all_results: List[dict] = []
        all_results.extend(run_sweep("small", args.dims, BASE_SEED))

        if args.full:
            full_dims = [dim for dim in args.dims if dim >= 64]
            all_results.extend(run_sweep("full", full_dims, BASE_SEED))

        os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
        with open(args.results_file, "w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2)
        print(f"\nResults saved to {args.results_file}")
    else:
        print(f"Loading results from {args.results_file}")
        with open(args.results_file, encoding="utf-8") as handle:
            all_results = json.load(handle)

    generate_plots(all_results, args.output_dir)


if __name__ == "__main__":
    main()
