"""Shared LiMIT retrieval metrics (PyTorch)."""

from __future__ import annotations

from typing import Any

import torch


def build_qrels_tensor(
    qrels: list[dict],
    query_ids: list[str],
    corpus_ids: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Binary relevance matrix (Q, N)."""
    qi = {q: i for i, q in enumerate(query_ids)}
    di = {c: j for j, c in enumerate(corpus_ids)}
    y = torch.zeros(len(query_ids), len(corpus_ids), dtype=torch.bool, device=device)
    for r in qrels:
        if int(r["score"]) <= 0:
            continue
        i = qi.get(r["query_id"])
        j = di.get(r["corpus_id"])
        if i is None or j is None:
            continue
        y[i, j] = True
    return y


@torch.no_grad()
def retrieval_metrics_from_logits(
    logits: torch.Tensor,
    y_full: torch.Tensor,
) -> dict[str, float]:
    """
    ``logits``: (Q, N) similarity scores.
    ``y_full``: (Q, N) bool relevance.

    LiMIT: each query has exactly two positives in official splits; primary
    diagnostic: ``top2_exact_match`` when ``pos.numel() == 2``.
    """
    num_queries = logits.shape[0]
    ranks: list[int] = []
    hits1 = 0
    top2_hits = 0
    n_top2_eval = 0
    for qi in range(num_queries):
        scores = logits[qi]
        pos = y_full[qi].nonzero(as_tuple=False).squeeze(-1)
        if pos.numel() == 0:
            continue
        order = torch.argsort(scores, descending=True)
        rank_1b = torch.empty_like(order)
        rank_1b[order] = torch.arange(
            1, scores.numel() + 1, device=scores.device, dtype=order.dtype
        )
        best_rank = int(rank_1b[pos].min().item())
        ranks.append(best_rank)
        top = int(scores.argmax().item())
        if bool(y_full[qi, top].item()):
            hits1 += 1
        if pos.numel() == 2:
            n_top2_eval += 1
            top2_idx = torch.topk(scores, k=2, largest=True).indices
            if set(top2_idx.tolist()) == set(pos.tolist()):
                top2_hits += 1
    mean_rank = sum(ranks) / len(ranks) if ranks else 0.0
    recall1 = hits1 / len(ranks) if ranks else 0.0
    top2_exact_match = top2_hits / n_top2_eval if n_top2_eval else 0.0
    return {
        "mean_rank": mean_rank,
        "recall_at_1": recall1,
        "top2_exact_match": top2_exact_match,
        "num_queries_eval": float(len(ranks)),
        "num_queries_top2_eval": float(n_top2_eval),
    }


@torch.no_grad()
def metrics_tqdm_postfix(logits: torch.Tensor, y: torch.Tensor) -> dict[str, str]:
    """Compact metric strings for :mod:`tqdm` ``set_postfix`` (partial logits slice)."""
    m = retrieval_metrics_from_logits(logits, y)
    return {
        "R@1": f"{m['recall_at_1']:.3f}",
        "top2EM": f"{m['top2_exact_match']:.3f}",
    }


__all__ = [
    "build_qrels_tensor",
    "metrics_tqdm_postfix",
    "retrieval_metrics_from_logits",
]
