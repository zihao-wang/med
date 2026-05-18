"""Shared LIMIT retrieval metrics (PyTorch)."""

from __future__ import annotations

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

    LIMIT official splits have two positives per query, so ``recall_at_2``
    reports the fraction of target items retrieved in the top two positions.
    """
    num_queries = logits.shape[0]
    ranks: list[int] = []
    hits1 = 0
    recall2_sum = 0.0
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
        k2 = min(2, scores.numel())
        if k2 > 0:
            top2_idx = torch.topk(scores, k=k2, largest=True).indices
            recall2_sum += float(y_full[qi, top2_idx].sum().item()) / float(pos.numel())
    mean_rank = sum(ranks) / len(ranks) if ranks else 0.0
    recall1 = hits1 / len(ranks) if ranks else 0.0
    recall2 = recall2_sum / len(ranks) if ranks else 0.0
    return {
        "mean_rank": mean_rank,
        "recall_at_1": recall1,
        "recall_at_2": recall2,
        "num_queries_eval": float(len(ranks)),
    }


__all__ = [
    "build_qrels_tensor",
    "retrieval_metrics_from_logits",
]
