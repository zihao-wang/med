"""Tests for retrieval metrics."""

import pytest
import torch

from unlimit.retrieval.metrics import (
    build_qrels_tensor,
    retrieval_metrics_from_logits,
)


def test_build_qrels_tensor():
    qrels = [
        {"query_id": "q0", "corpus_id": "c0", "score": 1},
        {"query_id": "q0", "corpus_id": "c1", "score": 1},
        {"query_id": "q1", "corpus_id": "c2", "score": 1},
    ]
    query_ids = ["q0", "q1"]
    corpus_ids = ["c0", "c1", "c2"]
    y = build_qrels_tensor(qrels, query_ids, corpus_ids, torch.device("cpu"))
    assert y.shape == (2, 3)
    assert y[0, 0].item()
    assert y[0, 1].item()
    assert not y[0, 2].item()
    assert not y[1, 0].item()
    assert not y[1, 1].item()
    assert y[1, 2].item()


def test_build_qrels_ignores_zero_scores():
    qrels = [
        {"query_id": "q0", "corpus_id": "c0", "score": 0},
    ]
    y = build_qrels_tensor(qrels, ["q0"], ["c0"], torch.device("cpu"))
    assert not y[0, 0].item()


def test_build_qrels_missing_ids():
    qrels = [
        {"query_id": "qX", "corpus_id": "c0", "score": 1},
    ]
    y = build_qrels_tensor(qrels, ["q0"], ["c0"], torch.device("cpu"))
    assert not y[0, 0].item()


def test_retrieval_metrics_perfect():
    """Perfect ranking: positives at positions 0, 1 for q0; pos at 1 for q1."""
    scores = torch.tensor([[10.0, 9.0, 1.0], [5.0, 10.0, 3.0]])
    y = torch.tensor([[True, True, False], [False, True, False]])
    m = retrieval_metrics_from_logits(scores, y)
    assert m["mean_rank"] == 1.0
    # q0: top-1 is idx 0 (score 10), positive → hit
    # q1: top-1 is idx 1 (score 10), positive → hit
    assert m["recall_at_1"] == 1.0
    assert m["recall_at_2"] == 1.0
    assert m["top2_exact_match"] == 1.0  # q0 has exactly 2 pos, both in top 2


def test_retrieval_metrics_worst():
    """Worst ranking: positives at the bottom (lowest scores)."""
    scores = torch.tensor([[1.0, 9.0, 0.2, 0.5]])
    y = torch.tensor([[True, False, True, True]])  # pos at 0, 2, 3
    m = retrieval_metrics_from_logits(scores, y)
    # Sorted: idx 1 (9.0, rank 1), idx 0 (1.0, rank 2), idx 3 (0.5, rank 3), idx 2 (0.2, rank 4)
    # Pos ranks: {2, 3, 4}, min = 2
    assert m["mean_rank"] == 2.0
    assert m["recall_at_1"] == 0.0  # top-1 is idx 1, not positive
    assert m["recall_at_2"] == pytest.approx(1.0 / 3.0)
    assert m["top2_exact_match"] == 0.0  # >2 pos, skipped


def test_retrieval_metrics_empty_positives():
    scores = torch.tensor([[10.0, 9.0]])
    y = torch.tensor([[False, False]])
    m = retrieval_metrics_from_logits(scores, y)
    assert m["num_queries_eval"] == 0.0
    assert m["mean_rank"] == 0.0
    assert m["recall_at_2"] == 0.0


def test_retrieval_metrics_single_positive():
    scores = torch.tensor([[2.0, 10.0, 1.0]])
    y = torch.tensor([[False, True, False]])
    m = retrieval_metrics_from_logits(scores, y)
    assert m["mean_rank"] == 1.0
    assert m["recall_at_1"] == 1.0
    assert m["recall_at_2"] == 1.0
    # top2_exact_match skipped (not exactly 2 pos)
    assert m["num_queries_top2_eval"] == 0.0
