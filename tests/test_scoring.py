"""Tests for scoring.py — shared inner product scoring."""

import pytest
import torch

from med.scoring import compute_scores


B = 4
n = 6
d = 3


@pytest.fixture
def vectors():
    torch.manual_seed(0)
    return torch.randn(n, d)


@pytest.fixture
def subset_sums():
    torch.manual_seed(1)
    return torch.randn(B, d)


def test_inner_product_shape(subset_sums, vectors):
    scores = compute_scores(subset_sums, vectors, "inner_product")
    assert scores.shape == (B, n)


def test_inner_product_equals_matmul(subset_sums, vectors):
    scores = compute_scores(subset_sums, vectors, "inner_product")
    expected = torch.matmul(subset_sums, vectors.T)
    assert torch.allclose(scores, expected)


def test_inner_product_higher_is_better(vectors):
    """A vector aligned with the subset sum should score higher."""
    q = vectors[0].clone()
    scores = compute_scores(q.unsqueeze(0), vectors, "inner_product")
    assert scores[0, 0] > scores[0, 1:].mean()


@pytest.mark.parametrize("scoring_function", ["l2", "cosine", "l1", "manhattan"])
def test_non_inner_product_scoring_raises(scoring_function):
    with pytest.raises(NotImplementedError):
        compute_scores(torch.randn(2, 3), torch.randn(4, 3), scoring_function)
