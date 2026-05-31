"""Tests for scoring.py — shared scoring functions."""

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


def test_cosine_shape(subset_sums, vectors):
    scores = compute_scores(subset_sums, vectors, "cosine")
    assert scores.shape == (B, n)


def test_cosine_range(subset_sums, vectors):
    scores = compute_scores(subset_sums, vectors, "cosine")
    assert (scores >= -1.0001).all()
    assert (scores <= 1.0001).all()


def test_inner_product_higher_is_better(vectors):
    """A vector aligned with the subset sum should score higher."""
    q = vectors[0].clone()
    scores = compute_scores(q.unsqueeze(0), vectors, "inner_product")
    assert scores[0, 0] > scores[0, 1:].mean()


def test_unknown_scoring_raises():
    with pytest.raises(NotImplementedError):
        compute_scores(torch.randn(2, 3), torch.randn(4, 3), "manhattan")


@pytest.mark.parametrize("removed_scoring", ["l1", "l2"])
def test_removed_distance_scoring_raises(removed_scoring):
    with pytest.raises(NotImplementedError):
        compute_scores(torch.randn(2, 3), torch.randn(4, 3), removed_scoring)
