"""Tests for random-token LiMIT retrieval helpers."""

import torch

from unlimit.retrieval.random_embeddings import (
    build_random_token_matrix,
    score_random_embeddings,
    sum_token_rows,
)


def test_build_random_token_matrix_is_seeded():
    a = build_random_token_matrix(5, 3, seed=17)
    b = build_random_token_matrix(5, 3, seed=17)
    c = build_random_token_matrix(5, 3, seed=18)

    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_sum_token_rows_counts_repeated_tokens():
    token_matrix = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 4.0],
        ]
    )

    out = sum_token_rows([[0, 1, 1], [2], []], token_matrix)

    expected = torch.tensor(
        [
            [1.0, 4.0],
            [3.0, 4.0],
            [0.0, 0.0],
        ]
    )
    assert torch.equal(out, expected)


def test_score_random_embeddings_uses_query_document_inner_product():
    token_matrix = torch.eye(3)
    scores = score_random_embeddings(
        corpus_tokens=[[0, 1], [2]],
        query_tokens=[[1], [2]],
        token_matrix=token_matrix,
    )

    expected = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    assert torch.equal(scores, expected)
