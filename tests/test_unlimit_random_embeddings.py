"""Tests for random-token LIMIT retrieval helpers."""

import torch

from unlimit.retrieval.cover_free import (
    build_phrase_cover_free_codes,
    recall_at_2_phrase_cover_free_chunked,
    score_phrase_cover_free,
)
from unlimit.retrieval.random_embeddings import (
    build_random_token_matrix,
    recall_at_2_random_embeddings_chunked,
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


def test_chunked_recall_at_2_matches_full_scores():
    token_matrix = torch.tensor([[4.0], [3.0], [2.0], [1.0]])
    corpus_tokens = [[0], [1], [2], [3]]
    query_tokens = [[0], [3]]
    labels = torch.tensor(
        [
            [True, True, False, False],
            [False, False, True, True],
        ]
    )

    metrics = recall_at_2_random_embeddings_chunked(
        corpus_tokens,
        query_tokens,
        token_matrix,
        labels,
        doc_chunk_size=2,
    )

    assert metrics["recall_at_2"] == 0.5
    assert metrics["num_queries_eval"] == 2.0


def test_phrase_cover_free_scores_centered_query_against_document_codes():
    codes = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
    )

    scores = score_phrase_cover_free(
        corpus_tokens=[[0], [1], [2]],
        query_tokens=[[0]],
        codes=codes,
        p=0.5,
    )

    expected = torch.tensor([[0.5, -0.5, 0.0]])
    assert torch.equal(scores, expected)


def test_phrase_cover_free_chunked_recall_uses_tokenized_text_not_qrels_for_scores():
    codes = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
    )
    labels = torch.tensor([[True, False, True]])

    metrics = recall_at_2_phrase_cover_free_chunked(
        corpus_tokens=[[0], [1], [2]],
        query_tokens=[[0]],
        codes=codes,
        labels=labels,
        p=0.5,
        doc_chunk_size=2,
    )

    assert metrics["recall_at_2"] == 1.0
    assert metrics["num_queries_eval"] == 1.0


def test_phrase_cover_free_prefix_is_stable_across_requested_dimensions():
    short = build_phrase_cover_free_codes(5, 2, seed=7)
    long = build_phrase_cover_free_codes(5, 4, seed=7)

    assert torch.equal(short, long[:, :2])
