"""Tests for random-token LiMIT retrieval helpers."""

import torch

from scripts import limit_random_embeddings as random_embedding_sweep
from med.unlimit.retrieval.metrics import build_qrels_tensor, retrieval_metrics_from_logits

from med.unlimit.retrieval.random_embeddings import (
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


def test_chunked_metrics_match_full_score_metrics():
    token_matrix = torch.tensor(
        [
            [3.0, 0.0],
            [2.0, 0.0],
            [0.0, 3.0],
            [0.0, 2.0],
        ]
    )
    corpus_tokens = [[0], [1], [2], [3]]
    query_tokens = [[0], [2]]
    corpus_ids = ["d0", "d1", "d2", "d3"]
    query_ids = ["q0", "q1"]
    qrels = [
        {"query_id": "q0", "corpus_id": "d0", "score": 1},
        {"query_id": "q0", "corpus_id": "d1", "score": 1},
        {"query_id": "q1", "corpus_id": "d2", "score": 1},
        {"query_id": "q1", "corpus_id": "d3", "score": 1},
    ]

    full_scores = score_random_embeddings(corpus_tokens, query_tokens, token_matrix)
    labels = build_qrels_tensor(qrels, query_ids, corpus_ids, torch.device("cpu"))
    expected = retrieval_metrics_from_logits(full_scores, labels)
    actual = random_embedding_sweep._retrieval_metrics_chunked(
        corpus_tokens,
        query_tokens,
        qrels,
        query_ids,
        corpus_ids,
        token_matrix,
        score_chunk_size=2,
    )

    assert actual == expected


def test_log_axis_values_floor_nonpositive_recall_values():
    floor = random_embedding_sweep._positive_log_floor([0.0, 0.001, 0.1])

    assert floor == 0.0005
    assert random_embedding_sweep._values_for_log_axis([0.0, 0.001], floor) == [
        floor,
        0.001,
    ]


def test_requested_keys_use_seed_override_for_limit_vanilla_dim64():
    keys = random_embedding_sweep._requested_keys(
        [64], ["full"], ["vanilla"], base_seed=42
    )

    assert keys == {("limit", "full", "vanilla", 64, 107)}


def test_write_pdf_figure_filters_to_dataset_with_reference(tmp_path):
    rows = [
        {"dataset": "limit", "tokenizer": "vanilla", "dim": 32, "recall_at_2": 0.0},
        {"dataset": "limit", "tokenizer": "vanilla", "dim": 64, "recall_at_2": 0.01},
        {
            "dataset": "limit-small",
            "tokenizer": "vanilla",
            "dim": 32,
            "recall_at_2": 0.5,
        },
    ]
    path = tmp_path / "limit.pdf"

    random_embedding_sweep._write_pdf_figure(
        rows,
        path,
        dataset="limit",
        reference=random_embedding_sweep.WELLER_PROMPTRIEVER_RECALL_AT_2_REFERENCE["limit"],
    )

    assert path.exists()
    assert path.read_bytes().startswith(b"%PDF")


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
