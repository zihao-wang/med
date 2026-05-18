"""Retrieval models and metrics."""

from __future__ import annotations

from unlimit.retrieval.cover_free import (
    build_phrase_cover_free_codes,
    center_phrase_cover_free_codes,
    phrase_cover_free_recall_at_2_curve,
    recall_at_2_phrase_cover_free_chunked,
    score_phrase_cover_free,
)
from unlimit.retrieval.metrics import build_qrels_tensor, retrieval_metrics_from_logits
from unlimit.retrieval.random_embeddings import (
    build_random_token_matrix,
    recall_at_2_random_embeddings_chunked,
    run_random_embedding_eval,
    score_random_embeddings,
    sum_token_rows,
)

__all__ = [
    "build_qrels_tensor",
    "build_phrase_cover_free_codes",
    "build_random_token_matrix",
    "center_phrase_cover_free_codes",
    "phrase_cover_free_recall_at_2_curve",
    "recall_at_2_random_embeddings_chunked",
    "recall_at_2_phrase_cover_free_chunked",
    "retrieval_metrics_from_logits",
    "run_random_embedding_eval",
    "score_phrase_cover_free",
    "score_random_embeddings",
    "sum_token_rows",
]
