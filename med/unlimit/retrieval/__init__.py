"""Retrieval models and metrics (PyTorch)."""

from __future__ import annotations

from med.unlimit.retrieval.metrics import build_qrels_tensor, retrieval_metrics_from_logits
from med.unlimit.retrieval.rp_omp import (
    build_token_matrix,
    omp_pair_doclocal,
    run_rp_omp_eval,
    scores_query_local_omp,
    sum_token_rows,
)

__all__ = [
    "build_qrels_tensor",
    "build_token_matrix",
    "omp_pair_doclocal",
    "retrieval_metrics_from_logits",
    "run_rp_omp_eval",
    "scores_query_local_omp",
    "sum_token_rows",
]
