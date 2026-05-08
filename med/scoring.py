"""Shared scoring functions for embedding-based top-k retrieval."""

from typing import Literal

import torch

ScoringFn = Literal["inner_product", "l2", "cosine", "l1"]


def compute_scores(
    subset_sums: torch.Tensor,
    vectors: torch.Tensor,
    scoring_function: ScoringFn,
) -> torch.Tensor:
    """Compute scores between subset centroid sums and all element vectors.

    Args:
        subset_sums: [B, d] centroid embeddings of subsets.
        vectors: [n, d] embeddings of all elements.
        scoring_function: one of "inner_product", "l2", "cosine", "l1".

    Returns:
        [B, n] score matrix.
    """
    if scoring_function == "inner_product":
        return torch.matmul(subset_sums, vectors.T)
    if scoring_function == "l2":
        return torch.norm(subset_sums.unsqueeze(1) - vectors, p=2, dim=-1)
    if scoring_function == "cosine":
        return torch.cosine_similarity(
            subset_sums.unsqueeze(1), vectors.unsqueeze(0), dim=-1
        )
    if scoring_function == "l1":
        return torch.norm(subset_sums.unsqueeze(1) - vectors, p=1, dim=-1)
    raise NotImplementedError(f"Unknown scoring_function: {scoring_function}")
