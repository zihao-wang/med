"""Label-unaware phrase cover-free retrieval construction for LIMIT."""

from __future__ import annotations

import torch

from unlimit.dtype import DEFAULT_FLOAT_DTYPE
from unlimit.retrieval.random_embeddings import sum_token_rows


def build_phrase_cover_free_codes(
    num_types: int,
    dim: int,
    seed: int,
    *,
    p: float = 0.5,
    device: torch.device | None = None,
    dtype: torch.dtype = DEFAULT_FLOAT_DTYPE,
) -> torch.Tensor:
    """Return deterministic Bernoulli phrase codes with shape ``(V, d)``.

    Columns use independent seeds so prefixes are stable when evaluating nested
    dimensions. The construction depends only on phrase ids, never on qrels.
    """
    if num_types <= 0:
        raise ValueError("num_types must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")
    if not 0.0 < p < 1.0:
        raise ValueError("p must be strictly between 0 and 1")

    generator = torch.Generator(device="cpu")
    columns: list[torch.Tensor] = []
    for col in range(dim):
        generator.manual_seed(seed + col)
        column = torch.rand(num_types, generator=generator) < p
        columns.append(column)
    codes = torch.stack(columns, dim=1).to(dtype=dtype)
    if device is None:
        return codes
    return codes.to(device=device, dtype=dtype)


def center_phrase_cover_free_codes(
    codes: torch.Tensor,
    *,
    p: float = 0.5,
) -> torch.Tensor:
    """Return query witness codes with zero-mean unrelated inner products."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be strictly between 0 and 1")
    return codes - float(p)


def score_phrase_cover_free(
    corpus_tokens: list[list[int]],
    query_tokens: list[list[int]],
    codes: torch.Tensor,
    *,
    p: float = 0.5,
) -> torch.Tensor:
    """Score documents by summed phrase codes against centered query witnesses."""
    docs = sum_token_rows(corpus_tokens, codes)
    queries = sum_token_rows(query_tokens, center_phrase_cover_free_codes(codes, p=p))
    return queries @ docs.T


@torch.no_grad()
def recall_at_2_phrase_cover_free_chunked(
    corpus_tokens: list[list[int]],
    query_tokens: list[list[int]],
    codes: torch.Tensor,
    labels: torch.Tensor,
    *,
    p: float = 0.5,
    doc_chunk_size: int = 2048,
) -> dict[str, float]:
    """Compute Recall@2 for phrase cover-free codes without full score materialization."""
    if doc_chunk_size <= 0:
        raise ValueError("doc_chunk_size must be positive")

    num_queries = len(query_tokens)
    num_docs = len(corpus_tokens)
    if num_queries == 0 or num_docs == 0:
        return {"recall_at_2": 0.0, "num_queries_eval": 0.0}

    labels = labels.to(device=codes.device)
    k = min(2, num_docs)
    queries = sum_token_rows(
        query_tokens,
        center_phrase_cover_free_codes(codes, p=p),
    )
    top_values = torch.full(
        (num_queries, k),
        -torch.inf,
        device=codes.device,
        dtype=codes.dtype,
    )
    top_indices = torch.full(
        (num_queries, k),
        -1,
        device=codes.device,
        dtype=torch.long,
    )

    for start in range(0, num_docs, doc_chunk_size):
        end = min(start + doc_chunk_size, num_docs)
        docs = sum_token_rows(corpus_tokens[start:end], codes)
        scores = queries @ docs.T
        chunk_indices = torch.arange(start, end, device=scores.device, dtype=torch.long)
        chunk_indices = chunk_indices.unsqueeze(0).expand(num_queries, -1)

        combined_values = torch.cat([top_values, scores], dim=1)
        combined_indices = torch.cat([top_indices, chunk_indices], dim=1)
        top_values, local_top = torch.topk(combined_values, k=k, dim=1, largest=True)
        top_indices = torch.gather(combined_indices, 1, local_top)

    recall2_sum = 0.0
    num_eval = 0
    for qi in range(num_queries):
        pos = labels[qi].nonzero(as_tuple=False).squeeze(-1)
        if pos.numel() == 0:
            continue
        recall2_sum += float(labels[qi, top_indices[qi]].sum().item()) / float(
            pos.numel()
        )
        num_eval += 1

    return {
        "recall_at_2": recall2_sum / num_eval if num_eval else 0.0,
        "num_queries_eval": float(num_eval),
    }


def phrase_cover_free_recall_at_2_curve(
    corpus_tokens: list[list[int]],
    query_tokens: list[list[int]],
    labels: torch.Tensor,
    num_types: int,
    dims: list[int],
    seed: int,
    *,
    p: float = 0.5,
    device: torch.device | None = None,
    dtype: torch.dtype = DEFAULT_FLOAT_DTYPE,
    doc_chunk_size: int = 2048,
) -> dict[int, dict[str, float]]:
    """Evaluate nested phrase cover-free prefixes for requested dimensions."""
    if not dims:
        return {}

    sorted_dims = sorted({int(dim) for dim in dims})
    if any(dim <= 0 for dim in sorted_dims):
        raise ValueError("all dimensions must be positive")

    codes = build_phrase_cover_free_codes(
        num_types,
        max(sorted_dims),
        seed,
        p=p,
        device=device,
        dtype=dtype,
    )
    curve: dict[int, dict[str, float]] = {}
    for dim in sorted_dims:
        curve[dim] = recall_at_2_phrase_cover_free_chunked(
            corpus_tokens,
            query_tokens,
            codes[:, :dim],
            labels,
            p=p,
            doc_chunk_size=doc_chunk_size,
        )
    return curve


__all__ = [
    "build_phrase_cover_free_codes",
    "center_phrase_cover_free_codes",
    "phrase_cover_free_recall_at_2_curve",
    "recall_at_2_phrase_cover_free_chunked",
    "score_phrase_cover_free",
]
