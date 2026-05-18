"""Training-free LIMIT retrieval with random token embeddings."""

from __future__ import annotations

from typing import Literal

import torch

from unlimit.datasets.limit import load_limit
from unlimit.device import resolve_torch_device
from unlimit.dtype import DEFAULT_FLOAT_DTYPE
from unlimit.retrieval.defaults import RANDOM_EMBED_DIM
from unlimit.retrieval.metrics import build_qrels_tensor, retrieval_metrics_from_logits
from unlimit.tokenizers.types import LimitTokenizer


def build_random_token_matrix(
    num_types: int,
    dim: int,
    seed: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = DEFAULT_FLOAT_DTYPE,
) -> torch.Tensor:
    """Return a deterministic Gaussian token matrix with shape ``(num_types, dim)``."""
    if num_types <= 0:
        raise ValueError("num_types must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn((num_types, dim), generator=generator, dtype=dtype)
    if device is None:
        return matrix
    return matrix.to(device=device, dtype=dtype)


def sum_token_rows(token_lists: list[list[int]], token_matrix: torch.Tensor) -> torch.Tensor:
    """Embed each record by summing the rows indexed by its token ids."""
    n_records = len(token_lists)
    dim = token_matrix.shape[1]
    out = torch.zeros(
        n_records,
        dim,
        device=token_matrix.device,
        dtype=token_matrix.dtype,
    )
    if n_records == 0:
        return out

    rows: list[int] = []
    cols: list[int] = []
    max_token_id = token_matrix.shape[0] - 1
    for row_idx, token_ids in enumerate(token_lists):
        for token_id in token_ids:
            rows.append(row_idx)
            cols.append(max(0, min(int(token_id), max_token_id)))
    if not rows:
        return out

    device = token_matrix.device
    row_idx = torch.tensor(rows, device=device, dtype=torch.long)
    token_idx = torch.tensor(cols, device=device, dtype=torch.long)
    values = token_matrix[token_idx]
    out.scatter_add_(0, row_idx.unsqueeze(1).expand(-1, dim), values)
    return out


def score_random_embeddings(
    corpus_tokens: list[list[int]],
    query_tokens: list[list[int]],
    token_matrix: torch.Tensor,
) -> torch.Tensor:
    """Score queries against documents using inner products of summed token vectors."""
    docs = sum_token_rows(corpus_tokens, token_matrix)
    queries = sum_token_rows(query_tokens, token_matrix)
    return queries @ docs.T


@torch.no_grad()
def recall_at_2_random_embeddings_chunked(
    corpus_tokens: list[list[int]],
    query_tokens: list[list[int]],
    token_matrix: torch.Tensor,
    labels: torch.Tensor,
    *,
    doc_chunk_size: int = 2048,
) -> dict[str, float]:
    """Compute Recall@2 without materializing the full query-document matrix."""
    if doc_chunk_size <= 0:
        raise ValueError("doc_chunk_size must be positive")

    num_queries = len(query_tokens)
    num_docs = len(corpus_tokens)
    if num_queries == 0 or num_docs == 0:
        return {"recall_at_2": 0.0, "num_queries_eval": 0.0}

    k = min(2, num_docs)
    queries = sum_token_rows(query_tokens, token_matrix)
    top_values = torch.full(
        (num_queries, k),
        -torch.inf,
        device=token_matrix.device,
        dtype=token_matrix.dtype,
    )
    top_indices = torch.full(
        (num_queries, k),
        -1,
        device=token_matrix.device,
        dtype=torch.long,
    )

    for start in range(0, num_docs, doc_chunk_size):
        end = min(start + doc_chunk_size, num_docs)
        docs = sum_token_rows(corpus_tokens[start:end], token_matrix)
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


def run_random_embedding_eval(
    tokenizer: LimitTokenizer,
    *,
    split: Literal["small", "full"] = "small",
    embed_dim: int = RANDOM_EMBED_DIM,
    base_seed: int = 42,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Evaluate random-token embedding retrieval on an official LIMIT split."""
    dev = resolve_torch_device(device)
    seed = base_seed + embed_dim

    if verbose:
        print(
            "[random-emb] hyperparameters: "
            f"tokenizer={tokenizer.name!r}  split={split!r}  "
            f"embed_dim={embed_dim}  seed={seed}  device={dev}",
            flush=True,
        )

    corpus, queries, qrels = load_limit(split)
    tokenized_corpus = tokenizer.tokenize_corpus_records(corpus)
    tokenized_queries = tokenizer.tokenize_query_records(queries)

    corpus_ids = [record["_id"] for record in tokenized_corpus]
    query_ids = [record["_id"] for record in tokenized_queries]
    corpus_tokens = [record["token_ids"] for record in tokenized_corpus]
    query_tokens = [record["token_ids"] for record in tokenized_queries]
    num_types = tokenizer.num_token_types()

    if verbose:
        print(
            "[random-emb] data: "
            f"corpus_docs={len(corpus_tokens)}  queries={len(query_tokens)}  "
            f"num_token_types={num_types}",
            flush=True,
        )

    labels = build_qrels_tensor(qrels, query_ids, corpus_ids, dev)
    token_matrix = build_random_token_matrix(
        num_types,
        embed_dim,
        seed,
        device=dev,
        dtype=DEFAULT_FLOAT_DTYPE,
    )
    scores = score_random_embeddings(corpus_tokens, query_tokens, token_matrix)
    metrics = retrieval_metrics_from_logits(scores, labels)
    metrics["dim"] = float(embed_dim)

    if verbose:
        print(
            "[random-emb] metrics: "
            f"recall_at_2={metrics['recall_at_2']:.4f}",
            flush=True,
        )

    return metrics


__all__ = [
    "build_random_token_matrix",
    "recall_at_2_random_embeddings_chunked",
    "run_random_embedding_eval",
    "score_random_embeddings",
    "sum_token_rows",
]
