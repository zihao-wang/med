"""Training-free LiMIT retrieval with random token embeddings."""

from __future__ import annotations

from typing import Literal

import torch

from med.unlimit.datasets.limit import load_limit
from med.unlimit.device import resolve_torch_device
from med.unlimit.dtype import DEFAULT_FLOAT_DTYPE
from med.unlimit.retrieval.defaults import RANDOM_EMBED_DIM
from med.unlimit.retrieval.metrics import build_qrels_tensor, retrieval_metrics_from_logits
from med.unlimit.tokenizers.types import LimitTokenizer


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


def run_random_embedding_eval(
    tokenizer: LimitTokenizer,
    *,
    split: Literal["small", "full"] = "small",
    embed_dim: int = RANDOM_EMBED_DIM,
    base_seed: int = 42,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Evaluate random-token embedding retrieval on an official LiMIT split."""
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
            f"mean_rank={metrics['mean_rank']:.4f}  "
            f"recall_at_1={metrics['recall_at_1']:.4f}  "
            f"recall_at_2={metrics['recall_at_2']:.4f}  "
            f"top2_exact_match={metrics['top2_exact_match']:.4f}",
            flush=True,
        )

    return metrics


__all__ = [
    "build_random_token_matrix",
    "run_random_embedding_eval",
    "score_random_embeddings",
    "sum_token_rows",
]
