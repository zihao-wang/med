"""
RP+OMP retrieval **facade**: picks NumPy or PyTorch implementation from ``backend``
/ device, then runs :func:`run_rp_omp_eval`.

- **NumPy:** :mod:`unlimit.retrieval.rp_omp_numpy` (CPU; fast LAPACK loops).
- **PyTorch:** :mod:`unlimit.retrieval.rp_omp_torch` (CUDA/MPS/CPU reference).
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
import torch

from unlimit.device import resolve_torch_device
from unlimit.dtype import DEFAULT_FLOAT_DTYPE
from unlimit.datasets.limit import load_limit
from unlimit.retrieval.defaults import RP_OMP_EMBED_DIM, RP_OMP_STEPS
from unlimit.retrieval.metrics import (
    build_qrels_tensor,
    retrieval_metrics_from_logits,
)
from unlimit.retrieval.rp_omp_torch import (
    build_token_matrix,
    omp_pair_doclocal,
    row_normalize,
    scores_query_local_omp,
    sum_token_rows,
)
from unlimit.tokenizers.types import LimitTokenizer


def _resolve_rp_backend(
    backend: Literal["auto", "torch", "numpy"],
    dev: torch.device,
) -> Literal["torch", "numpy"]:
    """NumPy path is much faster on CPU; PyTorch is used for CUDA/MPS."""
    env = os.environ.get("LIMIT_RP_BACKEND", "").strip().lower()
    if env in ("torch", "numpy"):
        backend = env  # type: ignore[assignment]
    if backend == "auto":
        return "numpy" if dev.type == "cpu" else "torch"

    print(f"[RP+OMP] using {backend} backend for device {dev}")
    return backend


def _resolve_n_jobs(requested: int | None) -> int:
    """
    Default: use up to 32 CPU threads (cap avoids huge pools on fat servers).
    ``-1`` / env ``auto`` / ``all`` mean “cap to cpu count”.
    If ``requested is None``, ``LIMIT_RP_N_JOBS`` may set the pool size (integer or
    ``auto`` / ``-1`` / ``all``). Passing ``n_jobs`` explicitly to ``run_rp_omp_eval``
    overrides the environment.
    """
    if requested is not None:
        if requested == -1:
            c = os.cpu_count() or 1
            return max(1, min(32, c))
        return max(1, requested)
    env = os.environ.get("LIMIT_RP_N_JOBS", "").strip()
    if env:
        if env.lower() in ("-1", "auto", "all"):
            c = os.cpu_count() or 1
            return max(1, min(32, c))
        return max(1, int(env))
    c = os.cpu_count() or 1
    return max(1, min(32, c))


def run_rp_omp_eval(
    tokenizer: LimitTokenizer,
    *,
    split: Literal["small", "full"] = "small",
    omp_steps: int = RP_OMP_STEPS,
    embed_dim: int = RP_OMP_EMBED_DIM,
    base_seed: int = 42,
    device: str | torch.device | None = None,
    verbose: bool = True,
    backend: Literal["auto", "torch", "numpy"] = "auto",
    n_jobs: int | None = None,
) -> dict[str, float]:
    """
    Evaluate RP+OMP on the **official LiMIT split** from :func:`~unlimit.datasets.load_limit`:
    full corpus, all queries, and all qrels.

    Dispatches scoring to :mod:`~unlimit.retrieval.rp_omp_numpy` or
    :mod:`~unlimit.retrieval.rp_omp_torch` according to ``backend`` and ``device``.

    Key knobs: ``omp_steps`` (OMP depth); ``embed_dim`` for **Gaussian** token rows
    (ignored when the tokenizer provides ``rp_omp_token_matrix``, e.g. Qwen3).
    Defaults match :mod:`unlimit.retrieval.defaults`.
    ``verbose`` defaults to ``True``; set ``verbose=False`` to silence logs.

    **backend:** ``\"auto\"`` uses **NumPy** on **CPU** and **PyTorch** on CUDA/MPS.
    Override with ``LIMIT_RP_BACKEND=torch`` or ``numpy``, or pass ``backend=`` explicitly.

    **n_jobs:** thread pool size for document precompute and per-query scoring (NumPy
    backend and **CPU** PyTorch only). ``None`` uses ``LIMIT_RP_N_JOBS`` if set, else up
    to 32 workers. ``-1`` means “use capped CPU count”. CUDA/MPS PyTorch stays sequential.
    """
    dev = resolve_torch_device(device)
    impl = _resolve_rp_backend(backend, dev)
    nj = _resolve_n_jobs(n_jobs)
    torch_nj = nj if dev.type == "cpu" else 1
    pretrained_dim = getattr(tokenizer, "pretrained_embedding_dim", None)
    dim_log = (
        f"{int(pretrained_dim)} (pretrained)"
        if pretrained_dim is not None
        else str(embed_dim)
    )

    if verbose:
        print(
            "[RP+OMP] hyperparameters: "
            f"tokenizer={tokenizer.name!r}  split={split!r}  "
            f"omp_steps={omp_steps}  embed_dim={dim_log}  base_seed={base_seed}  "
            f"device={dev}  backend={impl}  n_jobs={nj}  torch_n_jobs={torch_nj}",
            flush=True,
        )

    corpus, queries, qrels = load_limit(split)
    tc = tokenizer.tokenize_corpus_records(corpus)
    tq_te = tokenizer.tokenize_query_records(queries)
    query_ids = [r["_id"] for r in tq_te]
    query_tokens = [r["token_ids"] for r in tq_te]

    corpus_ids = [r["_id"] for r in tc]
    corpus_tokens = [r["token_ids"] for r in tc]

    n_corpus = len(corpus_tokens)
    n_queries = len(query_tokens)
    n_types = tokenizer.rp_omp_num_token_types()

    if verbose:
        print(
            "[RP+OMP] data: "
            f"corpus_docs={n_corpus}  eval_queries={n_queries}  "
            f"rp_omp_num_token_types={n_types}",
            flush=True,
        )

    y = build_qrels_tensor(qrels, query_ids, corpus_ids, dev)
    progress_y = y if verbose else None

    build_pretrained = getattr(tokenizer, "rp_omp_token_matrix", None)

    if impl == "numpy":
        from unlimit.retrieval import rp_omp_numpy as rpn

        if callable(build_pretrained):
            if verbose:
                print(
                    "[RP+OMP] building token matrix from pretrained LM embeddings "
                    f"({n_types} rows) on cpu (numpy backend) …",
                    flush=True,
                )
            X_np = np.asarray(
                build_pretrained(
                    device=torch.device("cpu"),
                    dtype=DEFAULT_FLOAT_DTYPE,
                ).numpy(),
                dtype=np.float32,
            )
            eff_dim = int(X_np.shape[1])
            if embed_dim != eff_dim and verbose:
                print(
                    f"[RP+OMP] note: passed embed_dim={embed_dim} ignored; "
                    f"pretrained width is {eff_dim}",
                    flush=True,
                )
        else:
            if verbose:
                print(
                    "[RP+OMP] building random Gaussian token matrix "
                    f"({n_types} × {embed_dim}) (numpy backend) …",
                    flush=True,
                )
            X_np = rpn.build_token_matrix_np(n_types, embed_dim, base_seed + embed_dim)

        if verbose:
            print(
                "[RP+OMP] summing token rows → document / query vectors …", flush=True
            )
        D_np = rpn.sum_token_rows_np(corpus_tokens, X_np)
        Q_np = rpn.sum_token_rows_np(query_tokens, X_np)

        scores_np = rpn.scores_query_local_omp_np(
            D_np,
            Q_np,
            corpus_tokens,
            query_tokens,
            X_np,
            omp_steps,
            verbose=verbose,
            log_prefix="[RP+OMP]",
            n_jobs=nj,
            progress_y=progress_y,
        )
        scores = torch.from_numpy(scores_np).to(device=dev, dtype=DEFAULT_FLOAT_DTYPE)
    else:
        if callable(build_pretrained):
            if verbose:
                print(
                    "[RP+OMP] building token matrix from pretrained LM embeddings "
                    f"({n_types} rows) on {dev} …",
                    flush=True,
                )
            X = build_pretrained(device=dev, dtype=DEFAULT_FLOAT_DTYPE)
            eff_dim = int(X.shape[1])
            if embed_dim != eff_dim and verbose:
                print(
                    f"[RP+OMP] note: passed embed_dim={embed_dim} ignored; "
                    f"pretrained width is {eff_dim}",
                    flush=True,
                )
        else:
            if verbose:
                print(
                    "[RP+OMP] building random Gaussian token matrix "
                    f"({n_types} × {embed_dim}) on {dev} …",
                    flush=True,
                )
            g = torch.Generator()
            g.manual_seed(base_seed + embed_dim)
            X = build_token_matrix(n_types, embed_dim, generator=g, device=dev)

        if verbose:
            print(
                "[RP+OMP] summing token rows → document / query vectors …", flush=True
            )

        D = sum_token_rows(corpus_tokens, X)
        Q = sum_token_rows(query_tokens, X)

        scores = scores_query_local_omp(
            D,
            Q,
            corpus_tokens,
            query_tokens,
            X,
            omp_steps,
            verbose=verbose,
            log_prefix="[RP+OMP]",
            n_jobs=torch_nj,
            progress_y=progress_y,
        )
    metrics = retrieval_metrics_from_logits(scores, y)
    metrics["dim"] = embed_dim
    metrics["omp_steps"] = omp_steps

    if verbose:
        print(
            "[RP+OMP] metrics: "
            f"mean_rank={metrics['mean_rank']:.4f}  "
            f"recall_at_1={metrics['recall_at_1']:.4f}  "
            f"top2_exact_match={metrics['top2_exact_match']:.4f}",
            flush=True,
        )

    return metrics


__all__ = [
    "build_token_matrix",
    "omp_pair_doclocal",
    "run_rp_omp_eval",
    "scores_query_local_omp",
    "row_normalize",
    "sum_token_rows",
]
