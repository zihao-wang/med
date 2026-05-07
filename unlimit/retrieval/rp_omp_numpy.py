"""
NumPy/LAPACK implementation of RP+OMP scoring (same algorithm as ``rp_omp.py``).

Used on **CPU** for much better throughput than the PyTorch reference: tight loops
over NumPy vector ops and ``np.linalg.lstsq`` call optimized BLAS without per-step tensor allocator overhead.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from tqdm import tqdm

from med.unlimit.retrieval.metrics import metrics_tqdm_postfix


def _precompute_corpus_doc_np(
    j: int,
    D: np.ndarray,
    corpus_tokens: list[list[int]],
    Xn: np.ndarray,
    n_steps: int,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    dt = corpus_tokens[j]
    y = D[j]
    r_empty = omp_pair_doclocal_np(y, Xn, dt, set(), n_steps)
    r_forbid = {t: omp_pair_doclocal_np(y, Xn, dt, {t}, n_steps) for t in set(dt)}
    return r_empty, r_forbid


def _score_query_row_np(
    i: int,
    D: np.ndarray,
    Q: np.ndarray,
    corpus_tokens: list[list[int]],
    query_tokens: list[list[int]],
    Xn: np.ndarray,
    n_steps: int,
    r_empty: list[np.ndarray],
    r_forbid: list[dict[int, np.ndarray]],
    nc: int,
) -> np.ndarray:
    row = np.empty(nc, dtype=np.float32)
    forb = set(query_tokens[i])
    qi = Q[i]
    for j in range(nc):
        dt_set = set(corpus_tokens[j])
        fb = forb & dt_set
        if not fb:
            r = r_empty[j]
        elif len(fb) == 1:
            t0 = next(iter(fb))
            r = r_forbid[j][t0]
        else:
            r = omp_pair_doclocal_np(D[j], Xn, corpus_tokens[j], fb, n_steps)
        row[j] = float(qi @ r)
    return row


def build_token_matrix_np(num_types: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_types, dim)).astype(np.float32, copy=False)


def sum_token_rows_np(token_lists: list[list[int]], X: np.ndarray) -> np.ndarray:
    n = len(token_lists)
    d = X.shape[1]
    if n == 0:
        return np.zeros((0, d), dtype=np.float32)
    out = np.zeros((n, d), dtype=np.float32)
    vmax = X.shape[0] - 1
    for i, toks in enumerate(token_lists):
        if not toks:
            continue
        idx = np.clip(np.asarray(toks, dtype=np.intp), 0, vmax)
        out[i] = X[idx].sum(axis=0)
    return out


def row_normalize_np(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, eps)


def omp_pair_doclocal_np(
    y: np.ndarray,
    Xn: np.ndarray,
    doc_tokens: list[int],
    forbid: set[int],
    n_steps: int,
) -> np.ndarray:
    uniq = list(dict.fromkeys(doc_tokens))
    support: list[int] = []
    y_work = y.astype(np.float32, copy=False)
    r = y_work.copy()
    for _ in range(n_steps):
        best_t, best_c = -1, -np.inf
        for t in uniq:
            if t in support or t in forbid:
                continue
            c = float(Xn[t] @ r)
            if c > best_c:
                best_c, best_t = c, t
        if best_t < 0:
            break
        support.append(best_t)
        A = Xn[support].T.astype(np.float32, copy=False)
        coeff, *_ = np.linalg.lstsq(A, y_work, rcond=None)
        r = y_work - A @ coeff
    return r.astype(y.dtype, copy=False)


def scores_query_local_omp_np(
    D: np.ndarray,
    Q: np.ndarray,
    corpus_tokens: list[list[int]],
    query_tokens: list[list[int]],
    X_raw: np.ndarray,
    n_steps: int,
    *,
    verbose: bool = False,
    log_prefix: str = "[RP+OMP]",
    n_jobs: int = 1,
    progress_y: torch.Tensor | None = None,
) -> np.ndarray:
    """
    ``n_jobs``: number of worker threads for precompute (per document) and scoring
    (per query). ``1`` uses :mod:`tqdm` bars. Values ``>1`` use
    :class:`~concurrent.futures.ThreadPoolExecutor` (NumPy releases the GIL in
    BLAS/LAPACK). If scoring is slow, try ``OPENBLAS_NUM_THREADS=1`` (or
    ``MKL_NUM_THREADS=1``) to avoid oversubscription with many workers.

    ``progress_y``: optional qrels ``(Q, N)``; with ``verbose`` and ``n_jobs==1``, the
    query bar postfix shows partial **R@1** and **top2EM** (metrics on rows finished so far).
    """
    Xn = row_normalize_np(X_raw)
    nq, nc = len(query_tokens), len(corpus_tokens)

    def _pre_j(j: int) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        return _precompute_corpus_doc_np(j, D, corpus_tokens, Xn, n_steps)

    def _score_i(i: int) -> np.ndarray:
        return _score_query_row_np(
            i,
            D,
            Q,
            corpus_tokens,
            query_tokens,
            Xn,
            n_steps,
            r_empty,
            r_forbid,
            nc,
        )

    r_empty: list[np.ndarray]
    r_forbid: list[dict[int, np.ndarray]]

    if n_jobs <= 1:
        r_empty = []
        r_forbid = []
        doc_iter = range(nc)
        if verbose:
            doc_iter = tqdm(
                doc_iter,
                desc=f"{log_prefix} precompute docs",
                leave=False,
                unit="doc",
            )
        for j in doc_iter:
            re, rf = _pre_j(j)
            r_empty.append(re)
            r_forbid.append(rf)
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            if nc and verbose:
                pairs = list(
                    tqdm(
                        ex.map(_pre_j, range(nc)),
                        total=nc,
                        desc=f"{log_prefix} precompute docs",
                        leave=False,
                        unit="doc",
                    )
                )
            else:
                pairs = list(ex.map(_pre_j, range(nc))) if nc else []
        r_empty = [p[0] for p in pairs]
        r_forbid = [p[1] for p in pairs]

    if n_jobs <= 1:
        out = np.zeros((nq, nc), dtype=np.float32)
        postfix_iv = max(1, nq // 100) if nq else 1
        q_iter = range(nq)
        if verbose:
            q_iter = tqdm(
                q_iter,
                desc=f"{log_prefix} score queries",
                leave=False,
                unit="query",
            )
        for i in q_iter:
            out[i] = _score_i(i)
            if (
                progress_y is not None
                and verbose
                and (i % postfix_iv == 0 or i == nq - 1)
            ):
                lt = torch.from_numpy(np.ascontiguousarray(out[: i + 1])).to(
                    device=progress_y.device,
                    dtype=torch.float32,
                )
                q_iter.set_postfix(
                    metrics_tqdm_postfix(lt, progress_y[: i + 1]),
                    refresh=False,
                )
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            if nq and verbose:
                rows = list(
                    tqdm(
                        ex.map(_score_i, range(nq)),
                        total=nq,
                        desc=f"{log_prefix} score queries (parallel)",
                        leave=False,
                        unit="query",
                    )
                )
            else:
                rows = list(ex.map(_score_i, range(nq))) if nq else []
        out = np.stack(rows, axis=0) if rows else np.zeros((0, nc), dtype=np.float32)

    return out


__all__ = [
    "build_token_matrix_np",
    "omp_pair_doclocal_np",
    "scores_query_local_omp_np",
    "sum_token_rows_np",
    "row_normalize_np",
]
