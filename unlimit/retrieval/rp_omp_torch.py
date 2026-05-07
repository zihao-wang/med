"""
PyTorch implementation of RP+OMP: random token rows, ``scatter_add_`` sums,
vectorized OMP atom selection (``U @ r``), and batched query–residual dots (``R @ q``).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import torch
from tqdm import tqdm

from med.unlimit.dtype import DEFAULT_FLOAT_DTYPE
from med.unlimit.retrieval.metrics import metrics_tqdm_postfix


def build_token_matrix(
    num_types: int,
    dim: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = DEFAULT_FLOAT_DTYPE,
) -> torch.Tensor:
    g = generator or torch.Generator()
    if device is None:
        device = torch.device("cpu")
    t = torch.randn((num_types, dim), generator=g, dtype=dtype)
    return t.to(device=device, dtype=dtype)


def sum_token_rows(token_lists: list[list[int]], X: torch.Tensor) -> torch.Tensor:
    """``X``: (V, d) — rows indexed by token id. Returns (len(lists), d)."""
    n = len(token_lists)
    d = X.shape[1]
    if n == 0:
        return torch.zeros(0, d, device=X.device, dtype=X.dtype)
    rows: list[int] = []
    cols: list[int] = []
    for i, toks in enumerate(token_lists):
        for t in toks:
            rows.append(i)
            cols.append(t)
    out = torch.zeros(n, d, device=X.device, dtype=X.dtype)
    if not rows:
        return out
    dev = X.device
    doc_idx = torch.tensor(rows, device=dev, dtype=torch.long)
    tok_idx = torch.tensor(cols, device=dev, dtype=torch.long).clamp(0, X.shape[0] - 1)
    parts = X[tok_idx]
    out.scatter_add_(0, doc_idx.unsqueeze(1).expand(-1, d), parts)
    return out


def row_normalize(M: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    n = torch.linalg.norm(M, dim=1, keepdim=True).clamp_min(eps)
    return M / n


def _lstsq_coeff(A: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``A``: (d, k), ``y``: (d,) -> coeff (k,)"""
    if A.shape[1] == 0:
        return torch.zeros(0, device=A.device, dtype=A.dtype)
    y_col = y.unsqueeze(-1)
    if A.device.type == "mps":
        sol = torch.linalg.lstsq(A.cpu(), y_col.cpu()).solution
        return sol.squeeze(-1).to(device=A.device, dtype=A.dtype)
    sol = torch.linalg.lstsq(A, y_col).solution
    return sol.squeeze(-1)


def omp_pair_doclocal(
    y: torch.Tensor,
    Xn: torch.Tensor,
    doc_tokens: list[int],
    forbid: set[int],
    n_steps: int,
) -> torch.Tensor:
    """
    Greedy OMP with **vectorized** correlation over distinct doc tokens: one ``U @ r``
    per step instead of a Python loop over ``uniq``.
    """
    uniq = list(dict.fromkeys(doc_tokens))
    y_work = y.to(dtype=DEFAULT_FLOAT_DTYPE)
    if not uniq:
        return y_work.to(dtype=y.dtype)

    dev = Xn.device
    pos = {t: i for i, t in enumerate(uniq)}
    idx_u = torch.tensor(uniq, device=dev, dtype=torch.long).clamp(0, Xn.shape[0] - 1)
    U = Xn[idx_u].to(dtype=DEFAULT_FLOAT_DTYPE)
    n_u = U.shape[0]
    r = y_work.clone()
    sup_idx: list[int] = []

    forbid_idx = torch.tensor(
        [pos[t] for t in forbid if t in pos],
        device=dev,
        dtype=torch.long,
    )

    for _ in range(n_steps):
        scores = U @ r
        mask = torch.ones(n_u, dtype=torch.bool, device=dev)
        if sup_idx:
            si = torch.tensor(sup_idx, device=dev, dtype=torch.long)
            mask[si] = False
        if forbid_idx.numel() > 0:
            mask[forbid_idx] = False
        if not mask.any():
            break
        scores = scores.masked_fill(~mask, float("-inf"))
        best_i = int(scores.argmax().item())
        sup_idx.append(best_i)
        si = torch.tensor(sup_idx, device=dev, dtype=torch.long)
        A = U[si].T
        coeff = _lstsq_coeff(A, y_work)
        r = y_work - A @ coeff
    return r.to(dtype=y.dtype)


def _precompute_corpus_doc_torch(
    j: int,
    D: torch.Tensor,
    corpus_tokens: list[list[int]],
    Xn: torch.Tensor,
    n_steps: int,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    dt = corpus_tokens[j]
    y = D[j]
    r_empty = omp_pair_doclocal(y, Xn, dt, set(), n_steps)
    r_forbid = {t: omp_pair_doclocal(y, Xn, dt, {t}, n_steps) for t in set(dt)}
    return r_empty, r_forbid


def _score_query_row_torch(
    i: int,
    D: torch.Tensor,
    Q: torch.Tensor,
    corpus_tokens: list[list[int]],
    query_tokens: list[list[int]],
    Xn: torch.Tensor,
    n_steps: int,
    r_empty: list[torch.Tensor],
    r_forbid: list[dict[int, torch.Tensor]],
    nc: int,
) -> torch.Tensor:
    if nc == 0:
        return torch.empty(0, device=D.device, dtype=D.dtype)
    forb = set(query_tokens[i])
    qi = Q[i].to(dtype=DEFAULT_FLOAT_DTYPE)
    rs: list[torch.Tensor] = []
    for j in range(nc):
        dt_set = set(corpus_tokens[j])
        fb = forb & dt_set
        if not fb:
            r = r_empty[j]
        elif len(fb) == 1:
            t0 = next(iter(fb))
            r = r_forbid[j][t0]
        else:
            r = omp_pair_doclocal(D[j], Xn, corpus_tokens[j], fb, n_steps)
        rs.append(r.to(dtype=DEFAULT_FLOAT_DTYPE))
    R = torch.stack(rs, dim=0)
    return (R @ qi).to(dtype=D.dtype)


def scores_query_local_omp(
    D: torch.Tensor,
    Q: torch.Tensor,
    corpus_tokens: list[list[int]],
    query_tokens: list[list[int]],
    X_raw: torch.Tensor,
    n_steps: int,
    *,
    verbose: bool = False,
    log_prefix: str = "[RP+OMP]",
    n_jobs: int = 1,
    progress_y: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    ``n_jobs``: worker threads for document precompute and per-query scoring.
    Only **CPU** tensors use ``n_jobs>1``; CUDA/MPS falls back to sequential.

    ``progress_y``: optional qrels matrix (Q, N) on the same device as ``D``; when
    ``verbose`` and ``n_jobs==1``, the query bar postfix shows partial **R@1** and **top2EM**.
    """
    if D.device.type != "cpu":
        n_jobs = 1
    Xn = row_normalize(X_raw)
    nq, nc = len(query_tokens), len(corpus_tokens)

    def _pre_j(j: int) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        return _precompute_corpus_doc_torch(j, D, corpus_tokens, Xn, n_steps)

    def _score_i(i: int) -> torch.Tensor:
        return _score_query_row_torch(
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

    r_empty: list[torch.Tensor]
    r_forbid: list[dict[int, torch.Tensor]]

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
        out = torch.zeros(nq, nc, device=D.device, dtype=D.dtype)
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
                q_iter.set_postfix(
                    metrics_tqdm_postfix(out[: i + 1], progress_y[: i + 1]),
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
        out = (
            torch.stack(rows, dim=0)
            if rows
            else torch.zeros(0, nc, device=D.device, dtype=D.dtype)
        )

    return out


__all__ = [
    "build_token_matrix",
    "omp_pair_doclocal",
    "scores_query_local_omp",
    "row_normalize",
    "sum_token_rows",
]
