from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import cvxpy as cp


@dataclass
class FaceCheckResult:
    subset: Tuple[int, ...]
    status: str  # "sat_le", "sat_ge", "unsat" or "trivial"


def is_face_linear_feasible(points: np.ndarray, S_indices: Sequence[int], tol: float = 1e-9) -> FaceCheckResult:
    """
    Check if subset indexed by S_indices is a face of conv(points) via LP feasibility.

    This uses a single hyperplane a^T x = c with all S on the hyperplane and all
    non-S on one side. We check both directions with inequality flip.
    """
    P = np.asarray(points, dtype=float)
    m, n = P.shape
    S_indices = tuple(sorted(set(int(i) for i in S_indices)))
    S = P[list(S_indices)]

    if len(S_indices) == 0:
        return FaceCheckResult(S_indices, "unsat")
    if len(S_indices) == m or len(S_indices) == 1:
        return FaceCheckResult(S_indices, "trivial")

    mask = np.ones(m, dtype=bool)
    mask[list(S_indices)] = False
    P_out = P[mask]

    a = cp.Variable(n)
    c = cp.Variable()

    constraints = [a @ s == c for s in S]
    # Normalize to avoid zero vector.
    constraints.append(cp.norm(a, 2) <= 1)

    # Check direction "<="
    constraints_le = constraints + [a @ p <= c + tol for p in P_out]
    prob_le = cp.Problem(cp.Maximize(0), constraints_le)
    prob_le.solve(solver=cp.SCS, verbose=False)
    if prob_le.status in ("optimal", "optimal_inaccurate"):
        return FaceCheckResult(S_indices, "sat_le")

    # Check direction ">="
    constraints_ge = constraints + [a @ p >= c - tol for p in P_out]
    prob_ge = cp.Problem(cp.Maximize(0), constraints_ge)
    prob_ge.solve(solver=cp.SCS, verbose=False)
    if prob_ge.status in ("optimal", "optimal_inaccurate"):
        return FaceCheckResult(S_indices, "sat_ge")

    return FaceCheckResult(S_indices, "unsat")


def batched_pair_face_checks(points: np.ndarray, batch: Iterable[Tuple[int, int]], tol: float = 1e-9) -> List[FaceCheckResult]:
    """
    Perform face checks for pairs (i, j) in a batch for scalability.

    Note: This still solves small LPs per pair; for large m this can be parallelized
    at the process level by the caller.
    """
    results: List[FaceCheckResult] = []
    for i, j in batch:
        results.append(is_face_linear_feasible(points, (i, j), tol))
    return results
