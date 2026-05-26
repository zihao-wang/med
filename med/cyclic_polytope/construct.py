from __future__ import annotations

import itertools
from typing import Optional

import numpy as np


def generate_cyclic_polytope_configuration(
    m: int, n: int, t: np.ndarray | None = None
) -> np.ndarray:
    """Generate the vertex set of a cyclic polytope C(m, n) using the moment curve.

    Parameters
    ----------
    m : int
        Number of points (vertices).
    n : int
        Dimension of the ambient space.
    t : np.ndarray | None
        Optional strictly increasing parameter values of shape (m,). If None, uses arange(m).

    Returns
    -------
    np.ndarray
        Array of shape (m, n) with vertex coordinates.
    """
    if t is None:
        t = np.arange(m, dtype=float)
    else:
        t = np.asarray(t, dtype=float)
        assert t.shape == (m,), "t must have shape (m,)"
        assert np.all(np.diff(t) > 0), "t must be strictly increasing"

    t_col = t.reshape(m, 1)
    coords = [t_col**i for i in range(1, n + 1)]
    X = np.concatenate(coords, axis=1)
    return X


def construct_query_for_subset(
    t_values: np.ndarray, subset_indices: list[int], n: int
) -> np.ndarray:
    """Construct query vector that makes *subset_indices* the top-k.

    Uses the squared-polynomial construction on the moment curve:
    P(t) = Π_{j∈S}(t - t_j), then q = -(coeff of t, …, coeff of t^n) from P(t)².

    When n ≥ 2k, this guarantees perfect separation. When n < 2k, the
    truncated coefficients may or may not work — caller should verify.
    """
    k = len(subset_indices)
    roots = t_values[list(subset_indices)]
    # P(t) = Π(t - t_j)  →  monic polynomial of degree k
    p_coeffs = np.poly(roots)  # [a_k, ..., a_1, a_0], a_k = 1
    # P(t)² via coefficient convolution
    p2_coeffs = np.convolve(p_coeffs, p_coeffs)  # length 2k+1

    # q_j = -coeff of t^j  for j = 1..n
    q = np.zeros(n)
    for j in range(1, min(n, 2 * k) + 1):
        idx = 2 * k - j  # coeff of t^j is at this position in p2_coeffs
        q[j - 1] = -p2_coeffs[idx]
    return q


def check_subset_retrieval(
    points: np.ndarray,
    subset_indices: list[int],
    query: np.ndarray,
) -> bool:
    """Return True if *subset_indices* are exactly the top-k by inner product."""
    scores = points @ query  # (m,)
    threshold = np.partition(scores, -len(subset_indices))[-len(subset_indices)]
    top_indices = set(np.where(scores >= threshold)[0])
    return top_indices == set(subset_indices)


def verify_construction(
    m: int,
    k: int,
    n: int,
    t: Optional[np.ndarray] = None,
    max_checks: Optional[int] = None,
    progress: bool = False,
) -> tuple[bool, int, int]:
    """Check whether the polynomial construction works for *all* k-subsets.

    Generates m moment-curve points in Rⁿ, enumerates every k-subset,
    constructs the squared-polynomial query, and verifies exact top-k retrieval.
    If max_checks is set, verifies only the first max_checks subsets.

    Returns (all_passed, num_checked, num_failed).
    """
    points = generate_cyclic_polytope_configuration(m, n, t)
    t_values = t if t is not None else np.arange(m, dtype=float)

    checked = 0
    failed = 0
    subsets = itertools.combinations(range(m), k)
    if max_checks is not None:
        if max_checks < 0:
            raise ValueError("max_checks must be non-negative")
        subsets = itertools.islice(subsets, max_checks)

    for subset in subsets:
        query = construct_query_for_subset(t_values, list(subset), n)
        if not check_subset_retrieval(points, list(subset), query):
            failed += 1
        checked += 1
        if progress and checked % 1000 == 0:
            print(f"[VERIFY] checked {checked} subsets")
    return failed == 0, checked, failed
