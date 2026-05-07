from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np


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
    max_checks: Optional[int] = 2000,
    seed: int = 42,
    progress: bool = True,
) -> tuple[bool, int, int]:
    """Check whether the polynomial construction works for *all* k-subsets.

    Generates m moment-curve points in Rⁿ, then samples k-subsets (or
    enumerates all if binom(m,k) ≤ max_checks).  For each subset, constructs
    the squared-polynomial query and verifies exact top-k retrieval.

    Returns (all_passed, num_checked, num_failed).
    """
    from .generator import generate_cyclic_polytope_configuration

    points = generate_cyclic_polytope_configuration(m, n, t)
    t_values = t if t is not None else np.arange(m, dtype=float)

    total = math.comb(m, k)
    rng = random.Random(seed)

    if max_checks and max_checks < total:
        indices = list(range(m))
        for i in range(max_checks):
            subset = sorted(rng.sample(indices, k))
            query = construct_query_for_subset(t_values, subset, n)
            if not check_subset_retrieval(points, subset, query):
                return False, i + 1, 1
        return True, max_checks, 0

    import itertools

    checked = 0
    for subset in itertools.combinations(range(m), k):
        query = construct_query_for_subset(t_values, list(subset), n)
        if not check_subset_retrieval(points, list(subset), query):
            return False, checked + 1, 1
        checked += 1
    return True, checked, 0
