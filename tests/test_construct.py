"""Tests for construct.py — moment-curve points and query construction."""

import numpy as np
import pytest

from med.cyclic_polytope.construct import (
    check_subset_retrieval,
    construct_query_for_subset,
    generate_cyclic_polytope_configuration,
    verify_construction,
)


def test_generate_default_t_gives_shape():
    points = generate_cyclic_polytope_configuration(5, 3)
    assert points.shape == (5, 3)


def test_generate_default_t_uses_arange():
    points = generate_cyclic_polytope_configuration(4, 2)
    np.testing.assert_allclose(points[:, 0], np.arange(4, dtype=float))


def test_generate_custom_t():
    t = np.array([0.0, 0.5, 1.0])
    points = generate_cyclic_polytope_configuration(3, 2, t)
    np.testing.assert_allclose(points[:, 0], t)
    np.testing.assert_allclose(points[:, 1], t**2)


def test_generate_t_must_be_strictly_increasing():
    t = np.array([0.0, 1.0, 0.5])
    with pytest.raises(AssertionError, match="strictly increasing"):
        generate_cyclic_polytope_configuration(3, 2, t)


def test_generate_t_shape_mismatch():
    t = np.array([0.0, 1.0])
    with pytest.raises(AssertionError, match="shape"):
        generate_cyclic_polytope_configuration(3, 2, t)


def test_generate_higher_dimensions():
    points = generate_cyclic_polytope_configuration(3, 4)
    assert points.shape == (3, 4)
    np.testing.assert_allclose(points[2, 3], 2**4)


def test_construct_query_shape():
    t = np.arange(5, dtype=float)
    q = construct_query_for_subset(t, [0, 2], n=4)
    assert q.shape == (4,)


def test_construct_query_values_small():
    """Known case: m=4, k=2, n=4, subset {0,2}.
    t = [0, 1, 2, 3], roots at t=0, t=2.
    P(t) = (t-0)(t-2) = t^2 - 2t → coeffs [1, -2, 0]
    P^2(t) = (t^2 - 2t)^2 = t^4 - 4t^3 + 4t^2 → coeffs [1, -4, 4, 0, 0]
    q_j = -coeff of t^j where idx = 2k - j = 4 - j
    j=1: idx=3, coeff=0  → q[0] = 0
    j=2: idx=2, coeff=4  → q[1] = -4
    j=3: idx=1, coeff=-4 → q[2] = 4
    j=4: idx=0, coeff=1  → q[3] = -1
    """
    t = np.arange(4, dtype=float)
    q = construct_query_for_subset(t, [0, 2], n=4)
    expected = np.array([0, -4, 4, -1], dtype=float)
    np.testing.assert_allclose(q, expected)


def test_check_subset_retrieval_passes_for_known_case():
    """With the query from above, verify {0, 2} are the top-2."""
    t = np.arange(4, dtype=float)
    points = generate_cyclic_polytope_configuration(4, 4, t)
    q = construct_query_for_subset(t, [0, 2], n=4)
    assert check_subset_retrieval(points, [0, 2], q)


def test_verify_construction_exhaustive_small():
    """For m=4, k=2, n=4 (n=2k), should pass all 6 subsets."""
    ok, checked, failed = verify_construction(4, 2, 4, max_checks=None, progress=False)
    assert ok, f"Expected all {checked} subsets to pass, but {failed} failed"
    assert checked == 6


def test_verify_construction_with_sampling():
    """With max_checks=3, should only check 3 subsets."""
    ok, checked, failed = verify_construction(4, 2, 4, max_checks=3, progress=False)
    assert checked == 3
    # For n=2k, all subsets should work
    assert ok


def test_verify_construction_insufficient_dimension():
    """For m=5, k=2, n=1 (n < k), should fail."""
    ok, checked, failed = verify_construction(5, 2, 1, max_checks=None, progress=False)
    assert not ok
    assert failed == 1
