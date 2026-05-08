"""Tests for generator.py — moment curve point generation."""

import numpy as np
import pytest

from med.cyclic_polytope.generator import generate_cyclic_polytope_configuration


def test_default_t_gives_shape():
    X = generate_cyclic_polytope_configuration(5, 3)
    assert X.shape == (5, 3)


def test_default_t_uses_arange():
    X = generate_cyclic_polytope_configuration(4, 2)
    # First column should be t^1 = [0, 1, 2, 3]
    np.testing.assert_allclose(X[:, 0], np.arange(4, dtype=float))


def test_custom_t():
    t = np.array([0.0, 0.5, 1.0])
    X = generate_cyclic_polytope_configuration(3, 2, t)
    np.testing.assert_allclose(X[:, 0], t)
    np.testing.assert_allclose(X[:, 1], t**2)


def test_t_must_be_strictly_increasing():
    t = np.array([0.0, 1.0, 0.5])  # not sorted
    with pytest.raises(AssertionError, match="strictly increasing"):
        generate_cyclic_polytope_configuration(3, 2, t)


def test_t_shape_mismatch():
    t = np.array([0.0, 1.0])  # length 2, but m=3
    with pytest.raises(AssertionError, match="shape"):
        generate_cyclic_polytope_configuration(3, 2, t)


def test_higher_dimensions():
    X = generate_cyclic_polytope_configuration(3, 4)
    # Columns: t^1, t^2, t^3, t^4 for t = [0, 1, 2]
    assert X.shape == (3, 4)
    np.testing.assert_allclose(X[2, 3], 2**4)  # t=2, power=4
