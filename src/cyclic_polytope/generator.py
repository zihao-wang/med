from __future__ import annotations

import numpy as np


def generate_cyclic_polytope_configuration(m: int, n: int, t: np.ndarray | None = None) -> np.ndarray:
    """
    Generate the vertex set of a cyclic polytope C(m, n) using the moment curve.

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
    coords = [t_col ** i for i in range(1, n + 1)]
    X = np.concatenate(coords, axis=1)
    return X
