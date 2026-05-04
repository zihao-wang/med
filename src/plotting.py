"""Common plotting utilities for MED experiments.

Includes the WBNL fitted curve from Weller et al. (2025) and plot styling.
"""

import numpy as np


def wbnl_critical_m(d: np.ndarray) -> np.ndarray:
    """WBNL fitted curve: max number of elements m* supported in dimension d.

    From Weller et al. (2025), Equation (approximate):
        m(d) = -10.5322 + 4.0309 d + 0.0520 d^2 + 0.0037 d^3
    """
    return -10.5322 + 4.0309 * d + 0.0520 * d**2 + 0.0037 * d**3


def invert_wbnl_curve(m: np.ndarray, d_range: tuple = (1, 100)) -> np.ndarray:
    """Numerically invert the WBNL curve to get d*(m).

    Given m values, find the corresponding d values such that
    wbnl_critical_m(d) ≈ m by solving the cubic equation.

    Args:
        m: array of m values (number of elements).
        d_range: (d_min, d_max) search range.

    Returns:
        array of d values (dimensions).
    """
    d_fine = np.linspace(d_range[0], d_range[1], 10000)
    m_fine = wbnl_critical_m(d_fine)

    d_result = np.empty_like(m, dtype=float)
    for i, mi in enumerate(m):
        idx = np.searchsorted(m_fine, mi)
        idx = np.clip(idx, 0, len(d_fine) - 1)
        d_result[i] = d_fine[idx]

    return d_result


def set_paper_style():
    """Configure matplotlib for paper-consistent figure styling."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
