from __future__ import annotations

from math import comb
from typing import Optional

import numpy as np

from ..checker import CheckResult, FeasibilityChecker
from .construct import verify_construction_up_to_k


class CyclicPolytopeChecker(FeasibilityChecker):
    """Feasibility checker using the moment-curve / squared-polynomial construction."""

    def __init__(
        self,
        m: int,
        k: int,
        t: Optional[np.ndarray] = None,
    ):
        self.m = m
        self.k = k
        self.t = t

    def check(self, d: int) -> CheckResult:
        ok, checked, failed, checks_by_size = verify_construction_up_to_k(
            self.m, self.k, d,
            t=self.t,
        )
        total_queries = sum(
            comb(self.m, subset_size) for subset_size in range(1, self.k + 1)
        )
        checked_fraction = checked / total_queries if total_queries else 0.0
        return CheckResult(
            feasible=ok,
            details={
                "checks": checked,
                "total_queries": total_queries,
                "checked_fraction": checked_fraction,
                "checks_by_size": checks_by_size,
                "failures": failed,
            },
        )
