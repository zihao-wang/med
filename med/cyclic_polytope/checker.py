from __future__ import annotations

from math import comb
from typing import Optional

import numpy as np

from ..checker import CheckResult, FeasibilityChecker
from .construct import verify_construction


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
        ok, checked, failed = verify_construction(
            self.m, self.k, d,
            t=self.t,
        )
        total_queries = comb(self.m, self.k)
        checked_fraction = checked / total_queries if total_queries else 0.0
        return CheckResult(
            feasible=ok,
            details={
                "checks": checked,
                "total_queries": total_queries,
                "checked_fraction": checked_fraction,
            },
        )
