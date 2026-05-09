from __future__ import annotations

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
        return CheckResult(feasible=ok, details={"checks": checked})
