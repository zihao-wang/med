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
        max_checks: Optional[int] = 2000,
        seed: int = 42,
    ):
        self.m = m
        self.k = k
        self.t = t
        self.max_checks = max_checks
        self.seed = seed

    def check(self, d: int) -> CheckResult:
        ok, checked, failed = verify_construction(
            self.m, self.k, d,
            t=self.t,
            max_checks=self.max_checks,
            seed=self.seed,
            progress=False,
        )
        return CheckResult(feasible=ok, details={"checks": checked})
