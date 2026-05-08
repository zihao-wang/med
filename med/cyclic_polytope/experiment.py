from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ..checker import FeasibilityChecker
from ..experiment import Experiment as _SharedExperiment
from .checker import CyclicPolytopeChecker


def _make_checker_factory(
    max_subset_checks: Optional[int] = 2000,
    seed: int = 42,
) -> Callable[[int, int], FeasibilityChecker]:
    def factory(m: int, k: int) -> FeasibilityChecker:
        return CyclicPolytopeChecker(
            m=m, k=k, max_checks=max_subset_checks, seed=seed,
        )
    return factory


class Experiment(_SharedExperiment):
    """Orchestrates MED binary search for the cyclic polytope construction."""

    def __init__(
        self,
        max_subset_checks: Optional[int] = 2000,
        seed: int = 42,
    ):
        super().__init__(
            checker_factory=_make_checker_factory(
                max_subset_checks=max_subset_checks, seed=seed,
            ),
        )
