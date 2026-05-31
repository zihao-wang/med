from __future__ import annotations

from typing import Callable

from ..checker import FeasibilityChecker
from ..experiment import Experiment as _SharedExperiment
from .checker import CyclicPolytopeChecker


def _make_checker_factory() -> Callable[[int, int], FeasibilityChecker]:
    def factory(m: int, k: int) -> FeasibilityChecker:
        return CyclicPolytopeChecker(m=m, k=k)
    return factory


class Experiment(_SharedExperiment):
    """Orchestrates MED binary search for the cyclic polytope construction."""

    def __init__(self):
        super().__init__(checker_factory=_make_checker_factory())
