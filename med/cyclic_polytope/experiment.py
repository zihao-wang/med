from __future__ import annotations

from math import comb

from ..experiment import Experiment as _SharedExperiment
from .construct import verify_construction


def _make_check_dimension(m: int, k: int):
    total_queries = comb(m, k)

    def check_dimension(d: int) -> bool:
        ok, checked, failed = verify_construction(m, k, d)
        assert checked == total_queries, (
            f"cyclic construction checked {checked} queries, expected {total_queries}"
        )
        assert ok == (failed == 0)
        return ok

    return check_dimension


class Experiment(_SharedExperiment):
    """Orchestrates MED binary search for the cyclic polytope construction."""

    def __init__(self):
        super().__init__(check_factory=_make_check_dimension)
