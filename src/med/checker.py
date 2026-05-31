from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class CheckResult:
    """Result of a single feasibility check at dimension d."""
    feasible: bool
    details: dict = field(default_factory=dict)


class FeasibilityChecker(ABC):
    """Check whether a given dimension d admits perfect top-k retrieval.

    Created once per (m, k) pair, then ``check(d)`` is called for different
    candidate dimensions during binary search.
    """

    @abstractmethod
    def check(self, d: int) -> CheckResult:
        ...
