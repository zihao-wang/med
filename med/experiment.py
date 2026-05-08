from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List

from .checker import FeasibilityChecker
from .search import binary_search_med


@dataclass
class Experiment:
    """Orchestrates MED binary search sweeps over m (and optionally k) values.

    checker_factory is a callable that takes (m, k) and returns a
    FeasibilityChecker instance.
    """

    checker_factory: Callable[[int, int], FeasibilityChecker]
    search_paths: Dict[int, List[dict]] = field(default_factory=dict)
    minimal_dimensions: Dict[int, int] = field(default_factory=dict)
    timings: Dict[int, float] = field(default_factory=dict)
    # Grid results when sweeping over both k and m
    grid_minimal_dimensions: Dict[int, Dict[int, int]] = field(default_factory=dict)
    grid_search_paths: Dict[int, Dict[int, List[dict]]] = field(default_factory=dict)
    grid_timings: Dict[int, Dict[int, float]] = field(default_factory=dict)

    def find_minimal_dimension(
        self,
        k: int,
        m_values: List[int],
        left0: int = 0,
    ) -> Dict[int, int]:
        """Binary search for MED over a list of m values, warm-starting between m."""
        last_minimal = left0
        for m in m_values:
            t0 = time.perf_counter()
            print("#" * 10 + " new task " + "#" * 10)
            print(f"[EXP] Finding minimal dimension for m={m}, k={k}")
            print("#" * 30)

            checker = self.checker_factory(m, k)
            result = binary_search_med(checker, left0=last_minimal, verbose=True)

            self.search_paths[m] = result["search_path"]
            med = result.get("med")
            self.minimal_dimensions[m] = med if med is not None else -1
            if med is not None and med > 0:
                last_minimal = max(0, med - 1)
            self.timings[m] = result.get("total_time", round(time.perf_counter() - t0, 1))

            print(f"[RESULT] minimal dimension @ k={k} & m={m} is {med}")
            print("#" * 30)

        return self.minimal_dimensions

    def find_minimal_dimension_grid(
        self,
        k_values: List[int],
        m_values: List[int],
        left0: int = 0,
    ) -> Dict[int, Dict[int, int]]:
        """Grid search over both k and m values, warm-starting between k."""
        self.grid_minimal_dimensions = {}
        self.grid_search_paths = {}
        self.grid_timings = {}

        warm_start = left0
        for kval in k_values:
            print("#" * 10 + f" Starting k={kval} grid row " + "#" * 10)
            self.search_paths = {}
            self.minimal_dimensions = {}
            self.timings = {}

            minimal_for_k = self.find_minimal_dimension(
                k=kval,
                m_values=m_values,
                left0=warm_start,
            )

            self.grid_minimal_dimensions[kval] = dict(minimal_for_k)
            self.grid_search_paths[kval] = dict(self.search_paths)
            self.grid_timings[kval] = dict(self.timings)

            feasible_ds = [d for d in minimal_for_k.values() if d != -1]
            if feasible_ds:
                warm_start = min(feasible_ds)

        return self.grid_minimal_dimensions
