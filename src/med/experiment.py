from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List

from .checker import FeasibilityChecker
from .search import binary_search_med


@dataclass
class Experiment:
    """Orchestrates MED binary search sweeps over k and m values.

    checker_factory is a callable that takes (m, k) and returns a
    FeasibilityChecker instance.
    """

    checker_factory: Callable[[int, int], FeasibilityChecker]
    search_paths: Dict[int, Dict[int, List[dict]]] = field(default_factory=dict)
    minimal_dimensions: Dict[int, Dict[int, int]] = field(default_factory=dict)
    timings: Dict[int, Dict[int, float]] = field(default_factory=dict)

    def find_minimal_dimension(
        self,
        k_values: List[int],
        m_values: List[int],
        left0: int = 0,
    ) -> Dict[int, Dict[int, int]]:
        """Binary search for MED over k_values x m_values, warm-starting between k."""
        self.search_paths.clear()
        self.minimal_dimensions.clear()
        self.timings.clear()

        warm_start = left0
        for k in k_values:
            print("#" * 10 + f" Starting k={k} " + "#" * 10)
            self.search_paths[k] = {}
            self.minimal_dimensions[k] = {}
            self.timings[k] = {}

            last_minimal = warm_start
            for m in m_values:
                t0 = time.perf_counter()
                print("#" * 10 + " new task " + "#" * 10)
                print(f"[EXP] Finding minimal dimension for m={m}, k={k}")
                print("#" * 30)

                checker = self.checker_factory(m, k)
                result = binary_search_med(checker, left0=last_minimal, verbose=True)

                self.search_paths[k][m] = result["search_path"]
                med = result.get("med")
                self.minimal_dimensions[k][m] = med if med is not None else -1
                if med is not None and med > 0:
                    last_minimal = max(0, med - 1)
                self.timings[k][m] = result.get("total_time", round(time.perf_counter() - t0, 1))

                print(f"[RESULT] minimal dimension @ k={k} & m={m} is {med}")
                print("#" * 30)

            feasible_ds = [d for d in self.minimal_dimensions[k].values() if d != -1]
            if feasible_ds:
                warm_start = min(feasible_ds)

        return self.minimal_dimensions
