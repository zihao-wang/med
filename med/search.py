from __future__ import annotations

import time
from collections.abc import Callable
from typing import Optional


def binary_search_med(
    check_dimension: Callable[[int], bool],
    left0: int = 0,
    max_range: int = 40,
    verbose: bool = True,
) -> dict:
    """Binary search for the minimal dimension d such that ``check_dimension(d)`` passes.

    Search interval is [left0 + 1, left0 + max_range].

    Returns a dict with keys:
        med: int or None  — minimal feasible dimension (None if none found)
        search_path: list[dict] — each checked dimension with feasibility and timing
        total_time: float
    """
    lo = left0 + 1
    hi = left0 + max_range

    search_path: list[dict] = []
    result: dict = {"med": None, "search_path": search_path}

    t_start = time.perf_counter()

    minimal_d = hi + 1  # sentinel — if unchanged, upper bound was infeasible

    while lo <= hi:
        mid = (lo + hi) // 2
        if mid == 0:
            mid = 1

        t0 = time.perf_counter()
        feasible = check_dimension(mid)
        elapsed = time.perf_counter() - t0

        entry = {"dimension": mid, "feasible": feasible, "time": round(elapsed, 3)}
        search_path.append(entry)

        if verbose:
            status = "OK" if feasible else "FAIL"
            print(f"  d={mid}: {status}  ({elapsed:.1f}s)")

        if feasible:
            minimal_d = mid
            hi = mid - 1
        else:
            lo = mid + 1

    if minimal_d <= left0 + max_range:
        result["med"] = minimal_d
    result["total_time"] = round(time.perf_counter() - t_start, 1)

    if verbose:
        if result["med"] is not None:
            print(f"  MED = {result['med']}  (total {result['total_time']:.1f}s)")
        else:
            print(f"  MED not found in [{left0 + 1}, {left0 + max_range}] — infeasible")

    return result
