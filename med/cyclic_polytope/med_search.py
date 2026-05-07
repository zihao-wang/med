from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Optional

import numpy as np

from .construct import verify_construction


def find_minimal_dimension(
    m: int,
    k: int,
    t: Optional[np.ndarray] = None,
    max_subset_checks: Optional[int] = 2000,
    seed: int = 42,
    left0: int = 0,
    verbose: bool = True,
) -> dict:
    """Binary search for smallest n such that every k-subset of C(m,n) is a face.

    Search range is [left0+1, left0+40], matching the MED-C convention.
    Warm-start left0 from the MED found for a smaller m value.
    """
    lo = left0 + 1
    hi = left0 + 40

    result: dict = {
        "m": m,
        "k": k,
        "search_path": [],
        "med": None,
    }

    if verbose:
        total = math.comb(m, k)
        effective = min(max_subset_checks, total) if max_subset_checks else total
        print(f"Cyclic polytope MED search: m={m}, k={k}")
        print(f"  Search range: n ∈ [{lo}, {hi}]")
        print(f"  Subset checks per n: {effective} / {total} total")

    def _check_n(n: int) -> tuple[bool, float]:
        t0 = time.perf_counter()
        ok, checked, failed = verify_construction(
            m, k, n, t=t, max_checks=max_subset_checks, seed=seed, progress=verbose
        )
        elapsed = time.perf_counter() - t0
        result["search_path"].append(
            {"dimension": n, "feasible": ok, "checks": checked, "time": round(elapsed, 3)}
        )
        if verbose:
            status = "OK" if ok else "FAIL"
            print(f"  n={n}: {status}  ({checked} checked, {elapsed:.1f}s)")
        return ok, elapsed

    t_start = time.perf_counter()

    minimal_n = hi + 1  # sentinel: if unchanged, upper bound was infeasible

    while lo <= hi:
        mid = (lo + hi) // 2
        if mid == 0:
            mid = 1
        ok, _ = _check_n(mid)
        if ok:
            minimal_n = mid
            hi = mid - 1
        else:
            lo = mid + 1

    if minimal_n <= left0 + 40:
        result["med"] = minimal_n
    result["total_time"] = round(time.perf_counter() - t_start, 1)
    if verbose:
        if result["med"] is not None:
            print(f"  MED = {result['med']}  (total {result['total_time']:.1f}s)")
        else:
            print(f"  MED not found in [{left0 + 1}, {left0 + 40}] — infeasible")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find minimal dimension for cyclic polytope construction"
    )
    p.add_argument("--m", type=int, required=True, help="Number of points")
    p.add_argument("--k", type=int, required=True, help="Subset size")
    p.add_argument(
        "--max_checks",
        type=int,
        default=2000,
        help="Max random k-subset checks per n (default: 2000; 0 = exhaustive)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--left0", type=int, default=0, help="Warm-start lower bound (MED-C style)")
    p.add_argument(
        "--output", type=str, default=None, help="Save result as JSON to path"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    max_checks = args.max_checks if args.max_checks > 0 else None

    result = find_minimal_dimension(
        m=args.m,
        k=args.k,
        max_subset_checks=max_checks,
        seed=args.seed,
        left0=args.left0,
        verbose=True,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
