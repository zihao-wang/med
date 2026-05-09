from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import torch

from med.defaults import DEFAULT_M_VALUES

from .experiment import Experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run minimal-embedding-dimension experiments (cyclic polytope)"
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="*",
        default=[2],
        help="List of k values (default: [2])",
    )
    parser.add_argument(
        "--m_values",
        type=int,
        nargs="*",
        default=DEFAULT_M_VALUES,
        help="List of m values (number of points)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[MAIN] Starting cyclic polytope MED experiment...")

    env_info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
    }
    print(f"[ENV] {env_info}")

    experiment = Experiment()

    run_config = {
        "m_values": args.m_values,
        "k_values": args.k_values,
        "env": env_info,
        "cwd": os.getcwd(),
    }
    try:
        with open("config.json", "w") as f:
            json.dump(run_config, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write config.json: {e}")

    t0 = time.perf_counter()
    print(
        f"[RUN] Running over k_values={args.k_values} and m_values={args.m_values}"
    )
    grid = experiment.find_minimal_dimension(
        k_values=args.k_values,
        m_values=args.m_values,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n[RUN] Minimal dimensions found: {grid}")
    print(f"[RUN] Total elapsed: {elapsed:.2f}s")

    results: dict = {}
    for kval in args.k_values:
        sk = str(kval)
        results[sk] = []
        for m in args.m_values:
            results[sk].append({
                "m": m,
                "med": experiment.minimal_dimensions.get(kval, {}).get(m, -1),
                "search_path": experiment.search_paths.get(kval, {}).get(m, []),
                "time": experiment.timings.get(kval, {}).get(m, -1),
            })

    unified: dict = {
        "experiment": "med",
        "results": results,
    }
    try:
        with open("results.json", "w") as f:
            json.dump(unified, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write results.json: {e}")


if __name__ == "__main__":
    main()
