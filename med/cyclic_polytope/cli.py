from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

from .experiment import Experiment

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run minimal-embedding-dimension experiments (cyclic polytope)"
    )
    parser.add_argument("--k", type=int, default=None, help="Subset size k (single)")
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="*",
        default=None,
        help="List of k values (grid mode)",
    )
    parser.add_argument(
        "--m_values",
        type=int,
        nargs="*",
        default=[8, 16, 32, 64, 128, 256, 512, 1024],
        help="List of m values (number of points)",
    )
    parser.add_argument(
        "--max_checks",
        type=int,
        default=2000,
        help="Max random k-subset checks per n (default: 2000)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[MAIN] Starting cyclic polytope MED experiment...")

    env_info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if torch is not None:
        try:
            env_info.update(
                {
                    "torch": getattr(torch, "__version__", "unknown"),
                    "cuda_available": bool(torch.cuda.is_available()),
                    "cuda_device_count": int(torch.cuda.device_count()),
                    "cuda_device_name": torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None,
                }
            )
        except Exception:
            pass
    print(f"[ENV] {env_info}")

    experiment = Experiment(
        max_subset_checks=args.max_checks,
        seed=args.seed,
    )

    run_config = {
        "max_checks": args.max_checks,
        "seed": args.seed,
        "m_values": args.m_values,
        "k": args.k,
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
    if args.k_values is not None and len(args.k_values) > 0:
        print(
            f"[GRID] Running grid over k_values={args.k_values} and m_values={args.m_values}"
        )
        grid = experiment.find_minimal_dimension_grid(
            k_values=args.k_values,
            m_values=args.m_values,
        )
        elapsed = time.perf_counter() - t0
        print(f"\n[GRID] Minimal dimensions found: {grid}")
        print(f"[GRID] Total elapsed: {elapsed:.2f}s")

        grid_results: dict = {}
        for kval in args.k_values:
            sk = str(kval)
            grid_results[sk] = []
            for m in args.m_values:
                grid_results[sk].append({
                    "m": m,
                    "med": experiment.grid_minimal_dimensions.get(kval, {}).get(m, -1),
                    "search_path": experiment.grid_search_paths.get(kval, {}).get(m, []),
                    "time": experiment.grid_timings.get(kval, {}).get(m, -1),
                })

        unified: dict = {
            "experiment": "med",
            "results": grid_results,
        }
        try:
            with open("results.json", "w") as f:
                json.dump(unified, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to write results.json: {e}")
    else:
        k_value = args.k if args.k is not None else 2
        print(f"[RUN] Running single-k search: k={k_value}, m_values={args.m_values}")
        minimal_dims = experiment.find_minimal_dimension(
            k=k_value,
            m_values=args.m_values,
        )

        elapsed = time.perf_counter() - t0
        print("\n[RUN] Minimal dimensions found:", minimal_dims)
        print(f"[RUN] Total elapsed: {elapsed:.2f}s")

        results_list = []
        for m in args.m_values:
            results_list.append({
                "m": m,
                "med": minimal_dims.get(m, -1),
                "search_path": experiment.search_paths.get(m, []),
                "time": experiment.timings.get(m, -1),
            })

        unified = {
            "experiment": "med",
            "k": k_value,
            "results": results_list,
        }
        try:
            with open("results.json", "w") as f:
                json.dump(unified, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to write results.json: {e}")


if __name__ == "__main__":
    main()
