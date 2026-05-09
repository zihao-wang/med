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
from .lr_scaling import LR_SCALING_CHOICES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run centroid-embedding GD minimal-dimension experiments"
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="*",
        default=[2],
        help="List of k values (default: [2])",
    )
    parser.add_argument(
        "--n_values",
        type=int,
        nargs="*",
        default=DEFAULT_M_VALUES,
        help="List of n values",
    )
    parser.add_argument(
        "--scoring_function",
        type=str,
        default="inner_product",
        choices=["inner_product", "l2", "cosine", "l1"],
    )

    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument(
        "--patience",
        type=int,
        default=1000,
        help="Early-stopping patience for full-batch GD",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1,
        help="Base GD learning rate; --lr_scaling controls any m-dependent scaling",
    )
    parser.add_argument(
        "--lr_scaling",
        type=str,
        default="constant",
        choices=LR_SCALING_CHOICES,
        help="How to scale --learning_rate as m grows",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[MAIN] Starting experiment with early stopping...")

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

    experiment = Experiment(
        scoring_function=args.scoring_function,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scaling=args.lr_scaling,
        patience=args.patience,
    )

    run_config = {
        "trainer": "gd",
        "scoring_function": args.scoring_function,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "lr_scaling": args.lr_scaling,
        "n_values": args.n_values,
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
        f"[RUN] Running over k_values={args.k_values} and n_values={args.n_values}"
    )
    grid = experiment.find_minimal_dimension(
        k_values=args.k_values,
        m_values=args.n_values,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n[RUN] Minimal dimensions found: {grid}")
    print(f"[RUN] Total elapsed: {elapsed:.2f}s")

    results: dict = {}
    for kval in args.k_values:
        sk = str(kval)
        results[sk] = []
        for n in args.n_values:
            results[sk].append({
                "m": n,
                "med": experiment.minimal_dimensions.get(kval, {}).get(n, -1),
                "search_path": experiment.search_paths.get(kval, {}).get(n, []),
                "time": experiment.timings.get(kval, {}).get(n, -1),
            })

    unified: dict = {
        "experiment": "medc",
        "scoring_function": args.scoring_function,
        "trainer": "gd",
        "results": results,
    }
    try:
        with open("results.json", "w") as f:
            json.dump(unified, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write results.json: {e}")


if __name__ == "__main__":
    main()
