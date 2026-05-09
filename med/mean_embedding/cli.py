from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import torch

from .experiment import Experiment
from .trainer_sgd import SGDConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal-dimension experiments")
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
        default=[8, 16, 32, 64, 128, 256, 512, 1024],
        help="List of n values",
    )
    parser.add_argument(
        "--scoring_function",
        type=str,
        default="inner_product",
        choices=["inner_product", "l2", "cosine", "l1"],
    )

    parser.add_argument(
        "--trainer",
        type=str,
        default="gd",
        choices=["gd", "sgd"],
        help="Training algorithm: full-batch GD or SGD",
    )

    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument(
        "--patience",
        type=int,
        default=1000,
        help="GD patience; SGD uses config.patience",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1,
        help="GD learning rate; SGD uses config.learning_rate",
    )

    parser.add_argument("--sgd_batch_size", type=int, default=256)
    parser.add_argument("--sgd_num_samples", type=int, default=50_000)
    parser.add_argument("--sgd_workers", type=int, default=0)
    parser.add_argument("--sgd_patience", type=int, default=500)
    parser.add_argument("--sgd_lr", type=float, default=1e-2)
    parser.add_argument("--sgd_max_steps", type=int, default=None)
    parser.add_argument("--sgd_negative_sampling", type=int, default=None)

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

    sgd_config = None
    if args.trainer == "sgd":
        sgd_config = SGDConfig(
            batch_size=args.sgd_batch_size,
            num_loader_workers=args.sgd_workers,
            num_samples=args.sgd_num_samples,
            learning_rate=args.sgd_lr,
            patience=args.sgd_patience,
            max_steps=args.sgd_max_steps,
            negative_sampling=args.sgd_negative_sampling,
        )

    experiment = Experiment(
        scoring_function=args.scoring_function,
        trainer_type=args.trainer,
        sgd_config=sgd_config,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )

    run_config = {
        "trainer": args.trainer,
        "scoring_function": args.scoring_function,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "n_values": args.n_values,
        "k_values": args.k_values,
        "sgd": {
            "batch_size": args.sgd_batch_size,
            "num_samples": args.sgd_num_samples,
            "workers": args.sgd_workers,
            "patience": args.sgd_patience,
            "lr": args.sgd_lr,
            "max_steps": args.sgd_max_steps,
            "negative_sampling": args.sgd_negative_sampling,
        },
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
        "trainer": args.trainer,
        "results": results,
    }
    try:
        with open("results.json", "w") as f:
            json.dump(unified, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write results.json: {e}")


if __name__ == "__main__":
    main()
