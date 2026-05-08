from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

from .experiment import Experiment
from .trainer_sgd import SGDConfig

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional logging only
    torch = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal-dimension experiments")
    parser.add_argument("--k", type=int, default=None, help="Subset size k (single)")
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="*",
        default=None,
        help="List of k values (grid mode)",
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
        "k": args.k,
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
    except Exception as e:  # pragma: no cover
        print(f"[WARN] Failed to write config.json: {e}")

    t0 = time.perf_counter()
    if args.k_values is not None and len(args.k_values) > 0:
        print(
            f"[GRID] Running grid over k_values={args.k_values} and n_values={args.n_values}"
        )
        grid = experiment.find_minimal_dimension_grid(
            k_values=args.k_values,
            m_values=args.n_values,
        )
        elapsed = time.perf_counter() - t0
        print(f"\n[GRID] Minimal dimensions found: {grid}")
        print(f"[GRID] Total elapsed: {elapsed:.2f}s")

        grid_results: dict = {}
        for kval in args.k_values:
            sk = str(kval)
            grid_results[sk] = []
            for n in args.n_values:
                grid_results[sk].append({
                    "m": n,
                    "med": experiment.grid_minimal_dimensions.get(kval, {}).get(n, -1),
                    "search_path": experiment.grid_search_paths.get(kval, {}).get(n, []),
                    "time": experiment.grid_timings.get(kval, {}).get(n, -1),
                })

        unified: dict = {
            "experiment": "medc",
            "scoring_function": args.scoring_function,
            "trainer": args.trainer,
            "results": grid_results,
        }
        try:
            with open("results.json", "w") as f:
                json.dump(unified, f, indent=2)
        except Exception as e:  # pragma: no cover
            print(f"[WARN] Failed to write results.json: {e}")
    else:
        k_value = args.k if args.k is not None else 2
        print(f"[RUN] Running single-k search: k={k_value}, n_values={args.n_values}")
        minimal_dims = experiment.find_minimal_dimension(
            k=k_value,
            m_values=args.n_values,
        )

        elapsed = time.perf_counter() - t0
        print("\n[RUN] Minimal dimensions found:", minimal_dims)
        print(f"[RUN] Total elapsed: {elapsed:.2f}s")

        results_list = []
        for n in args.n_values:
            results_list.append({
                "m": n,
                "med": minimal_dims.get(n, -1),
                "search_path": experiment.search_paths.get(n, []),
                "time": experiment.timings.get(n, -1),
            })

        unified = {
            "experiment": "medc",
            "k": k_value,
            "scoring_function": args.scoring_function,
            "trainer": args.trainer,
            "results": results_list,
        }
        try:
            with open("results.json", "w") as f:
                json.dump(unified, f, indent=2)
        except Exception as e:  # pragma: no cover
            print(f"[WARN] Failed to write results.json: {e}")


if __name__ == "__main__":
    main()
