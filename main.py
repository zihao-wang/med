from __future__ import annotations

import argparse

from src.mean_embedding.experiment import Experiment
from src.mean_embedding.trainer_sgd import SGDConfig


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
        default=[5 * (2 ** i) for i in [0, 1, 2, 3, 4, 5, 6, 7]],
        help="List of n values",
    )
    parser.add_argument(
        "--scoring_function",
        type=str,
        default="inner_product",
        choices=["inner_product", "l2", "cosine", "l1"],
    )

    # Trainer selection
    parser.add_argument(
        "--trainer",
        type=str,
        default="gd",
        choices=["gd", "sgd"],
        help="Training algorithm: full-batch GD or SGD",
    )

    # Common hyperparameters
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=1000, help="GD patience; SGD uses config.patience")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="GD learning rate; SGD uses config.learning_rate")

    # SGD-specific options
    parser.add_argument("--sgd_batch_size", type=int, default=256)
    parser.add_argument("--sgd_num_samples", type=int, default=50_000)
    parser.add_argument("--sgd_workers", type=int, default=0)
    parser.add_argument("--sgd_patience", type=int, default=500)
    parser.add_argument("--sgd_lr", type=float, default=1e-2)
    parser.add_argument("--sgd_max_steps", type=int, default=None)
    parser.add_argument("--sgd_negative_sampling", type=int, default=None)

    parser.add_argument("--plot", action="store_true", help="Generate plots after run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Starting experiment with early stopping...")

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
    )

    # Dispatch: grid mode if k_values provided, else single-k mode
    if args.k_values is not None and len(args.k_values) > 0:
        grid = experiment.find_minimal_dimension_grid(
            k_values=args.k_values,
            n_values=args.n_values,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            patience=args.patience,
        )
        print("\n Grid minimal dimensions found:", grid)
    else:
        k_value = args.k if args.k is not None else 2
        minimal_dims = experiment.find_minimal_dimension(
            k_value,
            args.n_values,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            patience=args.patience,
        )

        print("\n Minimal dimensions found:", minimal_dims)
        if args.plot:
            experiment.plot_minimal_dimension_vs_n(k_value)
            experiment.plot_violations_vs_d(k_value)


if __name__ == "__main__":
    main()
