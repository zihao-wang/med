from __future__ import annotations

import argparse

from src.mean_embedding.experiment import Experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal-dimension experiments")
    parser.add_argument("--k", type=int, default=2, help="Subset size k")
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
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--plot", action="store_true", help="Generate plots after run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Starting experiment with early stopping...")
    experiment = Experiment(args.scoring_function)
    minimal_dims = experiment.find_minimal_dimension(
        args.k,
        args.n_values,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience,
    )

    print("\n Minimal dimensions found:", minimal_dims)
    if args.plot:
        experiment.plot_minimal_dimension_vs_n(args.k)
        experiment.plot_violations_vs_d(args.k)


if __name__ == "__main__":
    main()
