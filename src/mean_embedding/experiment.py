from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
import matplotlib.pyplot as plt

from .train_gd import Trainer, ScoringFn
from .trainer_sgd import SGDTrainer, SGDConfig


@dataclass
class Experiment:
    scoring_function: ScoringFn
    trainer_type: Literal["gd", "sgd"] = "gd"
    sgd_config: Optional[SGDConfig] = None
    search_paths: Dict[int, List[dict]] = field(default_factory=dict)
    minimal_dimensions: Dict[int, int] = field(default_factory=dict)
    # Grid results when sweeping over both k and n
    grid_minimal_dimensions: Dict[int, Dict[int, int]] = field(default_factory=dict)
    grid_search_paths: Dict[int, Dict[int, List[dict]]] = field(default_factory=dict)

    def find_minimal_dimension(
        self,
        k: int,
        n_values: List[int],
        left0: int = 0,
        num_epochs: int = 100,
        learning_rate: float = 0.1,
        patience: int = 10,
    ) -> Dict[int, int]:
        last_minimal = left0
        for n in n_values:
            print("#" * 10 + " new task " + "#" * 10)
            print(f"[EXP] Finding minimal dimension for n={n}, k={k}")
            print("#" * 30)
            if self.trainer_type == "gd":
                trainer = Trainer(n, k, self.scoring_function)
            elif self.trainer_type == "sgd":
                trainer = SGDTrainer(n, k, self.scoring_function, config=self.sgd_config or SGDConfig())
            else:
                raise ValueError(f"Unknown trainer_type: {self.trainer_type}")

            self.search_paths[n] = []
            left, right = last_minimal + 1, last_minimal + 6
            minimal_d = n + 1

            while left <= right:
                mid = (left + right) // 2
                if mid == 0:
                    mid = 1
                print(f"\t[EXP] Testing dimension d={mid}")

                if self.trainer_type == "gd":
                    violations = trainer.train(
                        d=mid,
                        num_epochs=num_epochs,
                        learning_rate=learning_rate / np.log2(n),
                        patience=patience,
                    )
                else:
                    # SGD trainer uses its own config for lr/patience/steps
                    violations = trainer.train(
                        d=mid,
                        num_epochs=num_epochs,
                    )

                print(f"\t[EXP] Violations for d={mid}: {violations}")
                self.search_paths[n].append({"dimension": mid, "violations": violations})

                if violations == 0:
                    minimal_d = mid
                    right = mid - 1
                else:
                    left = mid + 1

            self.minimal_dimensions[n] = minimal_d if minimal_d <= n else -1
            last_minimal = minimal_d

            with open("minimal_dem_log.txt", "at") as f:
                f.write(f"[RESULT] minimal dimension @ k={k} & n={n} is {minimal_d}\n")
            print("#" * 10 + " Task Finished " + "#" * 10)
            print(f"[RESULT] minimal dimension @ k={k} & n={n} is {minimal_d}\n")
            print("#" * 30)

        return self.minimal_dimensions

    def find_minimal_dimension_grid(
        self,
        k_values: List[int],
        n_values: List[int],
        left0: int = 0,
        num_epochs: int = 100,
        learning_rate: float = 0.1,
        patience: int = 10,
    ) -> Dict[int, Dict[int, int]]:
        """
        Run minimal dimension search across a grid of k and n values.

        For each k in k_values, this method leverages the existing
        find_minimal_dimension routine to sweep over n_values while
        reusing a warm-start lower bound for the searched dimension.

        Returns a nested mapping {k: {n: minimal_d}}.
        """
        self.grid_minimal_dimensions = {}
        self.grid_search_paths = {}

        warm_start = left0
        for k in k_values:
            print("#" * 10 + f" Starting k={k} grid row " + "#" * 10)
            # Reset per-k accumulators so search paths and results are isolated
            self.search_paths = {}
            self.minimal_dimensions = {}
            minimal_for_k = self.find_minimal_dimension(
                k=k,
                n_values=n_values,
                left0=warm_start,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                patience=patience,
            )

            # Persist per-k results and search paths
            self.grid_minimal_dimensions[k] = dict(minimal_for_k)
            self.grid_search_paths[k] = dict(self.search_paths)

            # Update warm-start for next k using smallest feasible d found
            feasible_ds = [d for d in minimal_for_k.values() if d != -1]
            if feasible_ds:
                warm_start = min(feasible_ds)

        return self.grid_minimal_dimensions

    def plot_minimal_dimension_vs_n(self, k: int) -> None:
        if not self.minimal_dimensions:
            print("No experiment results to plot. Run find_minimal_dimension first.")
            return

        n_plot = list(self.minimal_dimensions.keys())
        d_plot = list(self.minimal_dimensions.values())

        plt.figure(figsize=(8, 6))
        plt.plot(n_plot, d_plot, marker="o", linestyle="-")
        plt.xlabel("Number of Vectors (n)")
        plt.ylabel("Minimal Dimension (d)")
        plt.xscale("log")
        plt.title(f"Minimal Dimension vs. Number of Vectors for k={k}")
        plt.grid(True)
        plt.show()

        plt.savefig("med_vs_n.png")

    def plot_violations_vs_d(self, k: int) -> None:
        if not self.search_paths:
            print("No experiment results to plot. Run find_minimal_dimension first.")
            return

        for n, path in self.search_paths.items():
            sorted_path = sorted(path, key=lambda x: x["dimension"])

            dimensions = [item["dimension"] for item in sorted_path]
            violations = [item["violations"] for item in sorted_path]

            plt.figure(figsize=(8, 6))
            plt.plot(dimensions, violations, marker="o", linestyle="-")
            plt.xlabel("Dimension (d)")
            plt.ylabel("Number of Violations")
            plt.title(f"Violations vs. Dimension for n={n}, k={k}")
            plt.grid(True)
            plt.show()
