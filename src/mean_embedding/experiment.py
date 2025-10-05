from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from .trainer import Trainer, ScoringFn


@dataclass
class Experiment:
    scoring_function: ScoringFn
    search_paths: Dict[int, List[dict]] = field(default_factory=dict)
    minimal_dimensions: Dict[int, int] = field(default_factory=dict)

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
            print("#" * 10 + "new task" + "#" * 10)
            print(f"Finding minimal dimension for n={n}, k={k}")
            print("#" * 30)
            trainer = Trainer(n, k, self.scoring_function)

            self.search_paths[n] = []
            left, right = last_minimal + 1, last_minimal + 6
            minimal_d = n + 1

            while left <= right:
                mid = (left + right) // 2
                if mid == 0:
                    mid = 1
                print(f"\t>>>>Testing dimension d={mid}")

                violations = trainer.train(
                    d=mid,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate / np.log2(n),
                    patience=patience,
                )

                print(f"\t<<<<Violations for d={mid}: {violations}")
                self.search_paths[n].append({"dimension": mid, "violations": violations})

                if violations == 0:
                    minimal_d = mid
                    right = mid - 1
                else:
                    left = mid + 1

            self.minimal_dimensions[n] = minimal_d if minimal_d <= n else -1
            last_minimal = minimal_d

            with open("minimal_dem_log.txt", "at") as f:
                f.write(f"minimal dimension @ k={k}&n={n} is {minimal_d}\n")
            print("#" * 10 + " Task Finished" + "#" * 10)
            print(f"minimal dimension @ k={k}&n={n} is {minimal_d}\n")
            print("#" * 30)

        return self.minimal_dimensions

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
