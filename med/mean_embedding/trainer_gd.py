import itertools
import time
from typing import Tuple

import torch
from torch import optim
from tqdm import trange

from med._device import resolve_device
from med.scoring import ScoringFn, compute_scores


class Trainer:
    def __init__(
        self,
        n: int,
        k: int,
        scoring_function: ScoringFn,
    ):
        self.n = n
        self.k = k
        self.scoring_function = scoring_function
        self.device = resolve_device()
        print(f"[GD] Using device: {self.device}")

        self.all_combinations = list(itertools.combinations(range(self.n), self.k))
        self.subset_indices_tensor = torch.tensor(
            [list(s) for s in self.all_combinations], device=self.device
        )
        self.subset_excluded_tensor = torch.tensor(
            [
                [i for i in range(self.n) if i not in subset_indices]
                for subset_indices in self.all_combinations
            ],
            device=self.device,
        )
        self.total_violations = len(self.all_combinations) * self.k * (self.n - self.k)

        self.vector_embeddings: torch.Tensor | None = None
        self.last_train_stats: dict[str, float | int | None] = {}

    def calculate_loss(self) -> Tuple[torch.Tensor, int]:
        assert self.vector_embeddings is not None, (
            "Call train() first to initialize embeddings"
        )

        subset_sums = self.vector_embeddings[self.subset_indices_tensor].mean(dim=1)

        dot_products_all = compute_scores(
            subset_sums, self.vector_embeddings, self.scoring_function
        )

        dot_products_xi = torch.gather(dot_products_all, 1, self.subset_indices_tensor)
        dot_products_xj = torch.gather(dot_products_all, 1, self.subset_excluded_tensor)

        differences = dot_products_xi.unsqueeze(1) - dot_products_xj.unsqueeze(2)
        total_loss = torch.relu(-differences).sum((-1, -2)).mean()
        num_violations = (differences < 0).sum().item()
        return total_loss, num_violations

    def train(
        self,
        d: int,
        num_epochs: int,
        learning_rate: float = 2.0,
        patience: int = 1000,
        show_progress: bool = True,
    ) -> int:
        self.vector_embeddings = torch.randn(
            self.n, d, device=self.device, requires_grad=True
        )

        max_lr = learning_rate
        optimizer = optim.Adam([self.vector_embeddings], lr=max_lr)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=num_epochs,
            pct_start=0.3,
        )

        min_violations = self.total_violations
        step_no_improve = 0
        best_step = 0
        witness_step: int | None = None
        final_violations = min_violations
        steps_run = 0
        t0 = time.perf_counter()

        with trange(
            num_epochs,
            desc=f"d={d}, m={self.n}, k={self.k}",
            dynamic_ncols=True,
            unit="step",
            disable=not show_progress,
        ) as step_iterator:
            for step in step_iterator:
                optimizer.zero_grad()
                loss, violations = self.calculate_loss()
                loss.backward()
                optimizer.step()
                scheduler.step()
                steps_run = step + 1
                final_violations = violations

                if violations < min_violations:
                    min_violations = violations
                    best_step = steps_run
                    step_no_improve = 0
                else:
                    step_no_improve += 1

                step_iterator.set_postfix(
                    {
                        "#vio": violations,
                        "step_no_improve": step_no_improve,
                    }
                )

                if violations == 0:
                    witness_step = steps_run
                    print("[GD] Early stopping: No violations found.")
                    break
                if step_no_improve >= patience:
                    print(
                        f"[GD] Early stopping: No improvement in violations for {patience} steps."
                    )
                    break

        self.last_train_stats = {
            "steps_run": steps_run,
            "best_step": best_step,
            "witness_step": witness_step,
            "min_violations": min_violations,
            "final_violations": final_violations,
            "elapsed": round(time.perf_counter() - t0, 3),
            "lr": learning_rate,
            "max_lr": max_lr,
        }
        return min_violations
