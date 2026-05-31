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
        margin: float = 1e-6,
        constraint_chunk_size: int = 8192,
    ):
        if margin < 0:
            raise ValueError("margin must be non-negative")
        if constraint_chunk_size <= 0:
            raise ValueError("constraint_chunk_size must be positive")
        self.n = n
        self.k = k
        self.scoring_function = scoring_function
        self.margin = margin
        self.constraint_chunk_size = constraint_chunk_size
        self.device = resolve_device()
        print(f"[GD] Using device: {self.device}")

        self.all_combinations: dict[int, list[tuple[int, ...]]] = {}
        self.subset_groups: list[tuple[int, torch.Tensor]] = []
        self.total_subsets = 0
        self.total_violations = 0
        for subset_size in range(1, self.k + 1):
            if subset_size >= self.n:
                continue
            combinations = list(itertools.combinations(range(self.n), subset_size))
            if not combinations:
                continue
            self.all_combinations[subset_size] = combinations
            subset_indices_tensor = torch.tensor(
                [list(s) for s in combinations],
                device=self.device,
                dtype=torch.long,
            )
            self.subset_groups.append((subset_size, subset_indices_tensor))
            self.total_subsets += len(combinations)
            self.total_violations += (
                len(combinations) * subset_size * (self.n - subset_size)
            )

        self.vector_embeddings: torch.Tensor | None = None
        self.last_train_stats: dict[str, object] = {}

    def _iter_subset_chunks(self):
        for subset_size, subset_indices_tensor in self.subset_groups:
            for start in range(
                0,
                subset_indices_tensor.shape[0],
                self.constraint_chunk_size,
            ):
                end = min(
                    start + self.constraint_chunk_size,
                    subset_indices_tensor.shape[0],
                )
                yield subset_size, subset_indices_tensor[start:end]

    def _chunk_loss_and_violations(
        self,
        subset_size: int,
        subset_indices_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        assert self.vector_embeddings is not None, (
            "Call train() first to initialize embeddings"
        )

        subset_sums = self.vector_embeddings[subset_indices_tensor].mean(dim=1)
        scores = compute_scores(
            subset_sums,
            self.vector_embeddings,
            self.scoring_function,
        )
        selected_scores = torch.gather(scores, 1, subset_indices_tensor)
        margin_deficits = (
            self.margin - selected_scores.unsqueeze(2) + scores.unsqueeze(1)
        )
        selected_positions = subset_indices_tensor.unsqueeze(1).expand(
            -1,
            subset_size,
            subset_size,
        )
        margin_deficits.scatter_(2, selected_positions, -torch.inf)
        num_violations = int((margin_deficits > 0).sum().item())
        return torch.relu(margin_deficits).sum(), num_violations

    def calculate_loss(self) -> Tuple[torch.Tensor, int]:
        assert self.vector_embeddings is not None, (
            "Call train() first to initialize embeddings"
        )

        total_loss = self.vector_embeddings.sum() * 0.0
        num_violations = 0
        for subset_size, subset_indices_tensor in self._iter_subset_chunks():
            chunk_loss, chunk_violations = self._chunk_loss_and_violations(
                subset_size,
                subset_indices_tensor,
            )
            total_loss = total_loss + chunk_loss
            num_violations += chunk_violations
        total_loss = total_loss / max(1, self.total_subsets)
        return total_loss, num_violations

    def _backward_full_batch_loss(self) -> tuple[float, int]:
        total_loss_value = 0.0
        num_violations = 0
        loss_scale = float(max(1, self.total_subsets))
        for subset_size, subset_indices_tensor in self._iter_subset_chunks():
            chunk_loss, chunk_violations = self._chunk_loss_and_violations(
                subset_size,
                subset_indices_tensor,
            )
            (chunk_loss / loss_scale).backward()
            total_loss_value += float(chunk_loss.detach().item()) / loss_scale
            num_violations += chunk_violations
        return total_loss_value, num_violations

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
                _loss, violations = self._backward_full_batch_loss()
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
            "margin": self.margin,
            "subset_sizes": list(self.all_combinations),
            "total_subsets": self.total_subsets,
            "total_constraints": self.total_violations,
            "constraint_chunk_size": self.constraint_chunk_size,
        }
        return min_violations
