import itertools
from typing import Literal, Tuple

import torch
from torch import optim
from tqdm import trange

ScoringFn = Literal["inner_product", "l2", "cosine", "l1"]


class Trainer:
    def __init__(self, n: int, k: int, scoring_function: ScoringFn):
        self.n = n
        self.k = k
        self.scoring_function = scoring_function
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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

    def _compute_scores(self, subset_sums: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        if self.scoring_function == "inner_product":
            return torch.matmul(subset_sums, vectors.T)
        if self.scoring_function == "l2":
            return torch.norm(subset_sums.unsqueeze(1) - vectors, p=2, dim=-1)
        if self.scoring_function == "cosine":
            return torch.cosine_similarity(subset_sums.unsqueeze(1), vectors)
        if self.scoring_function == "l1":
            return torch.norm(subset_sums.unsqueeze(1) - vectors, p=1, dim=-1)
        raise NotImplementedError(f"Unknown scoring_function: {self.scoring_function}")

    def calculate_loss(self) -> Tuple[torch.Tensor, int]:
        assert self.vector_embeddings is not None, "Call train() first to initialize embeddings"

        subset_sums = self.vector_embeddings[self.subset_indices_tensor].mean(dim=1)

        dot_products_all = self._compute_scores(subset_sums, self.vector_embeddings)

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
        learning_rate: float = 0.1,
        patience: int = 20,
    ) -> int:
        self.vector_embeddings = torch.randn(self.n, d, device=self.device, requires_grad=True)

        optimizer = optim.Adam([self.vector_embeddings], lr=learning_rate)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=learning_rate, total_steps=num_epochs, pct_start=0.0
        )

        min_violations = self.total_violations
        epochs_no_improve = 0

        with trange(num_epochs, desc=f"\t\t [GD] n={self.n}, k={self.k}, d={d}") as epoch_iterator:
            for _ in epoch_iterator:
                optimizer.zero_grad()
                loss, violations = self.calculate_loss()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_iterator.set_postfix(
                    {
                        "n": self.n,
                        "k": self.k,
                        "loss": loss.item(),
                        "#vio rate": violations / self.total_violations,
                        "min #vio": min_violations,
                        "epochs_no_improve": epochs_no_improve,
                    }
                )

                if violations < min_violations:
                    min_violations = violations
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if violations == 0:
                    print("[GD] Early stopping: No violations found.")
                    break
                if epochs_no_improve >= patience:
                    print(f"[GD] Early stopping: No improvement in violations for {patience} epochs.")
                    break

        return min_violations
