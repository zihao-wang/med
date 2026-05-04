from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
from tqdm import trange

from med.scoring import ScoringFn, compute_scores


class RandomSubsetDataset(Dataset):
    """
    Samples random k-subsets (as index tensors) without precomputing all combinations.

    Each __getitem__ returns a tuple (subset_indices, excluded_indices) where:
    - subset_indices: LongTensor[k]
    - excluded_indices: LongTensor[n-k]

    This avoids O(n choose k) memory by generating on-the-fly.
    """

    def __init__(self, n: int, k: int, num_samples: int):
        assert 1 <= k <= n
        self.n = n
        self.k = k
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate a fresh random subset for each sample
        subset = torch.randperm(self.n)[: self.k]
        # Compute excluded indices efficiently
        mask = torch.ones(self.n, dtype=torch.bool)
        mask[subset] = False
        excluded = torch.arange(self.n)[mask]
        return subset.long(), excluded.long()


@dataclass
class SGDConfig:
    batch_size: int = 256
    num_loader_workers: int = 0
    num_samples: int = 50_000  # total samples to draw from dataset
    learning_rate: float = 1e-2
    patience: int = 500
    max_steps: Optional[int] = None  # override total optimization steps
    negative_sampling: Optional[int] = None  # if set, sample this many negatives per positive


class SGDTrainer:
    def __init__(self, n: int, k: int, scoring_function: ScoringFn, config: Optional[SGDConfig] = None):
        self.n = n
        self.k = k
        self.scoring_function = scoring_function
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"[SGD] Using device: {self.device}")

        self.config = config or SGDConfig()

        # Prebuild tensor of all indices for negative sampling
        self.all_indices = torch.arange(n, device=self.device)
        self.vector_embeddings: Optional[torch.Tensor] = None

        # Estimated total violations for logging (k * (n-k) per subset)
        # We do not enumerate all combinations here.
        self.estimated_total_violations = self.k * (self.n - self.k) * self.config.batch_size

    def _loss_from_batch(self, subset_batch: torch.Tensor, excluded_batch: torch.Tensor) -> Tuple[torch.Tensor, int]:
        assert self.vector_embeddings is not None
        # subset_batch: [B, k]
        # excluded_batch: [B, n-k]
        subset_sums = self.vector_embeddings[subset_batch].mean(dim=1)  # [B, d]

        # Compute scores vs all vectors
        scores_all = compute_scores(subset_sums, self.vector_embeddings, self.scoring_function)  # [B, n]
        pos_scores = torch.gather(scores_all, 1, subset_batch)  # [B, k]

        if self.config.negative_sampling is not None:
            # Sample negatives per batch item for efficiency
            B = subset_batch.size(0)
            num_neg = min(self.config.negative_sampling, excluded_batch.size(1))
            # Choose random negatives among excluded per row
            rand_idx = torch.randint(low=0, high=excluded_batch.size(1), size=(B, num_neg), device=excluded_batch.device)
            sampled_negatives = torch.gather(excluded_batch, 1, rand_idx)  # [B, num_neg]
            neg_scores = torch.gather(scores_all, 1, sampled_negatives)  # [B, num_neg]
        else:
            neg_scores = torch.gather(scores_all, 1, excluded_batch)  # [B, n-k]

        # Broadcast pos vs neg, hinge on differences
        differences = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(2)  # [B, neg, pos]
        loss = torch.relu(-differences).sum((-1, -2)).mean()
        num_violations = (differences < 0).sum().item()
        return loss, num_violations

    def train(self, d: int, num_epochs: int) -> int:
        # Initialize learnable embeddings
        self.vector_embeddings = torch.randn(self.n, d, device=self.device, requires_grad=True)

        optimizer = optim.Adam([self.vector_embeddings], lr=self.config.learning_rate)

        dataset = RandomSubsetDataset(n=self.n, k=self.k, num_samples=self.config.num_samples)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_loader_workers,
            pin_memory=(self.device == "cuda"),
            drop_last=True,
            collate_fn=self._collate_to_device,
        )

        min_violations = math.inf
        epochs_no_improve = 0
        total_steps = self.config.max_steps or (num_epochs * max(1, len(loader)))

        step = 0
        with trange(total_steps, desc=f"\t\t [SGD] n={self.n}, k={self.k}, d={d}") as pbar:
            data_iter = iter(loader)
            while step < total_steps:
                try:
                    subset_batch, excluded_batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    subset_batch, excluded_batch = next(data_iter)

                optimizer.zero_grad()
                loss, violations = self._loss_from_batch(subset_batch, excluded_batch)
                loss.backward()
                optimizer.step()

                pbar.set_postfix(
                    {
                        "loss": float(loss.detach().cpu()),
                        "#vio": violations,
                        "best #vio": min_violations if min_violations < math.inf else -1,
                        "no_improve": epochs_no_improve,
                    }
                )
                pbar.update(1)

                if violations < min_violations:
                    min_violations = violations
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if min_violations == 0:
                    print("[SGD] Early stopping: No violations found.")
                    break
                if epochs_no_improve >= self.config.patience:
                    print(f"[SGD] Early stopping: No improvement for {self.config.patience} steps.")
                    break

                step += 1

        return int(min_violations if min_violations < math.inf else -1)

    def _collate_to_device(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        subsets, excluded = zip(*batch)
        subset_batch = torch.stack(subsets, dim=0).to(self.device)
        excluded_batch = torch.stack(excluded, dim=0).to(self.device)
        return subset_batch, excluded_batch
