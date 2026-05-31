from __future__ import annotations

from ..checker import CheckResult, FeasibilityChecker
from ..scoring import ScoringFn
from .trainer_gd import Trainer


class MeanEmbeddingChecker(FeasibilityChecker):
    """Feasibility checker using learned centroid embeddings with full-batch GD."""

    def __init__(
        self,
        m: int,
        k: int,
        scoring_function: ScoringFn,
        num_epochs: int = 1000,
        learning_rate: float = 2.0,
        patience: int = 1000,
        margin: float = 1e-6,
        constraint_chunk_size: int = 8192,
        show_progress: bool = True,
    ):
        self.m = m
        self.k = k
        self.scoring_function = scoring_function
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.margin = margin
        self.constraint_chunk_size = constraint_chunk_size
        self.show_progress = show_progress
        self._trainer = Trainer(
            m,
            k,
            scoring_function,
            margin=margin,
            constraint_chunk_size=constraint_chunk_size,
        )

    def check(self, d: int) -> CheckResult:
        violations = self._trainer.train(
            d=d,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            patience=self.patience,
            show_progress=self.show_progress,
        )

        return CheckResult(
            feasible=(violations == 0),
            details={
                "violations": violations,
                "train_stats": self._trainer.last_train_stats,
            },
        )
