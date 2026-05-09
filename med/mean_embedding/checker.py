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
    ):
        self.m = m
        self.k = k
        self.scoring_function = scoring_function
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self._trainer = Trainer(m, k, scoring_function)

    def check(self, d: int) -> CheckResult:
        violations = self._trainer.train(
            d=d,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            patience=self.patience,
        )

        return CheckResult(
            feasible=(violations == 0),
            details={
                "violations": violations,
                "train_stats": self._trainer.last_train_stats,
            },
        )
