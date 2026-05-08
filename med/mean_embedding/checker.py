from __future__ import annotations

from typing import Optional

import numpy as np

from ..checker import CheckResult, FeasibilityChecker
from ..scoring import ScoringFn
from .trainer_gd import Trainer
from .trainer_sgd import SGDConfig, SGDTrainer


class MeanEmbeddingChecker(FeasibilityChecker):
    """Feasibility checker using learned centroid embeddings (GD or SGD)."""

    def __init__(
        self,
        m: int,
        k: int,
        scoring_function: ScoringFn,
        trainer_type: str = "gd",
        sgd_config: Optional[SGDConfig] = None,
        num_epochs: int = 1000,
        learning_rate: float = 1,
        patience: int = 1000,
    ):
        self.m = m
        self.k = k
        self.scoring_function = scoring_function
        self.trainer_type = trainer_type
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.patience = patience

        if trainer_type == "gd":
            self._trainer = Trainer(m, k, scoring_function)
        elif trainer_type == "sgd":
            self._trainer = SGDTrainer(m, k, scoring_function, config=sgd_config or SGDConfig())
        else:
            raise ValueError(f"Unknown trainer_type: {trainer_type}")

    def check(self, d: int) -> CheckResult:
        if self.trainer_type == "gd":
            lr = self.learning_rate / np.log2(self.m)
            violations = self._trainer.train(
                d=d,
                num_epochs=self.num_epochs,
                learning_rate=lr,
                patience=self.patience,
            )
        else:
            violations = self._trainer.train(d=d, num_epochs=self.num_epochs)

        return CheckResult(
            feasible=(violations == 0),
            details={"violations": violations},
        )
