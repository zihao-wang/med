from __future__ import annotations

from typing import Callable, Optional

from ..checker import FeasibilityChecker
from ..experiment import Experiment as _SharedExperiment
from ..scoring import ScoringFn
from .checker import MeanEmbeddingChecker
from .trainer_sgd import SGDConfig


def _make_checker_factory(
    scoring_function: ScoringFn,
    trainer_type: str,
    sgd_config: Optional[SGDConfig],
    num_epochs: int,
    learning_rate: float,
    patience: int,
) -> Callable[[int, int], FeasibilityChecker]:
    def factory(m: int, k: int) -> FeasibilityChecker:
        return MeanEmbeddingChecker(
            m=m, k=k,
            scoring_function=scoring_function,
            trainer_type=trainer_type,
            sgd_config=sgd_config,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            patience=patience,
        )
    return factory


class Experiment(_SharedExperiment):
    """Orchestrates MED binary search for centroid embedding."""

    def __init__(
        self,
        scoring_function: ScoringFn,
        trainer_type: str = "gd",
        sgd_config: Optional[SGDConfig] = None,
        num_epochs: int = 1000,
        learning_rate: float = 1,
        patience: int = 1000,
    ):
        super().__init__(
            checker_factory=_make_checker_factory(
                scoring_function=scoring_function,
                trainer_type=trainer_type,
                sgd_config=sgd_config,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                patience=patience,
            ),
        )
