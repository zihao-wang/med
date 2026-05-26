from __future__ import annotations

from ..experiment import Experiment as _SharedExperiment
from ..scoring import ScoringFn
from .trainer_gd import Trainer


def _make_check_factory(
    scoring_function: ScoringFn,
    num_epochs: int,
    learning_rate: float,
    patience: int,
):
    def factory(m: int, k: int):
        trainer = Trainer(m, k, scoring_function)

        def check_dimension(d: int) -> bool:
            violations = trainer.train(
                d=d,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                patience=patience,
            )
            return violations == 0

        return check_dimension

    return factory


class Experiment(_SharedExperiment):
    """Orchestrates MED binary search for centroid embedding."""

    def __init__(
        self,
        scoring_function: ScoringFn,
        num_epochs: int = 1000,
        learning_rate: float = 2.0,
        patience: int = 1000,
    ):
        super().__init__(
            check_factory=_make_check_factory(
                scoring_function=scoring_function,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                patience=patience,
            ),
        )
