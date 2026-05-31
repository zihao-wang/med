from __future__ import annotations

from typing import Callable

from ..checker import FeasibilityChecker
from ..experiment import Experiment as _SharedExperiment
from ..scoring import ScoringFn
from .checker import MeanEmbeddingChecker


def _make_checker_factory(
    scoring_function: ScoringFn,
    num_epochs: int,
    learning_rate: float,
    patience: int,
    margin: float,
    constraint_chunk_size: int,
    show_progress: bool,
) -> Callable[[int, int], FeasibilityChecker]:
    def factory(m: int, k: int) -> FeasibilityChecker:
        return MeanEmbeddingChecker(
            m=m, k=k,
            scoring_function=scoring_function,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            patience=patience,
            margin=margin,
            constraint_chunk_size=constraint_chunk_size,
            show_progress=show_progress,
        )
    return factory


class Experiment(_SharedExperiment):
    """Orchestrates MED binary search for centroid embedding."""

    def __init__(
        self,
        scoring_function: ScoringFn,
        num_epochs: int = 1000,
        learning_rate: float = 2.0,
        patience: int = 1000,
        margin: float = 1e-6,
        constraint_chunk_size: int = 8192,
        show_progress: bool = True,
    ):
        super().__init__(
            checker_factory=_make_checker_factory(
                scoring_function=scoring_function,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                patience=patience,
                margin=margin,
                constraint_chunk_size=constraint_chunk_size,
                show_progress=show_progress,
            ),
        )
