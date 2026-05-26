from math import log2

import pytest
import torch

from med.mean_embedding.experiment import _make_check_factory
from med.mean_embedding.trainer_gd import Trainer


def test_current_train_smoke():
    torch.manual_seed(4)
    trainer = Trainer(8, 2, "inner_product")

    violations = trainer.train(
        d=4,
        num_epochs=2,
        learning_rate=0.05,
        patience=2,
        show_progress=False,
    )

    assert 0 <= violations <= trainer.total_violations
    assert trainer.last_train_stats["steps_run"] > 0
    assert trainer.last_train_stats["max_lr"] == pytest.approx(0.05 / log2(8))


def test_check_factory_captures_learning_rate():
    check_factory = _make_check_factory(
        scoring_function="inner_product",
        num_epochs=1,
        learning_rate=1.0,
        patience=1,
    )

    check_dimension = check_factory(16, 2)

    assert check_dimension.__closure__ is not None


def test_check_factory_builds_callable():
    check_factory = _make_check_factory(
        scoring_function="inner_product",
        num_epochs=1,
        learning_rate=2.0,
        patience=1,
    )

    assert callable(check_factory(16, 2))
