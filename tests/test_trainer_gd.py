import torch

from med.mean_embedding.checker import MeanEmbeddingChecker
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
    assert trainer.last_train_stats["max_lr"] == 0.05


def test_checker_uses_raw_learning_rate():
    checker = MeanEmbeddingChecker(
        m=16,
        k=2,
        scoring_function="inner_product",
        learning_rate=1.0,
    )

    assert checker.learning_rate == 1.0
    assert checker.m == 16


def test_checker_default_learning_rate_absorbs_best_scale():
    checker = MeanEmbeddingChecker(
        m=16,
        k=2,
        scoring_function="inner_product",
    )

    assert checker.learning_rate == 2.0
