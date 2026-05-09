import torch

from med.mean_embedding.checker import MeanEmbeddingChecker
from med.mean_embedding.lr_scaling import scaled_learning_rate
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
    assert trainer.last_train_stats["max_lr"] == 0.1


def test_checker_uses_constant_lr_by_default():
    checker = MeanEmbeddingChecker(
        m=16,
        k=2,
        scoring_function="inner_product",
        learning_rate=1.0,
    )

    assert checker.lr_scaling == "constant"
    assert (
        scaled_learning_rate(checker.learning_rate, checker.m, checker.lr_scaling)
        == 1.0
    )


def test_lr_scaling_schemes():
    assert scaled_learning_rate(1.0, 16, "log2") == 0.25
    assert scaled_learning_rate(1.0, 16, "sqrt_log2") == 0.5
    assert scaled_learning_rate(1.0, 16, "constant") == 1.0
    assert scaled_learning_rate(1.0, 16, "fourth_root_m") == 0.5
    assert scaled_learning_rate(1.0, 16, "sqrt_m") == 0.25
    assert scaled_learning_rate(1.0, 16, "linear_m") == 0.0625
