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


def test_zero_embedding_violates_positive_margin():
    trainer = Trainer(5, 2, "inner_product", margin=1e-6)
    trainer.vector_embeddings = torch.zeros(
        5,
        1,
        device=trainer.device,
        requires_grad=True,
    )

    loss, violations = trainer.calculate_loss()

    assert loss.item() > 0
    assert violations == trainer.total_violations


def test_negative_margin_is_rejected():
    try:
        Trainer(5, 2, "inner_product", margin=-1e-6)
    except ValueError as exc:
        assert "margin" in str(exc)
    else:
        raise AssertionError("negative margin should be rejected")
