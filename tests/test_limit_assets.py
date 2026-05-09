"""Tests for packaged LiMIT dataset assets."""

from unlimit.datasets.limit import load_limit, qrels_positive_distribution


def test_load_limit_small_from_packaged_assets():
    corpus, queries, qrels = load_limit("small")

    assert len(corpus) == 46
    assert len(queries) == 1000
    assert len(qrels) == 2000
    assert qrels_positive_distribution(qrels) == {2: 1000}


def test_load_limit_full_from_packaged_assets():
    corpus, queries, qrels = load_limit("full")

    assert len(corpus) == 50000
    assert len(queries) == 1000
    assert len(qrels) == 2000
    assert qrels_positive_distribution(qrels) == {2: 1000}
