from unlimit import random_embedding_sweep


def test_recover_rows_from_run_log(tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "[DATA] limit-small  tokenizer=handmade",
                "  corpus_docs=46  queries=1000  qrels=2000  token_types=1849",
                "  dim    recall@2",
                "  32     0.1715",
                "  64     0.2670",
                "",
                "[DATA] limit  tokenizer=qwen",
                "  corpus_docs=50000  queries=1000  qrels=2000  token_types=11545",
                "  dim    recall@2",
                "  1024   0.0570",
            ]
        ),
        encoding="utf-8",
    )

    rows = random_embedding_sweep._recover_rows_from_log(log_path, base_seed=42)

    assert len(rows) == 3
    assert rows[0]["dataset"] == "limit-small"
    assert rows[0]["split"] == "small"
    assert rows[0]["tokenizer"] == "handmade"
    assert rows[0]["dim"] == 32
    assert rows[0]["seed"] == 74
    assert rows[0]["num_queries_eval"] == 1000
    assert rows[2]["dataset"] == "limit"
    assert rows[2]["split"] == "full"
    assert rows[2]["tokenizer"] == "qwen"
    assert rows[2]["recall_at_2"] == 0.057


def test_load_resume_rows_deduplicates_by_key(tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "[DATA] limit-small  tokenizer=handmade",
                "  corpus_docs=46  queries=1000  qrels=2000  token_types=1849",
                "  32     0.1715",
            ]
        ),
        encoding="utf-8",
    )
    csv_path = tmp_path / "summary.csv"
    csv_path.write_text(
        "\n".join(
            [
                ",".join(random_embedding_sweep.RESULT_FIELDNAMES),
                "limit-small,small,handmade,32,74,0.2,1000",
            ]
        ),
        encoding="utf-8",
    )

    rows = random_embedding_sweep._load_resume_rows(str(tmp_path), base_seed=42)

    assert len(rows) == 1
    assert rows[0]["recall_at_2"] == 0.2
    assert "mean_rank" not in rows[0]


def test_recover_rows_from_legacy_run_log_ignores_exact_match(tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "[DATA] limit-small  tokenizer=handmade",
                "  corpus_docs=46  queries=1000  qrels=2000  token_types=1849",
                "  dim    recall@2    top2_exact    recall@1    mean_rank",
                "  32     0.1715      0.0120       0.1990     6.66",
            ]
        ),
        encoding="utf-8",
    )

    rows = random_embedding_sweep._recover_rows_from_log(log_path, base_seed=42)

    assert len(rows) == 1
    assert rows[0]["recall_at_2"] == 0.1715
    assert "recall_at_1" not in rows[0]
    assert "top2_exact_match" not in rows[0]


def test_filter_rows_for_requested_keys():
    rows = [
        {
            "dataset": "limit-small",
            "split": "small",
            "tokenizer": "handmade",
            "dim": 32,
            "seed": 74,
        },
        {
            "dataset": "limit",
            "split": "full",
            "tokenizer": "qwen",
            "dim": 4096,
            "seed": 4138,
        },
    ]
    requested = random_embedding_sweep._requested_keys(
        [32],
        ["small"],
        ["handmade"],
        base_seed=42,
    )

    filtered = random_embedding_sweep._filter_rows_for_request(rows, requested)

    assert len(filtered) == 1
    assert next(iter(filtered.values()))["dataset"] == "limit-small"


def test_requested_keys_use_base_seed_for_phrase_cover_free():
    requested = random_embedding_sweep._requested_keys(
        [32, 64],
        ["small"],
        ["phrase-cover-free"],
        base_seed=42,
    )

    assert ("limit-small", "small", "phrase-cover-free", 32, 42) in requested
    assert ("limit-small", "small", "phrase-cover-free", 64, 42) in requested


def test_legacy_cover_free_log_rows_are_not_reused_as_phrase_cover_free(tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "[DATA] limit-small  tokenizer=cover-free",
                "  corpus_docs=46  queries=1000  qrels=2000  construction=cover-free",
                "  dim    recall@2",
                "  32     0.8975",
            ]
        ),
        encoding="utf-8",
    )

    requested = random_embedding_sweep._requested_keys(
        [32],
        ["small"],
        ["phrase-cover-free"],
        base_seed=42,
    )
    rows = random_embedding_sweep._recover_rows_from_log(log_path, base_seed=42)

    assert rows[0]["tokenizer"] == "cover-free"
    assert not random_embedding_sweep._filter_rows_for_request(rows, requested)
