from scripts import limit_random_embeddings as random_embedding_sweep


def test_recover_rows_from_run_log(tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "[DATA] limit-small  tokenizer=handmade",
                "  corpus_docs=46  queries=1000  qrels=2000  token_types=1849",
                "  dim    recall@2    top2_exact    recall@1    mean_rank",
                "  32     0.1715      0.0120       0.1990     6.66",
                "  64     0.2670      0.0480       0.3370     4.65",
                "",
                "[DATA] limit  tokenizer=qwen",
                "  corpus_docs=50000  queries=1000  qrels=2000  token_types=11545",
                "  dim    recall@2    top2_exact    recall@1    mean_rank",
                "  1024   0.0570      0.0060       0.0750     1472.07",
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
    assert rows[2]["mean_rank"] == 1472.07


def test_load_resume_rows_can_recover_recall_only_latex_table(tmp_path):
    table_path = tmp_path / "limit_retrieval_table.tex"
    row_end = chr(92) * 2
    table_path.write_text(
        "\n".join(
            [
                "Dataset & Tokenizer & dim & Recall " + row_end,
                "LIMIT-small & qwen & 32 & 0.1120 " + row_end,
                "LIMIT & handmade & 2048 & 0.9915 " + row_end,
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = random_embedding_sweep._load_resume_rows(str(tmp_path), base_seed=42)
    requested = random_embedding_sweep._requested_keys(
        [32],
        ["small"],
        ["qwen"],
        base_seed=42,
    )

    assert len(rows) == 2
    assert rows[1]["dataset"] == "limit-small"
    assert rows[1]["seed"] == 74
    assert rows[1]["recall_at_2"] == 0.112
    assert not random_embedding_sweep._filter_rows_for_request(
        rows,
        requested,
        require_complete=True,
    )
    assert random_embedding_sweep._filter_rows_for_request(
        rows,
        requested,
        require_complete=True,
        required_fields=random_embedding_sweep.PLOT_REQUIRED_FIELDS,
    )


def test_load_resume_rows_deduplicates_by_key(tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "[DATA] limit-small  tokenizer=handmade",
                "  corpus_docs=46  queries=1000  qrels=2000  token_types=1849",
                "  32     0.1715      0.0120       0.1990     6.66",
            ]
        ),
        encoding="utf-8",
    )
    csv_path = tmp_path / "summary.csv"
    csv_path.write_text(
        "\n".join(
            [
                ",".join(random_embedding_sweep.RESULT_FIELDNAMES),
                "limit-small,small,handmade,32,74,0.2,0.1,0.3,5.5,1000,1000",
            ]
        ),
        encoding="utf-8",
    )

    rows = random_embedding_sweep._load_resume_rows(str(tmp_path), base_seed=42)

    assert len(rows) == 1
    assert rows[0]["recall_at_2"] == 0.2
    assert rows[0]["mean_rank"] == 5.5


def test_load_resume_rows_keeps_complete_log_metrics_over_legacy_json(tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "[DATA] limit-small  tokenizer=handmade",
                "  corpus_docs=46  queries=1000  qrels=2000  token_types=1849",
                "  32     0.1715      0.0120       0.1990     6.66",
            ]
        ),
        encoding="utf-8",
    )
    results_path = tmp_path / "results.json"
    results_path.write_text(
        "["
        '{"dataset":"limit-small","split":"small","tokenizer":"handmade",'
        '"dim":32,"seed":74,"recall_at_2":0.1715,"num_queries_eval":1000}'
        "]",
        encoding="utf-8",
    )

    rows = random_embedding_sweep._load_resume_rows(str(tmp_path), base_seed=42)

    assert len(rows) == 1
    assert rows[0]["top2_exact_match"] == 0.012
    assert rows[0]["recall_at_1"] == 0.199
    assert rows[0]["mean_rank"] == 6.66


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


def test_filter_rows_for_requested_keys_can_require_complete_rows():
    requested = random_embedding_sweep._requested_keys(
        [32],
        ["small"],
        ["handmade"],
        base_seed=42,
    )
    incomplete = {
        "dataset": "limit-small",
        "split": "small",
        "tokenizer": "handmade",
        "dim": 32,
        "seed": 74,
        "recall_at_2": 0.2,
        "num_queries_eval": 1000,
    }
    complete = dict(incomplete)
    complete.update(
        {
            "top2_exact_match": 0.1,
            "recall_at_1": 0.3,
            "mean_rank": 5.5,
            "num_queries_top2_eval": 1000,
        }
    )

    assert not random_embedding_sweep._filter_rows_for_request(
        [incomplete],
        requested,
        require_complete=True,
    )
    assert random_embedding_sweep._filter_rows_for_request(
        [complete],
        requested,
        require_complete=True,
    )

def test_latex_table_reports_recall_at_2_only(tmp_path):
    rows = [
        {
            'dataset': 'limit-small',
            'split': 'small',
            'tokenizer': 'handmade',
            'dim': 32,
            'seed': 74,
            'recall_at_2': 0.2,
            'top2_exact_match': 0.1,
            'recall_at_1': 0.3,
            'mean_rank': 5.5,
            'num_queries_eval': 1000,
            'num_queries_top2_eval': 1000,
        }
    ]
    path = tmp_path / 'table.tex'

    random_embedding_sweep._write_latex_table(rows, path)
    text = path.read_text(encoding='utf-8')

    assert 'Recall@2' in text
    assert 'Mean rank' not in text
    assert 'Top-2 EM' not in text
    assert '0.2000' in text
    assert '5.50' not in text


def test_promptriever_crossing_table_is_exported_from_rows(tmp_path):
    rows = [
        {
            "dataset": "limit",
            "split": "full",
            "tokenizer": "vanilla",
            "dim": 256,
            "seed": 298,
            "recall_at_2": 0.02,
        },
        {
            "dataset": "limit",
            "split": "full",
            "tokenizer": "vanilla",
            "dim": 512,
            "seed": 554,
            "recall_at_2": 0.063,
        },
        {
            "dataset": "limit",
            "split": "full",
            "tokenizer": "vanilla",
            "dim": 4096,
            "seed": 4138,
            "recall_at_2": 0.706,
        },
    ]
    path = tmp_path / random_embedding_sweep.PROMPTRIEVER_CROSSING_TABLE_NAME

    random_embedding_sweep._write_promptriever_crossing_table(rows, path)

    text = path.read_text(encoding="utf-8")
    assert "LIMIT & 0.030 & vanilla & 512 & 0.7060" in text
    assert r"\label{tab:limit-promptriever-crossing}" in text
