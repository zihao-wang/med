from unlimit import cyclic_overfit


def test_limit_small_cyclic_overfit_reaches_four_dimensions():
    row = cyclic_overfit.evaluate_split("small", max_dim=4)

    assert row["dataset"] == "limit-small"
    assert row["corpus_docs"] == 46
    assert row["minimal_overfit_dim"] == 4
    assert row["recall_at_2"] == 1.0
    assert row["exact_top2_match"] == 1.0


def test_limit_small_embedding_exports_have_expected_sizes():
    doc_rows = cyclic_overfit.limit_small_document_embedding_rows()
    query_rows = cyclic_overfit.limit_small_query_embedding_rows(num_queries=50)

    assert len(doc_rows) == 46
    assert len(query_rows) == 50
    assert {"doc_id", "profile_id", "x1", "x2", "x3", "x4"} <= set(doc_rows[0])
    assert {"query_id", "positive_doc_ids", "positive_profile_ids", "q1", "q2"} <= set(
        query_rows[0]
    )
    for row in doc_rows:
        assert all(-1.0 <= float(row[f"x{i}"]) <= 1.0 for i in range(1, 5))
    for row in query_rows:
        assert all(-1.0 <= float(row[f"q{i}"]) <= 1.0 for i in range(1, 5))


def test_limit_small_exported_cube_embeddings_retrieve_selected_queries():
    doc_rows = cyclic_overfit.limit_small_document_embedding_rows()
    query_rows = cyclic_overfit.limit_small_query_embedding_rows(num_queries=50)
    docs = [
        [float(row["x1"]), float(row["x2"]), float(row["x3"]), float(row["x4"])]
        for row in doc_rows
    ]

    for row in query_rows:
        query = [float(row["q1"]), float(row["q2"]), float(row["q3"]), float(row["q4"])]
        scores = [sum(a * b for a, b in zip(doc, query, strict=True)) for doc in docs]
        top2 = set(sorted(range(len(scores)), key=lambda idx: scores[idx])[-2:])
        positives = {int(idx) for idx in str(row["positive_doc_ids"]).split("|")}
        assert top2 == positives


def test_cyclic_overfit_writes_latex_tables(tmp_path):
    cyclic_overfit.write_artifacts(
        tmp_path,
        paper_table_dir=None,
        max_dim=4,
        num_query_rows=50,
    )

    doc_table = tmp_path / cyclic_overfit.LIMIT_SMALL_DOC_TEX_NAME
    query_table = tmp_path / cyclic_overfit.LIMIT_SMALL_QUERY_TEX_NAME

    assert doc_table.exists()
    assert query_table.exists()
    assert r"\begin{longtable}" in doc_table.read_text(encoding="utf-8")
    assert r"query\_0" in query_table.read_text(encoding="utf-8")
