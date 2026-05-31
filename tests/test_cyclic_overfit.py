from scripts import limit_cyclic_overfit as cyclic_overfit


def test_limit_small_cyclic_overfit_is_exact_in_four_dimensions():
    row = cyclic_overfit.evaluate_cyclic_overfit("small", 4)

    assert row["dataset"] == "limit-small"
    assert row["corpus_docs"] == 46
    assert row["recall_at_2"] == 1.0
    assert row["exact_top2_match"] == 1.0


def test_limit_small_query_rows_are_evenly_spaced_and_rescaled():
    rows = cyclic_overfit._small_query_rows(3)

    assert [row["query_index"] for row in rows] == [0, 499, 999]
    for row in rows:
        assert max(abs(float(row[f"q{i}"])) for i in range(1, 5)) == 1.0


def test_summary_row_finds_dimension_four_for_limit_small():
    row = cyclic_overfit.summary_row_for_split("small", max_dim=4)

    assert row["minimal_overfit_dim"] == 4
    assert row["recall_at_2"] == 1.0
