from argparse import Namespace

from scripts import paper_upper_bounds as paper_pipeline


def _summary_rows():
    return [
        {
            "m": 10,
            "k": 2,
            "top_k_queries": 55,
            "cyclic_polytope_d": 4,
            "cyclic_queries_checked": 45,
            "cyclic_total_queries": 45,
            "cyclic_checked_fraction": 1.0,
            "cyclic_time": 0.0,
            "mean_embedding_d": 1,
            "mean_embedding_violations": 0,
            "mean_embedding_time": 7.1,
        },
        {
            "m": 20,
            "k": 2,
            "top_k_queries": 210,
            "cyclic_polytope_d": 4,
            "cyclic_queries_checked": 190,
            "cyclic_total_queries": 190,
            "cyclic_checked_fraction": 1.0,
            "cyclic_time": 0.0,
            "mean_embedding_d": 7,
            "mean_embedding_violations": 0,
            "mean_embedding_time": 9.4,
        },
    ]


def test_load_payload_accepts_summary_csv(tmp_path):
    csv_path = tmp_path / paper_pipeline.SUMMARY_CSV_NAME
    paper_pipeline._write_table_csv(csv_path, _summary_rows())

    payload = paper_pipeline.load_payload(csv_path)

    assert payload["k"] == 2
    assert payload["m_values"] == [10, 20]
    assert payload["config"]["resume_format"] == "summary_csv"
    assert payload["summary_rows"][0]["m"] == 10
    assert payload["summary_rows"][0]["mean_embedding_d"] == 1
    assert payload["summary_rows"][0]["mean_embedding_time"] == 7.1


def test_load_plot_payload_falls_back_to_summary_csv(tmp_path):
    json_path = tmp_path / paper_pipeline.RESULTS_JSON_NAME
    csv_path = tmp_path / paper_pipeline.SUMMARY_CSV_NAME
    json_path.write_text("{bad json")
    paper_pipeline._write_table_csv(csv_path, _summary_rows())

    args = Namespace(results_file=None, output_root=tmp_path)
    payload, run_dir, loaded_from = paper_pipeline._load_plot_payload(args)

    assert run_dir == tmp_path
    assert loaded_from == csv_path
    assert payload["summary_rows"][1]["mean_embedding_d"] == 7


def test_generate_figures_exports_paper_dimension_fit(tmp_path):
    output_dir = tmp_path / "results"
    paper_figure_dir = tmp_path / "paper" / "figures"

    paper_pipeline.generate_figures(
        {"summary_rows": _summary_rows()},
        output_dir,
        paper_figure_dir,
    )

    assert (output_dir / paper_pipeline.PAPER_DIMENSION_FIT_NAME).exists()
    assert (paper_figure_dir / paper_pipeline.PAPER_DIMENSION_FIT_NAME).exists()
