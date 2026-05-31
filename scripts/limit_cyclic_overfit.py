"""Generate cyclic-polytope exact-overfit tables for packaged LIMIT assets."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np

from med.cyclic_polytope.construct import (
    construct_query_for_subset,
    generate_cyclic_polytope_configuration,
)
from med.unlimit.datasets import load_limit

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "unlimit" / "cyclic_overfit"
DEFAULT_PAPER_TABLE_DIR = None
DEFAULT_MAX_DIM = 4
DEFAULT_QUERY_ROWS = 50

SplitName = Literal["small", "full"]


def _normalize_split(split: str) -> SplitName:
    value = split.strip().lower()
    if value in {"small", "limit-small", "limit_small"}:
        return "small"
    if value in {"full", "limit"}:
        return "full"
    raise ValueError(f"unknown split {split!r}; expected limit-small or limit")


def _dataset_name(split: SplitName) -> str:
    return "limit-small" if split == "small" else "limit"


def _display_dataset_name(dataset: str) -> str:
    return "LIMIT-small" if dataset == "limit-small" else "LIMIT"


def _moment_parameters(num_docs: int) -> np.ndarray:
    if num_docs <= 1:
        return np.zeros(num_docs, dtype=float)
    return np.linspace(-1.0, 1.0, num_docs, dtype=float)


def _positive_doc_indices(
    qrels: list[dict],
    query_ids: list[str],
    corpus_ids: list[str],
) -> dict[str, list[int]]:
    doc_index = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}
    positives: dict[str, list[int]] = defaultdict(list)
    query_id_set = set(query_ids)
    for row in qrels:
        if int(row["score"]) <= 0 or row["query_id"] not in query_id_set:
            continue
        positives[row["query_id"]].append(doc_index[row["corpus_id"]])
    return {query_id: sorted(indices) for query_id, indices in positives.items()}


def evaluate_cyclic_overfit(
    split: SplitName,
    dim: int,
) -> dict[str, float | int | str]:
    corpus, queries, qrels = load_limit(split)
    corpus_ids = [record["_id"] for record in corpus]
    query_ids = [record["_id"] for record in queries]
    positives_by_query = _positive_doc_indices(qrels, query_ids, corpus_ids)
    t_values = np.arange(len(corpus), dtype=float)
    points = generate_cyclic_polytope_configuration(len(corpus), dim, t_values)

    recall_at_2_sum = 0.0
    exact_top2 = 0
    evaluated = 0
    for query in queries:
        positives = positives_by_query[query["_id"]]
        query_vec = construct_query_for_subset(t_values, positives, dim)
        scores = points @ query_vec
        top = np.argsort(scores)[-len(positives) :]
        top_set = set(int(idx) for idx in top)
        positive_set = set(positives)
        recall_at_2_sum += len(top_set & positive_set) / len(positive_set)
        exact_top2 += int(top_set == positive_set)
        evaluated += 1

    return {
        "dataset": _dataset_name(split),
        "corpus_docs": len(corpus),
        "queries": len(queries),
        "qrels": len(qrels),
        "positive_docs_per_query": len(next(iter(positives_by_query.values()))),
        "dim": dim,
        "recall_at_2": recall_at_2_sum / evaluated if evaluated else 0.0,
        "exact_top2_match": exact_top2 / evaluated if evaluated else 0.0,
    }


def summary_row_for_split(
    split: SplitName,
    *,
    max_dim: int,
) -> dict[str, float | int | str]:
    best: dict[str, float | int | str] | None = None
    minimal_dim: int | None = None
    for dim in range(1, max_dim + 1):
        row = evaluate_cyclic_overfit(split, dim)
        best = row
        if row["recall_at_2"] == 1.0 and row["exact_top2_match"] == 1.0:
            minimal_dim = dim
            break
    if best is None:
        raise RuntimeError("no dimensions were evaluated")
    return {
        "dataset": best["dataset"],
        "corpus_docs": best["corpus_docs"],
        "queries": best["queries"],
        "qrels": best["qrels"],
        "positive_docs_per_query": best["positive_docs_per_query"],
        "max_dim_checked": max_dim,
        "minimal_overfit_dim": minimal_dim if minimal_dim is not None else "",
        "recall_at_2": best["recall_at_2"],
        "exact_top2_match": best["exact_top2_match"],
    }


def _format_float(value: float | int | str, digits: int = 4) -> str:
    if isinstance(value, float):
        if abs(abs(value) - 1.0) < 10 ** -digits:
            return str(int(round(value)))
        return f"{value:.{digits}f}"
    return str(value)


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _write_summary(rows: list[dict[str, float | int | str]], output_dir: Path) -> None:
    fieldnames = [
        "dataset",
        "corpus_docs",
        "queries",
        "qrels",
        "positive_docs_per_query",
        "max_dim_checked",
        "minimal_overfit_dim",
        "recall_at_2",
        "exact_top2_match",
    ]
    csv_path = output_dir / "limit_cyclic_overfit_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Dataset & Docs & Queries & Qrels & Minimal $d$ & Recall@2 \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_display_dataset_name(str(row['dataset']))} & "
            f"{row['corpus_docs']} & "
            f"{row['queries']} & "
            f"{row['qrels']} & "
            f"{row['minimal_overfit_dim']} & "
            f"{_format_float(row['recall_at_2'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (output_dir / "limit_cyclic_overfit_summary.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _small_document_rows() -> list[dict[str, float | int | str]]:
    corpus, _queries, _qrels = load_limit("small")
    t_values = _moment_parameters(len(corpus))
    points = generate_cyclic_polytope_configuration(len(corpus), 4, t_values)
    rows: list[dict[str, float | int | str]] = []
    for idx, record in enumerate(corpus):
        rows.append(
            {
                "doc_id": idx,
                "profile_id": record["_id"],
                "t": t_values[idx],
                "x1": points[idx, 0],
                "x2": points[idx, 1],
                "x3": points[idx, 2],
                "x4": points[idx, 3],
            }
        )
    return rows


def _write_small_document_table(output_dir: Path) -> None:
    rows = _small_document_rows()
    csv_path = output_dir / "limit_small_cyclic_document_embeddings.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{@{}r l r r r r@{}}",
        r"\caption{LIMIT-small cyclic-polytope document embeddings in $[-1,1]^4$.}\label{tab:limit-small-cyclic-doc-embeddings}\\",
        r"\toprule",
        r"Doc id & Profile id & $x_1$ & $x_2$ & $x_3$ & $x_4$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{LIMIT-small cyclic-polytope document embeddings (continued).}\\",
        r"\toprule",
        r"Doc id & Profile id & $x_1$ & $x_2$ & $x_3$ & $x_4$ \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        lines.append(
            f"{row['doc_id']} & "
            f"{_latex_escape(str(row['profile_id']))} & "
            f"{_format_float(row['x1'])} & "
            f"{_format_float(row['x2'])} & "
            f"{_format_float(row['x3'])} & "
            f"{_format_float(row['x4'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"}", ""])
    (output_dir / "limit_small_cyclic_document_embeddings_table.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _small_query_rows(num_rows: int) -> list[dict[str, float | int | str]]:
    corpus, queries, qrels = load_limit("small")
    corpus_ids = [record["_id"] for record in corpus]
    query_ids = [record["_id"] for record in queries]
    positives_by_query = _positive_doc_indices(qrels, query_ids, corpus_ids)
    t_values = _moment_parameters(len(corpus))
    selected = np.linspace(0, len(queries) - 1, num_rows, dtype=int)
    rows: list[dict[str, float | int | str]] = []
    for query_index in selected:
        query = queries[int(query_index)]
        positives = positives_by_query[query["_id"]]
        query_vec = construct_query_for_subset(t_values, positives, 4)
        scale = float(np.max(np.abs(query_vec))) or 1.0
        query_vec = query_vec / scale
        rows.append(
            {
                "query_index": int(query_index),
                "query_id": query["_id"],
                "query_text": query["text"],
                "positive_doc_ids": "|".join(str(idx) for idx in positives),
                "positive_profile_ids": "|".join(corpus_ids[idx] for idx in positives),
                "q1": query_vec[0],
                "q2": query_vec[1],
                "q3": query_vec[2],
                "q4": query_vec[3],
            }
        )
    return rows


def _write_small_query_table(output_dir: Path, *, num_rows: int) -> None:
    rows = _small_query_rows(num_rows)
    csv_path = output_dir / "limit_small_cyclic_query_embeddings.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{@{}r l l r r r r@{}}",
        r"\caption{Selected LIMIT-small squared-polynomial query embeddings normalized into $[-1,1]^4$.}\label{tab:limit-small-cyclic-query-embeddings}\\",
        r"\toprule",
        r"Query & Query id & Positive doc ids & $q_1$ & $q_2$ & $q_3$ & $q_4$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{Selected LIMIT-small squared-polynomial query embeddings (continued).}\\",
        r"\toprule",
        r"Query & Query id & Positive doc ids & $q_1$ & $q_2$ & $q_3$ & $q_4$ \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        positive_doc_ids = str(row["positive_doc_ids"]).replace("|", ", ")
        lines.append(
            f"{row['query_index']} & "
            f"{_latex_escape(str(row['query_id']))} & "
            f"{_latex_escape(positive_doc_ids)} & "
            f"{_format_float(row['q1'])} & "
            f"{_format_float(row['q2'])} & "
            f"{_format_float(row['q3'])} & "
            f"{_format_float(row['q4'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"}", ""])
    (output_dir / "limit_small_cyclic_query_embeddings_table.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, float | int | str]],
    *,
    query_rows: int,
    paper_table_dir: Path | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary(rows, output_dir)
    _write_small_document_table(output_dir)
    _write_small_query_table(output_dir, num_rows=query_rows)

    if paper_table_dir is not None:
        paper_table_dir.mkdir(parents=True, exist_ok=True)
        for path in output_dir.glob("limit_*.csv"):
            shutil.copy2(path, paper_table_dir / path.name)
        for path in output_dir.glob("limit_*.tex"):
            shutil.copy2(path, paper_table_dir / path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cyclic-polytope LIMIT exact-overfit paper tables."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["limit-small", "limit"],
        help="Datasets to summarize: limit-small and/or limit.",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=DEFAULT_MAX_DIM,
        help="Largest cyclic-polytope dimension to check.",
    )
    parser.add_argument(
        "--query-rows",
        type=int,
        default=DEFAULT_QUERY_ROWS,
        help="Number of evenly spaced LIMIT-small query rows to export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated CSV and TeX files.",
    )
    parser.add_argument(
        "--paper-table-dir",
        type=Path,
        default=DEFAULT_PAPER_TABLE_DIR,
        help="Optional directory receiving copied LaTeX tables.",
    )
    parser.add_argument(
        "--no-paper-copy",
        action="store_true",
        help="Only write generated files under --output-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_dim < 1:
        raise ValueError("--max-dim must be positive")
    if args.query_rows < 1:
        raise ValueError("--query-rows must be positive")

    splits = [_normalize_split(split) for split in args.splits]
    rows = [
        summary_row_for_split(split, max_dim=args.max_dim)
        for split in splits
    ]
    paper_table_dir = None if args.no_paper_copy else args.paper_table_dir
    write_outputs(
        args.output_dir,
        rows,
        query_rows=args.query_rows,
        paper_table_dir=paper_table_dir,
    )
    print(f"[DONE] cyclic-overfit tables saved to {args.output_dir}")
    if paper_table_dir is not None:
        print(f"[DONE] paper tables copied to {paper_table_dir}")


if __name__ == "__main__":
    main()
