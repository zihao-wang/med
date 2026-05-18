"""Cyclic-polytope overfit tables for packaged LIMIT assets."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Literal

import numpy as np

from med.cyclic_polytope.construct import (
    construct_query_for_subset,
    generate_cyclic_polytope_configuration,
)
from unlimit.datasets import load_limit

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "unlimit" / "cyclic_overfit"
DEFAULT_PAPER_TABLE_DIR = REPO_ROOT / "paper" / "table"

SplitName = Literal["small", "full"]

SUMMARY_CSV_NAME = "limit_cyclic_overfit_summary.csv"
LIMIT_SMALL_DOC_CSV_NAME = "limit_small_cyclic_document_embeddings.csv"
LIMIT_SMALL_QUERY_CSV_NAME = "limit_small_cyclic_query_embeddings.csv"
SUMMARY_TEX_NAME = "limit_cyclic_overfit_summary.tex"
LIMIT_SMALL_DOC_TEX_NAME = "limit_small_cyclic_document_embeddings_table.tex"
LIMIT_SMALL_QUERY_TEX_NAME = "limit_small_cyclic_query_embeddings_table.tex"


def _dataset_name(split: SplitName) -> str:
    return "limit-small" if split == "small" else "limit"


def _display_dataset_name(dataset: object) -> str:
    value = str(dataset)
    if value == "limit-small":
        return "LIMIT-small"
    if value == "limit":
        return "LIMIT"
    return value


def _t_values(num_docs: int) -> np.ndarray:
    return np.arange(1, num_docs + 1, dtype=np.float64)


def _cube_t_values(num_docs: int) -> np.ndarray:
    if num_docs <= 0:
        raise ValueError("num_docs must be positive")
    if num_docs == 1:
        return np.array([0.0], dtype=np.float64)
    return np.linspace(-1.0, 1.0, num_docs, dtype=np.float64)


def _positive_indices_by_query(
    qrels: list[dict],
    query_ids: list[str],
    corpus_ids: list[str],
) -> dict[str, list[int]]:
    corpus_index = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}
    positives = {query_id: [] for query_id in query_ids}
    for row in qrels:
        if int(row["score"]) <= 0:
            continue
        query_id = row["query_id"]
        corpus_id = row["corpus_id"]
        if query_id in positives and corpus_id in corpus_index:
            positives[query_id].append(corpus_index[corpus_id])
    return positives


def _evaluate_dimension(
    points: np.ndarray,
    t_values: np.ndarray,
    query_ids: list[str],
    positives_by_query: dict[str, list[int]],
    dim: int,
) -> tuple[float, float]:
    recall_sum = 0.0
    exact_hits = 0
    num_eval = 0
    for query_id in query_ids:
        positives = positives_by_query[query_id]
        if not positives:
            continue
        query = construct_query_for_subset(t_values, positives, dim)
        scores = points @ query
        k = len(positives)
        top_indices = np.argpartition(scores, -k)[-k:]
        top_set = set(int(idx) for idx in top_indices)
        pos_set = set(positives)
        recall_sum += len(top_set & pos_set) / float(k)
        exact_hits += int(top_set == pos_set)
        num_eval += 1
    if num_eval == 0:
        return 0.0, 0.0
    return recall_sum / num_eval, exact_hits / num_eval


def evaluate_split(split: SplitName, *, max_dim: int = 4) -> dict[str, int | float | str]:
    corpus, queries, qrels = load_limit(split)
    corpus_ids = [record["_id"] for record in corpus]
    query_ids = [record["_id"] for record in queries]
    positives_by_query = _positive_indices_by_query(qrels, query_ids, corpus_ids)
    t_values = _t_values(len(corpus))

    minimal_dim = -1
    recall_at_min = 0.0
    exact_at_min = 0.0
    for dim in range(1, max_dim + 1):
        points = generate_cyclic_polytope_configuration(len(corpus), dim, t_values)
        recall, exact = _evaluate_dimension(
            points,
            t_values,
            query_ids,
            positives_by_query,
            dim,
        )
        if exact == 1.0:
            minimal_dim = dim
            recall_at_min = recall
            exact_at_min = exact
            break

    if minimal_dim < 0:
        points = generate_cyclic_polytope_configuration(len(corpus), max_dim, t_values)
        recall_at_min, exact_at_min = _evaluate_dimension(
            points,
            t_values,
            query_ids,
            positives_by_query,
            max_dim,
        )

    positive_counts = sorted({len(v) for v in positives_by_query.values()})
    return {
        "dataset": _dataset_name(split),
        "corpus_docs": len(corpus),
        "queries": len(queries),
        "qrels": len(qrels),
        "positive_docs_per_query": "|".join(str(value) for value in positive_counts),
        "max_dim_checked": max_dim,
        "minimal_overfit_dim": minimal_dim,
        "recall_at_2": recall_at_min,
        "exact_top2_match": exact_at_min,
    }


def limit_small_document_embedding_rows() -> list[dict[str, int | float | str]]:
    corpus, _queries, _qrels = load_limit("small")
    t_values = _cube_t_values(len(corpus))
    points = generate_cyclic_polytope_configuration(len(corpus), 4, t_values)
    rows: list[dict[str, int | float | str]] = []
    for idx, (record, point) in enumerate(zip(corpus, points, strict=True)):
        rows.append(
            {
                "doc_id": idx,
                "profile_id": record["_id"],
                "t": t_values[idx],
                "x1": point[0],
                "x2": point[1],
                "x3": point[2],
                "x4": point[3],
            }
        )
    return rows


def limit_small_query_embedding_rows(
    *,
    num_queries: int = 50,
) -> list[dict[str, int | float | str]]:
    corpus, queries, qrels = load_limit("small")
    corpus_ids = [record["_id"] for record in corpus]
    query_ids = [record["_id"] for record in queries]
    positives_by_query = _positive_indices_by_query(qrels, query_ids, corpus_ids)
    t_values = _cube_t_values(len(corpus))

    rows: list[dict[str, int | float | str]] = []
    if num_queries <= 0:
        raise ValueError("num_queries must be positive")
    selected_indices = np.linspace(
        0,
        len(queries) - 1,
        num=min(num_queries, len(queries)),
        dtype=int,
    )
    for query_index in selected_indices:
        record = queries[int(query_index)]
        positives = positives_by_query[record["_id"]]
        query = construct_query_for_subset(t_values, positives, 4)
        max_abs = float(np.max(np.abs(query)))
        if max_abs > 0.0:
            query = query / max_abs
        rows.append(
            {
                "query_index": int(query_index),
                "query_id": record["_id"],
                "query_text": record["text"],
                "positive_doc_ids": "|".join(str(idx) for idx in positives),
                "positive_profile_ids": "|".join(corpus_ids[idx] for idx in positives),
                "q1": query[0],
                "q2": query[1],
                "q3": query[2],
                "q4": query[3],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _latex_escape(value: object) -> str:
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
    return "".join(replacements.get(char, char) for char in str(value))


def _format_number(value: object) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.4f}"


def _positive_indices_for_tex(value: object) -> str:
    return _latex_escape(str(value).replace("|", ", "))


def _write_tex(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_tex(
    path: Path,
    rows: list[dict[str, int | float | str]],
) -> None:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Dataset & Docs & Queries & Qrels & Minimal $d$ & Recall@2 \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _latex_escape(_display_dataset_name(row["dataset"])),
                    _format_number(row["corpus_docs"]),
                    _format_number(row["queries"]),
                    _format_number(row["qrels"]),
                    _format_number(row["minimal_overfit_dim"]),
                    _format_number(row["recall_at_2"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tex(path, lines)


def _write_limit_small_document_tex(
    path: Path,
    rows: list[dict[str, int | float | str]],
) -> None:
    lines = [
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{@{}r l r r r r@{}}",
        (
            r"\caption{LIMIT-small cyclic-polytope document embeddings in "
            r"$[-1,1]^4$.}\label{tab:limit-small-cyclic-doc-embeddings}\\"
        ),
        r"\toprule",
        r"Doc id & Profile id & $x_1$ & $x_2$ & $x_3$ & $x_4$ \\",
        r"\midrule",
        r"\endfirsthead",
        (
            r"\caption[]{LIMIT-small cyclic-polytope document embeddings "
            r"(continued).}\\"
        ),
        r"\toprule",
        r"Doc id & Profile id & $x_1$ & $x_2$ & $x_3$ & $x_4$ \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _format_number(row["doc_id"]),
                    _latex_escape(row["profile_id"]),
                    _format_number(row["x1"]),
                    _format_number(row["x2"]),
                    _format_number(row["x3"]),
                    _format_number(row["x4"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"}"])
    _write_tex(path, lines)


def _write_limit_small_query_tex(
    path: Path,
    rows: list[dict[str, int | float | str]],
) -> None:
    lines = [
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{@{}r l l r r r r@{}}",
        (
            r"\caption{Selected LIMIT-small squared-polynomial query embeddings "
            r"normalized into $[-1,1]^4$.}\label{tab:limit-small-cyclic-query-embeddings}\\"
        ),
        r"\toprule",
        r"Query & Query id & Positive doc ids & $q_1$ & $q_2$ & $q_3$ & $q_4$ \\",
        r"\midrule",
        r"\endfirsthead",
        (
            r"\caption[]{Selected LIMIT-small squared-polynomial query embeddings "
            r"(continued).}\\"
        ),
        r"\toprule",
        r"Query & Query id & Positive doc ids & $q_1$ & $q_2$ & $q_3$ & $q_4$ \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _format_number(row["query_index"]),
                    _latex_escape(row["query_id"]),
                    _positive_indices_for_tex(row["positive_doc_ids"]),
                    _format_number(row["q1"]),
                    _format_number(row["q2"]),
                    _format_number(row["q3"]),
                    _format_number(row["q4"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"}"])
    _write_tex(path, lines)


def write_artifacts(
    output_dir: Path,
    *,
    paper_table_dir: Path | None,
    max_dim: int,
    num_query_rows: int,
) -> None:
    summary_rows = [
        evaluate_split("small", max_dim=max_dim),
        evaluate_split("full", max_dim=max_dim),
    ]
    doc_rows = limit_small_document_embedding_rows()
    query_rows = limit_small_query_embedding_rows(num_queries=num_query_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / SUMMARY_CSV_NAME,
        output_dir / LIMIT_SMALL_DOC_CSV_NAME,
        output_dir / LIMIT_SMALL_QUERY_CSV_NAME,
    ]
    tex_paths = [
        output_dir / SUMMARY_TEX_NAME,
        output_dir / LIMIT_SMALL_DOC_TEX_NAME,
        output_dir / LIMIT_SMALL_QUERY_TEX_NAME,
    ]
    _write_csv(paths[0], summary_rows)
    _write_csv(paths[1], doc_rows)
    _write_csv(paths[2], query_rows)
    _write_summary_tex(tex_paths[0], summary_rows)
    _write_limit_small_document_tex(tex_paths[1], doc_rows)
    _write_limit_small_query_tex(tex_paths[2], query_rows)

    if paper_table_dir is not None:
        paper_table_dir.mkdir(parents=True, exist_ok=True)
        for path in paths + tex_paths:
            shutil.copy2(path, paper_table_dir / path.name)

    print(f"[DONE] Wrote cyclic overfit CSVs and LaTeX tables to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write cyclic-polytope LIMIT overfit appendix CSVs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated CSV files",
    )
    parser.add_argument(
        "--paper-table-dir",
        type=Path,
        default=DEFAULT_PAPER_TABLE_DIR,
        help="Directory receiving paper CSV copies",
    )
    parser.add_argument(
        "--no-paper-copy",
        action="store_true",
        help="Only write CSV files under --output-dir",
    )
    parser.add_argument("--max-dim", type=int, default=4)
    parser.add_argument("--num-query-rows", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_artifacts(
        args.output_dir,
        paper_table_dir=None if args.no_paper_copy else args.paper_table_dir,
        max_dim=int(args.max_dim),
        num_query_rows=int(args.num_query_rows),
    )


if __name__ == "__main__":
    main()
