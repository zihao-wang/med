"""Run random-token embedding sweeps on packaged LiMIT assets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Literal

import torch

from unlimit.datasets import load_limit
from unlimit.device import resolve_torch_device
from unlimit.dtype import DEFAULT_FLOAT_DTYPE
from unlimit.retrieval.metrics import build_qrels_tensor, retrieval_metrics_from_logits
from unlimit.retrieval.random_embeddings import (
    build_random_token_matrix,
    score_random_embeddings,
)
from unlimit.tokenizers.handmade import HandmadeTokenizer
from unlimit.tokenizers.qwen import QwenSubwordTokenizer
from unlimit.tokenizers.types import LimitTokenizer

DEFAULT_DIMS = [32, 64, 128, 256, 512, 1024, 2048, 4096]
DEFAULT_SPLITS = ["limit-small", "limit"]
DEFAULT_TOKENIZERS = ["handmade", "qwen"]
BASE_SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "unlimit" / "random_embeddings"
DEFAULT_PAPER_TABLE_DIR = REPO_ROOT / "paper" / "table"
DEFAULT_PAPER_FIGURE_DIR = REPO_ROOT / "paper" / "figure"

SplitName = Literal["small", "full"]
ResultRow = dict[str, float | int | str]

RESULT_FIELDNAMES = [
    "dataset",
    "split",
    "tokenizer",
    "dim",
    "seed",
    "recall_at_2",
    "top2_exact_match",
    "recall_at_1",
    "mean_rank",
    "num_queries_eval",
    "num_queries_top2_eval",
]

INT_RESULT_FIELDS = {
    "dim",
    "seed",
    "num_queries_eval",
    "num_queries_top2_eval",
}
FLOAT_RESULT_FIELDS = {
    "recall_at_2",
    "top2_exact_match",
    "recall_at_1",
    "mean_rank",
}


def _normalize_split(split: str) -> SplitName:
    value = split.strip().lower()
    if value in {"small", "limit-small", "limit_small"}:
        return "small"
    if value in {"full", "limit"}:
        return "full"
    raise ValueError(f"unknown split {split!r}; expected limit-small or limit")


def _dataset_name(split: SplitName) -> str:
    return "limit-small" if split == "small" else "limit"


def _normalize_tokenizer(name: str) -> str:
    value = name.strip().lower()
    if value in {"handmade", "vocab", "vocab.txt"}:
        return "handmade"
    if value in {"qwen", "qwen3"}:
        return "qwen"
    raise ValueError(f"unknown tokenizer {name!r}; expected handmade or qwen")


def _build_tokenizer(
    name: str,
    *,
    qwen_model: str,
    qwen_local_files_only: bool,
) -> LimitTokenizer:
    if name == "handmade":
        return HandmadeTokenizer()
    if name == "qwen":
        return QwenSubwordTokenizer(
            qwen_model,
            local_files_only=qwen_local_files_only,
            compact_token_ids=True,
        )
    raise ValueError(f"unknown tokenizer {name!r}")


def _default_output_dir() -> str:
    return str(DEFAULT_OUTPUT_DIR)


@torch.no_grad()
def evaluate_split(
    split: SplitName,
    dims: list[int],
    *,
    tokenizer_name: str,
    qwen_model: str,
    qwen_local_files_only: bool,
    base_seed: int,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    """Evaluate one packaged LiMIT split over token embedding dimensions."""
    dataset = _dataset_name(split)
    tokenizer = _build_tokenizer(
        tokenizer_name,
        qwen_model=qwen_model,
        qwen_local_files_only=qwen_local_files_only,
    )

    print(f"\n[DATA] {dataset}  tokenizer={tokenizer.name}", flush=True)
    corpus, queries, qrels = load_limit(split)
    tokenized_corpus = tokenizer.tokenize_corpus_records(corpus)
    tokenized_queries = tokenizer.tokenize_query_records(queries)

    corpus_ids = [record["_id"] for record in tokenized_corpus]
    query_ids = [record["_id"] for record in tokenized_queries]
    corpus_tokens = [record["token_ids"] for record in tokenized_corpus]
    query_tokens = [record["token_ids"] for record in tokenized_queries]
    labels = build_qrels_tensor(qrels, query_ids, corpus_ids, device)

    print(
        "  "
        f"corpus_docs={len(corpus_tokens)}  queries={len(query_tokens)}  "
        f"qrels={len(qrels)}  token_types={tokenizer.num_token_types()}",
        flush=True,
    )
    print("  dim    recall@2    top2_exact    recall@1    mean_rank", flush=True)

    rows: list[ResultRow] = []
    for dim in dims:
        seed = base_seed + dim
        token_matrix = build_random_token_matrix(
            tokenizer.num_token_types(),
            dim,
            seed,
            device=device,
            dtype=DEFAULT_FLOAT_DTYPE,
        )
        scores = score_random_embeddings(corpus_tokens, query_tokens, token_matrix)
        metrics = retrieval_metrics_from_logits(scores, labels)
        row: ResultRow = {
            "dataset": dataset,
            "split": split,
            "tokenizer": tokenizer.name,
            "dim": dim,
            "seed": seed,
            "recall_at_2": metrics["recall_at_2"],
            "top2_exact_match": metrics["top2_exact_match"],
            "recall_at_1": metrics["recall_at_1"],
            "mean_rank": metrics["mean_rank"],
            "num_queries_eval": metrics["num_queries_eval"],
            "num_queries_top2_eval": metrics["num_queries_top2_eval"],
        }
        rows.append(row)
        print(
            f"  {dim:<6d} "
            f"{metrics['recall_at_2']:<11.4f} "
            f"{metrics['top2_exact_match']:<12.4f} "
            f"{metrics['recall_at_1']:<10.4f} "
            f"{metrics['mean_rank']:.2f}",
            flush=True,
        )
        del scores, token_matrix
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def _row_key(row: ResultRow) -> tuple[str, str, str, int, int]:
    return (
        str(row["dataset"]),
        str(row["split"]),
        str(row["tokenizer"]),
        int(row["dim"]),
        int(row["seed"]),
    )


def _sort_rows(rows: list[ResultRow]) -> list[ResultRow]:
    return sorted(
        rows,
        key=lambda row: (
            str(row["tokenizer"]),
            str(row["dataset"]),
            int(row["dim"]),
        ),
    )


def _parse_result_cell(value: str | None, field: str) -> float | int | str:
    if value is None:
        return ""
    if field in INT_RESULT_FIELDS:
        return int(float(value))
    if field in FLOAT_RESULT_FIELDS:
        return float(value)
    return value


def _load_json_rows(path: Path) -> list[ResultRow]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of result rows in {path}")
    return [dict(row) for row in payload]


def _load_summary_csv(path: Path) -> list[ResultRow]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {field: _parse_result_cell(value, field) for field, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _recover_rows_from_log(path: Path, *, base_seed: int) -> list[ResultRow]:
    data_re = re.compile(r"^\[DATA\]\s+(?P<dataset>\S+)\s+tokenizer=(?P<tokenizer>\S+)")
    stats_re = re.compile(
        r"^\s+corpus_docs=\d+\s+queries=(?P<queries>\d+)\s+qrels=\d+"
    )
    metric_re = re.compile(
        r"^\s+(?P<dim>\d+)\s+"
        r"(?P<recall2>[0-9.]+)\s+"
        r"(?P<top2>[0-9.]+)\s+"
        r"(?P<recall1>[0-9.]+)\s+"
        r"(?P<mean_rank>[0-9.]+)\s*$"
    )

    rows: list[ResultRow] = []
    dataset: str | None = None
    split: SplitName | None = None
    tokenizer: str | None = None
    num_queries = 0

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            data_match = data_re.match(line)
            if data_match:
                dataset = data_match.group("dataset")
                split = _normalize_split(dataset)
                tokenizer = _normalize_tokenizer(data_match.group("tokenizer"))
                num_queries = 0
                continue

            stats_match = stats_re.match(line)
            if stats_match and dataset is not None:
                num_queries = int(stats_match.group("queries"))
                continue

            metric_match = metric_re.match(line)
            if (
                not metric_match
                or dataset is None
                or split is None
                or tokenizer is None
            ):
                continue

            dim = int(metric_match.group("dim"))
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "tokenizer": tokenizer,
                    "dim": dim,
                    "seed": base_seed + dim,
                    "recall_at_2": float(metric_match.group("recall2")),
                    "top2_exact_match": float(metric_match.group("top2")),
                    "recall_at_1": float(metric_match.group("recall1")),
                    "mean_rank": float(metric_match.group("mean_rank")),
                    "num_queries_eval": num_queries,
                    "num_queries_top2_eval": num_queries,
                }
            )
    return rows


def _load_resume_rows(output_dir: str, *, base_seed: int) -> list[ResultRow]:
    output_path = Path(output_dir)
    rows_by_key: dict[tuple[str, str, str, int, int], ResultRow] = {}

    log_path = output_path / "run.log"
    if log_path.exists():
        for row in _recover_rows_from_log(log_path, base_seed=base_seed):
            rows_by_key[_row_key(row)] = row

    csv_path = output_path / "summary.csv"
    if csv_path.exists():
        for row in _load_summary_csv(csv_path):
            rows_by_key[_row_key(row)] = row

    results_path = output_path / "results.json"
    if results_path.exists():
        for row in _load_json_rows(results_path):
            rows_by_key[_row_key(row)] = row

    return _sort_rows(list(rows_by_key.values()))


def _requested_keys(
    dims: list[int],
    splits: list[SplitName],
    tokenizers: list[str],
    *,
    base_seed: int,
) -> set[tuple[str, str, str, int, int]]:
    return {
        (
            _dataset_name(split),
            split,
            tokenizer_name,
            dim,
            base_seed + dim,
        )
        for tokenizer_name in tokenizers
        for split in splits
        for dim in dims
    }


def _filter_rows_for_request(
    rows: list[ResultRow],
    requested_keys: set[tuple[str, str, str, int, int]],
) -> dict[tuple[str, str, str, int, int], ResultRow]:
    rows_by_key: dict[tuple[str, str, str, int, int], ResultRow] = {}
    for row in rows:
        key = _row_key(row)
        if key in requested_keys:
            rows_by_key[key] = row
    return rows_by_key


def _format_float(value: float | int | str, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _latex_escape(value: float | int | str) -> str:
    return str(value).replace("_", "\\_")


def _write_latex_table(
    rows: list[ResultRow],
    path: Path,
) -> None:
    sorted_rows = sorted(
        rows,
        key=lambda row: (str(row["dataset"]), str(row["tokenizer"]), int(row["dim"])),
    )
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & Tokenizer & $d$ & Recall@2 & Top-2 EM & Mean rank \\\\",
        "\\midrule",
    ]
    for row in sorted_rows:
        lines.append(
            f"{_latex_escape(row['dataset'])} & "
            f"{_latex_escape(row['tokenizer'])} & "
            f"{row['dim']} & "
            f"{_format_float(row['recall_at_2'])} & "
            f"{_format_float(row['top2_exact_match'])} & "
            f"{_format_float(row['mean_rank'], digits=2)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _set_plot_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def _write_pdf_figure(
    rows: list[ResultRow],
    path: Path,
) -> None:
    _set_plot_style()
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    groups = sorted({(str(row["dataset"]), str(row["tokenizer"])) for row in rows})
    markers = ["o", "s", "^", "D", "v", "P"]

    for idx, (dataset, tokenizer) in enumerate(groups):
        points = sorted(
            [
                row
                for row in rows
                if row["dataset"] == dataset and row["tokenizer"] == tokenizer
            ],
            key=lambda row: int(row["dim"]),
        )
        label = f"{dataset} / {tokenizer}"
        marker = markers[idx % len(markers)]
        dims = [int(point["dim"]) for point in points]
        axes[0].plot(
            dims,
            [float(point["top2_exact_match"]) for point in points],
            marker=marker,
            linewidth=1.7,
            markersize=5,
            label=label,
        )
        axes[1].plot(
            dims,
            [float(point["mean_rank"]) for point in points],
            marker=marker,
            linewidth=1.7,
            markersize=5,
            label=label,
        )

    axes[0].set_xlabel("Embedding dimension $d$")
    axes[0].set_ylabel("Top-2 exact match")
    axes[0].set_title("Top-2 exact match vs dimension")
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.25)

    axes[1].set_xlabel("Embedding dimension $d$")
    axes[1].set_ylabel("Mean rank")
    axes[1].set_title("Mean rank vs dimension")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.25)

    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_outputs(
    rows: list[ResultRow],
    output_dir: str,
    config: dict[str, object],
    *,
    paper_table_dir: str | None,
    paper_figure_dir: str | None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_path = output_path / "config.json"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    results_path = output_path / "results.json"
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    csv_path = output_path / "summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(_sort_rows(rows))

    table_path = output_path / "limit_retrieval_table.tex"
    figure_path = output_path / "limit_retrieval.pdf"
    sorted_rows = _sort_rows(rows)
    _write_latex_table(sorted_rows, table_path)
    _write_pdf_figure(sorted_rows, figure_path)

    if paper_table_dir is not None:
        paper_table_path = Path(paper_table_dir)
        paper_table_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(table_path, paper_table_path / table_path.name)
    if paper_figure_dir is not None:
        paper_figure_path = Path(paper_figure_dir)
        paper_figure_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(figure_path, paper_figure_path / figure_path.name)

    print(f"\n[DONE] Results saved to {output_path}", flush=True)
    print(f"[DONE] Table saved to {table_path}", flush=True)
    print(f"[DONE] Figure saved to {figure_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate random-token embeddings on LiMIT and LiMIT-small"
    )
    parser.add_argument(
        "--mode",
        choices=["run", "plot"],
        default="run",
        help="'run' evaluates missing rows; 'plot' only writes outputs from saved rows",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=DEFAULT_DIMS,
        help="Token embedding dimensions to evaluate",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help="Datasets to evaluate: limit-small and/or limit",
    )
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        default=DEFAULT_TOKENIZERS,
        help="Tokenizers to evaluate: handmade and/or qwen",
    )
    parser.add_argument(
        "--qwen-model",
        default="Qwen/Qwen3-0.6B",
        help="Hugging Face tokenizer name for --tokenizers qwen",
    )
    parser.add_argument(
        "--qwen-local-files-only",
        action="store_true",
        help="Load the Qwen tokenizer from the local Hugging Face cache only",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=BASE_SEED,
        help="Seed base; effective seed is base_seed + dim",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for evaluation",
    )
    parser.add_argument(
        "--output-dir",
        default=_default_output_dir(),
        help="Deterministic directory for config.json, results.json, summary.csv, table, and figure",
    )
    parser.add_argument(
        "--paper-table-dir",
        default=str(DEFAULT_PAPER_TABLE_DIR),
        help="Directory receiving the final LaTeX table",
    )
    parser.add_argument(
        "--paper-figure-dir",
        default=str(DEFAULT_PAPER_FIGURE_DIR),
        help="Directory receiving the final PDF figure",
    )
    parser.add_argument(
        "--no-paper-copy",
        action="store_true",
        help="Only write table and figure under --output-dir",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse completed rows from results.json, summary.csv, or run.log under "
            "--output-dir and only evaluate missing combinations"
        ),
    )
    return parser.parse_args()


def _print_missing_rows(
    missing_keys: set[tuple[str, str, str, int, int]],
) -> None:
    if not missing_keys:
        return
    missing = sorted(
        (dataset, tokenizer, dim)
        for dataset, _split, tokenizer, dim, _seed in missing_keys
    )
    print(f"[WARN] Writing partial outputs; missing {len(missing)} rows:", flush=True)
    for dataset, tokenizer, dim in missing:
        print(f"  - {dataset} tokenizer={tokenizer} dim={dim}", flush=True)


def main() -> None:
    args = parse_args()
    dims = [int(dim) for dim in args.dims]
    if any(dim <= 0 for dim in dims):
        raise ValueError("all dimensions must be positive")

    splits = [_normalize_split(split) for split in args.splits]
    tokenizers = [_normalize_tokenizer(name) for name in args.tokenizers]
    device = resolve_torch_device(args.device)
    config = {
        "dims": dims,
        "splits": [_dataset_name(split) for split in splits],
        "tokenizers": tokenizers,
        "qwen_model": args.qwen_model,
        "qwen_compact_token_ids": True,
        "qwen_local_files_only": bool(args.qwen_local_files_only),
        "base_seed": int(args.base_seed),
        "device": str(device),
        "mode": args.mode,
        "scoring": "inner_product(sum_random_token_embeddings)",
    }

    print("[INFO] Random-token LiMIT sweep", flush=True)
    print(f"  dims      = {' '.join(str(dim) for dim in dims)}", flush=True)
    print(f"  datasets  = {' '.join(config['splits'])}", flush=True)
    print(f"  tokenizers= {' '.join(tokenizers)}", flush=True)
    if "qwen" in tokenizers:
        print(f"  qwen_model= {args.qwen_model}", flush=True)
    print(f"  base_seed = {args.base_seed}", flush=True)
    print(f"  device    = {device}", flush=True)

    requested_keys = _requested_keys(
        dims,
        splits,
        tokenizers,
        base_seed=int(args.base_seed),
    )
    rows_by_key: dict[tuple[str, str, str, int, int], ResultRow] = {}
    if args.resume or args.mode == "plot":
        rows_by_key = _filter_rows_for_request(
            _load_resume_rows(args.output_dir, base_seed=int(args.base_seed)),
            requested_keys,
        )
        if rows_by_key:
            print(
                f"[RESUME] Recovered {len(rows_by_key)} completed rows from "
                f"{args.output_dir}",
                flush=True,
            )

    paper_table_dir = None if args.no_paper_copy else args.paper_table_dir
    paper_figure_dir = None if args.no_paper_copy else args.paper_figure_dir
    if args.mode == "plot":
        if not rows_by_key:
            raise RuntimeError(f"No saved rows found under {args.output_dir}")
        _print_missing_rows(requested_keys - set(rows_by_key))
        _write_outputs(
            _sort_rows(list(rows_by_key.values())),
            args.output_dir,
            config,
            paper_table_dir=paper_table_dir,
            paper_figure_dir=paper_figure_dir,
        )
        return

    for tokenizer_name in tokenizers:
        for split in splits:
            missing_dims = [
                dim
                for dim in dims
                if (
                    _dataset_name(split),
                    split,
                    tokenizer_name,
                    dim,
                    int(args.base_seed) + dim,
                )
                not in rows_by_key
            ]
            if not missing_dims:
                print(
                    f"[SKIP] {_dataset_name(split)} tokenizer={tokenizer_name}: "
                    "all requested dims are complete",
                    flush=True,
                )
                continue

            new_rows = evaluate_split(
                split,
                missing_dims,
                tokenizer_name=tokenizer_name,
                qwen_model=args.qwen_model,
                qwen_local_files_only=bool(args.qwen_local_files_only),
                base_seed=int(args.base_seed),
                device=device,
            )
            for row in new_rows:
                rows_by_key[_row_key(row)] = row

    missing_keys = requested_keys - set(rows_by_key)
    if missing_keys:
        raise RuntimeError(f"Sweep did not produce {len(missing_keys)} requested rows")

    rows = _sort_rows(list(rows_by_key.values()))
    _write_outputs(
        rows,
        args.output_dir,
        config,
        paper_table_dir=paper_table_dir,
        paper_figure_dir=paper_figure_dir,
    )


if __name__ == "__main__":
    main()
