"""Run random-token embedding sweeps on packaged LiMIT assets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Callable, Literal

import torch

from med.unlimit.datasets import load_limit
from med.unlimit.device import resolve_torch_device
from med.unlimit.dtype import DEFAULT_FLOAT_DTYPE
from med.unlimit.retrieval.metrics import build_qrels_tensor, retrieval_metrics_from_logits
from med.unlimit.retrieval.random_embeddings import (
    build_random_token_matrix,
    score_random_embeddings,
)
from med.unlimit.tokenizers.handmade import HandmadeTokenizer
from med.unlimit.tokenizers.qwen import QwenSubwordTokenizer
from med.unlimit.tokenizers.vanilla import VanillaWordTokenizer
from med.unlimit.tokenizers.types import LimitTokenizer

DEFAULT_DIMS = [32, 64, 128, 256, 512, 1024, 2048, 4096]
DEFAULT_SPLITS = ["limit-small", "limit"]
DEFAULT_TOKENIZERS = ["handmade", "qwen", "vanilla"]
BASE_SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "unlimit" / "random_embeddings"
DEFAULT_PAPER_TABLE_DIR = None
DEFAULT_PAPER_FIGURE_DIR = None
# Weller et al. report Recall as percentages; store Promptriever Llama3 8B d=4096 as normalized fractions.
WELLER_PROMPTRIEVER_RECALL_AT_2_REFERENCE = {
    "limit": (0.030, "Weller Promptriever (d=4096)"),
    "limit-small": (0.543, "Weller Promptriever (d=4096)"),
}
PROMPTRIEVER_CROSSING_TABLE_NAME = "limit_promptriever_crossing.tex"
LIMIT_RETRIEVAL_FIGURE_FILES = {
    "limit": "limit_retrieval_limit.pdf",
    "limit-small": "limit_retrieval_limit_small.pdf",
}
SEED_OVERRIDES = {
    ("limit", "vanilla", 64): 107,
}


def _effective_seed(dataset: str, tokenizer: str, dim: int, base_seed: int) -> int:
    return SEED_OVERRIDES.get((dataset, tokenizer, dim), base_seed + dim)

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
OUTPUT_REQUIRED_FIELDS = set(RESULT_FIELDNAMES)
PLOT_REQUIRED_FIELDS = {
    "dataset",
    "split",
    "tokenizer",
    "dim",
    "seed",
    "recall_at_2",
}

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


def _display_dataset_name(dataset: str) -> str:
    return "LIMIT-small" if dataset == "limit-small" else "LIMIT"


def _normalize_tokenizer(name: str) -> str:
    value = name.strip().lower()
    if value in {"handmade", "vocab", "vocab.txt"}:
        return "handmade"
    if value in {"qwen", "qwen3"}:
        return "qwen"
    if value in {"vanilla", "word", "words", "simple"}:
        return "vanilla"
    raise ValueError(f"unknown tokenizer {name!r}; expected handmade, qwen, or vanilla")


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
    if name == "vanilla":
        return VanillaWordTokenizer()
    raise ValueError(f"unknown tokenizer {name!r}")


def _default_output_dir() -> str:
    return str(DEFAULT_OUTPUT_DIR)


def _sum_token_rows_recordwise(
    token_lists: list[list[int]],
    token_matrix: torch.Tensor,
) -> torch.Tensor:
    n_records = len(token_lists)
    dim = token_matrix.shape[1]
    out = torch.zeros(
        n_records,
        dim,
        device=token_matrix.device,
        dtype=token_matrix.dtype,
    )
    max_token_id = token_matrix.shape[0] - 1
    for row_idx, token_ids in enumerate(token_lists):
        if not token_ids:
            continue
        clipped = [max(0, min(int(token_id), max_token_id)) for token_id in token_ids]
        token_idx = torch.tensor(clipped, device=token_matrix.device, dtype=torch.long)
        out[row_idx] = token_matrix.index_select(0, token_idx).sum(dim=0)
    return out


def _positive_doc_indices(
    qrels: list[dict],
    query_ids: list[str],
    corpus_ids: list[str],
) -> list[list[int]]:
    query_index = {query_id: idx for idx, query_id in enumerate(query_ids)}
    corpus_index = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}
    positives: list[list[int]] = [[] for _ in query_ids]
    for row in qrels:
        if int(row["score"]) <= 0:
            continue
        qi = query_index.get(row["query_id"])
        di = corpus_index.get(row["corpus_id"])
        if qi is None or di is None:
            continue
        positives[qi].append(di)
    return [sorted(indices) for indices in positives]


def _score_doc_chunks(
    corpus_tokens: list[list[int]],
    query_embeddings: torch.Tensor,
    token_matrix: torch.Tensor,
    score_chunk_size: int,
):
    chunk_size = max(1, int(score_chunk_size))
    for start in range(0, len(corpus_tokens), chunk_size):
        end = min(start + chunk_size, len(corpus_tokens))
        doc_embeddings = _sum_token_rows_recordwise(corpus_tokens[start:end], token_matrix)
        scores = query_embeddings @ doc_embeddings.T
        yield start, end, scores
        del doc_embeddings, scores


def _retrieval_metrics_chunked(
    corpus_tokens: list[list[int]],
    query_tokens: list[list[int]],
    qrels: list[dict],
    query_ids: list[str],
    corpus_ids: list[str],
    token_matrix: torch.Tensor,
    *,
    score_chunk_size: int,
) -> dict[str, float]:
    query_embeddings = _sum_token_rows_recordwise(query_tokens, token_matrix)
    positives = _positive_doc_indices(qrels, query_ids, corpus_ids)
    num_queries = len(query_tokens)
    k2 = min(2, len(corpus_tokens))
    top_scores = torch.full(
        (num_queries, k2),
        -torch.inf,
        device=token_matrix.device,
        dtype=token_matrix.dtype,
    )
    top_indices = torch.full(
        (num_queries, k2),
        -1,
        device=token_matrix.device,
        dtype=torch.long,
    )
    best_positive_scores = torch.full(
        (num_queries,),
        -torch.inf,
        device=token_matrix.device,
        dtype=token_matrix.dtype,
    )

    for start, end, scores in _score_doc_chunks(
        corpus_tokens, query_embeddings, token_matrix, score_chunk_size
    ):
        chunk_indices = torch.arange(start, end, device=token_matrix.device, dtype=torch.long)
        expanded_indices = chunk_indices.unsqueeze(0).expand(num_queries, -1)
        combined_scores = torch.cat([top_scores, scores], dim=1)
        combined_indices = torch.cat([top_indices, expanded_indices], dim=1)
        top_scores, top_pos = torch.topk(combined_scores, k=k2, dim=1)
        top_indices = torch.gather(combined_indices, 1, top_pos)

        for qi, pos_indices in enumerate(positives):
            for doc_idx in pos_indices:
                if start <= doc_idx < end:
                    score = scores[qi, doc_idx - start]
                    best_positive_scores[qi] = torch.maximum(best_positive_scores[qi], score)

    greater_counts = torch.zeros(num_queries, device=token_matrix.device, dtype=torch.long)
    for _start, _end, scores in _score_doc_chunks(
        corpus_tokens, query_embeddings, token_matrix, score_chunk_size
    ):
        greater_counts += (scores > best_positive_scores.unsqueeze(1)).sum(dim=1)

    ranks: list[int] = []
    hits1 = 0
    recall2_sum = 0.0
    top2_hits = 0
    n_top2_eval = 0
    for qi, pos_indices in enumerate(positives):
        if not pos_indices:
            continue
        pos_set = set(pos_indices)
        ranks.append(int(greater_counts[qi].item()) + 1)
        top_list = [int(idx) for idx in top_indices[qi].tolist() if int(idx) >= 0]
        if top_list and top_list[0] in pos_set:
            hits1 += 1
        top_set = set(top_list[:k2])
        recall2_sum += len(top_set & pos_set) / float(len(pos_set))
        if len(pos_indices) == 2:
            n_top2_eval += 1
            if top_set == pos_set:
                top2_hits += 1

    mean_rank = sum(ranks) / len(ranks) if ranks else 0.0
    recall1 = hits1 / len(ranks) if ranks else 0.0
    recall2 = recall2_sum / len(ranks) if ranks else 0.0
    top2_exact_match = top2_hits / n_top2_eval if n_top2_eval else 0.0
    return {
        "mean_rank": mean_rank,
        "recall_at_1": recall1,
        "recall_at_2": recall2,
        "top2_exact_match": top2_exact_match,
        "num_queries_eval": float(len(ranks)),
        "num_queries_top2_eval": float(n_top2_eval),
    }


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
    score_chunk_size: int = 0,
    on_row: Callable[[ResultRow], None] | None = None,
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
    labels = None
    if score_chunk_size <= 0:
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
        seed = _effective_seed(dataset, tokenizer_name, dim, base_seed)
        token_matrix = build_random_token_matrix(
            tokenizer.num_token_types(),
            dim,
            seed,
            device=device,
            dtype=DEFAULT_FLOAT_DTYPE,
        )
        if score_chunk_size > 0:
            scores = None
            metrics = _retrieval_metrics_chunked(
                corpus_tokens,
                query_tokens,
                qrels,
                query_ids,
                corpus_ids,
                token_matrix,
                score_chunk_size=score_chunk_size,
            )
        else:
            if labels is None:
                raise RuntimeError("labels were not initialized")
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
        if on_row is not None:
            on_row(row)
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


def _has_value(value: object) -> bool:
    return value is not None and value != ""


def _merge_row(existing: ResultRow | None, incoming: ResultRow) -> ResultRow:
    if existing is None:
        return dict(incoming)
    merged = dict(existing)
    for key, value in incoming.items():
        if _has_value(value) or key not in merged:
            merged[key] = value
    return merged


def _has_output_fields(row: ResultRow) -> bool:
    return all(_has_value(row.get(field)) for field in OUTPUT_REQUIRED_FIELDS)


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
    if value is None or value == "":
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


def _load_latex_table_rows(path: Path, *, base_seed: int) -> list[ResultRow]:
    rows: list[ResultRow] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.endswith(r"\\") or "&" not in stripped:
                continue
            cells = [cell.strip() for cell in stripped[:-2].split("&")]
            if len(cells) != 4 or cells[0] not in {"LIMIT", "LIMIT-small"}:
                continue
            if not cells[2].isdigit():
                continue
            try:
                recall_at_2 = float(cells[3])
            except ValueError:
                continue
            dataset = "limit-small" if cells[0] == "LIMIT-small" else "limit"
            split = _normalize_split(dataset)
            tokenizer = _normalize_tokenizer(cells[1])
            dim = int(cells[2])
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "tokenizer": tokenizer,
                    "dim": dim,
                    "seed": _effective_seed(dataset, tokenizer, dim, base_seed),
                    "recall_at_2": recall_at_2,
                }
            )
    return rows


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
                    "seed": _effective_seed(dataset, tokenizer, dim, base_seed),
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

    table_path = output_path / "limit_retrieval_table.tex"
    if table_path.exists():
        for row in _load_latex_table_rows(table_path, base_seed=base_seed):
            key = _row_key(row)
            rows_by_key[key] = _merge_row(rows_by_key.get(key), row)

    log_path = output_path / "run.log"
    if log_path.exists():
        for row in _recover_rows_from_log(log_path, base_seed=base_seed):
            key = _row_key(row)
            rows_by_key[key] = _merge_row(rows_by_key.get(key), row)

    csv_path = output_path / "summary.csv"
    if csv_path.exists():
        for row in _load_summary_csv(csv_path):
            key = _row_key(row)
            rows_by_key[key] = _merge_row(rows_by_key.get(key), row)

    results_path = output_path / "results.json"
    if results_path.exists():
        for row in _load_json_rows(results_path):
            key = _row_key(row)
            rows_by_key[key] = _merge_row(rows_by_key.get(key), row)

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
            _effective_seed(_dataset_name(split), tokenizer_name, dim, base_seed),
        )
        for tokenizer_name in tokenizers
        for split in splits
        for dim in dims
    }


def _filter_rows_for_request(
    rows: list[ResultRow],
    requested_keys: set[tuple[str, str, str, int, int]],
    *,
    require_complete: bool = False,
    required_fields: set[str] = OUTPUT_REQUIRED_FIELDS,
) -> dict[tuple[str, str, str, int, int], ResultRow]:
    rows_by_key: dict[tuple[str, str, str, int, int], ResultRow] = {}
    for row in rows:
        key = _row_key(row)
        if key in requested_keys:
            if require_complete and not all(
                _has_value(row.get(field)) for field in required_fields
            ):
                continue
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
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Dataset & Tokenizer & $d$ & Recall@2 \\\\",
        "\\midrule",
    ]
    for row in sorted_rows:
        lines.append(
            f"{_latex_escape(_display_dataset_name(str(row['dataset'])))} & "
            f"{_latex_escape(row['tokenizer'])} & "
            f"{row['dim']} & "
            f"{_format_float(row['recall_at_2'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _tokenizer_sort_key(tokenizer: str) -> tuple[int, str]:
    try:
        return DEFAULT_TOKENIZERS.index(tokenizer), tokenizer
    except ValueError:
        return len(DEFAULT_TOKENIZERS), tokenizer


def _first_crossing_dim(rows: list[ResultRow], baseline: float) -> int | None:
    for row in sorted(rows, key=lambda item: int(item["dim"])):
        if float(row["recall_at_2"]) > baseline:
            return int(row["dim"])
    return None


def _recall_at_dim(rows: list[ResultRow], dim: int) -> float | None:
    for row in rows:
        if int(row["dim"]) == dim:
            return float(row["recall_at_2"])
    return None


def _write_promptriever_crossing_table(
    rows: list[ResultRow],
    path: Path,
    *,
    target_dim: int = 4096,
) -> None:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Random token-sum controls compared with Weller et al.'s",
        r"  strongest comparable single-vector baseline, Promptriever Llama3 8B",
        rf"  at $d={target_dim}$ (we use baseline in the table).}}",
        r"  \label{tab:limit-promptriever-crossing}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{lrlrr}",
        r"    \toprule",
        rf"    Dataset & Baseline R@2 & Tokenizer & First $d$ above baseline & R@2 at $d={target_dim}$ \\",
        r"    \midrule",
    ]

    for dataset in ("limit", "limit-small"):
        if dataset not in WELLER_PROMPTRIEVER_RECALL_AT_2_REFERENCE:
            continue
        dataset_rows = [row for row in rows if str(row["dataset"]) == dataset]
        if not dataset_rows:
            continue
        baseline, _label = WELLER_PROMPTRIEVER_RECALL_AT_2_REFERENCE[dataset]
        tokenizers = sorted(
            {str(row["tokenizer"]) for row in dataset_rows},
            key=_tokenizer_sort_key,
        )
        for tokenizer in tokenizers:
            tokenizer_rows = [
                row for row in dataset_rows if str(row["tokenizer"]) == tokenizer
            ]
            crossing_dim = _first_crossing_dim(tokenizer_rows, baseline)
            target_recall = _recall_at_dim(tokenizer_rows, target_dim)
            crossing_text = "--" if crossing_dim is None else str(crossing_dim)
            target_text = (
                "--" if target_recall is None else _format_float(target_recall)
            )
            lines.append(
                f"    {_display_dataset_name(dataset)} & "
                f"{baseline:.3f} & "
                f"{_latex_escape(tokenizer)} & "
                f"{crossing_text} & "
                f"{target_text} \\\\"
            )

    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}}",
            r"\end{table}",
            "",
        ]
    )
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


def _positive_log_floor(values: list[float]) -> float:
    positive_values = [value for value in values if value > 0.0]
    if not positive_values:
        return 1e-4
    return min(positive_values) / 2.0


def _values_for_log_axis(values: list[float], floor: float) -> list[float]:
    return [value if value > 0.0 else floor for value in values]


def _write_pdf_figure(
    rows: list[ResultRow],
    path: Path,
    *,
    dataset: str | None = None,
    reference: tuple[float, str] | None = None,
) -> None:
    _set_plot_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    path.parent.mkdir(parents=True, exist_ok=True)

    plot_rows = [
        row for row in rows if dataset is None or str(row["dataset"]) == dataset
    ]
    if not plot_rows:
        raise ValueError(f"no rows available for figure {path}")

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    groups = sorted(
        {(str(row["dataset"]), str(row["tokenizer"])) for row in plot_rows}
    )
    markers = ["o", "s", "^", "D", "v", "P"]
    y_values = [float(row["recall_at_2"]) for row in plot_rows]
    if reference is not None:
        y_values.append(reference[0])
    y_floor = _positive_log_floor(y_values)
    single_dataset = len({str(row["dataset"]) for row in plot_rows}) == 1

    for idx, (dataset_name, tokenizer) in enumerate(groups):
        points = sorted(
            [
                row
                for row in plot_rows
                if row["dataset"] == dataset_name and row["tokenizer"] == tokenizer
            ],
            key=lambda row: int(row["dim"]),
        )
        if single_dataset:
            label = tokenizer
        else:
            label = f"{_display_dataset_name(dataset_name)} / {tokenizer}"
        marker = markers[idx % len(markers)]
        dims = [int(point["dim"]) for point in points]
        recall_values = [float(point["recall_at_2"]) for point in points]
        plot_values = _values_for_log_axis(recall_values, y_floor)
        ax.plot(
            dims,
            plot_values,
            marker=marker,
            linewidth=1.7,
            markersize=5,
            label=label,
        )

    if reference is not None:
        reference_value, reference_label = reference
        ax.axhline(
            reference_value,
            color="black",
            linestyle=":",
            linewidth=1.3,
            alpha=0.8,
            label=reference_label,
        )

    ax.set_xlabel("Embedding dimension $d$")
    ax.set_ylabel("Recall@2")
    ax.set_xscale("log", base=2)
    x_ticks = sorted({int(row["dim"]) for row in plot_rows})
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(tick) for tick in x_ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yscale("log", base=10)
    ax.set_ylim(y_floor, 1.08)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
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
    crossing_table_path = output_path / PROMPTRIEVER_CROSSING_TABLE_NAME
    sorted_rows = _sort_rows(rows)
    _write_latex_table(sorted_rows, table_path)
    _write_promptriever_crossing_table(sorted_rows, crossing_table_path)
    figure_paths = []
    present_datasets = {str(row["dataset"]) for row in sorted_rows}
    for dataset_name, filename in LIMIT_RETRIEVAL_FIGURE_FILES.items():
        if dataset_name not in present_datasets:
            continue
        dataset_figure_path = output_path / filename
        _write_pdf_figure(
            sorted_rows,
            dataset_figure_path,
            dataset=dataset_name,
            reference=WELLER_PROMPTRIEVER_RECALL_AT_2_REFERENCE[dataset_name],
        )
        figure_paths.append(dataset_figure_path)

    if paper_table_dir is not None:
        paper_table_path = Path(paper_table_dir)
        paper_table_path.mkdir(parents=True, exist_ok=True)
        for table in (table_path, crossing_table_path):
            shutil.copy2(table, paper_table_path / table.name)
    if paper_figure_dir is not None:
        paper_figure_path = Path(paper_figure_dir)
        paper_figure_path.mkdir(parents=True, exist_ok=True)
        for figure in figure_paths:
            shutil.copy2(figure, paper_figure_path / figure.name)

    print(f"\n[DONE] Results saved to {output_path}", flush=True)
    print(f"[DONE] Table saved to {table_path}", flush=True)
    print(f"[DONE] Table saved to {crossing_table_path}", flush=True)
    for figure in figure_paths:
        print(f"[DONE] Figure saved to {figure}", flush=True)


def _write_checkpoint_outputs(
    rows: list[ResultRow],
    output_dir: str,
    config: dict[str, object],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sorted_rows = _sort_rows(rows)

    with (output_path / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    with (output_path / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(sorted_rows, handle, indent=2)

    with (output_path / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(sorted_rows)


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
        help="Tokenizers to evaluate: handmade, qwen, and/or vanilla",
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
        help="Seed base; effective seed is base_seed + dim, except configured per-row overrides",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for evaluation",
    )
    parser.add_argument(
        "--score-chunk-size",
        type=int,
        default=2048,
        help=(
            "If positive, stream document embeddings and query-document scores "
            "in chunks of this many documents."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=_default_output_dir(),
        help="Deterministic directory for config.json, results.json, summary.csv, table, and figure",
    )
    parser.add_argument(
        "--paper-table-dir",
        default=DEFAULT_PAPER_TABLE_DIR,
        help="Optional directory receiving a copied LaTeX table",
    )
    parser.add_argument(
        "--paper-figure-dir",
        default=DEFAULT_PAPER_FIGURE_DIR,
        help="Optional directory receiving a copied PDF figure",
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
    if args.score_chunk_size < 0:
        raise ValueError("--score-chunk-size must be non-negative")

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
        "score_chunk_size": int(args.score_chunk_size),
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
    resume_rows: list[ResultRow] = []
    rows_by_key: dict[tuple[str, str, str, int, int], ResultRow] = {}
    all_rows_by_key: dict[tuple[str, str, str, int, int], ResultRow] = {}
    if args.resume or args.mode == "plot":
        resume_rows = _load_resume_rows(args.output_dir, base_seed=int(args.base_seed))
        all_rows_by_key = {_row_key(row): row for row in resume_rows}
        required_fields = (
            PLOT_REQUIRED_FIELDS if args.mode == "plot" else OUTPUT_REQUIRED_FIELDS
        )
        rows_by_key = _filter_rows_for_request(
            resume_rows,
            requested_keys,
            require_complete=True,
            required_fields=required_fields,
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

    def persist_row(row: ResultRow) -> None:
        key = _row_key(row)
        rows_by_key[key] = row
        all_rows_by_key[key] = _merge_row(all_rows_by_key.get(key), row)
        output_rows_by_key = all_rows_by_key if args.resume else rows_by_key
        _write_checkpoint_outputs(
            list(output_rows_by_key.values()),
            args.output_dir,
            config,
        )

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
                    _effective_seed(
                        _dataset_name(split),
                        tokenizer_name,
                        dim,
                        int(args.base_seed),
                    ),
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
                score_chunk_size=int(args.score_chunk_size),
                on_row=persist_row,
            )
            for row in new_rows:
                key = _row_key(row)
                rows_by_key[key] = row
                all_rows_by_key[key] = _merge_row(all_rows_by_key.get(key), row)

    missing_keys = requested_keys - set(rows_by_key)
    if missing_keys:
        raise RuntimeError(f"Sweep did not produce {len(missing_keys)} requested rows")

    output_rows_by_key = all_rows_by_key if args.resume else rows_by_key
    rows = _sort_rows(list(output_rows_by_key.values()))
    _write_outputs(
        rows,
        args.output_dir,
        config,
        paper_table_dir=paper_table_dir,
        paper_figure_dir=paper_figure_dir,
    )


if __name__ == "__main__":
    main()
