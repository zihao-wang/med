"""Load packaged LIMIT JSONL splits."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from importlib import resources
from typing import Any, Literal, TypedDict

LIMIT_ASSET_DIR_BY_SPLIT: dict[Literal["small", "full"], str] = {
    "small": "limit-small",
    "full": "limit",
}


class CorpusRecord(TypedDict):
    _id: str
    title: str
    text: str


class QueryRecord(TypedDict):
    _id: str
    text: str


class QrelRecord(TypedDict):
    """query-id and corpus-id join to queries._id and corpus._id."""

    query_id: str
    corpus_id: str
    score: int


def load_jsonl_from_asset(
    split: Literal["small", "full"],
    name: Literal["corpus", "queries", "qrels"],
) -> list[dict[str, Any]]:
    """Load one packaged LIMIT JSONL asset."""
    path = resources.files("unlimit").joinpath(
        "assets",
        LIMIT_ASSET_DIR_BY_SPLIT[split],
        f"{name}.jsonl",
    )
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalize_qrel_row(row: dict[str, Any]) -> QrelRecord:
    return {
        "query_id": row["query-id"],
        "corpus_id": row["corpus-id"],
        "score": int(row["score"]),
    }


def qrels_positive_distribution(qrels: list[QrelRecord]) -> Counter[int]:
    """
    Histogram of “how many positive (score > 0) qrels per query_id”.

    For the official ``limit-small`` and ``limit`` splits from DeepMind, this is
    ``Counter({2: 1000})``: every query has exactly **two** relevant corpus ids.
    """
    per_q: dict[str, int] = defaultdict(int)
    for r in qrels:
        if int(r["score"]) > 0:
            per_q[r["query_id"]] += 1
    return Counter(per_q.values())


def load_limit(
    split: Literal["small", "full"] = "small",
) -> tuple[list[CorpusRecord], list[QueryRecord], list[QrelRecord]]:
    """
    Load corpus, queries, and qrels from packaged LIMIT JSONL assets.

    Args:
        split: ``\"small\"`` — 46 docs / 1000 queries / 2000 qrels (quick demos).
               ``\"full\"`` — ~50k corpus docs, same query/qrel schema as the paper.

    Returns:
        corpus: documents with ``_id`` (person name) and ``text`` (comma-separated likes).
        queries: questions with ``_id`` (e.g. ``query_0``) and ``text``.
        qrels: relevance triples; ``query_id`` joins to ``queries._id``,
               ``corpus_id`` to ``corpus._id``.

    Note:
        The GitHub release is a **single** JSONL triple per scale; it does not
        ship separate ``train.jsonl`` / ``test.jsonl`` files.
    """
    corpus_raw = load_jsonl_from_asset(split, "corpus")
    queries_raw = load_jsonl_from_asset(split, "queries")
    qrels_raw = load_jsonl_from_asset(split, "qrels")

    corpus: list[CorpusRecord] = [r for r in corpus_raw]  # type: ignore[assignment]
    queries: list[QueryRecord] = [r for r in queries_raw]  # type: ignore[assignment]
    qrels = [_normalize_qrel_row(r) for r in qrels_raw]

    return corpus, queries, qrels


def load_limit_small() -> tuple[list[CorpusRecord], list[QueryRecord], list[QrelRecord]]:
    """Same as ``load_limit(\"small\")``."""
    return load_limit("small")


def load_limit_full() -> tuple[list[CorpusRecord], list[QueryRecord], list[QrelRecord]]:
    """Same as ``load_limit(\"full\")``."""
    return load_limit("full")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Load packaged LIMIT JSONL assets")
    ap.add_argument(
        "split",
        nargs="?",
        default="small",
        choices=("small", "full"),
        help="Dataset split (default: small)",
    )
    args = ap.parse_args()
    c, q, r = load_limit(args.split)
    print(f"split={args.split!r} | Corpus: {len(c)} | Queries: {len(q)} | Qrels: {len(r)}")
    print("Sample corpus:", c[0])
    print("Sample query:", q[0])
    print("Sample qrel:", r[0])
