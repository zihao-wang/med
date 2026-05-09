"""Load packaged LiMIT JSONL splits."""

from __future__ import annotations

import json
import random
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
    """Load one packaged LiMIT JSONL asset."""
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


def split_train_test_query_ids(
    query_ids: list[str],
    train_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """
    Deterministic train / test partition of query ids (for supervised training).

    The `google-deepmind/limit` JSONL release does **not** ship separate train/test
    files; Hugging Face / MTEB expose qrels under a ``test`` split name only. This
    helper creates a standard held-out query set for experiments.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    ids = sorted(query_ids)
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_train = int(round(len(shuffled) * train_fraction))
    n_train = max(1, min(len(shuffled) - 1, n_train))
    return shuffled[:n_train], shuffled[n_train:]


def load_limit_train_test(
    split: Literal["small", "full"] = "small",
    train_fraction: float = 0.8,
    split_seed: int = 42,
) -> tuple[
    list[CorpusRecord],
    list[QueryRecord],
    list[QueryRecord],
    list[QrelRecord],
    list[QrelRecord],
]:
    """
    Load LiMIT, then partition **queries** (and qrels) into train / test.

    The **corpus** is unchanged (full retrieval pool for both splits).
    """
    corpus, queries, qrels = load_limit(split)
    all_ids = [q["_id"] for q in queries]
    train_ids, test_ids = split_train_test_query_ids(all_ids, train_fraction, split_seed)
    tr_set, te_set = set(train_ids), set(test_ids)
    q_train = sorted((q for q in queries if q["_id"] in tr_set), key=lambda x: x["_id"])
    q_test = sorted((q for q in queries if q["_id"] in te_set), key=lambda x: x["_id"])
    qr_train = [r for r in qrels if r["query_id"] in tr_set]
    qr_test = [r for r in qrels if r["query_id"] in te_set]
    return corpus, q_train, q_test, qr_train, qr_test


def load_limit(
    split: Literal["small", "full"] = "small",
) -> tuple[list[CorpusRecord], list[QueryRecord], list[QrelRecord]]:
    """
    Load corpus, queries, and qrels from packaged LiMIT JSONL assets.

    Args:
        split: ``\"small\"`` — 46 docs / 1000 queries / 2000 qrels (quick demos).
               ``\"full\"`` — ~50k corpus docs, same query/qrel schema as the paper.

    Returns:
        corpus: documents with ``_id`` (person name) and ``text`` (comma-separated likes).
        queries: questions with ``_id`` (e.g. ``query_0``) and ``text``.
        qrels: relevance triples; ``query_id`` joins to ``queries._id``,
               ``corpus_id`` to ``corpus._id``.

    Note:
        The GitHub release is a **single** JSONL triple per scale (no separate
        ``train.jsonl`` / ``test.jsonl``). For train/test experiments, use
        :func:`load_limit_train_test`.
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

    ap = argparse.ArgumentParser(description="Load packaged LiMIT JSONL assets")
    ap.add_argument(
        "split",
        nargs="?",
        default="small",
        choices=("small", "full"),
        help="Dataset split (default: small)",
    )
    args = ap.parse_args()
    c, q, r = load_limit(args.split)
    tr, te = split_train_test_query_ids([x["_id"] for x in q], 0.8, 42)
    print(f"split={args.split!r} | Corpus: {len(c)} | Queries: {len(q)} | Qrels: {len(r)}")
    print(f"example train/test query partition (80/20, seed=42): {len(tr)} train, {len(te)} test")
    print("Sample corpus:", c[0])
    print("Sample query:", q[0])
    print("Sample qrel:", r[0])
