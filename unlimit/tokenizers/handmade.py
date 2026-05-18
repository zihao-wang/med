"""Hand-built LIMIT concept vocabulary (sorted ``vocab.txt``) + phrase tokenization."""

from __future__ import annotations

import re
from pathlib import Path

from unlimit.tokenizers.types import TokenizedCorpusRecord, TokenizedQueryRecord

_VOCAB_PATH = Path(__file__).resolve().parents[1] / "vocab.txt"


def _load_vocab_lines(path: Path | None = None) -> list[str]:
    p = path or _VOCAB_PATH
    raw = p.read_text(encoding="utf-8")
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]


_LINES = _load_vocab_lines()
_SORTED = sorted(_LINES)
token_to_id: dict[str, int] = {t: i for i, t in enumerate(_SORTED)}
id_to_token: list[str] = list(_SORTED)
VOCAB_SIZE = len(token_to_id)
UNK_TOKEN_ID = VOCAB_SIZE
PAD_TOKEN_ID = VOCAB_SIZE + 1
NUM_EMBEDDING = VOCAB_SIZE + 2


def token_id_for_unknown() -> int:
    return UNK_TOKEN_ID


def _split_likes_body(rest: str) -> list[str]:
    rest = rest.strip()
    if not rest:
        return []
    if rest.endswith("."):
        rest = rest[:-1].strip()
    parts = rest.split(", ")
    if len(parts) == 1:
        segment = parts[0]
        if " and " in segment:
            a, b = segment.rsplit(" and ", 1)
            return [a.strip(), b.strip()]
        return [segment.strip()] if segment.strip() else []
    *initial, last = parts
    items = [p.strip() for p in initial if p.strip()]
    if " and " in last:
        a, b = last.rsplit(" and ", 1)
        items.append(a.strip())
        items.append(b.strip())
    else:
        items.append(last.strip())
    return [x for x in items if x]


def extract_likes_phrases(corpus_text: str) -> list[str]:
    if " likes " not in corpus_text:
        return []
    rest = corpus_text.split(" likes ", 1)[1]
    return _split_likes_body(rest)


_QUERY_PATTERN = re.compile(r"^Who likes (.+)\?$")


def extract_query_phrase(query_text: str) -> str | None:
    m = _QUERY_PATTERN.match(query_text.strip())
    if not m:
        return None
    return m.group(1).strip()


def phrases_to_token_ids(phrases: list[str]) -> tuple[list[int], list[str]]:
    ids: list[int] = []
    unknown: list[str] = []
    for p in phrases:
        tid = token_to_id.get(p)
        if tid is None:
            ids.append(UNK_TOKEN_ID)
            unknown.append(p)
        else:
            ids.append(tid)
    return ids, unknown


def tokenize_corpus_text(corpus_text: str) -> tuple[list[int], list[str]]:
    return phrases_to_token_ids(extract_likes_phrases(corpus_text))


def tokenize_query_text(query_text: str) -> tuple[list[int], list[str]]:
    phrase = extract_query_phrase(query_text)
    if phrase is None:
        return [], [query_text.strip()]
    return phrases_to_token_ids([phrase])


def tokenize_corpus_records(records: list[dict]) -> list[TokenizedCorpusRecord]:
    out: list[TokenizedCorpusRecord] = []
    for r in records:
        ids, unk = tokenize_corpus_text(r["text"])
        out.append(
            {
                "_id": r["_id"],
                "title": r.get("title", ""),
                "text": r["text"],
                "token_ids": ids,
                "unknown_phrases": unk,
            }
        )
    return out


def tokenize_query_records(records: list[dict]) -> list[TokenizedQueryRecord]:
    out: list[TokenizedQueryRecord] = []
    for r in records:
        ids, unk = tokenize_query_text(r["text"])
        out.append(
            {
                "_id": r["_id"],
                "text": r["text"],
                "token_ids": ids,
                "unknown_phrases": unk,
            }
        )
    return out


class HandmadeTokenizer:
    """Concept-level tokenizer aligned with ``vocab.txt``."""

    name = "handmade"

    def num_token_types(self) -> int:
        return UNK_TOKEN_ID + 1

    def tokenize_corpus_records(
        self, records: list[dict]
    ) -> list[TokenizedCorpusRecord]:
        return tokenize_corpus_records(records)

    def tokenize_query_records(
        self, records: list[dict]
    ) -> list[TokenizedQueryRecord]:
        return tokenize_query_records(records)


__all__ = [
    "HandmadeTokenizer",
    "NUM_EMBEDDING",
    "PAD_TOKEN_ID",
    "UNK_TOKEN_ID",
    "VOCAB_SIZE",
    "extract_likes_phrases",
    "extract_query_phrase",
    "id_to_token",
    "phrases_to_token_ids",
    "token_id_for_unknown",
    "token_to_id",
    "tokenize_corpus_records",
    "tokenize_corpus_text",
    "tokenize_query_records",
    "tokenize_query_text",
]
