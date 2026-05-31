"""Vanilla word tokenizer for LiMIT random-token baselines."""

from __future__ import annotations

import re

from med.unlimit.tokenizers.types import TokenizedCorpusRecord, TokenizedQueryRecord

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "who",
    "with",
}

_WORD_RE = re.compile(r"[A-Za-z]+(?:[-\x27][A-Za-z]+)*")


class VanillaWordTokenizer:
    """Lowercase word tokenizer with local compact IDs and no fixed vocabulary."""

    name = "vanilla"

    def __init__(self, stop_words: set[str] | None = None) -> None:
        self._stop_words = STOP_WORDS if stop_words is None else set(stop_words)
        self._word_to_id: dict[str, int] = {}

    def num_token_types(self) -> int:
        return max(1, len(self._word_to_id))

    def _tokenize_text(self, text: str) -> list[str]:
        return [
            word
            for word in (match.group(0).lower() for match in _WORD_RE.finditer(text))
            if word not in self._stop_words
        ]

    def _id_for_word(self, word: str) -> int:
        token_id = self._word_to_id.get(word)
        if token_id is None:
            token_id = len(self._word_to_id)
            self._word_to_id[word] = token_id
        return token_id

    def _encode(self, text: str) -> list[int]:
        return [self._id_for_word(word) for word in self._tokenize_text(text)]

    def tokenize_corpus_records(
        self, records: list[dict]
    ) -> list[TokenizedCorpusRecord]:
        out: list[TokenizedCorpusRecord] = []
        for record in records:
            out.append(
                {
                    "_id": record["_id"],
                    "title": record.get("title", ""),
                    "text": record["text"],
                    "token_ids": self._encode(record["text"]),
                    "unknown_phrases": [],
                }
            )
        return out

    def tokenize_query_records(
        self, records: list[dict]
    ) -> list[TokenizedQueryRecord]:
        out: list[TokenizedQueryRecord] = []
        for record in records:
            out.append(
                {
                    "_id": record["_id"],
                    "text": record["text"],
                    "token_ids": self._encode(record["text"]),
                    "unknown_phrases": [],
                }
            )
        return out


__all__ = ["STOP_WORDS", "VanillaWordTokenizer"]
