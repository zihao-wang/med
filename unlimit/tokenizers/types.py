"""Shared record types for tokenized LIMIT data."""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable


class TokenizedCorpusRecord(TypedDict):
    _id: str
    title: str
    text: str
    token_ids: list[int]
    unknown_phrases: list[str]


class TokenizedQueryRecord(TypedDict):
    _id: str
    text: str
    token_ids: list[int]
    unknown_phrases: list[str]


@runtime_checkable
class LimitTokenizer(Protocol):
    """Maps raw LIMIT corpus / query dicts to token id sequences."""

    name: str

    def tokenize_corpus_records(
        self, records: list[dict]
    ) -> list[TokenizedCorpusRecord]: ...

    def tokenize_query_records(
        self, records: list[dict]
    ) -> list[TokenizedQueryRecord]: ...

    def num_token_types(self) -> int:
        """Number of token-id rows needed in the random token matrix."""


__all__ = [
    "LimitTokenizer",
    "TokenizedCorpusRecord",
    "TokenizedQueryRecord",
]
