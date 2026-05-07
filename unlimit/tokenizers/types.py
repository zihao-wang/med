"""Shared record types for tokenized LiMIT data."""

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
    """Maps raw LiMIT corpus / query dicts to token id sequences (RP+OMP and metrics)."""

    name: str

    def tokenize_corpus_records(
        self, records: list[dict]
    ) -> list[TokenizedCorpusRecord]: ...

    def tokenize_query_records(
        self, records: list[dict]
    ) -> list[TokenizedQueryRecord]: ...

    def rp_omp_num_token_types(self) -> int:
        """
        Number of token-id rows in the RP+OMP token matrix.

        For phrase vocab this is ``unk_token_id + 1`` (ids ``0..unk`` inclusive).
        For subword tokenizers it is typically the tokenizer table size.

        Optional: implement ``rp_omp_token_matrix(device=..., dtype=...) -> Tensor``
        to supply **frozen** rows (e.g. pretrained LM embeddings) instead of Gaussians.
        """


__all__ = [
    "LimitTokenizer",
    "TokenizedCorpusRecord",
    "TokenizedQueryRecord",
]
