"""Qwen subword tokenizer for random-token LiMIT baselines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from unlimit.tokenizers.types import TokenizedCorpusRecord, TokenizedQueryRecord

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class QwenSubwordTokenizer:
    """
    Tokenize raw LiMIT text with a Qwen tokenizer.

    The retrieval experiment still uses random token embeddings. By default this
    tokenizer compacts the raw Qwen ids observed in the evaluated split to a
    local contiguous id space, avoiding a large random matrix for unused Qwen
    vocabulary rows.
    """

    name = "qwen"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
        compact_token_ids: bool = True,
    ) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "QwenSubwordTokenizer requires transformers. Install the qwen "
                "extra or add transformers to the environment before using "
                "--tokenizers qwen."
            ) from exc

        self._model_name = model_name
        self._tok: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        self._compact_token_ids = compact_token_ids
        self._raw_to_compact: dict[int, int] = {}

        pad_id = self._tok.pad_token_id
        eos_id = self._tok.eos_token_id
        unk_id = self._tok.unk_token_id
        max_special = max(
            [value for value in (pad_id, eos_id, unk_id) if value is not None],
            default=-1,
        )
        self._full_num_token_types = max(
            len(self._tok),
            int(getattr(self._tok, "vocab_size", 0)),
            int(max_special) + 1,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def num_token_types(self) -> int:
        if self._compact_token_ids:
            return max(1, len(self._raw_to_compact))
        return self._full_num_token_types

    def _map_id(self, raw_id: int) -> int:
        if not self._compact_token_ids:
            return raw_id
        mapped = self._raw_to_compact.get(raw_id)
        if mapped is None:
            mapped = len(self._raw_to_compact)
            self._raw_to_compact[raw_id] = mapped
        return mapped

    def _encode(self, text: str) -> tuple[list[int], list[str]]:
        ids = self._tok.encode(text, add_special_tokens=False)
        return [self._map_id(int(token_id)) for token_id in ids], []

    def tokenize_corpus_records(
        self, records: list[dict]
    ) -> list[TokenizedCorpusRecord]:
        out: list[TokenizedCorpusRecord] = []
        for record in records:
            ids, unknown = self._encode(record["text"])
            out.append(
                {
                    "_id": record["_id"],
                    "title": record.get("title", ""),
                    "text": record["text"],
                    "token_ids": ids,
                    "unknown_phrases": unknown,
                }
            )
        return out

    def tokenize_query_records(
        self, records: list[dict]
    ) -> list[TokenizedQueryRecord]:
        out: list[TokenizedQueryRecord] = []
        for record in records:
            ids, unknown = self._encode(record["text"])
            out.append(
                {
                    "_id": record["_id"],
                    "text": record["text"],
                    "token_ids": ids,
                    "unknown_phrases": unknown,
                }
            )
        return out


__all__ = ["QwenSubwordTokenizer"]
