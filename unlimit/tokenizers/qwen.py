"""Qwen3 (Hugging Face) subword tokenizer for raw LiMIT text."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from med.unlimit.dtype import DEFAULT_FLOAT_DTYPE
from med.unlimit.tokenizers.types import TokenizedCorpusRecord, TokenizedQueryRecord

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class QwenSubwordTokenizer:
    """
    Wraps ``transformers.AutoTokenizer`` + **pretrained token embeddings** from the
    matching causal LM (default ``Qwen/Qwen3-0.6B``).

    Corpus and query records are encoded from the **raw** ``text`` field (full
    document / question string), not phrase-parsed like :class:`HandmadeTokenizer`.

    RP+OMP uses rows of the LM's input embedding matrix (not random Gaussians).
    :attr:`pretrained_embedding_dim` is the LM hidden size (token vector width).
    """

    name = "qwen"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        trust_remote_code: bool = True,
        *,
        load_pretrained_embeddings: bool = True,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model_name = model_name
        self._tok: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if self._tok.pad_token_id is None and self._tok.eos_token_id is not None:
            self._tok.pad_token = self._tok.eos_token
        _pad = int(self._tok.pad_token_id)
        unk_id = self._tok.unk_token_id
        _unk = int(unk_id) if unk_id is not None else _pad
        # Row count for RP+OMP matrix (covers all ids that may appear, incl. specials).
        self._rp_matrix_rows = max(
            len(self._tok),
            int(self._tok.vocab_size),
            _pad + 1,
            _unk + 1,
        )

        self._token_embed_weight: torch.Tensor | None = None
        if load_pretrained_embeddings:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            w = model.get_input_embeddings().weight.detach().float().cpu().clone()
            del model
            self._token_embed_weight = w
            self.pretrained_embedding_dim = int(w.shape[1])
        else:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
            self.pretrained_embedding_dim = int(cfg.hidden_size)

    def rp_omp_num_token_types(self) -> int:
        return self._rp_matrix_rows

    def rp_omp_token_matrix(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype = DEFAULT_FLOAT_DTYPE,
    ) -> torch.Tensor:
        """
        Token vectors for RP+OMP: frozen **pretrained** embedding rows ``(V, d)``.
        """
        if self._token_embed_weight is None:
            raise RuntimeError(
                "rp_omp_token_matrix requires load_pretrained_embeddings=True "
                "(full LM load) for QwenSubwordTokenizer."
            )
        W = self._token_embed_weight
        n_types = self._rp_matrix_rows
        n_vocab, _d = W.shape
        if n_types > n_vocab:
            pad = torch.zeros(
                n_types - n_vocab,
                W.shape[1],
                dtype=torch.float32,
                device="cpu",
            )
            W = torch.cat([W, pad], dim=0)
        elif n_types < n_vocab:
            W = W[:n_types]
        return W.to(device=device, dtype=dtype)

    def _encode(self, text: str) -> tuple[list[int], list[str]]:
        ids = self._tok.encode(text, add_special_tokens=False)
        return ids, []

    def tokenize_corpus_records(
        self, records: list[dict]
    ) -> list[TokenizedCorpusRecord]:
        out: list[TokenizedCorpusRecord] = []
        for r in records:
            ids, unk = self._encode(r["text"])
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

    def tokenize_query_records(
        self, records: list[dict]
    ) -> list[TokenizedQueryRecord]:
        out: list[TokenizedQueryRecord] = []
        for r in records:
            ids, unk = self._encode(r["text"])
            out.append(
                {
                    "_id": r["_id"],
                    "text": r["text"],
                    "token_ids": ids,
                    "unknown_phrases": unk,
                }
            )
        return out


__all__ = ["QwenSubwordTokenizer"]
