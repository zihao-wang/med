"""Default floating-point dtype: float32 (CUDA, CPU, and Apple MPS)."""

from __future__ import annotations

import torch

DEFAULT_FLOAT_DTYPE = torch.float32


def apply_default_float_dtype() -> None:
    """
    Set PyTorch's default dtype for floating-point tensors to float32.

    MPS does not support float64; using float32 everywhere avoids accidental
    promotions and keeps new modules (``nn.Linear``, etc.) on float32.
    """
    torch.set_default_dtype(DEFAULT_FLOAT_DTYPE)


__all__ = ["DEFAULT_FLOAT_DTYPE", "apply_default_float_dtype"]
