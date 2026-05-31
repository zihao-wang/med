"""Device resolution for PyTorch: CUDA, MPS, or CPU.

This is a local copy of the logic in med.unlimit.device, kept here so the
mean_embedding and cyclic_polytope modules don't need to depend on LIMIT code.
"""

from __future__ import annotations

import torch


def resolve_device(device: str | torch.device | None = None) -> str:
    """Return the best available torch device name.

    Preference: CUDA > MPS > CPU.
    """
    if device is not None:
        if isinstance(device, torch.device):
            return str(device)
        if device.strip():
            return device
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
