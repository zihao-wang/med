"""PyTorch device selection: CUDA, Apple Silicon MPS, or CPU."""

from __future__ import annotations

import torch

__all__ = ["best_available_device_name", "resolve_torch_device"]


def best_available_device_name() -> str:
    """
    Return ``\"cuda\"``, ``\"mps\"``, or ``\"cpu\"``.

    Preference order: NVIDIA CUDA, then Apple Metal (MPS) when built and available,
    else CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def resolve_torch_device(device: str | torch.device | None) -> torch.device:
    """
    Resolve a device for training/eval.

    * ``None`` or empty string → :func:`best_available_device_name`
    * Otherwise → ``torch.device(device)``
    """
    if device is None:
        return torch.device(best_available_device_name())
    if isinstance(device, str) and not device.strip():
        return torch.device(best_available_device_name())
    return torch.device(device)
