from __future__ import annotations

import math
from typing import Literal

LRScaling = Literal[
    "log2",
    "sqrt_log2",
    "constant",
    "fourth_root_m",
    "sqrt_m",
    "linear_m",
]

LR_SCALING_CHOICES: tuple[LRScaling, ...] = (
    "log2",
    "sqrt_log2",
    "constant",
    "fourth_root_m",
    "sqrt_m",
    "linear_m",
)


def scaled_learning_rate(base_lr: float, m: int, scheme: LRScaling) -> float:
    if m <= 1:
        return base_lr
    if scheme == "log2":
        return base_lr / math.log2(m)
    if scheme == "sqrt_log2":
        return base_lr / math.sqrt(math.log2(m))
    if scheme == "constant":
        return base_lr
    if scheme == "fourth_root_m":
        return base_lr / (m**0.25)
    if scheme == "sqrt_m":
        return base_lr / math.sqrt(m)
    if scheme == "linear_m":
        return base_lr / m
    raise ValueError(f"Unknown lr scaling scheme: {scheme}")
