"""Residual-stream interventions, shared by all backends.

Layer indices follow the HF ``hidden_states`` convention: index 0 is the
embedding output and index ``l`` (1..L) is the output of decoder block ``l-1``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Steer:
    """Add ``alpha * unit(vector)`` to the residual stream after block ``layer``.

    ``alpha`` is in units of the residual norm scale chosen by the caller; we
    always pass a unit vector so alphas are comparable across directions.
    """
    vector: np.ndarray
    layer: int
    alpha: float


@dataclass(frozen=True)
class Ablate:
    """Project ``vector`` out of the residual stream after the embeddings and
    after every block (directional ablation, Arditi et al. 2024)."""
    vector: np.ndarray


Intervention = Steer | Ablate


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n == 0:
        raise ValueError("cannot normalise a zero vector")
    return v / n
