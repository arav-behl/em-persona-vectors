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
    after every block (directional ablation, Arditi et al. 2024).

    With ``means`` (one value per hidden-state index 0..L), the component along
    the unit vector is set to that value instead of zero (mean-ablation). This
    removes variation along the direction without moving every activation when
    ordinary activations already sit far from zero along it.
    """
    vector: np.ndarray
    means: tuple[float, ...] | None = None

    def target(self, index: int) -> float:
        """Value the component is set to at hidden-state ``index``."""
        return 0.0 if self.means is None else float(self.means[index])


Intervention = Steer | Ablate


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n == 0:
        raise ValueError("cannot normalise a zero vector")
    return v / n
