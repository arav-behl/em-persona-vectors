"""Pure-numpy analysis helpers: directions, projections, AUROCs and their nulls.

Activation arrays are ``[n, L+1, d]``; a per-layer direction file is ``[L+1, d]``.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from .metrics import COH_THR, is_num


def unit_rows(v: np.ndarray) -> np.ndarray:
    """Normalise the last axis; zero rows stay zero."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > 0, v / np.where(n > 0, n, 1), 0)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def random_units(n: int, d: int, seed: int = 0) -> np.ndarray:
    r = np.random.default_rng(seed).standard_normal((n, d))
    return r / np.linalg.norm(r, axis=1, keepdims=True)


# --------------------------------------------------------------------- persona
def persona_selection(rows: list[dict], threshold: float = 50) -> np.ndarray:
    """Chen et al. filter: keep coherent pos responses that express the trait
    (evil >= threshold) and coherent neg responses that do not."""
    keep = []
    for r in rows:
        e, c = r.get("evil"), r.get("coherent")
        ok = is_num(e) and is_num(c) and c >= COH_THR
        keep.append(ok and ((e >= threshold) if r["pol"] == "pos" else (e < threshold)))
    return np.array(keep)


def persona_vector(acts: np.ndarray, rows: list[dict], keep: np.ndarray | None = None) -> np.ndarray:
    """Mean response activation of (kept) pos minus (kept) neg responses: ``[L+1, d]``."""
    pol = np.array([r["pol"] for r in rows])
    k = np.ones(len(rows), bool) if keep is None else keep
    pos, neg = k & (pol == "pos"), k & (pol == "neg")
    if pos.sum() < 2 or neg.sum() < 2:
        raise ValueError(f"too few responses after filtering (pos {pos.sum()}, neg {neg.sum()})")
    return acts[pos].mean(0) - acts[neg].mean(0)


# ------------------------------------------------------------------ geometry
def centre_by_question(X: np.ndarray, qids: list[str]) -> np.ndarray:
    """Subtract each question's mean so contrasts are not just 'which question'."""
    X = X.copy()
    q = np.asarray(qids)
    for qq in np.unique(q):
        X[q == qq] -= X[q == qq].mean(0)
    return X


def diff_of_means(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X[y == 1].mean(0) - X[y == 0].mean(0)


def auroc_with_null(X: np.ndarray, y: np.ndarray, direction: np.ndarray,
                    n_null: int = 200, seed: int = 0) -> dict:
    """AUROC of ``X @ direction`` for labels y, against random unit directions.
    The null uses max(AUC, 1-AUC) because random directions have arbitrary sign."""
    auc = roc_auc_score(y, X @ direction)
    null = np.array([roc_auc_score(y, X @ r) for r in random_units(n_null, X.shape[1], seed)])
    null = np.maximum(null, 1 - null)
    return {"auroc": float(auc), "null_p95": float(np.percentile(null, 95)),
            "null_mean": float(null.mean()), "p_value": float((1 + (null >= max(auc, 1 - auc)).sum()) / (1 + n_null)),
            "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())}


def proj_null_p95(shift: np.ndarray, n_null: int = 500, seed: int = 0) -> float:
    """95th percentile of |shift . r| over random unit directions r."""
    return float(np.percentile(np.abs(random_units(n_null, shift.shape[0], seed) @ shift), 95))


def cos_null_p95(d: int, n_null: int = 2000, seed: int = 0) -> float:
    """95th pct of |cos| between two random directions in d dims (about 2/sqrt(d))."""
    a, b = random_units(n_null, d, seed), random_units(n_null, d, seed + 1)
    return float(np.percentile(np.abs((a * b).sum(1)), 95))


def perm_null_dir_cos(X: np.ndarray, y: np.ndarray, target: np.ndarray, n_perm: int = 200,
                      seed: int = 0) -> np.ndarray:
    """|cos(target, diff-of-means)| under label permutation: how large the cosine
    gets by chance given these activations and class sizes."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        out.append(abs(cos(diff_of_means(X, yp), target)))
    return np.array(out)


def bootstrap_ci(x: np.ndarray, fn=np.mean, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def exact_perm_p_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided exact permutation p-value for Pearson r (use only for small n, e.g. <= 8 models)."""
    from itertools import permutations

    r0 = abs(np.corrcoef(x, y)[0, 1])
    rs = [abs(np.corrcoef(x, np.array(p))[0, 1]) for p in permutations(y)]
    return float(np.mean([r >= r0 - 1e-12 for r in rs]))
