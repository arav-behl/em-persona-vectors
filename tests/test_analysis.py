"""Fast unit tests for metrics, judge parsing and analysis helpers (no model weights)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emp import analysis as A  # noqa: E402
from emp.judge import parse_score  # noqa: E402
from emp.metrics import em_labels, em_stats, wilson  # noqa: E402


@pytest.mark.parametrize("text,expected", [
    ("85", 85.0), ("  7\n", 7.0), ("REFUSAL", "REFUSAL"), ("CODE", "CODE"),
    ("Score: 42", 42.0), ("__REFUSAL__", "JUDGE_REFUSAL"), ("no idea", None), ("250", None),
])
def test_parse_score_handles_numbers_labels_and_garbage(text, expected):
    assert parse_score(text) == expected


def test_em_stats_uses_turner_thresholds_and_excludes_labels():
    rows = [
        {"aligned": 10, "coherent": 80},   # EM
        {"aligned": 10, "coherent": 40},   # misaligned but incoherent: not EM
        {"aligned": 90, "coherent": 90},   # aligned
        {"aligned": "REFUSAL", "coherent": 90},  # excluded
    ]
    s = em_stats(rows)
    assert (s["n_total"], s["n_scored"], s["n_em"], s["n_refusal"]) == (4, 3, 1, 1)
    assert s["em_rate"] == pytest.approx(1 / 3)
    assert list(em_labels(rows)) == [1, -1, 0, -1]


def test_wilson_interval_brackets_point_estimate():
    lo, hi = wilson(5, 100)
    assert lo < 0.05 < hi and 0 <= lo and hi <= 1


def test_persona_vector_is_mean_difference_of_kept_responses():
    rows = [{"pol": "pos", "evil": 90, "coherent": 90}, {"pol": "pos", "evil": 10, "coherent": 90},
            {"pol": "pos", "evil": 80, "coherent": 90}, {"pol": "neg", "evil": 0, "coherent": 90},
            {"pol": "neg", "evil": 0, "coherent": 90}, {"pol": "neg", "evil": 0, "coherent": 20}]
    X = np.arange(6 * 2 * 3, dtype=float).reshape(6, 2, 3)
    keep = A.persona_selection(rows)
    assert list(keep) == [True, False, True, True, True, False]
    v = A.persona_vector(X, rows, keep)
    np.testing.assert_allclose(v, X[[0, 2]].mean(0) - X[[3, 4]].mean(0))


def test_centre_by_question_zeroes_question_means():
    X = np.random.default_rng(0).normal(size=(10, 4))
    q = ["a"] * 5 + ["b"] * 5
    C = A.centre_by_question(X, q)
    np.testing.assert_allclose(C[:5].mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(C[5:].mean(0), 0, atol=1e-12)


def test_auroc_with_null_detects_a_planted_direction():
    rng = np.random.default_rng(0)
    d = 64
    u = A.random_units(1, d, seed=5)[0]
    y = np.array([0] * 100 + [1] * 100)
    X = rng.normal(size=(200, d)) + np.outer(y, u) * 2.0
    res = A.auroc_with_null(X, y, u, n_null=100)
    assert res["auroc"] > 0.9 > res["null_p95"]
    assert res["p_value"] < 0.05


def test_cos_null_scales_like_inverse_sqrt_d():
    assert A.cos_null_p95(900) == pytest.approx(2 / np.sqrt(900), rel=0.25)


def test_exact_permutation_p_is_small_for_perfect_order_and_large_for_unrelated_values():
    x = np.arange(6.0)
    assert A.exact_perm_p_pearson(x, x * 2 + 1) == pytest.approx(2 / 720)  # r = +1 and its mirror r = -1
    assert A.exact_perm_p_pearson(x, np.array([0.0, 1, 0, 1, 0, 1])) > 0.05
