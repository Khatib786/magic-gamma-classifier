import numpy as np
import pytest

from magic_classifier.metrics import (best_q_factor, full_report, q_factor,
                                      threshold_at_fpr, tpr_at_fpr)


def test_perfect_classifier():
    y = np.array([0] * 500 + [1] * 500)
    s = np.concatenate([np.linspace(0, 0.4, 500), np.linspace(0.6, 1, 500)])
    r = full_report(y, s)
    assert r["auc"] == pytest.approx(1.0)
    assert r["tpr_at_fpr"] == pytest.approx(1.0)


def test_random_classifier_auc_near_half():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 5000)
    s = rng.random(5000)
    assert full_report(y, s)["auc"] == pytest.approx(0.5, abs=0.03)


def test_q_factor_formula():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1, 0.1, 0.1])
    r = q_factor(y, s, threshold=0.5)
    # eps_s = 3/4, eps_b = 1/4  ->  Q = 0.75 / sqrt(0.25) = 1.5
    assert r["q"] == pytest.approx(1.5, rel=1e-3)


def test_threshold_respects_fpr():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 4000)
    s = rng.random(4000) + y * 0.5
    thr = threshold_at_fpr(y, s, 0.01)
    fpr = np.mean(s[y == 0] >= thr)
    assert fpr <= 0.01 + 1e-9


def test_best_q_at_least_fixed_q():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 4000)
    s = rng.random(4000) + y * 0.8
    assert best_q_factor(y, s)["q"] >= q_factor(y, s)["q"] - 1e-6


def test_tpr_at_fpr_monotone():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 4000)
    s = rng.random(4000) + y * 0.8
    assert tpr_at_fpr(y, s, 0.1) >= tpr_at_fpr(y, s, 0.01)
