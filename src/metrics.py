"""Evaluation metrics, including the gamma-astronomy Q-factor."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def threshold_at_fpr(y_true: np.ndarray, y_score: np.ndarray,
                     target_fpr: float = 0.01) -> float:
    """Tightest score threshold keeping background acceptance <= target_fpr."""
    fpr, _, thr = roc_curve(y_true, y_score)
    ok = np.where(fpr <= target_fpr)[0]
    return float(np.max(thr)) if len(ok) == 0 else float(thr[ok[-1]])


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray,
               target_fpr: float = 0.01) -> float:
    """Signal efficiency retained at a fixed background acceptance."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.interp(target_fpr, fpr, tpr))


def q_factor(y_true: np.ndarray, y_score: np.ndarray,
             threshold: float | None = None, target_fpr: float = 0.01) -> dict:
    """Quality factor Q = eps_signal / sqrt(eps_background)."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    if threshold is None:
        threshold = threshold_at_fpr(y_true, y_score, target_fpr)
    pred = y_score >= threshold
    tp = np.sum((pred == 1) & (y_true == 1))
    fn = np.sum((pred == 0) & (y_true == 1))
    fp = np.sum((pred == 1) & (y_true == 0))
    tn = np.sum((pred == 0) & (y_true == 0))
    eps_s = tp / (tp + fn + 1e-12)
    eps_b = fp / (fp + tn + 1e-12)
    return {"q": eps_s / np.sqrt(eps_b + 1e-12),
            "eps_signal": eps_s, "eps_background": eps_b,
            "threshold": threshold}


def best_q_factor(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Largest Q-factor reachable across all thresholds."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = tpr / np.sqrt(fpr + 1e-12)
    q[~np.isfinite(q)] = 0.0
    valid = fpr > 0
    if not valid.any():
        return {"q": 0.0}
    i = np.argmax(np.where(valid, q, -np.inf))
    return {"q": float(q[i]), "threshold": float(thr[i])}


def full_report(y_true: np.ndarray, y_score: np.ndarray,
                target_fpr: float = 0.01) -> dict:
    """AUC, Q-factor (fixed FPR and best) and TPR at the target FPR."""
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "q_at_fixed_fpr": q_factor(y_true, y_score,
                                   target_fpr=target_fpr)["q"],
        "best_q": best_q_factor(y_true, y_score)["q"],
        "tpr_at_fpr": tpr_at_fpr(y_true, y_score, target_fpr),
        "target_fpr": target_fpr,
    }
