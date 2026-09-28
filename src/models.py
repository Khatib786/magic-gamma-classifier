"""Estimators, the sklearn Pipeline factory, cross-validation and the paired
t-test used to decide whether engineered features help."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RS
from .features import HillasFeatureBuilder
from .logging_utils import get_logger

logger = get_logger()

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:  # pragma: no cover - depends on environment
    lgb = None
    HAS_LGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except Exception:  # pragma: no cover - depends on environment
    optuna = None
    HAS_OPTUNA = False


if HAS_LGB:
    class NestedEarlyStoppingLGBM(lgb.LGBMClassifier):
        """LightGBM with an inner validation split used for early stopping.

        Each outer CV fold carves a stratified 20% validation set out of its
        own training data, so the number of trees is chosen without ever
        touching the outer test fold. LightGBM automatically predicts with
        ``best_iteration_`` after early stopping.
        """

        def fit(self, X, y, **kwargs):
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=RS)
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
            super().fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                        callbacks=callbacks, **kwargs)
            return self


def make_pipeline(estimator, add_features: bool = True) -> Pipeline:
    """Feature builder -> standardiser -> classifier, as one fitted unit."""
    return Pipeline([
        ("features", HillasFeatureBuilder(add_features=add_features)),
        ("scale", StandardScaler()),
        ("clf", estimator),
    ])


def base_estimators() -> dict:
    """The model zoo compared in every experiment."""
    est = {
        "logreg": LogisticRegression(max_iter=2000, C=1.0),
        "gboost": GradientBoostingClassifier(random_state=RS),
        "rforest": RandomForestClassifier(n_estimators=300, random_state=RS,
                                          n_jobs=-1),
        "mlp": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000,
                             random_state=RS, early_stopping=True),
    }
    if HAS_LGB:
        est["lightgbm"] = NestedEarlyStoppingLGBM(
            n_estimators=1000, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=RS, verbose=-1)
    return est


def tune_lgb_optuna(X, y, n_trials: int = 20):
    """Optional: minimal Optuna study returning a tuned LightGBM classifier.

    Returns None when LightGBM or Optuna is not installed.
    """
    if not (HAS_LGB and HAS_OPTUNA):
        logger.warning("Optuna/LightGBM not available; skipping tuning.")
        return None

    logger.info(f"Running {n_trials}-trial Optuna study for LightGBM...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RS)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2,
                                                 log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 90),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": RS, "verbose": -1, "n_jobs": -1,
        }
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_tr, y_tr)
        return roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RS))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(study.best_params, random_state=RS, verbose=-1, n_jobs=-1)
    logger.info(f"Optuna best validation AUC: {study.best_value:.4f}")
    return lgb.LGBMClassifier(**best)


def cv_auc_scores(X: pd.DataFrame, y, estimator, add_features: bool = True,
                  cv: int = 5) -> np.ndarray:
    """Per-fold ROC-AUC, kept fold-aligned for a paired comparison."""
    pipe = make_pipeline(estimator, add_features=add_features)
    skf = StratifiedKFold(cv, shuffle=True, random_state=RS)
    return cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)


def paired_ttest(scores_a, scores_b) -> dict:
    """Paired t-test of two fold-aligned score arrays (a minus b)."""
    a, b = np.asarray(scores_a), np.asarray(scores_b)
    t, p = stats.ttest_rel(a, b)
    return {"delta": float(a.mean() - b.mean()), "t": float(t), "p": float(p)}


def compare_features(X: pd.DataFrame, y, cv: int = 5,
                     estimators: dict | None = None) -> dict:
    """For each model, test whether the engineered features shift CV AUC by a
    statistically meaningful amount rather than by noise."""
    if estimators is None:
        estimators = base_estimators()

    results = {}
    for name, est in estimators.items():
        raw = cv_auc_scores(X, y, est, add_features=False, cv=cv)
        ext = cv_auc_scores(X, y, est, add_features=True, cv=cv)
        test = paired_ttest(ext, raw)
        results[name] = {
            "auc_raw": float(raw.mean()),
            "auc_extended": float(ext.mean()),
            "delta": test["delta"],
            "p_value": test["p"],
            "significant": bool(test["p"] < 0.05),
            "raw_folds": raw.tolist(),
            "ext_folds": ext.tolist(),
        }
    return results
