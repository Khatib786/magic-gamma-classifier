"""End-to-end experiment: load data, compare feature sets, evaluate, plot.

Run with:  python -m magic_classifier   (or)   python run.py
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .config import RS, ensure_dirs
from .data import load_magic
from .logging_utils import get_logger
from .metrics import full_report
from .models import (HAS_LGB, HAS_OPTUNA, base_estimators, compare_features,
                     make_pipeline, tune_lgb_optuna)
from .visualize import (fig_importance, fig_physics, fig_roc, fig_summary,
                        plot_embedding)

logger = get_logger()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MAGIC gamma/hadron separation experiment")
    p.add_argument("--data", default=None,
                   help="Path to magic04.data (otherwise auto-detected/downloaded)")
    p.add_argument("--out", default="outputs", help="Output directory")
    p.add_argument("--cv", type=int, default=5, help="CV folds (default 5)")
    p.add_argument("--test-size", type=float, default=0.25,
                   help="Hold-out fraction (default 0.25)")
    p.add_argument("--tune", action="store_true",
                   help="Add an Optuna-tuned LightGBM model (needs optuna)")
    p.add_argument("--trials", type=int, default=20,
                   help="Optuna trials when --tune is set")
    p.add_argument("--no-embedding", action="store_true",
                   help="Skip the (slow) t-SNE embedding figure")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config.OUT = Path(args.out)
    ensure_dirs()

    logger.info("MAGIC Gamma Telescope - gamma/hadron separation")
    logger.info(f"LightGBM available: {HAS_LGB}   Optuna available: {HAS_OPTUNA}")
    logger.info("-" * 60)

    df = load_magic(args.data)
    y = df["target"].values
    X = df.drop(columns=["target"])
    logger.info(f"Loaded {len(df)} events, gamma fraction {y.mean():.3f}")

    df.to_csv(config.OUT / "dataset.csv", index=False)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=RS)

    estimators = base_estimators()
    if args.tune:
        tuned = tune_lgb_optuna(X_tr, y_tr, n_trials=args.trials)
        if tuned is not None:
            estimators["lgb_tuned"] = tuned

    logger.info("Do engineered features help? (paired t-test on CV folds)")
    cmp = compare_features(X_tr, y_tr, cv=args.cv, estimators=estimators)
    for name, r in cmp.items():
        flag = "*" if r["significant"] else " "
        logger.info(f"  {name:10s} raw={r['auc_raw']:.4f} "
                    f"ext={r['auc_extended']:.4f}  d={r['delta']:+.4f} "
                    f"p={r['p_value']:.4f} {flag}")
    (config.OUT / "feature_significance.json").write_text(
        json.dumps(cmp, indent=2))

    logger.info("Hold-out evaluation:")
    rows, roc_data = [], {}
    for name, est in estimators.items():
        for add in (False, True):
            pipe = make_pipeline(est, add_features=add)
            pipe.fit(X_tr, y_tr)
            score = pipe.predict_proba(X_te)[:, 1]
            tag = "extended" if add else "raw"
            rows.append({"model": name, "features": tag,
                         **full_report(y_te, score, target_fpr=0.01)})
            roc_data[f"{name}_{tag}"] = (y_te, score)

    res = pd.DataFrame(rows).sort_values("auc", ascending=False)
    res.to_csv(config.OUT / "results.csv", index=False)
    logger.info("\n" + res.to_string(index=False))

    logger.info("Writing figures...")
    fig_roc(roc_data)
    fig_summary(res)
    fig_physics(df)
    fig_importance(X_tr, y_tr)
    if not args.no_embedding:
        plot_embedding(X, y, method="tsne")

    logger.info(f"Artifacts written to {config.OUT.resolve()}")
    for f in sorted(config.OUT.iterdir()):
        logger.info(f"  {f.name:32s} {f.stat().st_size:>8d} bytes")


def cli() -> None:
    warnings.filterwarnings("ignore")
    main()
