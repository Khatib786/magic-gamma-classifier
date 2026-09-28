"""All figure-generation code (headless Agg backend)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from . import config  # noqa: E402
from .config import ENGINEERED, RS  # noqa: E402
from .features import HillasFeatureBuilder  # noqa: E402
from .logging_utils import get_logger  # noqa: E402
from .models import make_pipeline  # noqa: E402

logger = get_logger()


def fig_roc(roc_data: dict) -> None:
    plt.figure(figsize=(7, 6))
    for label, (yt, ys) in roc_data.items():
        fpr, tpr, _ = roc_curve(yt, ys)
        plt.plot(fpr, tpr, lw=1.3,
                 label=f"{label} (AUC={roc_auc_score(yt, ys):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("Background acceptance (FPR)")
    plt.ylabel("Signal efficiency (TPR)")
    plt.title("ROC curves across model and feature configurations")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(config.OUT / "03_roc_curves.png", dpi=140)
    plt.close()


def fig_summary(res: pd.DataFrame) -> None:
    """Grouped bars using plain matplotlib so output is reliable everywhere."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    def grouped(axis, value_col, title, ylim=None):
        piv = res.pivot_table(index="model", columns="features",
                              values=value_col)
        models, feats = list(piv.index), list(piv.columns)
        x = np.arange(len(models))
        w = 0.8 / max(len(feats), 1)
        for i, f in enumerate(feats):
            axis.bar(x + i * w, piv[f].values, width=w, label=str(f))
        axis.set_xticks(x + w * (len(feats) - 1) / 2)
        axis.set_xticklabels(models, rotation=30, ha="right")
        axis.set_title(title)
        axis.legend()
        if ylim:
            axis.set_ylim(*ylim)

    grouped(ax[0], "auc", "ROC-AUC by model and features", ylim=(0.5, 1.0))
    ax[0].set_ylabel("ROC-AUC")
    grouped(ax[1], "q_at_fixed_fpr", "Q-factor at FPR = 0.01")
    ax[1].set_ylabel("Q-factor")
    plt.tight_layout()
    plt.savefig(config.OUT / "04_summary.png", dpi=140)
    plt.close()


def fig_physics(df: pd.DataFrame) -> None:
    builder = HillasFeatureBuilder().fit(df.drop(columns=["target"]))
    fdf = builder.transform(df.drop(columns=["target"])).copy()
    fdf["target"] = df["target"].values
    feats = [f for f in ENGINEERED if f in fdf.columns][:6]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, f in zip(axes.ravel(), feats):
        for t, lbl, c in [(1, "gamma", "tab:blue"), (0, "hadron", "tab:red")]:
            d = fdf[fdf.target == t][f]
            lo, hi = d.quantile([0.01, 0.99])
            ax.hist(d.clip(lo, hi), bins=50, alpha=0.5, label=lbl, color=c,
                    density=True)
        ax.set_title(f)
        ax.legend(fontsize=7)
    plt.suptitle("Engineered features: gamma vs hadron separation")
    plt.tight_layout()
    plt.savefig(config.OUT / "02_physics_features.png", dpi=140)
    plt.close()


def fig_importance(X_tr: pd.DataFrame, y_tr) -> None:
    pipe = make_pipeline(
        RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RS))
    pipe.fit(X_tr, y_tr)

    feature_names = pipe[:-1].get_feature_names_out()
    imp = pipe.named_steps["clf"].feature_importances_

    order = np.argsort(imp)
    colors = ["tab:red" if name in ENGINEERED else "tab:gray"
              for name in feature_names[order]]

    plt.figure(figsize=(8, 7))
    plt.barh(feature_names[order], imp[order], color=colors)
    plt.title("Random-forest feature importance (engineered features in red)")
    plt.tight_layout()
    plt.savefig(config.OUT / "05_feature_importance.png", dpi=140)
    plt.close()


def plot_embedding(X: pd.DataFrame, y: np.ndarray, method: str = "tsne") -> None:
    """2D projection of the raw vs engineered feature space."""
    logger.info(f"Generating {method.upper()} embedding for feature "
                "visualization...")

    builder = HillasFeatureBuilder(add_features=True)
    X_eng = builder.fit_transform(X)

    # Separate scalers so each space is standardised on its own statistics.
    X_raw_scaled = np.asarray(StandardScaler().fit_transform(X))
    X_eng_scaled = np.asarray(StandardScaler().fit_transform(X_eng))

    def project(arr):
        if method == "tsne":
            pca = PCA(n_components=min(50, arr.shape[1]), random_state=RS)
            reduced = np.asarray(pca.fit_transform(arr))
            return np.asarray(TSNE(n_components=2, random_state=RS, init="pca",
                                   learning_rate="auto").fit_transform(reduced))
        return np.asarray(PCA(n_components=2, random_state=RS).fit_transform(arr))

    emb_raw, emb_eng = project(X_raw_scaled), project(X_eng_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, emb, title in zip(axes, [emb_raw, emb_eng],
                              ["Raw Hillas Parameters",
                               "Engineered Feature Space"]):
        for target, label, color in [(1, "gamma", "tab:blue"),
                                     (0, "hadron", "tab:red")]:
            mask = y == target
            ax.scatter(emb[mask, 0], emb[mask, 1], c=color, label=label,
                       alpha=0.5, s=10, edgecolors="none")
        ax.set_title(f"{title} ({method.upper()})")
        ax.legend()
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    plt.tight_layout()
    plt.savefig(config.OUT / "06_feature_embedding.png", dpi=140)
    plt.close()
    logger.info(f"Saved feature embedding plot to "
                f"{config.OUT / '06_feature_embedding.png'}")
