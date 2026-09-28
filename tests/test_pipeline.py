import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from magic_classifier.models import (base_estimators, compare_features,
                                     make_pipeline, paired_ttest)


def split(df):
    y = (df["class"] == "g").astype(int).values
    return df.drop(columns=["class"]), y


def test_pipeline_fits_and_predicts(fake_df):
    X, y = split(fake_df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0, stratify=y)
    for add in (False, True):
        pipe = make_pipeline(RandomForestClassifier(n_estimators=30,
                                                    random_state=0), add)
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        assert p.shape == (len(Xte),) and ((p >= 0) & (p <= 1)).all()


def test_feature_names_propagate(fake_df):
    X, y = split(fake_df)
    pipe = make_pipeline(RandomForestClassifier(n_estimators=10, random_state=0))
    pipe.fit(X, y)
    assert len(pipe[:-1].get_feature_names_out()) == 16


def test_paired_ttest_detects_shift():
    a = np.array([0.90, 0.91, 0.92, 0.93, 0.94])
    b = a - 0.05 + np.array([0.001, -0.001, 0.0, 0.001, -0.001])
    r = paired_ttest(a, b)
    assert r["delta"] > 0.04 and r["p"] < 0.01


def test_compare_features_smoke(fake_df):
    X, y = split(fake_df)
    est = {"rf": RandomForestClassifier(n_estimators=20, random_state=0)}
    out = compare_features(X, y, cv=3, estimators=est)
    assert set(out["rf"]) >= {"auc_raw", "auc_extended", "delta", "p_value"}
    assert len(out["rf"]["raw_folds"]) == 3


def test_lightgbm_early_stopping_estimator(fake_df):
    est = base_estimators()
    assert {"logreg", "gboost", "rforest", "mlp"} <= set(est)
    if "lightgbm" in est:
        X, y = split(fake_df)
        pipe = make_pipeline(est["lightgbm"])
        pipe.fit(X, y)
        assert pipe.predict_proba(X).shape[1] == 2
