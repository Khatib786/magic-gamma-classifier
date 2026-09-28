"""Physics-motivated feature engineering for Hillas parameters."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import ENGINEERED, EPS


class HillasFeatureBuilder(BaseEstimator, TransformerMixin):
    """Derive physics-motivated ratios from the raw Hillas parameters.

    Implemented as a transformer so it sits inside the Pipeline and is only
    ever fitted on training folds, keeping the train/test split clean.
    """

    def __init__(self, add_features: bool = True):
        self.add_features = add_features

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns)
        else:
            self.feature_names_in_ = np.asarray(
                [f"x{i}" for i in range(X.shape[1])])
        return self

    def transform(self, X):
        if not self.add_features:
            return X

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        out = X.copy()
        # gammas are more elongated
        out["elongation"] = X["fLength"] / (X["fWidth"] + EPS)
        # hadrons are more asymmetric
        out["asym_ratio"] = X["fAsym"].abs() / (X["fLength"] + EPS)
        # light concentration
        out["conc_ratio"] = X["fConc1"] / (X["fConc"] + EPS)
        # orientation per unit intensity
        out["alpha_size"] = X["fAlpha"] / (X["fSize"].abs() + EPS)
        # ellipse area ~ shower size
        out["area_proxy"] = np.pi * X["fLength"] * X["fWidth"]
        out["m3_ratio"] = X["fM3Long"].abs() / (X["fM3Trans"].abs() + EPS)
        return out

    def get_feature_names_out(self, input_features=None):
        """Required by scikit-learn Pipeline to propagate feature names."""
        if input_features is None:
            input_features = getattr(
                self, "feature_names_in_",
                np.asarray([f"x{i}" for i in range(10)]))
        if not self.add_features:
            return np.asarray(input_features)
        return np.concatenate(
            [np.asarray(input_features), np.asarray(ENGINEERED)])
