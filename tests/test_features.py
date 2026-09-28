import numpy as np
import pandas as pd

from magic_classifier.config import ENGINEERED
from magic_classifier.features import HillasFeatureBuilder


def raw_X(df):
    return df.drop(columns=["class"])


def test_adds_all_engineered_columns(fake_df):
    out = HillasFeatureBuilder().fit_transform(raw_X(fake_df))
    assert list(out.columns[-6:]) == ENGINEERED
    assert out.shape == (len(fake_df), 10 + len(ENGINEERED))


def test_disabled_is_identity(fake_df):
    X = raw_X(fake_df)
    out = HillasFeatureBuilder(add_features=False).fit_transform(X)
    pd.testing.assert_frame_equal(out, X)


def test_values_match_definitions(fake_df):
    X = raw_X(fake_df)
    out = HillasFeatureBuilder().fit_transform(X)
    np.testing.assert_allclose(out["elongation"], X.fLength / (X.fWidth + 1e-9))
    np.testing.assert_allclose(out["area_proxy"], np.pi * X.fLength * X.fWidth)
    assert (out["asym_ratio"] >= 0).all()
    assert np.isfinite(out.values).all()


def test_feature_names_out(fake_df):
    b = HillasFeatureBuilder().fit(raw_X(fake_df))
    names = b.get_feature_names_out()
    assert len(names) == 16 and names[-1] == "m3_ratio"
