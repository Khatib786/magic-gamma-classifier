"""Shared fixtures. The synthetic data here exists ONLY for unit tests."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from magic_classifier.config import COLUMNS  # noqa: E402


def make_fake_magic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    n_g = int(n * 0.65)
    n_h = n - n_g

    def block(m, gamma):
        s = 1.0 if gamma else 1.6
        return pd.DataFrame({
            "fLength": rng.gamma(5, 8 * s, m) + 1,
            "fWidth": rng.gamma(3, 4 * s, m) + 1,
            "fSize": rng.normal(2.5, 0.3, m),
            "fConc": rng.beta(5 if gamma else 3, 4, m) + 0.01,
            "fConc1": rng.beta(3, 5, m) + 0.01,
            "fAsym": rng.normal(0, 25 * s, m),
            "fM3Long": rng.normal(0, 30 * s, m),
            "fM3Trans": rng.normal(0, 12, m),
            "fAlpha": np.abs(rng.normal(0, 12, m)) if gamma else rng.uniform(0, 90, m),
            "fDist": rng.normal(180, 50, m),
        })

    df = pd.concat([block(n_g, True), block(n_h, False)], ignore_index=True)
    df["class"] = ["g"] * n_g + ["h"] * n_h
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)[COLUMNS]


@pytest.fixture
def fake_df():
    return make_fake_magic()


@pytest.fixture
def fake_file(tmp_path, fake_df):
    """Headerless file, exactly like the raw UCI magic04.data."""
    p = tmp_path / "magic04.data"
    fake_df.to_csv(p, header=False, index=False)
    return p
