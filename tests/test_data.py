import pandas as pd
import pytest

from magic_classifier.data import _normalise, load_magic


def test_reads_headerless_raw_file(fake_file, fake_df):
    df = load_magic(str(fake_file))
    assert len(df) == len(fake_df)
    assert "target" in df.columns and "class" not in df.columns
    assert set(df["target"].unique()) == {0, 1}
    assert df["target"].mean() == pytest.approx(0.65, abs=0.01)


def test_reads_file_with_header(tmp_path, fake_df):
    p = tmp_path / "with_header.csv"
    fake_df.to_csv(p, index=False)
    df = load_magic(str(p))
    assert len(df) == len(fake_df)
    assert "fLength" in df.columns


def test_label_mapping():
    df = pd.DataFrame({"fLength": [1, 2, 3], "class": ["g", "h", " G "]})
    assert _normalise(df)["target"].tolist() == [1, 0, 1]


def test_missing_label_raises():
    with pytest.raises(ValueError):
        _normalise(pd.DataFrame({"fLength": [1, 2]}))
