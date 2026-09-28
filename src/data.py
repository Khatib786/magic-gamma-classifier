"""Data loading for the UCI MAGIC Gamma Telescope dataset.

There is deliberately no synthetic fallback: if the real data cannot be
obtained, a RuntimeError is raised so fake numbers can never be reported.
"""
from __future__ import annotations

import io
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .config import COLUMNS, UCI_ID, UCI_URL
from .logging_utils import get_logger

logger = get_logger()

LOCAL_NAMES = ("magic04.data", "magic04.csv", "MAGIC04.data",
               "data/magic04.data", "data/magic04.csv")


def load_magic(path: str | None = None) -> pd.DataFrame:
    """Load MAGIC data: raw Hillas params plus a 0/1 ``target`` column
    (1 = gamma signal, 0 = hadron background).

    Resolution order: explicit path, a local magic04 file (working directory
    or ``data/``), the ucimlrepo package, then a direct UCI download.
    """
    df = None

    if path and Path(path).exists():
        logger.info(f"Reading from explicit path: {path}")
        df = _read_any(path)

    if df is None:
        for name in LOCAL_NAMES:
            if Path(name).exists():
                logger.info(f"Found local file: {name}")
                df = _read_any(name)
                break

    if df is None:
        df = _download()

    return _normalise(df)


def _download(max_retries: int = 2) -> pd.DataFrame:
    """Fetch the dataset from UCI, raising RuntimeError on failure."""
    for attempt in range(max_retries):
        try:
            try:
                from ucimlrepo import fetch_ucirepo
                logger.info(f"Fetching from UCI via ucimlrepo (id={UCI_ID})...")
                ds = fetch_ucirepo(id=UCI_ID)
                feats = ds.data.features.copy()
                feats.columns = COLUMNS[:-1]
                feats["class"] = ds.data.targets.iloc[:, 0].values
                return feats
            except Exception as e:
                logger.warning(f"ucimlrepo unavailable: {e}")

            logger.info("Downloading magic04.data directly...")
            req = urllib.request.Request(
                UCI_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
            return pd.read_csv(io.StringIO(raw), header=None, names=COLUMNS)
        except Exception as e:
            logger.error(f"Download attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise RuntimeError(
                    "Failed to fetch UCI MAGIC dataset after multiple "
                    "attempts. Check your internet connection or download "
                    "magic04.data manually into the data/ folder."
                ) from e
            time.sleep(2)
    raise RuntimeError("unreachable")  # pragma: no cover


def _read_any(path: str) -> pd.DataFrame:
    """Read a MAGIC file whether or not it carries a header row."""
    df = pd.read_csv(path)
    # The raw UCI file has no header, so pandas would swallow the first data
    # row as column names. A numeric "column name" reveals that case.
    try:
        float(str(df.columns[0]))
        headerless = True
    except ValueError:
        headerless = False
    if headerless:
        df = pd.read_csv(path, header=None, names=COLUMNS)
    return df


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Map the textual class label to a 0/1 ``target`` and drop the original."""
    df = df.copy()
    if "label" in df.columns and "class" in df.columns:
        df = df.drop(columns=["label"])

    label_col = "class" if "class" in df.columns else "label"
    if label_col not in df.columns:
        raise ValueError("No class/label column found in the data.")

    vals = df[label_col].astype(str).str.strip().str.lower()
    mapping = {"g": 1, "gamma": 1, "1": 1, "1.0": 1,
               "h": 0, "hadron": 0, "0": 0, "0.0": 0}
    df["target"] = vals.map(mapping)
    if df["target"].isna().any():
        df["target"] = pd.to_numeric(df[label_col], errors="coerce").astype(int)

    return df.drop(columns=[label_col]).reset_index(drop=True)
