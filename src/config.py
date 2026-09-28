"""Global constants and paths shared by all modules."""
from __future__ import annotations

from pathlib import Path

EPS = 1e-9
RS = 42  # random seed used everywhere

OUT = Path("outputs")
LOG_DIR = Path("logs")

UCI_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "magic/magic04.data")
UCI_ID = 159

# The ten Hillas parameters plus the class label, in the dataset native order.
COLUMNS = ["fLength", "fWidth", "fSize", "fConc", "fConc1",
           "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]

ENGINEERED = ["elongation", "asym_ratio", "conc_ratio",
              "alpha_size", "area_proxy", "m3_ratio"]


def ensure_dirs() -> None:
    """Create output directories on demand (not at import time)."""
    OUT.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
