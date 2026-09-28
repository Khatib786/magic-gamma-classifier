"""Console + rotating-file logging."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from .config import LOG_DIR


def get_logger(name: str = "magic_classifier") -> logging.Logger:
    """Return a logger with console and rotating file handlers."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    LOG_DIR.mkdir(exist_ok=True)
    fh = RotatingFileHandler(LOG_DIR / "experiment.log",
                             maxBytes=5 * 1024 * 1024, backupCount=3)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    return logger
