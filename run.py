"""Convenience launcher: python run.py [--data PATH] [--tune] [--cv 5]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from magic_classifier.pipeline import cli  # noqa: E402

if __name__ == "__main__":
    cli()
