"""Fill paper.md result tables from a real experiment run.

Usage:
    python run.py                                   # produces outputs/
    python scripts/make_paper_tables.py             # updates paper.md in place

Replaces everything between the markers
    <!-- RESULTS_TABLE_START -->      ... <!-- RESULTS_TABLE_END -->
    <!-- SIGNIFICANCE_TABLE_START --> ... <!-- SIGNIFICANCE_TABLE_END -->
so it is safe to re-run after every experiment.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def results_table(csv_path: Path) -> str:
    df = pd.read_csv(csv_path).sort_values(["model", "features"])
    lines = [
        "| Model | Features | ROC-AUC | Q @ FPR=0.01 | Best Q | TPR @ FPR=0.01 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['model']} | {r['features']} | {r['auc']:.4f} | "
            f"{r['q_at_fixed_fpr']:.2f} | {r['best_q']:.2f} | "
            f"{r['tpr_at_fpr']:.4f} |")
    return "\n".join(lines)


def significance_table(json_path: Path) -> str:
    d = json.loads(json_path.read_text())
    lines = [
        "| Model | CV AUC (raw) | CV AUC (extended) | Delta | p-value | p < 0.05 |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for name, r in d.items():
        mark = "yes" if r["significant"] else "no"
        lines.append(
            f"| {name} | {r['auc_raw']:.4f} | {r['auc_extended']:.4f} | "
            f"{r['delta']:+.4f} | {r['p_value']:.4f} | {mark} |")
    return "\n".join(lines)


def replace_block(text: str, tag: str, body: str) -> str:
    pat = re.compile(rf"(<!-- {tag}_START -->)(.*?)(<!-- {tag}_END -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"Marker {tag} not found in paper file.")
    return pat.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--paper", default="paper.md")
    a = ap.parse_args()

    out = Path(a.outputs)
    for need in ("results.csv", "feature_significance.json"):
        if not (out / need).exists():
            raise SystemExit(f"{out / need} missing - run `python run.py` first.")

    paper = Path(a.paper)
    text = paper.read_text(encoding="utf-8")
    text = replace_block(text, "RESULTS_TABLE", results_table(out / "results.csv"))
    text = replace_block(text, "SIGNIFICANCE_TABLE",
                         significance_table(out / "feature_significance.json"))
    paper.write_text(text, encoding="utf-8")
    print(f"Updated tables in {paper}")


if __name__ == "__main__":
    main()
