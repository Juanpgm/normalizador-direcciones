"""Simulate the ambiguity abstention (``ambiguity_delta``) on the harness rows.

Reads ``artifacts/strict_eval_<tag>_rows.parquet`` (which carries ``margen``) and,
for each delta, reports how many OK rows would be abstained, how many of those
with ground truth were right or wrong (manzana / predio) and the resulting pooled
OK count and precisions. It also reports the precision of OK rows by margin bin,
which is the direct evidence of whether the margin carries signal. Simulation
only: no behaviour changes and no coordinates are used as a correctness criterion.

Usage
-----
    PYTHONPATH=src python scripts/margin_sweep.py --tag paso4
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELTAS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)
#: (label, lower bound inclusive, upper bound exclusive); "no competitor" is margen NaN.
BINS = (("no competitor", None, None), (">=0.10", 0.10, np.inf), ("0.05-0.10", 0.05, 0.10),
        ("0.02-0.05", 0.02, 0.05), ("<0.02", -np.inf, 0.02))


def _precision(rows: pd.DataFrame, col: str) -> float:
    values = rows[col].dropna().astype(float)
    return float(values.mean()) if len(values) else float("nan")


def sweep(rows: pd.DataFrame, deltas=DELTAS) -> pd.DataFrame:
    ok = rows[rows["estado"] == "OK"]
    out = []
    for delta in deltas:
        abst = ok["margen"].notna() & (ok["margen"] < delta) if delta > 0 else pd.Series(False, index=ok.index)
        gone, kept = ok[abst], ok[~abst]
        out.append({
            "section": "delta", "label": delta,
            "n_ok": len(kept), "n_abstained": len(gone),
            "abst_with_gt": int(gone["has_gt"].astype(bool).sum()),
            "abst_manzana_correct": int((gone["correct_manzana"] == 1).sum()),
            "abst_manzana_wrong": int((gone["correct_manzana"] == 0).sum()),
            "abst_predial_correct": int((gone["correct_predial"] == 1).sum()),
            "abst_predial_wrong": int((gone["correct_predial"] == 0).sum()),
            "manzana_precision": _precision(kept, "correct_manzana"),
            "predial_precision": _precision(kept, "correct_predial"),
        })
    for label, lo, hi in BINS:
        m = ok["margen"]
        sel = m.isna() if lo is None else (m >= lo) & (m < hi)
        part = ok[sel]
        out.append({
            "section": "bin", "label": label, "n_ok": len(part),
            "n_gt": int(part["has_gt"].astype(bool).sum()),
            "manzana_precision": _precision(part, "correct_manzana"),
            "predial_precision": _precision(part, "correct_predial"),
        })
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="paso4")
    ap.add_argument("--artifacts", default=os.path.join(ROOT, "artifacts"))
    args = ap.parse_args(argv)
    rows = pd.read_parquet(os.path.join(args.artifacts, f"strict_eval_{args.tag}_rows.parquet"))
    result = sweep(rows)
    path = os.path.join(args.artifacts, f"strict_eval_{args.tag}_margin_sweep.csv")
    result.to_csv(path, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(result.to_string(index=False))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
