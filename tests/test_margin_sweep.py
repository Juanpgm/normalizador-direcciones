"""The delta sweep simulation (scripts/margin_sweep.py)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from margin_sweep import sweep  # noqa: E402


def _rows():
    return pd.DataFrame({
        "estado": ["OK", "OK", "OK", "OK", "SIN_MATCH", "OK"],
        "margen": [0.004, 0.015, np.nan, 0.2, np.nan, 0.03],
        "has_gt": [True, True, True, False, True, True],
        "correct_manzana": [0.0, 1.0, 1.0, np.nan, np.nan, 1.0],
        "correct_predial": [0.0, 0.0, 1.0, np.nan, np.nan, 1.0],
    })


def test_delta_zero_abstains_nothing():
    row = sweep(_rows(), deltas=(0.0,)).iloc[0]
    assert row["n_ok"] == 5 and row["n_abstained"] == 0


def test_delta_counts_abstained_correct_and_wrong_and_ignores_non_ok_and_no_competitor():
    out = sweep(_rows(), deltas=(0.02,))
    row = out[out["section"] == "delta"].iloc[0]
    assert row["n_abstained"] == 2 and row["n_ok"] == 3
    assert (row["abst_manzana_wrong"], row["abst_manzana_correct"]) == (1, 1)
    assert (row["abst_predial_wrong"], row["abst_predial_correct"]) == (2, 0)
    assert row["manzana_precision"] == 1.0 and row["predial_precision"] == 1.0


def test_margin_equal_to_delta_is_not_abstained():
    out = sweep(_rows(), deltas=(0.03,))
    assert out.iloc[0]["n_abstained"] == 2  # 0.004 and 0.015, not the 0.03 row


def test_bins_partition_the_ok_rows():
    out = sweep(_rows(), deltas=())
    bins = out[out["section"] == "bin"]
    assert int(bins["n_ok"].sum()) == 5
    assert dict(zip(bins["label"], bins["n_ok"]))["no competitor"] == 1
