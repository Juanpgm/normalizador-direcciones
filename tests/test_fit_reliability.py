"""Offline helpers of scripts/fit_reliability.py (metrics and the sign-constrained fit)."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

pytest.importorskip("sklearn")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("fit_reliability", os.path.join(ROOT, "scripts", "fit_reliability.py"))
fr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fr)


def test_ece_perfectly_calibrated_bins_is_zero():
    p = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([0, 0, 1, 1])
    assert fr.ece(p, y, bins=2) == pytest.approx(0.0)


def test_ece_fully_overconfident_is_large():
    assert fr.ece(np.full(10, 0.99), np.zeros(10), bins=5) == pytest.approx(0.99)


def test_ece_empty_and_fewer_rows_than_bins_do_not_raise():
    assert np.isnan(fr.ece(np.array([]), np.array([]), bins=5))
    # singleton bins: |0.5-1| and |0.5-0| average to 0.5; empty bins are skipped, no NaN
    assert fr.ece(np.array([0.5, 0.5]), np.array([1, 0]), bins=5) == pytest.approx(0.5)


def test_metrics_single_class_gives_nan_auc_but_finite_rest():
    m = fr.metrics(np.array([0.2, 0.9, 0.5]), np.array([1, 1, 1]))
    assert np.isnan(m["auc"]) and np.isfinite(m["brier"]) and np.isfinite(m["logloss"])


def test_fit_recovers_positive_score_signal_and_standardizes():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.7, 1.0, 400)
    y = (rng.uniform(size=400) < (x - 0.7) / 0.3).astype(int)
    X = np.column_stack([x, rng.normal(size=400)])
    fit = fr.fit_logistic(X, y, C=1.0, score_index=0)
    assert fit["coef"][0] > 0 and fit["dropped_score"] is False


def test_fit_negative_score_coefficient_is_clamped_to_zero_and_flagged():
    rng = np.random.default_rng(1)
    x = rng.uniform(0.7, 1.0, 400)
    y = (rng.uniform(size=400) < (1.0 - (x - 0.7) / 0.3)).astype(int)  # inverse relation
    X = np.column_stack([x, rng.normal(size=400)])
    fit = fr.fit_logistic(X, y, C=1.0, score_index=0)
    assert fit["coef"][0] == 0.0 and fit["dropped_score"] is True


def test_fit_single_class_returns_finite_constant_model():
    X = np.random.default_rng(2).normal(size=(20, 3))
    fit = fr.fit_logistic(X, np.ones(20, dtype=int), C=1.0, score_index=0)
    assert np.all(np.isfinite(fit["coef"])) and np.isfinite(fit["intercept"])
    assert np.all(fit["coef"] == 0.0)


def test_standardize_constant_column_has_unit_std_and_zero_z():
    X = np.column_stack([np.ones(5), np.arange(5.0)])
    mean, std = fr.fit_scaler(X)
    assert std[0] == 1.0 and np.all(fr.apply_scaler(X, mean, std)[:, 0] == 0.0)


def test_apply_scaler_clips_to_five_sigma():
    Z = fr.apply_scaler(np.array([[1000.0]]), np.array([0.0]), np.array([1.0]))
    assert Z[0, 0] == 5.0


def test_fit_drops_features_with_too_little_support():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.uniform(0.7, 1.0, n)
    rare = np.zeros(n)
    rare[:3] = 1.0  # only 3 non-zero rows: cannot support a coefficient
    y = (rng.uniform(size=n) < 0.6).astype(int)
    y[:3] = 0
    X = np.column_stack([x, rare])
    mean, std = fr.fit_scaler(X)
    fit = fr.fit_logistic(fr.apply_scaler(X, mean, std), y, C=3.0, score_index=0,
                          support=(X != 0).sum(axis=0), min_support=10)
    assert fit["coef"][1] == 0.0


def test_fit_with_every_feature_dropped_is_a_finite_constant_model():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(50, 1))
    y = (X[:, 0] < 0).astype(int)  # negative relation -> score dropped -> no columns left
    fit = fr.fit_logistic(X, y, C=1.0, score_index=0)
    assert fit["dropped_score"] is True and fit["coef"][0] == 0.0 and np.isfinite(fit["intercept"])


# ---------------------------------------------------------------------------
# cell model: shrinkage, monotonicity, fit, out-of-fold
# ---------------------------------------------------------------------------
def test_shrink_empty_cell_returns_the_parent_rate():
    assert fr.shrink(0, 0, 0.42, 20) == pytest.approx(0.42)


def test_shrink_matches_hand_computation():
    # (3 + 4 * 0.5) / (4 + 4) = 0.625
    assert fr.shrink(3, 4, 0.5, 4) == pytest.approx(0.625)


def test_shrink_large_n_approaches_the_observed_rate():
    assert fr.shrink(700, 1000, 0.1, 20) == pytest.approx(0.7, abs=0.02)


def test_shrink_zero_prior_strength_is_the_raw_rate():
    assert fr.shrink(1, 4, 0.9, 0) == pytest.approx(0.25)
    assert fr.shrink(0, 0, 0.9, 0) == pytest.approx(0.9)  # no data and no prior: parent


def test_pav_leaves_a_monotone_sequence_alone():
    assert fr.pav_increasing([0.1, 0.2, 0.2, 0.9], [1, 1, 1, 1]) == pytest.approx([0.1, 0.2, 0.2, 0.9])


def test_pav_pools_violators_with_weights():
    assert fr.pav_increasing([0.5, 0.4, 0.6], [1, 1, 1]) == pytest.approx([0.45, 0.45, 0.6])
    assert fr.pav_increasing([0.8, 0.2], [3, 1]) == pytest.approx([0.65, 0.65])


def test_pav_cascading_violations_and_edge_sizes():
    assert fr.pav_increasing([0.9, 0.5, 0.1], [1, 1, 1]) == pytest.approx([0.5, 0.5, 0.5])
    assert fr.pav_increasing([], []) == []
    assert fr.pav_increasing([0.3], [2]) == [0.3]


def _arr(*xs):
    return np.array(xs, dtype=object)


def test_fit_cells_hand_computed_table_with_pooled_bands():
    kinds = _arr(*["exacto"] * 4)
    bands = _arr("0.05-0.10", "0.05-0.10", "holgado", "holgado")
    y_p = np.array([1, 1, 0, 0])
    table = fr.fit_cell_table(kinds, bands, y_p, np.ones(4, dtype=int), m=2)
    cells = {(c["kind"], c["band"]): c for c in table["cells"]}
    assert table["global"]["predio"] == pytest.approx(0.5)
    # the raw shrunk rates 0.75 (0.05-0.10) and 0.25 (holgado) violate monotonicity -> pooled
    assert cells[("exacto", "0.05-0.10")]["predio"] == pytest.approx(0.5)
    assert cells[("exacto", "holgado")]["predio"] == pytest.approx(0.5)
    assert cells[("exacto", "<0.02")]["predio"] == pytest.approx(0.5)  # empty -> parent
    assert cells[("exacto", "holgado")]["n"] == 2 and cells[("exacto", "holgado")]["hits_predio"] == 0


def test_fit_cells_emits_every_kind_and_band_even_when_empty():
    table = fr.fit_cell_table(_arr("exacto"), _arr("holgado"), np.array([1]), np.array([1]), m=20)
    assert len(table["cells"]) == 16
    empty = [c for c in table["cells"] if c["n"] == 0]
    assert len(empty) == 15
    assert all(0.0 <= c["predio"] <= c["manzana"] <= 1.0 for c in table["cells"])


def test_fit_cells_is_monotone_within_each_kind_and_manzana_dominates_predio():
    rng = np.random.default_rng(7)
    n = 400
    kinds = rng.choice(np.array(["exacto", "relajada", "compartida", "parcial"], dtype=object), n)
    bands = rng.choice(np.array(["<0.02", "0.02-0.05", "0.05-0.10", "holgado"], dtype=object), n)
    y_p = (rng.uniform(size=n) < 0.5).astype(int)
    y_m = np.maximum(y_p, (rng.uniform(size=n) < 0.6).astype(int))
    table = fr.fit_cell_table(kinds, bands, y_p, y_m, m=10)
    order = ["<0.02", "0.02-0.05", "0.05-0.10", "holgado"]
    for kind in ("exacto", "relajada", "compartida", "parcial"):
        for target in ("predio", "manzana"):
            vals = [next(c[target] for c in table["cells"] if c["kind"] == kind and c["band"] == b) for b in order]
            assert vals == sorted(vals)
    assert all(c["manzana"] >= c["predio"] for c in table["cells"])
    assert all(v["manzana"] >= v["predio"] for v in table["kinds"].values())


def test_fit_cells_manzana_is_lifted_to_predio_when_labels_disagree():
    table = fr.fit_cell_table(_arr("exacto", "exacto"), _arr("holgado", "holgado"),
                              np.array([1, 1]), np.array([0, 0]), m=1)
    cell = next(c for c in table["cells"] if c["kind"] == "exacto" and c["band"] == "holgado")
    assert cell["manzana"] == cell["predio"]


def test_fit_cells_empty_training_set_is_finite():
    table = fr.fit_cell_table(_arr(), _arr(), np.array([], dtype=int), np.array([], dtype=int), m=20)
    assert np.isfinite(table["global"]["predio"]) and len(table["cells"]) == 16


def test_predict_cells_uses_the_table_with_kind_and_global_fallbacks():
    table = {"global": {"predio": 0.5, "manzana": 0.7},
             "kinds": {"exacto": {"predio": 0.6, "manzana": 0.8}},
             "cells": [{"kind": "exacto", "band": "holgado", "predio": 0.9, "manzana": 0.95}]}
    kinds = _arr("exacto", "exacto", "parcial")
    bands = _arr("holgado", "<0.02", "holgado")
    pp, pm = fr.predict_cells(table, kinds, bands)
    assert pp.tolist() == [0.9, 0.6, 0.5] and pm.tolist() == [0.95, 0.8, 0.7]


def test_cell_oof_never_sees_the_held_out_dataset():
    rng = np.random.default_rng(11)
    n = 300
    kinds = rng.choice(np.array(["exacto", "relajada"], dtype=object), n)
    bands = rng.choice(np.array(["holgado", "0.02-0.05"], dtype=object), n)
    groups = rng.choice(np.array(["a", "b", "c"], dtype=object), n)
    y_p = (rng.uniform(size=n) < 0.5).astype(int)
    y_m = np.maximum(y_p, (rng.uniform(size=n) < 0.5).astype(int))
    base = fr.cell_oof(kinds, bands, y_p, y_m, groups)
    flipped_p = np.where(groups == "a", 1 - y_p, y_p)
    flipped_m = np.where(groups == "a", 1 - y_m, y_m)
    other = fr.cell_oof(kinds, bands, flipped_p, flipped_m, groups)
    a = groups == "a"
    assert np.array_equal(base["predio"][a], other["predio"][a])  # held-out labels never used
    assert np.array_equal(base["manzana"][a], other["manzana"][a])
    assert set(base["chosen_m"].values()) <= set(fr.M_GRID)
    assert np.all(base["manzana"] >= base["predio"])


def test_cell_oof_single_group_falls_back_to_global_without_raising():
    out = fr.cell_oof(_arr("exacto", "exacto"), _arr("holgado", "holgado"),
                      np.array([1, 0]), np.array([1, 1]), _arr("a", "a"))
    assert np.all(np.isfinite(out["predio"]))


def test_ship_rule_brier_tolerance_and_auc():
    ok = {"brier": 0.2410, "auc": 0.60}
    base = {"brier": 0.2409, "auc": 0.49}
    assert fr.ships({"predio": ok, "manzana": ok}, {"predio": base, "manzana": base})
    worse_brier = {"brier": 0.2409 + 0.0021, "auc": 0.60}
    assert not fr.ships({"predio": worse_brier, "manzana": ok}, {"predio": base, "manzana": base})
    lower_auc = {"brier": 0.24, "auc": 0.48}
    assert not fr.ships({"predio": ok, "manzana": lower_auc}, {"predio": base, "manzana": base})
    # exactly at the tolerance boundary is allowed
    edge = {"brier": 0.2409 + 0.002, "auc": 0.5}
    assert fr.ships({"predio": edge, "manzana": ok}, {"predio": base, "manzana": base})
    nan_auc = {"brier": 0.24, "auc": float("nan")}
    assert not fr.ships({"predio": nan_auc, "manzana": ok}, {"predio": base, "manzana": base})


def _synthetic_rows(path, seed=5):
    import pandas as pd

    from cali_address.reliability import FEATURE_NAMES

    rng = np.random.default_rng(seed)
    n = 180
    df = pd.DataFrame({f"feat_{name}": rng.uniform(0, 1, n) for name in FEATURE_NAMES})
    df["feat_margen_manzana"] = rng.choice([0.03, 0.07, 0.3], n)
    for col in ("relaxed_letter", "relaxed_plate", "nivel_esquina", "nivel_via", "nivel_manzana", "nivel_direccion"):
        df[f"feat_{col}"] = (rng.uniform(size=n) < 0.15).astype(float)
    df["dataset"] = np.repeat(["a", "b", "c"], n // 3)
    df["estado"] = "OK"
    df["has_gt"] = True
    df["confianza"] = rng.uniform(0.6, 1.0, n)
    df["correct_predial"] = (rng.uniform(size=n) < 0.5).astype(float)
    df["correct_manzana"] = np.maximum(df["correct_predial"], (rng.uniform(size=n) < 0.6).astype(float))
    df["motivo"] = ""
    df.to_parquet(path, index=False)


def test_run_cells_force_writes_a_loadable_artifact(tmp_path):
    import argparse

    from cali_address.reliability import CellReliability, load_reliability

    rows, out = tmp_path / "rows.parquet", tmp_path / "reliability.json"
    _synthetic_rows(rows)
    fr.run_cells(argparse.Namespace(rows=str(rows), out=str(out), force=True))
    model = load_reliability(str(out))
    assert isinstance(model, CellReliability) and model.version.startswith("reliability-v3-")
    import json

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["kind"] == "cells" and payload["m"] in fr.M_GRID and len(payload["cells"]) == 16
    assert isinstance(payload["ship_rule_passed"], bool)


def test_run_cells_without_force_never_writes_when_the_rule_fails(tmp_path, monkeypatch):
    import argparse

    rows, out = tmp_path / "rows.parquet", tmp_path / "reliability.json"
    _synthetic_rows(rows)
    monkeypatch.setattr(fr, "ships", lambda *_: False)
    assert fr.run_cells(argparse.Namespace(rows=str(rows), out=str(out), force=False)) == 2
    assert not out.exists()
