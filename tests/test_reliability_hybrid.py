"""Hybrid reliability: isotonic step function on the raw score x per-kind conditional predio rate."""

from __future__ import annotations

import json
import math

import pytest

from cali_address.reliability import (
    HYBRID_CLIP, KINDS, CellReliability, HybridReliability, Reliability, interpolate_knots,
    load_reliability,
)
from cali_address.service import normalize_strict, reliability_features

LO, HI = HYBRID_CLIP


def _f(conf=0.9, nivel="predio", relaxed_letter=False, relaxed_plate=False, n_predios=1) -> dict:
    return reliability_features(conf, 1.0, relaxed_letter, relaxed_plate, n_predios, nivel, False)


def _model(q=None, q_global=0.7) -> HybridReliability:
    return HybridReliability(
        xs=[0.74, 0.85, 0.95], ys=[0.60, 0.80, 0.95],
        q=q if q is not None else {"exacto": 0.80, "relajada": 0.60, "compartida": 0.70, "parcial": 0.50},
        q_global=q_global, version="hybrid-test",
    )


# ---------------------------------------------------------------------------
# interpolation
# ---------------------------------------------------------------------------
def test_interpolation_hits_knots_and_is_linear_between():
    xs, ys = [0.7, 0.8, 1.0], [0.5, 0.6, 0.9]
    assert interpolate_knots(xs, ys, 0.7) == pytest.approx(0.5)
    assert interpolate_knots(xs, ys, 0.75) == pytest.approx(0.55)
    assert interpolate_knots(xs, ys, 0.9) == pytest.approx(0.75)
    assert interpolate_knots(xs, ys, 1.0) == pytest.approx(0.9)


def test_interpolation_clamps_at_both_ends():
    xs, ys = [0.7, 0.8], [0.5, 0.6]
    assert interpolate_knots(xs, ys, -5.0) == 0.5
    assert interpolate_knots(xs, ys, 9.0) == 0.6
    assert interpolate_knots(xs, ys, float("inf")) == 0.6
    assert interpolate_knots(xs, ys, float("-inf")) == 0.5


def test_interpolation_single_knot_is_constant():
    assert interpolate_knots([0.8], [0.66], 0.1) == 0.66
    assert interpolate_knots([0.8], [0.66], 0.99) == 0.66


def test_interpolation_is_non_decreasing_on_a_fine_grid():
    xs, ys = [0.7, 0.75, 0.9, 1.0], [0.3, 0.3, 0.7, 0.9]  # includes a flat stretch
    grid = [0.6 + i * 0.001 for i in range(500)]
    values = [interpolate_knots(xs, ys, g) for g in grid]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------
def test_manzana_is_the_interpolated_isotonic_value():
    pp, pm = _model().predict(_f(conf=0.85))
    assert pm == pytest.approx(0.80)
    assert pp == pytest.approx(0.80 * 0.80)  # exacto q


def test_manzana_is_non_decreasing_in_confianza():
    m = _model()
    pms = [m.predict(_f(conf=0.70 + i * 0.005))[1] for i in range(70)]
    assert all(b >= a - 1e-12 for a, b in zip(pms, pms[1:]))


def test_output_clipped_to_bounds():
    m = HybridReliability(xs=[0.7, 0.9], ys=[0.0, 1.0], q={}, q_global=1.0)
    assert m.predict(_f(conf=0.1))[1] == LO
    assert m.predict(_f(conf=5.0))[1] == HI
    pp, pm = m.predict(_f(conf=5.0))
    assert pp <= pm <= HI and pp >= 0.001


def test_predio_never_exceeds_manzana_on_any_row():
    m = _model()
    for conf in (0.0, 0.5, 0.74, 0.8, 0.9, 1.0, 3.0):
        for kw in (dict(), dict(relaxed_plate=True), dict(nivel="esquina"), dict(nivel="manzana", n_predios=3)):
            pp, pm = m.predict(_f(conf=conf, **kw))
            assert 0.0 < pp <= pm <= HI


def test_relaxed_row_scores_lower_than_exact_at_same_confianza():
    m = _model()
    exact = m.predict(_f(conf=0.9))
    relaxed = m.predict(_f(conf=0.9, relaxed_letter=True))
    assert relaxed[1] == exact[1]  # manzana depends on the score only
    assert relaxed[0] < exact[0]


@pytest.mark.parametrize("kwargs,kind", [
    (dict(), "exacto"),
    (dict(relaxed_plate=True), "relajada"),
    (dict(nivel="manzana", n_predios=3), "compartida"),
    (dict(nivel="esquina"), "parcial"),
    (dict(nivel="via"), "parcial"),
])
def test_each_kind_uses_its_own_q(kwargs, kind):
    q = {"exacto": 0.9, "relajada": 0.5, "compartida": 0.6, "parcial": 0.4}
    pp, pm = _model(q=q).predict(_f(conf=0.85, **kwargs))
    assert pp == pytest.approx(pm * q[kind])


def test_missing_kind_falls_back_to_q_global():
    pp, pm = _model(q={"exacto": 0.9}, q_global=0.55).predict(_f(conf=0.85, nivel="esquina"))
    assert pp == pytest.approx(pm * 0.55)


@pytest.mark.parametrize("bad", [None, float("nan"), "x"])
def test_undefined_score_gets_the_lowest_step(bad):
    feats = _f()
    feats["confidence"] = bad
    pp, pm = _model().predict(feats)
    assert pm == pytest.approx(0.60) and math.isfinite(pp) and pp <= pm


def test_missing_score_key_gets_the_lowest_step():
    assert _model().predict({})[1] == pytest.approx(0.60)


def test_infinite_scores_are_clamped_to_the_ends():
    feats = _f()
    feats["confidence"] = float("inf")
    assert _model().predict(feats)[1] == pytest.approx(0.95)
    feats["confidence"] = float("-inf")
    assert _model().predict(feats)[1] == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# artifact loading
# ---------------------------------------------------------------------------
def _payload() -> dict:
    return {
        "kind": "hybrid", "version": "reliability-v4-test",
        "isotonic": {"x": [0.74, 0.85, 0.95], "y": [0.6, 0.8, 0.95], "n": [50, 50, 50]},
        "q": {"exacto": {"n": 10, "hits": 8, "q": 0.8}, "relajada": {"n": 0, "hits": 0, "q": 0.7}},
        "q_global": 0.7, "m": 20,
    }


def _write(tmp_path, payload) -> str:
    path = tmp_path / "reliability.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_load_hybrid_artifact(tmp_path):
    model = load_reliability(_write(tmp_path, _payload()))
    assert isinstance(model, HybridReliability) and model.version == "reliability-v4-test"
    pp, pm = model.predict(_f(conf=0.85))
    assert pm == pytest.approx(0.8) and pp == pytest.approx(0.64)
    assert model.predict(_f(conf=0.85, nivel="via"))[0] == pytest.approx(0.8 * 0.7)  # q_global


def test_old_logistic_and_cells_artifacts_still_load(tmp_path):
    from cali_address.reliability import FEATURE_NAMES
    n = len(FEATURE_NAMES)
    logistic = {"version": "t", "feature_names": list(FEATURE_NAMES), "means": [0.0] * n, "stds": [1.0] * n,
                "predio": {"coef": [0.1] * n, "intercept": -0.2}, "manzana": {"coef": [0.2] * n, "intercept": 0.1}}
    assert isinstance(load_reliability(_write(tmp_path, logistic)), Reliability)
    cells = {"kind": "cells", "global": {"predio": 0.6, "manzana": 0.8}, "kinds": {}, "cells": []}
    assert isinstance(load_reliability(_write(tmp_path, cells)), CellReliability)


def _mutations():
    return {
        "no_isotonic": lambda p: p.pop("isotonic"),
        "isotonic_not_dict": lambda p: p.__setitem__("isotonic", [1]),
        "no_x": lambda p: p["isotonic"].pop("x"),
        "empty_knots": lambda p: p["isotonic"].update(x=[], y=[]),
        "length_mismatch": lambda p: p["isotonic"].update(y=[0.6, 0.8]),
        "x_not_increasing": lambda p: p["isotonic"].update(x=[0.74, 0.74, 0.95]),
        "x_decreasing": lambda p: p["isotonic"].update(x=[0.9, 0.85, 0.8]),
        "y_decreasing": lambda p: p["isotonic"].update(y=[0.9, 0.8, 0.95]),
        "y_above_one": lambda p: p["isotonic"].update(y=[0.6, 0.8, 1.2]),
        "y_negative": lambda p: p["isotonic"].update(y=[-0.1, 0.8, 0.9]),
        "nan_x": lambda p: p["isotonic"].update(x=[0.7, float("nan"), 0.9]),
        "string_y": lambda p: p["isotonic"].update(y=["a", 0.8, 0.9]),
        "no_q_global": lambda p: p.pop("q_global"),
        "q_global_above_one": lambda p: p.__setitem__("q_global", 1.5),
        "q_not_dict": lambda p: p.__setitem__("q", [1]),
        "q_unknown_kind": lambda p: p["q"].__setitem__("zzz", {"q": 0.5}),
        "q_entry_not_dict": lambda p: p["q"].__setitem__("exacto", 0.5),
        "q_missing_value": lambda p: p["q"]["exacto"].pop("q"),
        "q_negative": lambda p: p["q"]["exacto"].__setitem__("q", -0.2),
        "q_nan": lambda p: p["q"]["exacto"].__setitem__("q", float("nan")),
    }


@pytest.mark.parametrize("name", sorted(_mutations()))
def test_malformed_hybrid_artifact_returns_none_without_raising(tmp_path, name):
    payload = _payload()
    _mutations()[name](payload)
    assert load_reliability(_write(tmp_path, payload)) is None


def test_hybrid_artifact_without_q_block_uses_q_global(tmp_path):
    payload = _payload()
    del payload["q"]
    model = load_reliability(_write(tmp_path, payload))
    assert model is not None and model.predict(_f(conf=0.85))[0] == pytest.approx(0.8 * 0.7)


def test_missing_and_non_json_files_return_none(tmp_path):
    assert load_reliability(str(tmp_path / "nope.json")) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_reliability(str(bad)) is None


def test_known_kinds_constant_unchanged():
    assert KINDS == ("exacto", "relajada", "compartida", "parcial")


# ---------------------------------------------------------------------------
# through normalize_strict
# ---------------------------------------------------------------------------
def test_hybrid_model_serves_through_build_row(stub_normalizer):
    stub = stub_normalizer([{"direccion": "CL 5 # 38 - 10"}])
    stub.reliability = HybridReliability(xs=[0.5], ys=[0.9], q={"exacto": 0.5}, q_global=0.5)
    row = normalize_strict(stub, ["CL 5 # 38 - 10"], gazetteer=None).iloc[0]
    assert row["estado"] == "OK"
    assert row["confiabilidad_manzana"] == 0.9 and row["confiabilidad"] == 0.45
    assert row["confiabilidad"] <= row["confiabilidad_manzana"]


def test_missing_artifact_leaves_columns_none_on_ok_rows(stub_normalizer):
    stub = stub_normalizer([{"direccion": "CL 5 # 38 - 10"}])
    stub.reliability = None
    row = normalize_strict(stub, ["CL 5 # 38 - 10"], gazetteer=None).iloc[0]
    assert row["estado"] == "OK"
    assert row["confiabilidad"] is None and row["confiabilidad_manzana"] is None
