"""Cell-based reliability: cell derivation, serving lookup with fallbacks and artifact loading."""

from __future__ import annotations

import json
import math

import pytest

from cali_address.reliability import (
    BANDS, FEATURE_NAMES, KINDS, CellReliability, Reliability, load_reliability, reliability_cell,
)
from cali_address.service import normalize_strict, reliability_features


def _f(margen=None, nivel="predio", relaxed_letter=False, relaxed_plate=False, n_predios=1) -> dict:
    return reliability_features(
        0.9, 1.0, relaxed_letter, relaxed_plate, n_predios, nivel, False,
        margen_manzana=margen, margen_predio=margen,
    )


# ---------------------------------------------------------------------------
# reliability_cell: kind
# ---------------------------------------------------------------------------
def test_kinds_and_bands_are_the_documented_vocabulary():
    assert KINDS == ("exacto", "relajada", "compartida", "parcial")
    assert BANDS == ("<0.02", "0.02-0.05", "0.05-0.10", "holgado")


def test_exact_row_is_exacto():
    assert reliability_cell(_f())[0] == "exacto"


@pytest.mark.parametrize("kwargs", [
    dict(relaxed_letter=True, nivel="manzana"),
    dict(relaxed_plate=True, nivel="manzana"),
    dict(relaxed_plate=True, nivel="direccion"),
    dict(relaxed_plate=True, nivel="esquina"),
    dict(relaxed_plate=True, nivel="manzana", n_predios=4),  # relaxed wins over shared
])
def test_relaxed_rows_are_relajada(kwargs):
    assert reliability_cell(_f(**kwargs))[0] == "relajada"


@pytest.mark.parametrize("nivel", ["manzana", "direccion"])
def test_shared_address_rows_are_compartida(nivel):
    assert reliability_cell(_f(nivel=nivel, n_predios=3))[0] == "compartida"


@pytest.mark.parametrize("nivel", ["esquina", "via"])
def test_partial_levels_are_parcial(nivel):
    assert reliability_cell(_f(nivel=nivel))[0] == "parcial"


def test_unknown_nivel_is_treated_as_exacto():
    assert reliability_cell(_f(nivel="weird"))[0] == "exacto"


# ---------------------------------------------------------------------------
# reliability_cell: band (boundaries 0.02 / 0.05 / 0.10, None, NaN)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("margen,band", [
    (None, "holgado"), (float("nan"), "holgado"), (0.3, "holgado"), (5.0, "holgado"),
    (float("inf"), "holgado"),
    (0.10, "holgado"), (0.1000001, "holgado"), (0.0999, "0.05-0.10"),
    (0.05, "0.05-0.10"), (0.0499, "0.02-0.05"),
    (0.02, "0.02-0.05"), (0.0199, "<0.02"),
    (0.0, "<0.02"), (-0.5, "<0.02"),
])
def test_band_boundaries(margen, band):
    assert reliability_cell(_f(margen=margen))[1] == band


@pytest.mark.parametrize("bad", ["x", [], object()])
def test_unparseable_margin_is_treated_as_no_competitor(bad):
    feats = _f()
    feats["margen_manzana"] = bad
    assert reliability_cell(feats)[1] == "holgado"


def test_missing_features_do_not_raise_and_give_exacto_holgado():
    assert reliability_cell({}) == ("exacto", "holgado")


def test_every_cell_is_in_the_vocabulary():
    for nivel in ("predio", "esquina", "via", "manzana", "direccion"):
        for margen in (None, 0.0, 0.03, 0.07, 0.2):
            kind, band = reliability_cell(_f(nivel=nivel, margen=margen))
            assert kind in KINDS and band in BANDS


# ---------------------------------------------------------------------------
# serving model
# ---------------------------------------------------------------------------
def _model() -> CellReliability:
    return CellReliability(
        cells={("exacto", "holgado"): (0.70, 0.90), ("exacto", "0.02-0.05"): (0.30, 0.55)},
        kinds={"exacto": (0.65, 0.85), "relajada": (0.50, 0.75)},
        global_rate=(0.60, 0.80), version="cells-test",
    )


def test_known_cell_returns_its_rates():
    assert _model().predict(_f(margen=None)) == (0.70, 0.90)
    assert _model().predict(_f(margen=0.03)) == (0.30, 0.55)


def test_unknown_cell_falls_back_to_the_kind_marginal():
    assert _model().predict(_f(margen=0.07)) == (0.65, 0.85)  # exacto x 0.05-0.10 not in table
    assert _model().predict(_f(relaxed_plate=True, nivel="manzana")) == (0.50, 0.75)


def test_unknown_kind_falls_back_to_the_global_rate():
    assert _model().predict(_f(nivel="esquina")) == (0.60, 0.80)


def test_predict_bounds_and_manzana_never_below_predio():
    m = CellReliability(cells={("exacto", "holgado"): (1.0, 0.2)}, kinds={}, global_rate=(0.0, 0.0), version="")
    pp, pm = m.predict(_f())
    assert 0.001 <= pp <= pm <= 0.999
    pp, pm = m.predict(_f(nivel="via"))  # global 0/0 is clipped inside (0, 1)
    assert 0.001 <= pp <= pm <= 0.999


def test_predict_nan_margin_is_no_competitor():
    assert _model().predict(_f(margen=float("nan"))) == _model().predict(_f(margen=None))


def test_predict_ignores_extra_and_missing_keys():
    assert _model().predict({}) == (0.70, 0.90)
    assert _model().predict({**_f(), "unknown": 1.0}) == _model().predict(_f())


# ---------------------------------------------------------------------------
# artifact loading
# ---------------------------------------------------------------------------
def _cell_payload() -> dict:
    return {
        "kind": "cells", "version": "reliability-v3-test",
        "global": {"predio": 0.6, "manzana": 0.8},
        "kinds": {"exacto": {"predio": 0.65, "manzana": 0.85, "n": 10}},
        "cells": [
            {"kind": "exacto", "band": "holgado", "n": 6, "predio": 0.7, "manzana": 0.9},
            {"kind": "exacto", "band": "<0.02", "n": 0, "predio": 0.4, "manzana": 0.6},
        ],
    }


def _logistic_payload() -> dict:
    n = len(FEATURE_NAMES)
    return {
        "version": "t", "feature_names": list(FEATURE_NAMES), "means": [0.0] * n, "stds": [1.0] * n,
        "predio": {"coef": [0.1] * n, "intercept": -0.2}, "manzana": {"coef": [0.2] * n, "intercept": 0.1},
    }


def _write(tmp_path, payload) -> str:
    path = tmp_path / "reliability.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_load_cells_artifact(tmp_path):
    model = load_reliability(_write(tmp_path, _cell_payload()))
    assert isinstance(model, CellReliability) and model.version == "reliability-v3-test"
    assert model.predict(_f()) == (0.7, 0.9)
    assert model.predict(_f(margen=0.0)) == (0.4, 0.6)
    assert model.predict(_f(margen=0.03)) == (0.65, 0.85)  # kind marginal
    assert model.predict(_f(nivel="via")) == (0.6, 0.8)  # global


def test_artifact_without_kind_is_the_old_logistic_format(tmp_path):
    model = load_reliability(_write(tmp_path, _logistic_payload()))
    assert isinstance(model, Reliability)


def test_explicit_logistic_kind_still_loads(tmp_path):
    payload = {**_logistic_payload(), "kind": "logistic"}
    assert isinstance(load_reliability(_write(tmp_path, payload)), Reliability)


def test_unknown_kind_returns_none(tmp_path):
    assert load_reliability(_write(tmp_path, {**_cell_payload(), "kind": "forest"})) is None


def _mutations():
    def drop(key):
        def fn(p):
            del p[key]
        return fn

    def setp(fn_):
        return fn_

    return {
        "no_global": drop("global"),
        "no_cells": drop("cells"),
        "cells_not_list": lambda p: p.__setitem__("cells", {"a": 1}),
        "cell_not_dict": lambda p: p["cells"].__setitem__(0, 5),
        "unknown_kind_in_cell": lambda p: p["cells"][0].__setitem__("kind", "zzz"),
        "unknown_band_in_cell": lambda p: p["cells"][0].__setitem__("band", "0.3"),
        "missing_rate": lambda p: p["cells"][0].pop("predio"),
        "string_rate": lambda p: p["cells"][0].__setitem__("manzana", "x"),
        "nan_rate": lambda p: p["cells"][0].__setitem__("predio", float("nan")),
        "rate_above_one": lambda p: p["cells"][0].__setitem__("predio", 1.2),
        "negative_rate": lambda p: p["global"].__setitem__("manzana", -0.1),
        "global_missing_target": lambda p: p["global"].pop("manzana"),
        "kinds_bad_kind": lambda p: p["kinds"].__setitem__("zzz", {"predio": 0.5, "manzana": 0.5}),
        "kinds_not_dict": lambda p: p.__setitem__("kinds", [1]),
    }


@pytest.mark.parametrize("name", sorted(_mutations()))
def test_malformed_cell_artifact_returns_none_without_raising(tmp_path, name):
    payload = _cell_payload()
    _mutations()[name](payload)
    assert load_reliability(_write(tmp_path, payload)) is None


def test_cell_artifact_without_kinds_block_is_fine(tmp_path):
    payload = _cell_payload()
    del payload["kinds"]
    model = load_reliability(_write(tmp_path, payload))
    assert model is not None and model.predict(_f(margen=0.03)) == (0.6, 0.8)


# ---------------------------------------------------------------------------
# through normalize_strict
# ---------------------------------------------------------------------------
def test_cells_model_serves_through_normalize_strict(stub_normalizer):
    stub = stub_normalizer([{"direccion": "CL 5 # 38 - 10"}])
    stub.reliability = _model()
    row = normalize_strict(stub, ["CL 5 # 38 - 10"], gazetteer=None).iloc[0]
    assert row["estado"] == "OK"
    assert row["confiabilidad"] == 0.7 and row["confiabilidad_manzana"] == 0.9
    assert math.isfinite(row["confiabilidad"])
