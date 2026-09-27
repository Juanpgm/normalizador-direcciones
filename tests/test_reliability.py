"""Calibrated reliability: feature capture, serving model, loader and row plumbing."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from cali_address.reliability import FEATURE_NAMES, Reliability, load_reliability
from cali_address.service import OUTPUT_COLUMNS, normalize_strict, reliability_features


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _model(score_coef=2.0, intercept_p=-1.0, intercept_m=0.0, extra=None) -> Reliability:
    """Hand-built tiny model: only the raw score and log_n_predios carry weight."""
    n = len(FEATURE_NAMES)
    coef_p = [0.0] * n
    coef_m = [0.0] * n
    coef_p[FEATURE_NAMES.index("confidence")] = score_coef
    coef_m[FEATURE_NAMES.index("confidence")] = score_coef
    coef_p[FEATURE_NAMES.index("log_n_predios")] = -0.5
    for key, value in (extra or {}).items():
        coef_p[FEATURE_NAMES.index(key)] = value
    return Reliability(
        feature_names=list(FEATURE_NAMES), means=[0.0] * n, stds=[1.0] * n,
        coef_predio=coef_p, intercept_predio=intercept_p,
        coef_manzana=coef_m, intercept_manzana=intercept_m, version="test",
    )


def _feats(**over) -> dict:
    base = reliability_features(
        confidence=0.9, struct=1.0, relaxed_letter=False, relaxed_plate=False,
        n_predios=1, nivel="predio", gate_applied=False,
    )
    base.update(over)
    return base


def _payload(n=len(FEATURE_NAMES)) -> dict:
    return {
        "version": "t", "feature_names": list(FEATURE_NAMES),
        "means": [0.0] * n, "stds": [1.0] * n,
        "predio": {"coef": [0.1] * n, "intercept": -0.2},
        "manzana": {"coef": [0.2] * n, "intercept": 0.1},
    }


# ---------------------------------------------------------------------------
# reliability_features
# ---------------------------------------------------------------------------
def test_features_has_exactly_the_model_feature_names():
    assert set(_feats()) == set(FEATURE_NAMES)
    assert len(FEATURE_NAMES) == 13
    assert FEATURE_NAMES[-3:] == ("margen_manzana", "margen_predio", "n_competidores")


def test_features_default_to_no_competition():
    f = _feats()
    assert (f["margen_manzana"], f["margen_predio"], f["n_competidores"]) == (0.3, 0.3, 0.0)


@pytest.mark.parametrize("raw,expected", [
    (None, 0.3), (float("nan"), 0.3), (0.0, 0.0), (0.12, 0.12), (0.3, 0.3), (0.9, 0.3), (-0.2, 0.0),
    (float("inf"), 0.3),
])
def test_features_margins_are_clipped_to_0_03(raw, expected):
    f = reliability_features(0.9, 1.0, False, False, 1, "predio", False,
                             margen_manzana=raw, margen_predio=raw)
    assert f["margen_manzana"] == pytest.approx(expected) and f["margen_predio"] == pytest.approx(expected)


@pytest.mark.parametrize("raw,expected", [(0, 0.0), (1, 1.0), (5, 5.0), (9, 5.0), (-2, 0.0)])
def test_features_n_competidores_is_clipped_to_0_5(raw, expected):
    f = reliability_features(0.9, 1.0, False, False, 1, "predio", False, n_competidores=raw)
    assert f["n_competidores"] == expected


def test_features_base_level_predio_has_all_dummies_zero():
    f = _feats()
    assert [f[k] for k in ("nivel_esquina", "nivel_via", "nivel_manzana", "nivel_direccion")] == [0.0] * 4


@pytest.mark.parametrize("nivel", ["esquina", "via", "manzana", "direccion"])
def test_features_each_nivel_sets_only_its_dummy(nivel):
    f = _feats(**reliability_features(0.9, 1.0, False, False, 1, nivel, False))
    dummies = {k: f[k] for k in ("nivel_esquina", "nivel_via", "nivel_manzana", "nivel_direccion")}
    assert dummies == {f"nivel_{n}": (1.0 if n == nivel else 0.0)
                       for n in ("esquina", "via", "manzana", "direccion")}


def test_features_relaxed_flags_and_gate_are_zero_or_one():
    f = reliability_features(0.8, 0.7, True, False, 1, "manzana", True)
    assert (f["relaxed_letter"], f["relaxed_plate"], f["gate_applied"]) == (1.0, 0.0, 1.0)
    g = reliability_features(0.8, 0.7, False, True, 1, "manzana", False)
    assert (g["relaxed_letter"], g["relaxed_plate"], g["gate_applied"]) == (0.0, 1.0, 0.0)


def test_features_single_predio_has_log_zero_and_shared_grows():
    assert _feats()["log_n_predios"] == 0.0
    f = reliability_features(0.9, 1.0, False, False, 5, "manzana", False)
    assert f["log_n_predios"] == pytest.approx(math.log(5))


@pytest.mark.parametrize("n", [0, -3])
def test_features_nonpositive_n_predios_does_not_raise(n):
    assert reliability_features(0.9, 1.0, False, False, n, "predio", False)["log_n_predios"] == 0.0


def test_features_unknown_nivel_is_base_level():
    f = reliability_features(0.9, 1.0, False, False, 1, "weird", False)
    assert f["nivel_esquina"] == f["nivel_via"] == f["nivel_manzana"] == f["nivel_direccion"] == 0.0


def test_features_carry_confidence_and_struct_as_floats():
    f = reliability_features(0.87, 0.6, False, False, 1, "predio", False)
    assert f["confidence"] == 0.87 and f["struct"] == 0.6


# ---------------------------------------------------------------------------
# Reliability.predict
# ---------------------------------------------------------------------------
def test_predict_matches_hand_computation():
    model = _model(score_coef=2.0, intercept_p=-1.0, intercept_m=0.5)
    pp, pm = model.predict(_feats(confidence=0.5))
    assert pp == pytest.approx(1 / (1 + math.exp(-(-1.0 + 2.0 * 0.5))))
    assert pm == pytest.approx(1 / (1 + math.exp(-(0.5 + 2.0 * 0.5))))


def test_predict_bounds_and_manzana_not_below_predio_over_a_grid():
    model = _model(intercept_p=3.0, intercept_m=-3.0)  # manzana head deliberately lower
    for c in np.linspace(0.0, 1.0, 11):
        for n in (1, 2, 50):
            pp, pm = model.predict(_feats(confidence=float(c), log_n_predios=math.log(n)))
            assert 0.001 <= pp <= 0.999 and 0.001 <= pm <= 0.999
            assert pm >= pp


def test_predict_monotone_non_decreasing_in_raw_score():
    model = _model()
    prev_p = prev_m = -1.0
    for c in np.linspace(0.0, 1.0, 21):
        pp, pm = model.predict(_feats(confidence=float(c)))
        assert pp >= prev_p and pm >= prev_m
        prev_p, prev_m = pp, pm


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 1e300, -1e300, None])
def test_predict_bad_feature_values_stay_finite_and_bounded(bad):
    model = _model()
    pp, pm = model.predict(_feats(confidence=bad))
    assert math.isfinite(pp) and math.isfinite(pm)
    assert 0.001 <= pp <= pm <= 0.999


def test_predict_missing_feature_keys_are_treated_as_the_mean():
    model = _model()
    full = model.predict(_feats(confidence=0.0, log_n_predios=0.0))
    assert model.predict({}) == full  # means are all 0 here, so absent == 0


def test_predict_extra_keys_are_ignored():
    model = _model()
    assert model.predict({**_feats(), "unknown": 5.0}) == model.predict(_feats())


def test_predict_clips_standardized_features_at_5_sigma():
    model = _model(score_coef=1.0, intercept_p=0.0)
    at_clip = model.predict(_feats(confidence=5.0))
    far = model.predict(_feats(confidence=5000.0))
    assert far == at_clip


def test_predict_uses_means_and_stds():
    n = len(FEATURE_NAMES)
    coef = [0.0] * n
    coef[FEATURE_NAMES.index("confidence")] = 1.0
    means, stds = [0.0] * n, [1.0] * n
    means[FEATURE_NAMES.index("confidence")] = 0.5
    stds[FEATURE_NAMES.index("confidence")] = 0.25
    model = Reliability(list(FEATURE_NAMES), means, stds, coef, 0.0, coef, 0.0, "t")
    pp, _ = model.predict(_feats(confidence=0.75))  # z = 1
    assert pp == pytest.approx(1 / (1 + math.exp(-1.0)))


def test_zero_std_does_not_divide_by_zero():
    n = len(FEATURE_NAMES)
    model = Reliability(list(FEATURE_NAMES), [0.0] * n, [0.0] * n, [1.0] * n, 0.0, [1.0] * n, 0.0, "t")
    pp, pm = model.predict(_feats())
    assert math.isfinite(pp) and math.isfinite(pm)


# ---------------------------------------------------------------------------
# load_reliability
# ---------------------------------------------------------------------------
def test_load_valid_file(tmp_path):
    path = tmp_path / "reliability.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    model = load_reliability(str(path))
    assert isinstance(model, Reliability) and model.version == "t"
    pp, pm = model.predict(_feats())
    assert 0.0 < pp <= pm < 1.0


LEGACY_NAMES = ["confidence", "struct", "relaxed_letter", "relaxed_plate", "log_n_predios",
                "nivel_esquina", "nivel_via", "nivel_manzana", "nivel_direccion", "gate_applied"]


def _legacy_payload() -> dict:
    n = len(LEGACY_NAMES)
    return {
        "version": "old", "feature_names": list(LEGACY_NAMES),
        "means": [0.0] * n, "stds": [1.0] * n,
        "predio": {"coef": [0.1] * n, "intercept": -0.2},
        "manzana": {"coef": [0.2] * n, "intercept": 0.1},
    }


def test_load_old_artifact_without_margin_features_still_works(tmp_path):
    path = tmp_path / "reliability.json"
    path.write_text(json.dumps(_legacy_payload()), encoding="utf-8")
    model = load_reliability(str(path))
    assert model is not None and model.feature_names == LEGACY_NAMES
    # margin features are ignored by the old model: same output whatever their value
    a = model.predict(_feats(margen_manzana=0.0, margen_predio=0.0, n_competidores=5.0))
    b = model.predict(_feats(margen_manzana=0.3, margen_predio=0.3, n_competidores=0.0))
    assert a == b
    assert 0.0 < a[0] <= a[1] < 1.0


def test_old_artifact_serves_through_normalize_strict(stub_normalizer):
    stub = stub_normalizer([{"direccion": "CL 5 # 38 - 10"}])
    stub.reliability = Reliability(
        list(LEGACY_NAMES), [0.0] * 10, [1.0] * 10, [0.1] * 10, 0.0, [0.1] * 10, 0.0, "old")
    row = normalize_strict(stub, ["CL 5 # 38 - 10"], gazetteer=None).iloc[0]
    assert row["estado"] == "OK" and row["confiabilidad"] is not None


def test_load_artifact_with_duplicate_or_empty_feature_names_returns_none(tmp_path):
    path = tmp_path / "reliability.json"
    payload = _legacy_payload()
    payload["feature_names"][1] = "confidence"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_reliability(str(path)) is None
    empty = {"version": "e", "feature_names": [], "means": [], "stds": [],
             "predio": {"coef": [], "intercept": 0.0}, "manzana": {"coef": [], "intercept": 0.0}}
    path.write_text(json.dumps(empty), encoding="utf-8")
    assert load_reliability(str(path)) is None


def test_margin_features_move_a_model_that_uses_them():
    model = _model(extra={"margen_manzana": 3.0})
    tight = model.predict(_feats(margen_manzana=0.0))[0]
    loose = model.predict(_feats(margen_manzana=0.3))[0]
    assert loose > tight


def test_load_missing_file_returns_none(tmp_path):
    assert load_reliability(str(tmp_path / "nope.json")) is None


@pytest.mark.parametrize("text", ["", "{not json", "[]", "null", "42", '{"feature_names": 3}'])
def test_load_malformed_json_returns_none(tmp_path, text):
    path = tmp_path / "reliability.json"
    path.write_text(text, encoding="utf-8")
    assert load_reliability(str(path)) is None


def test_load_wrong_feature_count_returns_none(tmp_path):
    payload = _payload()
    payload["predio"]["coef"] = payload["predio"]["coef"][:-1]
    path = tmp_path / "reliability.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_reliability(str(path)) is None


def test_load_unknown_feature_names_returns_none(tmp_path):
    payload = _payload()
    payload["feature_names"][0] = "something_else"
    path = tmp_path / "reliability.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_reliability(str(path)) is None


def test_load_non_numeric_or_nan_coefficient_returns_none(tmp_path):
    payload = _payload()
    payload["manzana"]["coef"][0] = "x"
    path = tmp_path / "reliability.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_reliability(str(path)) is None
    payload = _payload()
    payload["predio"]["intercept"] = float("nan")
    path.write_text(json.dumps(payload), encoding="utf-8")  # json.dumps emits NaN
    assert load_reliability(str(path)) is None


def test_load_missing_target_block_returns_none(tmp_path):
    payload = _payload()
    del payload["manzana"]
    path = tmp_path / "reliability.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_reliability(str(path)) is None


# ---------------------------------------------------------------------------
# columns / rows through normalize_strict
# ---------------------------------------------------------------------------
def test_output_columns_put_reliability_right_after_confianza():
    i = OUTPUT_COLUMNS.index("confianza")
    assert OUTPUT_COLUMNS[i + 1: i + 5] == [
        "confiabilidad", "confiabilidad_manzana", "margen", "nivel_precision"]


def test_ok_row_without_model_has_none_reliability_and_same_estado(stub_normalizer):
    stub = stub_normalizer([{"direccion": "CL 5 # 38 - 10", "predial": "PXYZ"}])
    before = normalize_strict(stub, ["CL 5 # 38 - 10"], gazetteer=None).iloc[0]
    assert before["estado"] == "OK"
    assert before["confiabilidad"] is None and before["confiabilidad_manzana"] is None
    stub.reliability = _model()
    after = normalize_strict(stub, ["CL 5 # 38 - 10"], gazetteer=None).iloc[0]
    assert after["estado"] == before["estado"] == "OK"
    assert after["numero_predial_nacional"] == before["numero_predial_nacional"]


def test_ok_row_reliability_matches_hand_built_model(stub_normalizer):
    stub = stub_normalizer([{"direccion": "CL 5 # 38 - 10"}])
    stub.reliability = _model(score_coef=2.0, intercept_p=-1.0, intercept_m=0.5)
    sink: list = []
    row = normalize_strict(stub, ["CL 5 # 38 - 10"], gazetteer=None, feature_sink=sink).iloc[0]
    assert len(sink) == 1 and set(sink[0]) == set(FEATURE_NAMES)
    pp, pm = stub.reliability.predict(sink[0])
    assert row["confiabilidad"] == round(pp, 3)
    assert row["confiabilidad_manzana"] == round(pm, 3)
    assert row["confiabilidad_manzana"] >= row["confiabilidad"]


def test_non_ok_rows_have_none_reliability_and_none_sink_entries(stub_normalizer):
    stub = stub_normalizer([{"direccion": "CL 9 # 20 - 30"}], scores=[[0.1]], threshold=0.7)
    stub.reliability = _model()
    sink: list = []
    result = normalize_strict(stub, ["", "asdkjasd1234", "CL 9 # 20 - 30"], gazetteer=None,
                              feature_sink=sink)
    assert list(result["estado"]) == ["SIN_MATCH", "NO_PARSEABLE", "SIN_MATCH"]
    assert result["confiabilidad"].isna().all() and result["confiabilidad_manzana"].isna().all()
    assert sink == [None, None, None]


def test_sink_shared_address_features(stub_normalizer):
    stub = stub_normalizer([{"direccion": "CL 5 # 38 - 10", "n_predios": 4}])
    sink: list = []
    normalize_strict(stub, ["CL 5 # 38 - 10"], gazetteer=None, feature_sink=sink)
    assert sink[0]["log_n_predios"] == pytest.approx(math.log(4))
    assert sink[0]["nivel_manzana"] == 1.0 and sink[0]["gate_applied"] == 0.0


def test_sink_relaxed_plate_flag(stub_normalizer):
    stub = stub_normalizer([{"direccion": "CL 5 # 38 - 12"}])
    sink: list = []
    row = normalize_strict(stub, ["CL 5 # 38 - 10"], gazetteer=None, feature_sink=sink).iloc[0]
    assert row["estado"] == "OK" and row["nivel_precision"] == "manzana"
    assert sink[0]["relaxed_plate"] == 1.0 and sink[0]["relaxed_letter"] == 0.0


def test_reliability_columns_survive_dataframe_as_object_none(stub_normalizer):
    stub = stub_normalizer([{"direccion": "CL 5 # 38 - 10"}])
    df = normalize_strict(stub, ["CL 5 # 38 - 10"], gazetteer=None)
    assert isinstance(df, pd.DataFrame) and list(df.columns) == OUTPUT_COLUMNS


# ---------------------------------------------------------------------------
# Esri attributes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value,expected", [
    (0.734, 0.734), (None, None), (float("nan"), None), (float("inf"), None), ("x", None),
])
def test_esri_attributes_expose_confiabilidad(value, expected):
    pytest.importorskip("fastapi")
    from cali_address.api.esri import _attributes

    assert _attributes({"estado": "OK", "confiabilidad": value})["Confiabilidad"] == expected


@pytest.mark.parametrize("value,expected", [(0.0312, 0.0312), (None, None), (float("nan"), None)])
def test_esri_attributes_expose_margen(value, expected):
    pytest.importorskip("fastapi")
    from cali_address.api.esri import _attributes

    assert _attributes({"estado": "OK", "margen": value})["Margen"] == expected
