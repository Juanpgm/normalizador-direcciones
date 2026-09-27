"""Ambiguity margin: per-row competition signal and the optional soft abstention."""

from __future__ import annotations

import math

import pytest

from cali_address.service import normalize_strict

QUERY = "CL 5 # 38 - 10"
SAME = "CL 5 # 38 - 10"


def _run(stub_normalizer, docs, scores, threshold=0.5, **kwargs):
    # Margin math is tested with the abstention off unless a test opts in explicitly.
    kwargs.setdefault("ambiguity_delta", 0.0)
    stub = stub_normalizer(docs, scores=[scores], threshold=threshold)
    sink: list = []
    result = normalize_strict(stub, [QUERY], gazetteer=None, feature_sink=sink, **kwargs)
    return result.iloc[0], sink[0]


# ---------------------------------------------------------------------------
# margin math
# ---------------------------------------------------------------------------
def test_two_rule_passing_candidates_in_different_manzanas_give_the_gap(stub_normalizer):
    row, feats = _run(
        stub_normalizer,
        [{"direccion": SAME, "manzana": "M1", "predial": "P1"},
         {"direccion": SAME, "manzana": "M2", "predial": "P2"}],
        [0.90, 0.86],
    )
    assert row["estado"] == "OK" and row["manzana"] == "M1"
    assert row["margen"] == pytest.approx(0.04)
    assert feats["margen_manzana"] == pytest.approx(0.04)
    assert feats["margen_predio"] == pytest.approx(0.04)
    assert feats["n_competidores"] == 1.0


def test_competitor_violating_a_hard_rule_is_ignored(stub_normalizer):
    row, feats = _run(
        stub_normalizer,
        [{"direccion": SAME, "manzana": "M1"}, {"direccion": "CL 6 # 38 - 10", "manzana": "M2"}],
        [0.90, 0.89],
    )
    assert row["estado"] == "OK"
    assert row["margen"] is None
    assert feats["margen_manzana"] == 0.3 and feats["margen_predio"] == 0.3
    assert feats["n_competidores"] == 0.0


def test_same_manzana_competitor_only_affects_margen_predio(stub_normalizer):
    row, feats = _run(
        stub_normalizer,
        [{"direccion": SAME, "manzana": "M1", "predial": "P1"},
         {"direccion": SAME, "manzana": "M1", "predial": "P2"}],
        [0.90, 0.88],
    )
    assert row["margen"] is None
    assert feats["margen_manzana"] == 0.3
    assert feats["margen_predio"] == pytest.approx(0.02)
    assert feats["n_competidores"] == 0.0


def test_single_candidate_has_no_competition(stub_normalizer):
    row, feats = _run(stub_normalizer, [{"direccion": SAME}], [0.95])
    assert row["margen"] is None
    assert (feats["margen_manzana"], feats["margen_predio"], feats["n_competidores"]) == (0.3, 0.3, 0.0)


def test_exact_tie_gives_zero_margin(stub_normalizer):
    row, feats = _run(
        stub_normalizer,
        [{"direccion": SAME, "manzana": "M1"}, {"direccion": SAME, "manzana": "M2"}],
        [0.80, 0.80],
    )
    assert row["estado"] == "OK"
    assert row["margen"] == 0.0
    assert feats["margen_manzana"] == 0.0 and feats["n_competidores"] == 1.0


def test_soft_violating_competitor_counts(stub_normalizer):
    # plate delta 2 is a soft violation: the competitor would itself be OK (as 'aproximado').
    row, feats = _run(
        stub_normalizer,
        [{"direccion": SAME, "manzana": "M1"}, {"direccion": "CL 5 # 38 - 12", "manzana": "M2"}],
        [0.90, 0.88],
    )
    assert row["margen"] == pytest.approx(0.02)
    assert feats["n_competidores"] == 1.0


def test_competitor_below_the_confidence_threshold_is_ignored(stub_normalizer):
    row, _ = _run(
        stub_normalizer,
        [{"direccion": SAME, "manzana": "M1"}, {"direccion": SAME, "manzana": "M2"}],
        [0.90, 0.30], threshold=0.5,
    )
    assert row["margen"] is None


def test_nan_score_competitor_is_skipped(stub_normalizer):
    row, feats = _run(
        stub_normalizer,
        [{"direccion": SAME, "manzana": "M1"}, {"direccion": SAME, "manzana": "M2"}],
        [0.90, float("nan")],
    )
    assert row["estado"] == "OK" and row["margen"] is None
    assert feats["margen_manzana"] == 0.3 and feats["n_competidores"] == 0.0


def test_margin_is_the_gap_to_the_closest_competitor_and_rounded_to_4_decimals(stub_normalizer):
    row, feats = _run(
        stub_normalizer,
        [{"direccion": SAME, "manzana": "M1"}, {"direccion": SAME, "manzana": "M3"},
         {"direccion": SAME, "manzana": "M2"}],
        [0.90, 0.60, 0.812345],
    )
    assert row["margen"] == 0.0877
    assert feats["n_competidores"] == 0.0  # neither is within 0.05


def test_n_competidores_counts_within_005_inclusive_and_is_clipped_at_5(stub_normalizer):
    docs = [{"direccion": SAME, "manzana": f"M{i}"} for i in range(8)]
    scores = [0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.855, 0.50]
    _, feats = _run(stub_normalizer, docs, scores)
    assert feats["n_competidores"] == 5.0  # 6 competitors within 0.05, clipped for the model


def test_margin_of_exactly_005_counts_as_competitor(stub_normalizer):
    _, feats = _run(
        stub_normalizer,
        [{"direccion": SAME, "manzana": "M1"}, {"direccion": SAME, "manzana": "M2"}],
        [0.75, 0.70],  # 0.75 - 0.70 == 0.05000000000000004 in floating point
    )
    assert feats["n_competidores"] == 1.0


def test_margins_are_clipped_to_03_in_the_features(stub_normalizer):
    row, feats = _run(
        stub_normalizer,
        [{"direccion": SAME, "manzana": "M1"}, {"direccion": SAME, "manzana": "M2"}],
        [0.95, 0.55],
    )
    assert row["margen"] == pytest.approx(0.4)  # the column is the raw gap
    assert feats["margen_manzana"] == 0.3 and feats["margen_predio"] == 0.3


def test_non_ok_rows_have_none_margin_and_none_sink(stub_normalizer):
    stub = stub_normalizer(
        [{"direccion": SAME, "manzana": "M1"}, {"direccion": SAME, "manzana": "M2"}],
        scores=[[0.3, 0.29], [0.9, 0.89]], threshold=0.7,
    )
    sink: list = []
    result = normalize_strict(stub, [QUERY, "CL 9 # 20 - 30"], gazetteer=None, feature_sink=sink)
    assert result["margen"].isna().all()
    result2 = normalize_strict(stub, ["", "asdkjasd1234"], gazetteer=None)
    assert result2["margen"].isna().all()


# ---------------------------------------------------------------------------
# soft abstention (ambiguity_delta)
# ---------------------------------------------------------------------------
_TWO_MANZANAS = [{"direccion": SAME, "manzana": "M1", "predial": "P1"},
                 {"direccion": SAME, "manzana": "M2", "predial": "P2"}]


def _run_default(stub_normalizer, docs, scores, threshold=0.5):
    """Like ``_run`` but with the production default ``ambiguity_delta``."""
    stub = stub_normalizer(docs, scores=[scores], threshold=threshold)
    return normalize_strict(stub, [QUERY], gazetteer=None).iloc[0]


def test_default_delta_is_0_02():
    import inspect

    assert inspect.signature(normalize_strict).parameters["ambiguity_delta"].default == 0.02


def test_default_abstains_on_a_001_margin(stub_normalizer):
    row = _run_default(stub_normalizer, _TWO_MANZANAS, [0.90, 0.89])
    assert row["estado"] == "SIN_MATCH"
    assert row["motivo"] == "ambiguo: 1 candidatos cercanos en otras manzanas"


def test_default_keeps_ok_on_a_margin_equal_to_the_delta(stub_normalizer):
    row = _run_default(stub_normalizer, _TWO_MANZANAS, [0.75, 0.73])  # gap 0.02 (>= 0.02 in floats)
    assert row["estado"] == "OK" and row["margen"] == 0.02


def test_default_abstains_just_below_the_boundary(stub_normalizer):
    row = _run_default(stub_normalizer, _TWO_MANZANAS, [0.75, 0.7301])  # gap 0.0199
    assert row["estado"] == "SIN_MATCH"


def test_default_keeps_ok_without_a_competitor(stub_normalizer):
    row = _run_default(stub_normalizer, [{"direccion": SAME}], [0.9])
    assert row["estado"] == "OK" and row["margen"] is None


def test_default_keeps_ok_with_a_same_manzana_competitor(stub_normalizer):
    docs = [{"direccion": SAME, "manzana": "M1", "predial": "P1"},
            {"direccion": SAME, "manzana": "M1", "predial": "P2"}]
    assert _run_default(stub_normalizer, docs, [0.90, 0.90])["estado"] == "OK"


def test_explicit_zero_disables_the_abstention(stub_normalizer):
    row, _ = _run(stub_normalizer, _TWO_MANZANAS, [0.90, 0.90], ambiguity_delta=0.0)
    assert row["estado"] == "OK" and row["numero_predial_nacional"] == "P1"


def test_negative_delta_is_off(stub_normalizer):
    row, _ = _run(stub_normalizer, _TWO_MANZANAS, [0.90, 0.90], ambiguity_delta=-1.0)
    assert row["estado"] == "OK"


def test_positive_delta_abstains_with_the_ambiguity_motivo(stub_normalizer):
    row, feats = _run(stub_normalizer, _TWO_MANZANAS, [0.90, 0.88], ambiguity_delta=0.05)
    assert row["estado"] == "SIN_MATCH"
    assert row["motivo"] == "ambiguo: 1 candidatos cercanos en otras manzanas"
    assert row["numero_predial_nacional"] is None and row["manzana"] is None
    assert row["margen"] is None and row["confiabilidad"] is None
    assert feats is None


def test_margin_equal_to_delta_does_not_abstain(stub_normalizer):
    row, _ = _run(stub_normalizer, _TWO_MANZANAS, [0.75, 0.5], ambiguity_delta=0.25)  # gap == 0.25 exactly
    assert row["estado"] == "OK"
    assert row["margen"] == 0.25


def test_no_competitor_never_abstains(stub_normalizer):
    row, _ = _run(stub_normalizer, [{"direccion": SAME}], [0.9], ambiguity_delta=0.2)
    assert row["estado"] == "OK"


def test_same_manzana_competitor_never_abstains(stub_normalizer):
    docs = [{"direccion": SAME, "manzana": "M1", "predial": "P1"},
            {"direccion": SAME, "manzana": "M1", "predial": "P2"}]
    row, _ = _run(stub_normalizer, docs, [0.90, 0.90], ambiguity_delta=0.2)
    assert row["estado"] == "OK"


def test_abstention_reports_at_least_one_competitor_even_beyond_005(stub_normalizer):
    row, _ = _run(stub_normalizer, _TWO_MANZANAS, [0.90, 0.80], ambiguity_delta=0.2)
    assert row["estado"] == "SIN_MATCH"
    assert row["motivo"].startswith("ambiguo: 1 ")


def test_abstention_counts_the_close_competitors(stub_normalizer):
    docs = [{"direccion": SAME, "manzana": f"M{i}"} for i in range(4)]
    row, _ = _run(stub_normalizer, docs, [0.90, 0.89, 0.88, 0.87], ambiguity_delta=0.05)
    assert row["motivo"] == "ambiguo: 3 candidatos cercanos en otras manzanas"


def test_rows_already_rejected_by_rules_keep_their_motivo_under_delta(stub_normalizer):
    stub = stub_normalizer([{"direccion": "CL 6 # 38 - 10", "manzana": "M1"},
                            {"direccion": SAME, "manzana": "M2"}], scores=[[0.9, 0.89]])
    row = normalize_strict(stub, [QUERY], gazetteer=None, ambiguity_delta=0.2).iloc[0]
    assert row["estado"] == "SIN_MATCH" and not row["motivo"].startswith("ambiguo")


def test_margen_is_finite_when_present(stub_normalizer):
    row, _ = _run(stub_normalizer, _TWO_MANZANAS, [0.90, 0.80])
    assert math.isfinite(row["margen"])
