"""Stage 3 decision-layer options: ``soft_rules``, ``max_soft`` and ``gate_fallback``.

Every option defaults to today's behaviour (byte-for-byte); the tests below pin both the
defaults and each new behaviour, including empty / malformed / boundary values.
"""

from __future__ import annotations

import inspect
import os
import json
import math

import pytest

from cali_address import service
from cali_address.inference import load_tuning
from cali_address.parser import parse_address
from cali_address.service import (
    SOFT_RULE_NAMES,
    _classify_violations,
    normalize_soft_rules,
    normalize_strict,
)


def _cv(query, cadastral, soft_rules=frozenset(), plate_tolerance=0, min_struct=0.0, struct=1.0):
    q, c = parse_address(query), parse_address(cadastral)
    assert q.parse_ok and c.parse_ok
    return _classify_violations(q, c, min_struct, struct, plate_tolerance, soft_rules)


# ---------------------------------------------------------------------------
# normalize_soft_rules
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", [None, "", "  ", (), [], set(), frozenset()])
def test_empty_soft_rules_normalise_to_the_empty_set(value):
    assert normalize_soft_rules(value) == frozenset()


def test_soft_rules_accept_comma_separated_string_and_iterables():
    assert normalize_soft_rules("letra_cruce, bis_cruce") == frozenset({"letra_cruce", "bis_cruce"})
    assert normalize_soft_rules(["letra_cruce"]) == frozenset({"letra_cruce"})
    assert normalize_soft_rules(("letra_cruce", "letra_cruce")) == frozenset({"letra_cruce"})


def test_every_advertised_soft_rule_name_is_accepted():
    assert normalize_soft_rules(SOFT_RULE_NAMES) == frozenset(SOFT_RULE_NAMES)


@pytest.mark.parametrize("bad", ["nope", "letra_cruce,zzz", ["placa"], [1], "LETRA_CRUCE"])
def test_unknown_or_malformed_soft_rules_are_rejected(bad):
    with pytest.raises(ValueError):
        normalize_soft_rules(bad)


# ---------------------------------------------------------------------------
# _classify_violations: defaults unchanged
# ---------------------------------------------------------------------------
def test_default_keeps_cross_letter_mismatch_hard():
    assert _cv("KR 5 # 38 A-10", "KR 5 # 38-10") == (["letra de cruce A != -"], [])


def test_default_keeps_via_bis_mismatch_hard():
    assert _cv("CL 13 BIS # 4-10", "CL 13 # 4-10") == (["bis de via si != no"], [])


def test_empty_soft_rules_equal_the_legacy_call():
    q, c = parse_address("KR 5 # 38 A-10"), parse_address("KR 5 # 38-10")
    assert _classify_violations(q, c, 0.0, 1.0, 0) == _classify_violations(q, c, 0.0, 1.0, 0, frozenset())


# ---------------------------------------------------------------------------
# _classify_violations: each named class
# ---------------------------------------------------------------------------
def test_letra_cruce_makes_any_cross_letter_mismatch_soft():
    rules = frozenset({"letra_cruce"})
    assert _cv("KR 5 # 38 A-10", "KR 5 # 38-10", rules) == ([], ["letra de cruce A != -"])
    assert _cv("KR 5 # 38 A-10", "KR 5 # 38 B-10", rules) == ([], ["letra de cruce A != B"])


def test_letra_cruce_una_cara_only_relaxes_one_sided_letters():
    rules = frozenset({"letra_cruce_una_cara"})
    assert _cv("KR 5 # 38 A-10", "KR 5 # 38-10", rules) == ([], ["letra de cruce A != -"])
    assert _cv("KR 5 # 38-10", "KR 5 # 38 B-10", rules) == ([], ["letra de cruce - != B"])
    # both sides carry a (different) letter: still a real disagreement
    assert _cv("KR 5 # 38 A-10", "KR 5 # 38 B-10", rules) == (["letra de cruce A != B"], [])


def test_letra_cruce_cardinal_only_relaxes_a_lone_cardinal_letter():
    rules = frozenset({"letra_cruce_cardinal"})
    assert _cv("CL 12 # 4N-17", "CL 12 NORTE # 4-17", rules) == ([], ["letra de cruce N != -"])
    assert _cv("KR 5 # 38-10", "KR 5 # 38 N-10", rules) == ([], ["letra de cruce - != N"])
    # an ordinary street letter is not a cardinal
    assert _cv("KR 5 # 38 A-10", "KR 5 # 38-10", rules) == (["letra de cruce A != -"], [])
    # cardinal against another letter is a disagreement, not a missing quadrant
    assert _cv("KR 5 # 38 N-10", "KR 5 # 38 B-10", rules) == (["letra de cruce N != B"], [])


def test_letra_via_makes_two_sided_via_letter_mismatch_soft():
    assert _cv("KR 41 E # 5-20", "KR 41 F # 5-20") == (["letra de via E != F"], [])
    assert _cv("KR 41 E # 5-20", "KR 41 F # 5-20", frozenset({"letra_via"})) == ([], ["letra de via E != F"])


def test_bis_classes_are_independent():
    assert _cv("CL 13 BIS # 4-10", "CL 13 # 4-10", frozenset({"bis_via"})) == ([], ["bis de via si != no"])
    assert _cv("CL 13 BIS # 4-10", "CL 13 # 4-10", frozenset({"bis_cruce"})) == (["bis de via si != no"], [])
    assert _cv("KR 5 # 38-10", "KR 5 # 38 BIS-10", frozenset({"bis_cruce"})) == ([], ["bis de cruce no != si"])
    assert _cv("KR 5 # 38-10", "KR 5 # 38 BIS-10", frozenset({"bis_via"})) == (["bis de cruce no != si"], [])


def test_cuadrante_compl_relaxes_the_complement_quadrant_rule():
    q, c = "CL 5 # 38 - 10 NORTE", "CL 5 # 38 - 10"
    assert _cv(q, c) == (["cuadrante en complemento NORTE no esta en catastro"], [])
    assert _cv(q, c, frozenset({"cuadrante_compl"})) == ([], ["cuadrante en complemento NORTE no esta en catastro"])


@pytest.mark.parametrize("query, cadastral", [
    ("CL 5 # 38 - 10", "KR 5 # 38 - 10"),      # via type
    ("CL 5 # 38 - 10", "CL 6 # 38 - 10"),      # via number
    ("CL 5 # 38 - 10", "CL 5 # 39 - 10"),      # cross number
    ("CL 5 # 38 - 10", "CL 5 # 38 - 40"),      # plate beyond the soft window
])
def test_identity_rules_stay_hard_whatever_soft_rules_say(query, cadastral):
    hard, soft = _cv(query, cadastral, frozenset(SOFT_RULE_NAMES))
    assert hard and not soft


def test_struct_floor_stays_hard_whatever_soft_rules_say():
    hard, soft = _cv("CL 5 # 38 - 10", "CL 5 # 38 - 10", frozenset(SOFT_RULE_NAMES), min_struct=0.9, struct=0.5)
    assert hard and not soft


def test_unparseable_candidate_stays_hard():
    q, c = parse_address("CL 5 # 38 - 10"), parse_address("zzz")
    hard, soft = _classify_violations(q, c, 0.0, 1.0, 0, frozenset(SOFT_RULE_NAMES))
    assert hard == ["candidato catastral no parseable"] and soft == []


# ---------------------------------------------------------------------------
# normalize_strict: signature defaults + behaviour
# ---------------------------------------------------------------------------
def test_new_parameters_default_to_unset_so_serving_falls_back_to_the_artifact():
    params = inspect.signature(normalize_strict).parameters
    for name in ("soft_rules", "max_soft", "gate_fallback"):
        assert params[name].default is None


QUERY = "KR 5 # 38 A-10"
CAD = "KR 5 # 38 - 10"


def _one(stub_normalizer, query=QUERY, docs=None, scores=None, **kw):
    docs = docs or [{"direccion": CAD, "manzana": "M1", "predial": "P1"}]
    stub = stub_normalizer(docs, scores=[scores or [0.95] * len(docs)], threshold=0.5)
    kw.setdefault("ambiguity_delta", 0.0)
    return normalize_strict(stub, [query], gazetteer=None, **kw).iloc[0]


def test_default_rejects_a_cross_letter_mismatch(stub_normalizer):
    row = _one(stub_normalizer)
    assert row["estado"] == "SIN_MATCH" and "letra de cruce" in row["motivo"]


def test_soft_letra_cruce_yields_ok_at_manzana_level(stub_normalizer):
    row = _one(stub_normalizer, soft_rules="letra_cruce")
    assert row["estado"] == "OK"
    assert row["nivel_precision"] == "manzana"
    assert row["motivo"] == "aproximado: letra de cruce A != -"
    assert row["numero_predial_nacional"] == "P1"


def test_explicit_empty_soft_rules_override_the_artifact_default(stub_normalizer):
    stub = stub_normalizer([{"direccion": CAD}], scores=[[0.95]], threshold=0.5)
    stub.decision = {"soft_rules": ["letra_cruce"]}
    assert normalize_strict(stub, [QUERY], gazetteer=None, ambiguity_delta=0.0).iloc[0]["estado"] == "OK"
    assert normalize_strict(stub, [QUERY], gazetteer=None, ambiguity_delta=0.0, soft_rules=()).iloc[0][
        "estado"] == "SIN_MATCH"


def test_decision_block_is_ignored_when_the_normalizer_has_none(stub_normalizer):
    stub = stub_normalizer([{"direccion": CAD}], scores=[[0.95]], threshold=0.5)
    assert not hasattr(stub, "decision")
    assert normalize_strict(stub, [QUERY], gazetteer=None, ambiguity_delta=0.0).iloc[0]["estado"] == "SIN_MATCH"


def test_unknown_soft_rule_raises_before_any_scoring(stub_normalizer):
    with pytest.raises(ValueError):
        _one(stub_normalizer, soft_rules="nope")


@pytest.mark.parametrize("bad", [-1, 1.5, "2", True])
def test_max_soft_must_be_a_non_negative_int(stub_normalizer, bad):
    with pytest.raises(ValueError):
        _one(stub_normalizer, soft_rules="letra_cruce", max_soft=bad)


TWO_SOFT_QUERY = "KR 41 # 5 A-20"           # cross letter A missing in the cadastre
TWO_SOFT_CAD = "KR 41 E # 5-20"            # ...and a via letter missing in the query (soft today)


def test_two_soft_violations_are_rejected_by_default_cap(stub_normalizer):
    docs = [{"direccion": TWO_SOFT_CAD}]
    row = _one(stub_normalizer, TWO_SOFT_QUERY, docs, soft_rules="letra_cruce")
    assert row["estado"] == "SIN_MATCH" and "letra de via" in row["motivo"] and "letra de cruce" in row["motivo"]


def test_max_soft_two_accepts_two_soft_violations_and_lists_both(stub_normalizer):
    docs = [{"direccion": TWO_SOFT_CAD}]
    row = _one(stub_normalizer, TWO_SOFT_QUERY, docs, soft_rules="letra_cruce", max_soft=2)
    assert row["estado"] == "OK" and row["nivel_precision"] == "manzana"
    assert row["motivo"].startswith("aproximado: ") and row["motivo"].count("letra de") == 2


def test_max_soft_zero_turns_the_legacy_soft_rules_hard(stub_normalizer):
    docs = [{"direccion": "KR 41 E # 5-20"}]
    assert _one(stub_normalizer, "KR 41 # 5-20", docs)["estado"] == "OK"
    assert _one(stub_normalizer, "KR 41 # 5-20", docs, max_soft=0)["estado"] == "SIN_MATCH"


def test_max_soft_never_lets_a_hard_violation_through(stub_normalizer):
    docs = [{"direccion": "KR 5 # 39 - 10"}]
    row = _one(stub_normalizer, "KR 5 # 38 A-10", docs, soft_rules=SOFT_RULE_NAMES, max_soft=9)
    assert row["estado"] == "SIN_MATCH"


def test_relaxed_letter_feature_flag_covers_the_new_classes(stub_normalizer):
    sink: list = []
    stub = stub_normalizer([{"direccion": CAD}], scores=[[0.95]], threshold=0.5)
    normalize_strict(stub, [QUERY], gazetteer=None, ambiguity_delta=0.0, soft_rules="letra_cruce", feature_sink=sink)
    assert sink[0]["relaxed_letter"] == 1.0 and sink[0]["relaxed_plate"] == 0.0


def test_relaxed_rival_counts_as_a_competitor_only_when_its_rule_is_soft(stub_normalizer):
    docs = [{"direccion": "KR 5 # 38 A-10", "manzana": "M1", "predial": "P1"},
            {"direccion": CAD, "manzana": "M2", "predial": "P2"}]
    scores = [0.90, 0.89]
    strict = _one(stub_normalizer, QUERY, docs, scores)
    assert strict["estado"] == "OK" and strict["manzana"] == "M1" and strict["margen"] is None
    soft = _one(stub_normalizer, QUERY, docs, scores, soft_rules="letra_cruce")
    assert soft["estado"] == "OK" and math.isclose(soft["margen"], 0.01, abs_tol=1e-6)
    abstained = _one(stub_normalizer, QUERY, docs, scores, soft_rules="letra_cruce", ambiguity_delta=0.02)
    assert abstained["estado"] == "SIN_MATCH" and "ambiguo" in abstained["motivo"]


# ---------------------------------------------------------------------------
# gate_fallback
# ---------------------------------------------------------------------------
class _Place:
    barrio, comuna, corregimiento, vereda = "Zzz", None, None, None
    cleaned_text = "CL 5 # 38 - 10"
    notes: list = []
    detected = True


class _FakeGazetteer:
    """Detects a barrio for every string; ``inside`` decides whether candidates fall in it."""

    def __init__(self, inside: bool):
        self.inside = inside

    def detect(self, raw):
        return _Place()

    def lookup(self, kind, name):
        return object()

    def comuna_entry(self, value):
        return None

    def contains(self, kind, name, lon, lat, buffer_m=0.0):
        return self.inside

    def locate(self, lon, lat):
        return {}


def _gated(stub_normalizer, inside, **kw):
    stub = stub_normalizer([{"direccion": "CL 5 # 38 - 10"}], scores=[[0.95]], threshold=0.5)
    kw.setdefault("ambiguity_delta", 0.0)
    return normalize_strict(stub, ["CL 5 # 38 - 10"], gazetteer=_FakeGazetteer(inside), **kw).iloc[0]


def test_gate_default_rejects_when_every_candidate_is_outside(stub_normalizer):
    row = _gated(stub_normalizer, inside=False)
    assert row["estado"] == "SIN_MATCH" and row["motivo"].startswith("candidatos fuera de barrio Zzz")


def test_gate_fallback_keeps_the_text_match_but_downgrades_it(stub_normalizer):
    row = _gated(stub_normalizer, inside=False, gate_fallback=True)
    assert row["estado"] == "OK"
    assert row["nivel_precision"] == "manzana"
    assert "fuera de barrio Zzz" in row["motivo"]


def test_gate_fallback_is_a_no_op_when_the_gate_keeps_a_candidate(stub_normalizer):
    off = _gated(stub_normalizer, inside=True)
    on = _gated(stub_normalizer, inside=True, gate_fallback=True)
    assert off.to_dict() == on.to_dict()
    assert off["nivel_precision"] == "predio" and off["motivo"] == ""


def test_gate_fallback_still_applies_the_confidence_threshold(stub_normalizer):
    stub = stub_normalizer([{"direccion": "CL 5 # 38 - 10"}], scores=[[0.30]], threshold=0.5)
    row = normalize_strict(stub, ["CL 5 # 38 - 10"], gazetteer=_FakeGazetteer(False), gate_fallback=True).iloc[0]
    assert row["estado"] == "SIN_MATCH" and row["motivo"].startswith("confianza")


def test_gate_fallback_from_the_artifact_decision_block(stub_normalizer):
    stub = stub_normalizer([{"direccion": "CL 5 # 38 - 10"}], scores=[[0.95]], threshold=0.5)
    stub.decision = {"gate_fallback": True}
    row = normalize_strict(stub, ["CL 5 # 38 - 10"], gazetteer=_FakeGazetteer(False), ambiguity_delta=0.0).iloc[0]
    assert row["estado"] == "OK"


# ---------------------------------------------------------------------------
# load_tuning: the optional ``decision`` block
# ---------------------------------------------------------------------------
def _write_tuning(tmp_path, chosen_extra):
    chosen = {"weights": {"sim": 0.7, "fuzz": 0.1, "struct": 0.1, "geo": 0.1}, "threshold": 0.7, **chosen_extra}
    (tmp_path / "tuning.json").write_text(json.dumps({"chosen": chosen}), encoding="utf-8")
    return str(tmp_path)


def test_load_tuning_without_decision_block_gives_an_empty_decision(tmp_path):
    assert load_tuning(_write_tuning(tmp_path, {}))["decision"] == {}
    assert load_tuning(str(tmp_path / "missing"))["decision"] == {}


def test_load_tuning_reads_and_normalises_the_decision_block(tmp_path):
    out = load_tuning(_write_tuning(tmp_path, {"decision": {
        "soft_rules": ["bis_cruce", "letra_cruce_cardinal"], "max_soft": 2, "gate_fallback": True}}))
    assert out["decision"] == {"soft_rules": ["bis_cruce", "letra_cruce_cardinal"], "max_soft": 2, "gate_fallback": True}


@pytest.mark.parametrize("bad", [
    "not a dict", {"soft_rules": ["nope"]}, {"max_soft": -1}, {"max_soft": "1"}, {"gate_fallback": "yes"},
    {"unknown_key": 1},
])
def test_load_tuning_rejects_a_malformed_decision_block(tmp_path, bad):
    with pytest.raises(ValueError):
        load_tuning(_write_tuning(tmp_path, {"decision": bad}))


def test_soft_rule_names_are_exported_by_service():
    assert service.SOFT_RULE_NAMES == SOFT_RULE_NAMES
    assert len(set(SOFT_RULE_NAMES)) == len(SOFT_RULE_NAMES)


# ---------------------------------------------------------------------------
# scripts/eval_strict.py: decision overrides
# ---------------------------------------------------------------------------
def _eval_strict():
    import importlib.util
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("script_eval_strict_s3", os.path.join(root, "scripts", "eval_strict.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ns(**kw):
    import argparse

    base = dict(soft_rules=None, max_soft=None, gate_fallback=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_eval_strict_decision_kwargs_default_to_the_artifact():
    ev = _eval_strict()
    assert ev.decision_kwargs(_ns()) == {"soft_rules": None, "max_soft": None, "gate_fallback": None}


def test_eval_strict_decision_kwargs_pass_explicit_values_through():
    ev = _eval_strict()
    got = ev.decision_kwargs(_ns(soft_rules="letra_cruce,bis_via", max_soft=2, gate_fallback=1))
    assert got == {"soft_rules": frozenset({"letra_cruce", "bis_via"}), "max_soft": 2, "gate_fallback": True}
    assert ev.decision_kwargs(_ns(soft_rules="", gate_fallback=0))["soft_rules"] == frozenset()
    assert ev.decision_kwargs(_ns(gate_fallback=0))["gate_fallback"] is False


@pytest.mark.parametrize("bad", [dict(soft_rules="nope"), dict(max_soft=-1)])
def test_eval_strict_decision_kwargs_reject_malformed_values(bad):
    ev = _eval_strict()
    with pytest.raises(ValueError):
        ev.decision_kwargs(_ns(**bad))


# ---------------------------------------------------------------------------
# paths.model_path: decision-only experiments inherit the production model files
# ---------------------------------------------------------------------------
def _paths():
    from cali_address import paths

    return paths


def test_model_path_prefers_the_local_file(tmp_path):
    paths = _paths()
    (tmp_path / "model.pt").write_bytes(b"x")
    (tmp_path / "experiment.json").write_text(json.dumps({"inherits_model": True}))
    assert paths.model_path("model.pt", str(tmp_path)) == str(tmp_path / "model.pt")


def test_model_path_without_the_flag_never_falls_back(tmp_path):
    paths = _paths()
    assert paths.model_path("model.pt", str(tmp_path)) == str(tmp_path / "model.pt")
    (tmp_path / "experiment.json").write_text(json.dumps({"inherits_model": False}))
    assert paths.model_path("catastro_emb.pt", str(tmp_path)) == str(tmp_path / "catastro_emb.pt")


def test_model_path_inherits_production_files_when_declared(tmp_path):
    paths = _paths()
    (tmp_path / "experiment.json").write_text(json.dumps({"inherits_model": True}))
    for name in paths.MODEL_FILES:
        assert paths.model_path(name, str(tmp_path)) == os.path.join(paths.DEFAULT_ARTIFACTS_DIR, name)


@pytest.mark.parametrize("payload", ["not json", "[]", '{"inherits_model": "yes"}', '{"inherits_model": 1}', ""])
def test_model_path_ignores_malformed_or_non_boolean_flags(tmp_path, payload):
    paths = _paths()
    (tmp_path / "experiment.json").write_text(payload)
    assert paths.model_path("model.pt", str(tmp_path)) == str(tmp_path / "model.pt")


def test_model_path_rejects_files_that_are_not_model_files(tmp_path):
    with pytest.raises(ValueError):
        _paths().model_path("tuning.json", str(tmp_path))


def test_model_path_for_the_default_dir_is_itself():
    paths = _paths()
    assert paths.model_path("model.pt", paths.DEFAULT_ARTIFACTS_DIR) == os.path.join(paths.DEFAULT_ARTIFACTS_DIR, "model.pt")
