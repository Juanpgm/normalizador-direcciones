"""Realistic-noise measurement and the augmenter's ``op_scale`` hook."""

from __future__ import annotations

import random

from tiny import tiny_docs  # noqa: F401

from cali_address.augment import OPERATIONS, corrupt, corrupt_many
from cali_address.realistic_noise import (
    PATTERNS, SCALE_MAX, SCALE_MIN, compute_op_scale, estimate_op_scale, measure_patterns,
)


def test_patterns_detect_the_structures():
    f = measure_patterns([
        "CL 13 # 50 - 95", "KR 5BN # 33 - 01", "CL 13 50 95", "Carrera 5 Norte No. 3 - 4",
        "cl 5 # 3 - 4 apto 301 Cali", "CL 5 # 3 - 4 CL 5 # 3 - 4",
    ])
    assert set(f) == set(PATTERNS)
    assert f["has_hash"] == 4 / 6 and f["missing_separators"] == 1 / 6
    assert f["digit_letter_attached"] > 0 and f["via_type_spelled"] == 1 / 6
    assert f["has_lowercase"] > 0 and f["city_suffix"] == 1 / 6 and f["double_plate"] > 0
    assert f["trailing_complement"] == 1 / 6


def test_measure_ignores_empty_none_and_handles_no_input():
    assert all(v == 0.0 for v in measure_patterns([]).values())
    assert all(v == 0.0 for v in measure_patterns([None, "", "   "]).values())
    f = measure_patterns([None, "CL 1 # 2 - 3", ""])
    assert f["has_hash"] == 1.0


def test_op_scale_is_clamped_and_uses_known_operations():
    real = {k: 0.0 for k in PATTERNS}
    real["digit_letter_attached"] = 1.0
    real["has_lowercase"] = 0.0
    synth = {k: 0.5 for k in PATTERNS}
    scale = compute_op_scale(real, synth)
    assert set(scale) <= set(OPERATIONS)
    assert all(SCALE_MIN <= v <= SCALE_MAX for v in scale.values())
    assert scale["glue_letters"] > 1.0 and scale["lowercase"] < 1.0


def test_op_scale_missing_patterns_is_empty():
    assert compute_op_scale({}, {}) == {}


def test_corrupt_without_scale_is_bit_identical():
    a = [corrupt("CL 5 # 10 - 20", random.Random(i)) for i in range(50)]
    b = [corrupt("CL 5 # 10 - 20", random.Random(i), op_scale=None) for i in range(50)]
    c = [corrupt("CL 5 # 10 - 20", random.Random(i), op_scale={}) for i in range(50)]
    assert a == b == c


def test_scale_changes_operator_frequency():
    def rate(scale):
        rng = random.Random(0)
        outs = [corrupt("CL 5 # 10 - 20", rng, op_scale=scale) for _ in range(600)]
        return sum(o == o.lower() for o in outs)

    assert rate({"lowercase": 3.0}) > rate({"lowercase": 0.25})


def test_corrupt_many_forwards_scale_and_edge_inputs():
    assert len(corrupt_many("CL 5 # 10 - 20", 3, random.Random(1), op_scale={"lowercase": 2.0})) == 3
    assert corrupt("", random.Random(1), op_scale={"lowercase": 2.0}) == ""


def test_estimate_op_scale_end_to_end_and_no_text_leak():
    docs = tiny_docs(48)["direccion"].tolist()
    est = estimate_op_scale(["KR 5BN # 33 - 01", "CL 13 50 95", None, ""], docs, n_docs=30, seed=1)
    assert est["n_real"] == 2 and est["n_synthetic"] > 0 and est["op_scale"]
    blob = str(est)
    assert "KR 5BN" not in blob and "CL 13 50" not in blob     # numbers only, no address text


def test_estimate_op_scale_without_real_strings():
    est = estimate_op_scale([], tiny_docs(10)["direccion"].tolist(), n_docs=5)
    assert est["op_scale"] == {} and est["n_real"] == 0
