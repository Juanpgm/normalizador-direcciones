"""Offline fit helpers of the hybrid reliability model (scripts/fit_reliability.py)."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

pytest.importorskip("sklearn")

from cali_address.reliability import interpolate_knots  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("fit_reliability", os.path.join(ROOT, "scripts", "fit_reliability.py"))
fr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fr)


def _step_values(knots, x):
    return interpolate_knots(knots["x"], knots["y"], x)


# ---------------------------------------------------------------------------
# isotonic knots
# ---------------------------------------------------------------------------
def test_knots_are_non_decreasing_and_x_strictly_increasing_on_noisy_data():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.73, 1.0, 400)
    y = (rng.uniform(size=400) < 0.5 + 0.2 * (x - 0.73)).astype(int)
    k = fr.fit_isotonic_knots(x, y, min_support=15)
    assert all(b >= a for a, b in zip(k["y"], k["y"][1:]))
    assert all(b > a for a, b in zip(k["x"], k["x"][1:]))
    assert sum(k["n"]) == 400


def test_unsorted_input_gives_the_same_knots_as_sorted():
    rng = np.random.default_rng(1)
    x = rng.uniform(0.7, 1.0, 120)
    y = (rng.uniform(size=120) < x).astype(int)
    perm = rng.permutation(120)
    a = fr.fit_isotonic_knots(x, y, min_support=5)
    b = fr.fit_isotonic_knots(x[perm], y[perm], min_support=5)
    assert a == b


def test_pooling_violations():
    # labels decrease with score -> everything pools into one flat step at the mean
    x = np.linspace(0.7, 1.0, 40)
    y = np.array([1] * 20 + [0] * 20)
    k = fr.fit_isotonic_knots(x, y, min_support=1)
    assert len(k["y"]) == 1 and k["y"][0] == pytest.approx(0.5)


def test_perfectly_ordered_labels_give_two_steps_zero_and_one():
    x = np.linspace(0.7, 1.0, 40)
    y = (x > 0.85).astype(int)
    k = fr.fit_isotonic_knots(x, y, min_support=1)
    assert k["y"][0] == 0.0 and k["y"][-1] == 1.0 and len(k["y"]) == 2


def test_ties_share_one_value():
    x = np.array([0.8] * 10 + [0.9] * 10)
    y = np.array([1, 0] * 5 + [1] * 10)
    k = fr.fit_isotonic_knots(x, y, min_support=1)
    assert k["n"] == [10, 10] and k["y"] == pytest.approx([0.5, 1.0])


def test_single_point_and_single_value_inputs():
    k = fr.fit_isotonic_knots(np.array([0.8]), np.array([1]), min_support=15)
    assert k["x"] == [0.8] and k["y"] == [1.0]
    k = fr.fit_isotonic_knots(np.full(30, 0.9), np.zeros(30, dtype=int), min_support=15)
    assert k["x"] == [0.9] and k["y"] == [0.0]


@pytest.mark.parametrize("label", [0, 1])
def test_all_same_label_is_a_single_constant_step(label):
    x = np.linspace(0.7, 1.0, 50)
    k = fr.fit_isotonic_knots(x, np.full(50, label), min_support=15)
    assert len(k["y"]) == 1 and k["y"][0] == float(label)


def test_empty_input_raises_value_error():
    with pytest.raises(ValueError):
        fr.fit_isotonic_knots(np.array([]), np.array([]), min_support=15)


def test_small_top_and_bottom_steps_are_merged_into_their_neighbour():
    x = np.concatenate([np.full(5, 0.70), np.full(40, 0.80), np.full(40, 0.90), np.full(4, 1.00)])
    y = np.concatenate([np.zeros(5), [0] * 30 + [1] * 10, [0] * 10 + [1] * 30, np.ones(4)]).astype(int)
    unmerged = fr.fit_isotonic_knots(x, y, min_support=1)
    merged = fr.fit_isotonic_knots(x, y, min_support=15)
    assert len(unmerged["y"]) == 4 and min(unmerged["n"]) == 4
    assert len(merged["y"]) == 2 and all(n >= 15 for n in merged["n"])
    assert sum(merged["n"]) == 89
    assert all(b >= a for a, b in zip(merged["y"], merged["y"][1:]))


def test_merge_never_leaves_a_small_step_when_enough_rows_exist():
    rng = np.random.default_rng(3)
    x = rng.uniform(0.73, 1.0, 300)
    y = (rng.uniform(size=300) < 0.4 + 0.5 * (x - 0.73)).astype(int)
    k = fr.fit_isotonic_knots(x, y, min_support=15)
    assert k["n"][0] >= 15 and k["n"][-1] >= 15


def test_all_rows_below_support_collapse_to_one_step():
    k = fr.fit_isotonic_knots(np.array([0.7, 0.8, 0.9]), np.array([0, 1, 1]), min_support=15)
    assert len(k["y"]) == 1 and k["y"][0] == pytest.approx(2 / 3)


def test_served_step_function_is_monotone_and_clamped():
    x = np.linspace(0.73, 1.0, 100)
    y = (np.random.default_rng(4).uniform(size=100) < x - 0.1).astype(int)
    k = fr.fit_isotonic_knots(x, y, min_support=10)
    grid = np.linspace(0.5, 1.2, 300)
    vals = [_step_values(k, g) for g in grid]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
    assert vals[0] == k["y"][0] and vals[-1] == k["y"][-1]


# ---------------------------------------------------------------------------
# q table
# ---------------------------------------------------------------------------
def test_q_shrinkage_hand_computed():
    kinds = np.array(["exacto"] * 10 + ["relajada"] * 2 + ["parcial"] * 4 + ["exacto"] * 5, dtype=object)
    manz = np.array([1] * 10 + [1] * 2 + [1] * 4 + [0] * 5)
    pred = np.array([1] * 6 + [0] * 4 + [1, 0] + [0] * 4 + [1] * 5)  # last 5 are manzana-wrong: ignored
    t = fr.fit_q_table(kinds, manz, pred, m=10)
    # among manzana-correct: 16 rows, 7 hits -> q_global = 7/16
    assert t["q_global"] == pytest.approx(7 / 16)
    assert t["kinds"]["exacto"]["n"] == 10 and t["kinds"]["exacto"]["hits"] == 6
    assert t["kinds"]["exacto"]["q"] == pytest.approx((6 + 10 * 7 / 16) / 20)
    assert t["kinds"]["relajada"]["q"] == pytest.approx((1 + 10 * 7 / 16) / 12)
    assert t["kinds"]["parcial"]["q"] == pytest.approx((0 + 10 * 7 / 16) / 14)


def test_q_kind_without_rows_falls_back_to_q_global():
    kinds = np.array(["exacto"] * 4, dtype=object)
    t = fr.fit_q_table(kinds, np.ones(4, dtype=int), np.array([1, 1, 0, 1]), m=20)
    assert t["kinds"]["compartida"] == {"n": 0, "hits": 0, "q": pytest.approx(0.75)}
    assert t["q_global"] == 0.75


def test_q_large_n_approaches_observed_rate():
    kinds = np.array(["exacto"] * 1000, dtype=object)
    pred = np.array([1] * 800 + [0] * 200)
    t = fr.fit_q_table(kinds, np.ones(1000, dtype=int), pred, m=10)
    assert t["kinds"]["exacto"]["q"] == pytest.approx(0.8, abs=1e-9)


def test_q_no_manzana_correct_rows_is_defined():
    kinds = np.array(["exacto"] * 3, dtype=object)
    t = fr.fit_q_table(kinds, np.zeros(3, dtype=int), np.zeros(3, dtype=int), m=10)
    assert 0.0 <= t["q_global"] <= 1.0 and 0.0 <= t["kinds"]["exacto"]["q"] <= 1.0


# ---------------------------------------------------------------------------
# hybrid predictions and CV
# ---------------------------------------------------------------------------
def _toy(n=240, seed=5):
    rng = np.random.default_rng(seed)
    score = rng.uniform(0.73, 1.0, n)
    kinds = rng.choice(np.array(["exacto", "relajada", "compartida", "parcial"], dtype=object), n)
    manz = (rng.uniform(size=n) < 0.5 + 0.4 * (score - 0.73)).astype(int)
    pred = manz * (rng.uniform(size=n) < 0.75).astype(int)
    groups = np.array(["a", "b", "c"], dtype=object)[rng.integers(0, 3, n)]
    return score, kinds, manz, pred, groups


def test_hybrid_oof_predio_never_exceeds_manzana_and_is_bounded():
    score, kinds, manz, pred, groups = _toy()
    out = fr.hybrid_oof(score, kinds, manz, pred, groups)
    assert np.all(out["predio"] <= out["manzana"] + 1e-12)
    assert np.all((out["manzana"] >= 0.02) & (out["manzana"] <= 0.98))
    assert set(out["chosen_m"].values()) <= set(fr.M_GRID)


def test_hybrid_oof_does_not_use_the_held_out_group():
    score, kinds, manz, pred, groups = _toy()
    base = fr.hybrid_oof(score, kinds, manz, pred, groups)
    manz2, pred2 = manz.copy(), pred.copy()
    held = groups == "a"
    manz2[held], pred2[held] = 1 - manz2[held], 1 - pred2[held]  # flip labels of one group only
    flipped = fr.hybrid_oof(score, kinds, manz2, pred2, groups)
    # predictions for group "a" only depend on b and c labels, which did not change
    assert np.allclose(base["manzana"][held], flipped["manzana"][held])
    assert np.allclose(base["predio"][held], flipped["predio"][held])


def test_hybrid_predict_relaxed_lower_than_exact_when_q_lower():
    iso = {"x": [0.8], "y": [0.8], "n": [50]}
    q = {"q_global": 0.7, "kinds": {"exacto": {"q": 0.8}, "relajada": {"q": 0.5}}}
    pp, pm = fr.hybrid_predict(iso, q, np.array([0.9, 0.9]), np.array(["exacto", "relajada"], dtype=object))
    assert pm[0] == pm[1] and pp[1] < pp[0]


def test_serve_time_and_fit_time_predictions_agree(tmp_path):
    """The offline predictor must equal what HybridReliability serves."""
    from cali_address.reliability import HybridReliability
    score, kinds, manz, pred, _ = _toy()
    iso = fr.fit_isotonic_knots(score, manz, min_support=15)
    qt = fr.fit_q_table(kinds, manz, pred, m=20)
    pp, pm = fr.hybrid_predict(iso, qt, score, kinds)
    model = HybridReliability(
        xs=iso["x"], ys=iso["y"], q={k: v["q"] for k, v in qt["kinds"].items()}, q_global=qt["q_global"])
    from cali_address.service import reliability_features
    kind_flags = {"exacto": {}, "relajada": {"relaxed_plate": True}, "compartida": {"nivel": "manzana"},
                  "parcial": {"nivel": "esquina"}}
    for i in range(0, len(score), 17):
        kw = kind_flags[kinds[i]]
        feats = reliability_features(score[i], 1.0, kw.get("relaxed_letter", False), kw.get("relaxed_plate", False),
                                     3 if kw.get("nivel") == "manzana" else 1, kw.get("nivel", "predio"), False)
        sp, sm = model.predict(feats)
        assert sm == pytest.approx(pm[i]) and sp == pytest.approx(pp[i])


# ---------------------------------------------------------------------------
# ship rule
# ---------------------------------------------------------------------------
def _m(brier, auc):
    return {"brier": brier, "auc": auc}


def test_ship_rule_accepts_within_tolerance_and_rejects_worse_auc_or_brier():
    base = {"predio": _m(0.2409, 0.491), "manzana": _m(0.1422, 0.508)}
    good = {"predio": _m(0.2420, 0.495), "manzana": _m(0.1326, 0.582)}
    assert fr.ships(good, base)
    assert not fr.ships({**good, "predio": _m(0.2431, 0.495)}, base)  # Brier > +0.002
    assert not fr.ships({**good, "manzana": _m(0.13, 0.50)}, base)  # AUC lower
    assert not fr.ships({**good, "manzana": _m(0.13, float("nan"))}, base)  # NaN AUC fails
