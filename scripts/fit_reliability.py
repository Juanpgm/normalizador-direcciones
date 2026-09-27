"""Fit the calibrated reliability model (``artifacts/reliability.json``).

Three artifact kinds (``--kind``): ``hybrid`` (default), ``cells`` and ``logistic``.

``hybrid``: manzana is a monotone isotonic (PAV) fit of ``correct_manzana`` on the raw
score (steps with < 15 rows at either end merged inward, served as linear
interpolation between step midpoints, clipped to [0.02, 0.98]); predio is
``manzana * q_kind`` with ``q_kind`` the empirical-Bayes shrunk
P(predio correct | manzana correct) per row kind (m from {10, 20, 40}, nested CV).
Ships to ``--out`` only if, out-of-fold, Brier is at most 0.002 worse and AUC not lower
than logistic v2 on both targets; otherwise it goes to ``--candidate-out``.

``cells``: reliability per (row kind x margin band) cell, see
``cali_address.reliability.reliability_cell``. Per target (predio, manzana) the rate
of a cell is an empirical-Bayes shrunk hit rate ``(k + m * p_parent) / (n + m)`` with
hierarchical parents (cell -> kind marginal -> global rate). Within a kind the rates
are forced non-decreasing along the margin bands (pool-adjacent-violators) and
``manzana >= predio`` is enforced. The prior strength ``m`` is chosen from
{10, 20, 40} by leave-one-dataset-out log loss, nested inside every outer fold. The
ship rule compares the out-of-fold cell model with the out-of-fold logistic model on
the same rows: Brier at most 0.002 worse and AUC not lower, on both targets.

``logistic`` (the previous model): reads the rows parquet written by ``scripts/eval_strict.py`` (which stores the
``feat_*`` columns captured by ``service.reliability_features``), keeps ONLY OK
rows with ground truth, and fits one L2-regularized logistic regression per
target (``correct_predial`` and ``correct_manzana``) on standardized features.

Validation is leave-one-dataset-out (LODO) over the datasets that have ground
truth. The regularization strength C is chosen by an INNER leave-one-dataset-out
loop inside each outer fold (nested), so the reported out-of-fold numbers never
saw the held-out dataset, neither for fitting nor for choosing C. The shipped
model is then refit on all rows with the C chosen by the outer LODO.

Baselines, on the same out-of-fold rows: (b) the raw ``confianza`` used as a
probability, (c) isotonic regression on the raw score alone.

Caveat printed with the results: labels come from GPS point-in-polygon and
neighbouring parcels are ~6 m apart, so predio labels are noisy and the measured
P(predio) is a conservative estimate; the sample is small (a few hundred rows).

Usage
-----
    PYTHONPATH=src python scripts/fit_reliability.py --rows artifacts/strict_eval_paso7_rows.parquet
    PYTHONPATH=src python scripts/fit_reliability.py --kind logistic --out <path>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address.reliability import (  # noqa: E402
    BANDS, CLIP, FEATURE_NAMES, HYBRID_CLIP, KINDS, P_MAX, P_MIN, interpolate_knots, reliability_cell,
)

#: Ship rule: out-of-fold log loss must not exceed the raw-score baseline, and Brier may
#: exceed it by at most this much (a tie at n~500 is noise, not evidence of harm).
BRIER_TOLERANCE = 0.005
#: A feature needs this many non-zero training rows to get a coefficient; rarer ones
#: (e.g. shared addresses, ~7 rows) would be amplified by standardization and clipped
#: at 5 sigma into an absurd logit, so they are left out of the fit (coefficient 0).
MIN_SUPPORT = 10
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
TARGETS = {"predio": "correct_predial", "manzana": "correct_manzana"}
SCORE_INDEX = FEATURE_NAMES.index("confidence")
VERSION_PREFIX = "reliability-v2"
CELLS_VERSION_PREFIX = "reliability-v3"
#: Prior strengths tried for the cell model; ``DEFAULT_M`` is used when there are too few groups to choose.
M_GRID = (10, 20, 40)
DEFAULT_M = 20
#: Cell model ships only if its out-of-fold Brier is at most this much worse than the logistic model's.
SHIP_BRIER_TOLERANCE = 0.002
#: Out-of-fold numbers of the previous logistic v2 (paso6b rows), for reference in the report.
LOGISTIC_V2_REFERENCE = {"predio": {"brier": 0.2409, "ece": 0.084, "auc": 0.491},
                         "manzana": {"brier": 0.1422, "ece": 0.050, "auc": 0.508}}


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------
def fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return mean, std


def apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.clip((X - mean) / std, -CLIP, CLIP)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def fit_logistic(Z: np.ndarray, y: np.ndarray, C: float, score_index: int = SCORE_INDEX,
                 support: np.ndarray | None = None, min_support: int = MIN_SUPPORT) -> dict:
    """L2 logistic fit on standardized ``Z``; the raw-score coefficient is kept >= 0.

    If the fit gives the score a negative sign (a small-sample artifact), the score
    feature is dropped (coefficient forced to 0), the model is refit and
    ``dropped_score`` is True. A single-class target yields a constant model.
    Columns whose ``support`` (non-zero row count) is below ``min_support`` are
    excluded from the fit.
    """
    y = np.asarray(y, dtype=int)
    n_feat = Z.shape[1]
    if len(np.unique(y)) < 2:
        rate = float(np.clip(y.mean() if len(y) else 0.5, 0.01, 0.99))
        return {"coef": np.zeros(n_feat), "intercept": float(np.log(rate / (1 - rate))),
                "dropped_score": False}

    def _fit(cols):
        if not cols:  # nothing left to fit: constant (base-rate) model
            rate = float(np.clip(y.mean(), 0.01, 0.99))
            return np.zeros(n_feat), float(np.log(rate / (1 - rate)))
        clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=1000)
        clf.fit(Z[:, cols], y)
        coef = np.zeros(n_feat)
        coef[cols] = clf.coef_[0]
        return coef, float(clf.intercept_[0])

    cols = [j for j in range(n_feat) if support is None or support[j] >= min_support or j == score_index]
    coef, intercept = _fit(cols)
    dropped = False
    if coef[score_index] < 0:
        cols.remove(score_index)
        coef, intercept = _fit(cols)
        dropped = True
    return {"coef": coef, "intercept": intercept, "dropped_score": dropped}


def predict(fit: dict, Z: np.ndarray) -> np.ndarray:
    return np.clip(_sigmoid(Z @ fit["coef"] + fit["intercept"]), P_MIN, P_MAX)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 5) -> float:
    """Expected calibration error with equal-count bins (empty bins are skipped)."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    if len(p) == 0:
        return float("nan")
    order = np.argsort(p, kind="stable")
    total = 0.0
    for idx in np.array_split(order, bins):
        if len(idx):
            total += len(idx) / len(p) * abs(p[idx].mean() - y[idx].mean())
    return float(total)


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, dtype=float), P_MIN, P_MAX)
    y = np.asarray(y, dtype=float)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    return {
        "n": int(len(y)),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "ece": ece(p, y),
        "auc": auc,
    }


def reliability_table(p: np.ndarray, y: np.ndarray, bins: int = 5) -> pd.DataFrame:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    order = np.argsort(p, kind="stable")
    rows = []
    for b, idx in enumerate(np.array_split(order, bins)):
        if len(idx):
            rows.append({"bin": b + 1, "n": len(idx), "mean_pred": p[idx].mean(),
                         "observed": y[idx].mean(), "pred_range": f"{p[idx].min():.3f}-{p[idx].max():.3f}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# cross-validation
# ---------------------------------------------------------------------------
def _fit_scaled(X, y, C):
    mean, std = fit_scaler(X)
    fit = fit_logistic(apply_scaler(X, mean, std), y, C, support=(X != 0).sum(axis=0))
    return {**fit, "mean": mean, "std": std}


def _predict_raw(fit, X):
    return predict(fit, apply_scaler(X, fit["mean"], fit["std"]))


def _logloss(p, y):
    return metrics(p, y)["logloss"]


def choose_c(X, y, groups) -> float:
    """C minimizing pooled leave-one-group-out log loss."""
    best, best_ll = C_GRID[0], np.inf
    for C in C_GRID:
        oof = np.zeros(len(y))
        for g in np.unique(groups):
            tr, te = groups != g, groups == g
            oof[te] = _predict_raw(_fit_scaled(X[tr], y[tr], C), X[te])
        ll = _logloss(oof, y)
        if ll < best_ll:
            best, best_ll = C, ll
    return best


def nested_lodo(X, y, groups, base_p) -> dict:
    """Out-of-fold predictions: model (nested C), raw score, isotonic-on-score."""
    oof = {"model": np.zeros(len(y)), "raw": np.clip(base_p, P_MIN, P_MAX), "isotonic": np.zeros(len(y))}
    chosen = {}
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        C = choose_c(X[tr], y[tr], groups[tr])
        chosen[str(g)] = C
        oof["model"][te] = _predict_raw(_fit_scaled(X[tr], y[tr], C), X[te])
        iso = IsotonicRegression(y_min=P_MIN, y_max=P_MAX, out_of_bounds="clip", increasing=True)
        iso.fit(base_p[tr], y[tr])
        oof["isotonic"][te] = iso.predict(base_p[te])
    return {"oof": oof, "chosen_c": chosen}


# ---------------------------------------------------------------------------
# cell model
# ---------------------------------------------------------------------------
def shrink(hits: float, n: float, parent: float, m: float) -> float:
    """Empirical-Bayes shrunk rate ``(hits + m * parent) / (n + m)``; ``parent`` when there is no mass."""
    denom = n + m
    return float(parent) if denom <= 0 else float((hits + m * parent) / denom)


def pav_increasing(values, weights) -> list[float]:
    """Weighted pool-adjacent-violators: the closest non-decreasing sequence to ``values``."""
    blocks: list[list[float]] = []  # [weighted sum, weight, count]
    for v, w in zip(values, weights):
        w = max(float(w), 1e-12)
        blocks.append([float(v) * w, w, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            top = blocks.pop()
            blocks[-1][0] += top[0]
            blocks[-1][1] += top[1]
            blocks[-1][2] += top[2]
    out: list[float] = []
    for total, weight, count in blocks:
        out.extend([total / weight] * count)
    return out


def _target_rates(kinds: np.ndarray, bands: np.ndarray, y: np.ndarray, m: float) -> dict:
    y = np.asarray(y, dtype=float)
    base = float(y.mean()) if len(y) else 0.5
    out = {"global": base, "kinds": {}, "cells": {}}
    for kind in KINDS:
        in_kind = kinds == kind
        n_k, k_k = int(in_kind.sum()), float(y[in_kind].sum())
        kind_rate = shrink(k_k, n_k, base, m)
        out["kinds"][kind] = {"n": n_k, "hits": int(k_k), "rate": kind_rate}
        raw, weights, stats = [], [], []
        for band in BANDS:
            sel = in_kind & (bands == band)
            n_c, k_c = int(sel.sum()), float(y[sel].sum())
            stats.append((n_c, int(k_c)))
            raw.append(shrink(k_c, n_c, kind_rate, m))
            weights.append(n_c + m)
        for band, (n_c, k_c), rate in zip(BANDS, stats, pav_increasing(raw, weights)):
            out["cells"][(kind, band)] = {"n": n_c, "hits": k_c, "rate": rate}
    return out


def fit_cell_table(kinds, bands, y_predio, y_manzana, m: float) -> dict:
    """Serving-format table (``global`` / ``kinds`` / ``cells``) from labelled OK rows.

    Every (kind, band) cell is emitted, empty ones carrying their parent's rate.
    ``manzana >= predio`` is enforced per cell and per kind marginal.
    """
    kinds, bands = np.asarray(kinds, dtype=object), np.asarray(bands, dtype=object)
    rp = _target_rates(kinds, bands, np.asarray(y_predio), m)
    rm = _target_rates(kinds, bands, np.asarray(y_manzana), m)
    table = {
        "global": {"predio": rp["global"], "manzana": max(rm["global"], rp["global"])},
        "kinds": {}, "cells": [],
    }
    for kind in KINDS:
        p, q = rp["kinds"][kind], rm["kinds"][kind]
        table["kinds"][kind] = {
            "n": p["n"], "hits_predio": p["hits"], "hits_manzana": q["hits"],
            "predio": p["rate"], "manzana": max(q["rate"], p["rate"]),
        }
    for kind in KINDS:
        for band in BANDS:
            p, q = rp["cells"][(kind, band)], rm["cells"][(kind, band)]
            table["cells"].append({
                "kind": kind, "band": band, "n": p["n"], "hits_predio": p["hits"],
                "hits_manzana": q["hits"], "predio": p["rate"], "manzana": max(q["rate"], p["rate"]),
            })
    return table


def predict_cells(table: dict, kinds, bands) -> tuple[np.ndarray, np.ndarray]:
    """Look the rows up in ``table`` (cell -> kind marginal -> global), clipped like serving."""
    cell = {(c["kind"], c["band"]): (c["predio"], c["manzana"]) for c in table["cells"]}
    kind_rate = {k: (v["predio"], v["manzana"]) for k, v in table["kinds"].items()}
    glob = (table["global"]["predio"], table["global"]["manzana"])
    pp, pm = [], []
    for kind, band in zip(kinds, bands):
        a, b = cell.get((kind, band)) or kind_rate.get(kind) or glob
        a, b = min(max(a, P_MIN), P_MAX), min(max(b, P_MIN), P_MAX)
        pp.append(a)
        pm.append(max(a, b))
    return np.array(pp), np.array(pm)


def choose_m(kinds, bands, y_p, y_m, groups, grid=M_GRID) -> int:
    """Prior strength minimizing pooled leave-one-group-out log loss (both targets)."""
    if len(np.unique(groups)) < 2:
        return DEFAULT_M
    best, best_ll = DEFAULT_M, np.inf
    for m in grid:
        op, om = np.zeros(len(y_p)), np.zeros(len(y_p))
        for g in np.unique(groups):
            tr, te = groups != g, groups == g
            table = fit_cell_table(kinds[tr], bands[tr], y_p[tr], y_m[tr], m)
            op[te], om[te] = predict_cells(table, kinds[te], bands[te])
        ll = _logloss(op, y_p) + _logloss(om, y_m)
        if ll < best_ll:
            best, best_ll = m, ll
    return best


def cell_oof(kinds, bands, y_p, y_m, groups) -> dict:
    """Out-of-fold cell predictions; ``m`` is chosen inside each outer fold (nested)."""
    kinds, bands = np.asarray(kinds, dtype=object), np.asarray(bands, dtype=object)
    groups = np.asarray(groups, dtype=object)
    y_p, y_m = np.asarray(y_p), np.asarray(y_m)
    out = {"predio": np.zeros(len(y_p)), "manzana": np.zeros(len(y_p)), "chosen_m": {}}
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        m = choose_m(kinds[tr], bands[tr], y_p[tr], y_m[tr], groups[tr])
        out["chosen_m"][str(g)] = m
        table = fit_cell_table(kinds[tr], bands[tr], y_p[tr], y_m[tr], m)
        out["predio"][te], out["manzana"][te] = predict_cells(table, kinds[te], bands[te])
    return out


def ships(cell: dict, baseline: dict) -> bool:
    """Ship rule: per target, Brier <= baseline + 0.002 and AUC >= baseline AUC (NaN AUC fails)."""
    for target in TARGETS:
        c, b = cell[target], baseline[target]
        if not c["brier"] <= b["brier"] + SHIP_BRIER_TOLERANCE + 1e-9:
            return False
        if not c["auc"] >= b["auc"]:
            return False
    return True


# ---------------------------------------------------------------------------
# hybrid model: isotonic P(manzana | confianza) x per-kind q = P(predio | manzana)
# ---------------------------------------------------------------------------
#: Minimum training rows supporting the lowest / highest isotonic step (else merged inward).
MIN_STEP_SUPPORT = 15
HYBRID_VERSION_PREFIX = "reliability-v4"


def fit_isotonic_knots(x, y, min_support: int = MIN_STEP_SUPPORT) -> dict:
    """Monotone non-decreasing step fit of ``y`` on ``x`` as knots ``{"x", "y", "n"}``.

    Pool-adjacent-violators over the unique scores (ties share a value), then the
    lowest and highest steps with fewer than ``min_support`` rows are merged into
    their neighbour. Each step becomes one knot at the midpoint of its score range,
    so ``x`` is strictly increasing and ``y`` non-decreasing by construction.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) == 0:
        raise ValueError("cannot fit an isotonic step function on no rows")
    blocks: list[list[float]] = []  # [sum_y, n, x_min, x_max]
    for value in np.unique(x):
        sel = x == value
        blocks.append([float(y[sel].sum()), int(sel.sum()), float(value), float(value)])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
            top = blocks.pop()
            blocks[-1][0] += top[0]
            blocks[-1][1] += top[1]
            blocks[-1][3] = top[3]

    def _merge(i: int) -> None:  # merge block i+1 into block i
        blocks[i][0] += blocks[i + 1][0]
        blocks[i][1] += blocks[i + 1][1]
        blocks[i][3] = blocks[i + 1][3]
        del blocks[i + 1]

    while len(blocks) > 1 and blocks[0][1] < min_support:
        _merge(0)
    while len(blocks) > 1 and blocks[-1][1] < min_support:
        _merge(len(blocks) - 2)
    return {"x": [(b[2] + b[3]) / 2 for b in blocks], "y": [b[0] / b[1] for b in blocks],
            "n": [int(b[1]) for b in blocks]}


def fit_q_table(kinds, y_manzana, y_predio, m: float) -> dict:
    """Shrunk ``P(predio | manzana correct)`` per kind: ``(hits + m*q_global) / (n + m)``.

    Only rows whose manzana is right count. A kind with no such rows gets ``q_global``.
    """
    kinds = np.asarray(kinds, dtype=object)
    sel = np.asarray(y_manzana).astype(int) == 1
    yp = np.asarray(y_predio).astype(int)
    n_all = int(sel.sum())
    q_global = float(yp[sel].sum() / n_all) if n_all else 0.5
    table = {"q_global": q_global, "kinds": {}}
    for kind in KINDS:
        in_kind = sel & (kinds == kind)
        n, hits = int(in_kind.sum()), int(yp[in_kind].sum())
        table["kinds"][kind] = {"n": n, "hits": hits, "q": shrink(hits, n, q_global, m)}
    return table


def hybrid_predict(iso: dict, qtab: dict, scores, kinds) -> tuple[np.ndarray, np.ndarray]:
    """``(predio, manzana)`` exactly as ``HybridReliability.predict`` serves them."""
    lo, hi = HYBRID_CLIP
    pm = np.array([min(max(interpolate_knots(iso["x"], iso["y"], float(s)), lo), hi) for s in scores])
    q = np.array([min(max(qtab["kinds"].get(k, {}).get("q", qtab["q_global"]), 0.0), 1.0) for k in kinds])
    return np.minimum(np.maximum(pm * q, P_MIN), pm), pm


def choose_m_hybrid(score, kinds, y_m, y_p, groups, grid=M_GRID) -> int:
    """Prior strength minimizing pooled leave-one-group-out log loss of the predio score."""
    if len(np.unique(groups)) < 2:
        return DEFAULT_M
    best, best_ll = DEFAULT_M, np.inf
    for m in grid:
        oof = np.zeros(len(y_p))
        for g in np.unique(groups):
            tr, te = groups != g, groups == g
            iso = fit_isotonic_knots(score[tr], y_m[tr])
            oof[te] = hybrid_predict(iso, fit_q_table(kinds[tr], y_m[tr], y_p[tr], m), score[te], kinds[te])[0]
        ll = _logloss(oof, y_p)
        if ll < best_ll:
            best, best_ll = m, ll
    return best


def hybrid_oof(score, kinds, y_m, y_p, groups) -> dict:
    """Nested leave-one-group-out predictions; ``m`` is chosen inside each outer fold."""
    score = np.asarray(score, dtype=float)
    kinds, groups = np.asarray(kinds, dtype=object), np.asarray(groups, dtype=object)
    y_m, y_p = np.asarray(y_m).astype(int), np.asarray(y_p).astype(int)
    out = {"predio": np.zeros(len(y_p)), "manzana": np.zeros(len(y_p)), "chosen_m": {}}
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        m = choose_m_hybrid(score[tr], kinds[tr], y_m[tr], y_p[tr], groups[tr])
        out["chosen_m"][str(g)] = m
        iso = fit_isotonic_knots(score[tr], y_m[tr])
        out["predio"][te], out["manzana"][te] = hybrid_predict(
            iso, fit_q_table(kinds[tr], y_m[tr], y_p[tr], m), score[te], kinds[te])
    return out


def _clean(obj):
    """JSON-safe copy: NaN/inf become None."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def run_cells(args) -> int:
    df = load_training_rows(args.rows)
    groups = df["dataset"].to_numpy(dtype=object)
    X = df[[f"feat_{n}" for n in FEATURE_NAMES]].to_numpy(dtype=float)
    cells = [reliability_cell(dict(zip(FEATURE_NAMES, row))) for row in X]
    kinds = np.array([c[0] for c in cells], dtype=object)
    bands = np.array([c[1] for c in cells], dtype=object)
    y = {t: df[col].to_numpy(dtype=float).astype(int) for t, col in TARGETS.items()}
    base_p = df["confianza"].to_numpy(dtype=float)
    print(f"training rows (OK with GT): {len(df)}; per dataset: {dict(pd.Series(groups).value_counts())}")

    cv = cell_oof(kinds, bands, y["predio"], y["manzana"], groups)
    oof = {"predio": {"cells": cv["predio"]}, "manzana": {"cells": cv["manzana"]}}
    for target in TARGETS:
        lg = nested_lodo(X, y[target], groups, base_p)["oof"]
        if target == "manzana":  # same serve-time rule: P(manzana) >= P(predio)
            lg["model"] = np.maximum(lg["model"], oof["predio"]["logistic"])
        oof[target].update({"logistic": lg["model"], "raw": lg["raw"], "isotonic": lg["isotonic"]})
    labels = {"cells": "cell model", "logistic": "logistic (refit, same rows)", "raw": "raw confianza",
              "isotonic": "isotonic on score"}
    pooled = {}
    for target in TARGETS:
        pooled[target] = {k: metrics(v, y[target]) for k, v in oof[target].items()}
        _print_metrics(f"[{target}] pooled out-of-fold", {labels[k]: pooled[target][k] for k in labels})
        ref = LOGISTIC_V2_REFERENCE[target]
        print(f"  (logistic v2 on paso6b rows: brier {ref['brier']:.4f} ece {ref['ece']:.3f} auc {ref['auc']:.3f})")
        for g in np.unique(groups):
            mask = groups == g
            _print_metrics(f"[{target}] dataset={g}", {labels[k]: metrics(oof[target][k][mask], y[target][mask])
                                                      for k in ("cells", "logistic", "raw")})
    print(f"\nprior strength m chosen per outer fold: {cv['chosen_m']}")

    m = choose_m(kinds, bands, y["predio"], y["manzana"], groups)
    table = fit_cell_table(kinds, bands, y["predio"], y["manzana"], m)
    print(f"final m={m}")
    print(f"{'kind':<11}{'band':<11}{'n':>5}{'manz hits':>10}{'manz rate':>10}{'pred hits':>10}{'pred rate':>10}")
    for c in table["cells"]:
        print(f"{c['kind']:<11}{c['band']:<11}{c['n']:>5}{c['hits_manzana']:>10}{c['manzana']:>10.3f}"
              f"{c['hits_predio']:>10}{c['predio']:>10.3f}")

    ok = ships({t: pooled[t]["cells"] for t in TARGETS}, {t: pooled[t]["logistic"] for t in TARGETS})
    for t in TARGETS:
        c, b = pooled[t]["cells"], pooled[t]["logistic"]
        print(f"[{t}] cells vs logistic: brier {c['brier'] - b['brier']:+.4f}, auc {c['auc'] - b['auc']:+.3f}")
    if not ok:
        print("\nWARNING: cell model fails the ship rule (Brier within +0.002 and AUC not lower).")
        if not getattr(args, "force", False):
            print("Not writing (use --force to write a candidate artifact to --out anyway).")
            return 2
    payload = {
        "kind": "cells", "version": f"{CELLS_VERSION_PREFIX}-{dt.date.today():%Y%m%d}",
        "fit_date": dt.datetime.now().isoformat(timespec="seconds"),
        "m": m, "m_grid": list(M_GRID), "n_train": int(len(df)), "ship_rule_passed": bool(ok),
        "source_rows": os.path.basename(args.rows),
        **table,
        "cv": {"chosen_m_per_outer_fold": cv["chosen_m"],
               "pooled": {t: pooled[t] for t in TARGETS},
               "per_dataset": {t: {str(g): {k: metrics(oof[t][k][groups == g], y[t][groups == g])
                                            for k in ("cells", "logistic", "raw")}
                                   for g in np.unique(groups)} for t in TARGETS}},
        "notes": ["Rates are empirical-Bayes shrunk (cell -> kind -> global), monotone in the margin band within "
                  "a kind, and manzana >= predio.",
                  "Labels are GPS point-in-polygon on a small validation set; neighbouring parcels ~6 m apart make "
                  "predio labels noisy, so P(predio) is a conservative estimate."],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(_clean(payload), fh, indent=2)
    print(f"wrote {args.out}")
    return 0


def run_hybrid(args) -> int:
    df = load_training_rows(args.rows)
    groups = df["dataset"].to_numpy(dtype=object)
    X = df[[f"feat_{n}" for n in FEATURE_NAMES]].to_numpy(dtype=float)
    kinds = np.array([reliability_cell(dict(zip(FEATURE_NAMES, row)))[0] for row in X], dtype=object)
    cell_bands = np.array([reliability_cell(dict(zip(FEATURE_NAMES, row)))[1] for row in X], dtype=object)
    score = X[:, SCORE_INDEX]
    y = {t: df[col].to_numpy(dtype=float).astype(int) for t, col in TARGETS.items()}
    base_p = df["confianza"].to_numpy(dtype=float)
    print(f"training rows (OK with GT): {len(df)}; per dataset: {dict(pd.Series(groups).value_counts())}")

    hy = hybrid_oof(score, kinds, y["manzana"], y["predio"], groups)
    cv = cell_oof(kinds, cell_bands, y["predio"], y["manzana"], groups)
    oof = {"predio": {"hybrid": hy["predio"], "cells": cv["predio"]},
           "manzana": {"hybrid": hy["manzana"], "cells": cv["manzana"]}}
    for target in TARGETS:
        lg = nested_lodo(X, y[target], groups, base_p)["oof"]
        if target == "manzana":
            lg["model"] = np.maximum(lg["model"], oof["predio"]["logistic"])
        oof[target].update({"logistic": lg["model"], "raw": lg["raw"], "isotonic": lg["isotonic"]})
    labels = {"hybrid": "hybrid (v4)", "logistic": "logistic v2 (refit)", "cells": "cells (v3 cand.)",
              "raw": "raw confianza", "isotonic": "isotonic on score"}
    pooled = {}
    for target in TARGETS:
        pooled[target] = {k: metrics(oof[target][k], y[target]) for k in labels}
        _print_metrics(f"[{target}] pooled out-of-fold", {labels[k]: pooled[target][k] for k in labels})
        ref = LOGISTIC_V2_REFERENCE[target]
        print(f"  (logistic v2 reference: brier {ref['brier']:.4f} ece {ref['ece']:.3f} auc {ref['auc']:.3f})")
        for g in np.unique(groups):
            mask = groups == g
            _print_metrics(f"[{target}] dataset={g}", {labels[k]: metrics(oof[target][k][mask], y[target][mask])
                                                      for k in ("hybrid", "logistic", "raw")})
        print(f"\n[{target}] reliability table (hybrid OOF, 5 equal-count bins)")
        print(reliability_table(oof[target]["hybrid"], y[target]).to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nprior strength m chosen per outer fold: {hy['chosen_m']}")

    m = choose_m_hybrid(score, kinds, y["manzana"], y["predio"], groups)
    iso = fit_isotonic_knots(score, y["manzana"])
    qtab = fit_q_table(kinds, y["manzana"], y["predio"], m)
    print(f"final m={m}; isotonic steps={len(iso['y'])}, y range {min(iso['y']):.3f}-{max(iso['y']):.3f}")
    for x_, y_, n_ in zip(iso["x"], iso["y"], iso["n"]):
        print(f"  knot x={x_:.4f} y={y_:.3f} n={n_}")
    print(f"q_global={qtab['q_global']:.3f}")
    for kind in KINDS:
        row = qtab["kinds"][kind]
        print(f"  {kind:<11} n={row['n']:>4} hits={row['hits']:>4} q={row['q']:.3f}")

    ref_ok = ships({t: pooled[t]["hybrid"] for t in TARGETS}, LOGISTIC_V2_REFERENCE)
    same_rows_ok = ships({t: pooled[t]["hybrid"] for t in TARGETS}, {t: pooled[t]["logistic"] for t in TARGETS})
    for t in TARGETS:
        h, b = pooled[t]["hybrid"], pooled[t]["logistic"]
        print(f"[{t}] hybrid vs logistic v2 (refit): brier {h['brier'] - b['brier']:+.4f}, auc {h['auc'] - b['auc']:+.3f}")
    ok = ref_ok and same_rows_ok
    print(f"ship rule vs v2 reference: {ref_ok}; vs refit on same rows: {same_rows_ok}")
    payload = {
        "kind": "hybrid", "version": f"{HYBRID_VERSION_PREFIX}-{dt.date.today():%Y%m%d}",
        "fit_date": dt.datetime.now().isoformat(timespec="seconds"),
        "m": m, "m_grid": list(M_GRID), "n_train": int(len(df)), "ship_rule_passed": bool(ok),
        "source_rows": os.path.basename(args.rows), "clip": list(HYBRID_CLIP),
        "min_step_support": MIN_STEP_SUPPORT,
        "isotonic": iso, **qtab,
        "q": qtab["kinds"],
        "cv": {"chosen_m_per_outer_fold": hy["chosen_m"], "pooled": {t: pooled[t] for t in TARGETS},
               "logistic_v2_reference": LOGISTIC_V2_REFERENCE},
        "notes": ["manzana: isotonic (PAV) fit of correct_manzana on the raw score, served by linear "
                  "interpolation between step midpoints, clipped to [0.02, 0.98].",
                  "predio = manzana * q_kind, q_kind = shrunk P(predio correct | manzana correct) per row kind.",
                  "Labels are GPS point-in-polygon on a small validation set; neighbouring parcels ~6 m apart make "
                  "predio labels noisy, so P(predio) is a conservative estimate."],
    }
    payload.pop("kinds")
    target_path = args.out if (ok or args.force) else args.candidate_out
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as fh:
        json.dump(_clean(payload), fh, indent=2)
    print(f"{'wrote' if target_path == args.out else 'ship rule FAILED; wrote candidate'} {target_path}")
    return 0 if target_path == args.out else 2


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def load_training_rows(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    feat_cols = [f"feat_{n}" for n in FEATURE_NAMES]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"rows parquet lacks feature columns {missing}; rerun eval_strict.py")
    keep = (df["estado"] == "OK") & df["has_gt"].astype(bool)
    keep &= df["correct_predial"].notna() & df["correct_manzana"].notna()
    keep &= df[feat_cols].notna().all(axis=1)
    return df[keep].reset_index(drop=True)


def _print_metrics(title: str, table: dict[str, dict]) -> None:
    print(f"\n{title}")
    print(f"{'source':<22}{'n':>6}{'brier':>9}{'logloss':>9}{'ece5':>8}{'auc':>8}")
    for name, m in table.items():
        print(f"{name:<22}{m['n']:>6}{m['brier']:>9.4f}{m['logloss']:>9.4f}{m['ece']:>8.4f}{m['auc']:>8.3f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=("hybrid", "cells", "logistic"), default="hybrid")
    ap.add_argument("--artifacts-dir", default=None,
                    help="experiment directory: defaults of --rows/--out/--candidate-out live there "
                         "(env CALI_ARTIFACTS_DIR; default artifacts/)")
    ap.add_argument("--candidate-out", default=None,
                    help="hybrid only: where to write the artifact when it fails the ship rule")
    ap.add_argument("--rows", default=None)
    ap.add_argument("--force", action="store_true",
                    help="cells only: write the artifact even if it fails the ship rule (marked in the file)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    from cali_address.paths import resolve_artifacts_dir
    art = resolve_artifacts_dir(args.artifacts_dir)
    args.rows = args.rows or os.path.join(art, "strict_eval_paso7_rows.parquet")
    args.out = args.out or os.path.join(art, "reliability.json")
    args.candidate_out = args.candidate_out or os.path.join(art, "reliability_hybrid_candidate.json")
    warnings.filterwarnings("ignore")
    if args.kind == "hybrid":
        return run_hybrid(args)
    if args.kind == "cells":
        return run_cells(args)

    df = load_training_rows(args.rows)
    groups = df["dataset"].to_numpy()
    X = df[[f"feat_{n}" for n in FEATURE_NAMES]].to_numpy(dtype=float)
    relaxed = df["motivo"].fillna("").str.startswith("aproximado").to_numpy()
    base_p = df["confianza"].to_numpy(dtype=float)
    print(f"training rows (OK with GT): {len(df)}; per dataset: {dict(pd.Series(groups).value_counts())}")
    print(f"relaxed rows: {int(relaxed.sum())}")

    payload = {"kind": "logistic", "version": f"{VERSION_PREFIX}-{dt.date.today():%Y%m%d}",
               "fit_date": dt.datetime.now().isoformat(timespec="seconds"),
               "feature_names": list(FEATURE_NAMES), "n_train": int(len(df)),
               "source_rows": os.path.basename(args.rows), "cv": {}, "notes": []}
    oof_pred: dict[str, np.ndarray] = {}
    verdicts = {}
    for target, col in TARGETS.items():
        y = df[col].to_numpy(dtype=float).astype(int)
        cv = nested_lodo(X, y, groups, base_p)
        oof = cv["oof"]
        if target == "manzana":  # same serve-time rule: P(manzana) >= P(predio)
            oof["model"] = np.maximum(oof["model"], oof_pred["predio_model"])
        oof_pred[f"{target}_model"] = oof["model"]
        labels = {"model": "model (nested LODO)", "raw": "raw confianza", "isotonic": "isotonic on score"}
        _print_metrics(f"[{target}] pooled out-of-fold", {labels[k]: metrics(v, y) for k, v in oof.items()})
        for g in np.unique(groups):
            m = groups == g
            _print_metrics(f"[{target}] dataset={g}", {labels[k]: metrics(v[m], y[m]) for k, v in oof.items()})
        if relaxed.any():
            _print_metrics(f"[{target}] relaxed rows only",
                           {labels[k]: metrics(v[relaxed], y[relaxed]) for k, v in oof.items()})
        print(f"\n[{target}] reliability table (model OOF, all OK rows with GT)")
        print(reliability_table(oof["model"], y).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        if relaxed.any():
            print(f"[{target}] reliability table (model OOF, relaxed rows only, 3 bins)")
            print(reliability_table(oof["model"][relaxed], y[relaxed], bins=3).to_string(
                index=False, float_format=lambda v: f"{v:.3f}"))
        pooled = {k: metrics(v, y) for k, v in oof.items()}
        verdicts[target] = (pooled["model"]["logloss"] <= pooled["raw"]["logloss"]
                            and pooled["model"]["brier"] <= pooled["raw"]["brier"] + BRIER_TOLERANCE)
        payload["cv"][target] = {
            "chosen_c_per_outer_fold": cv["chosen_c"],
            "pooled": pooled,
            "per_dataset": {str(g): {k: metrics(v[groups == g], y[groups == g]) for k, v in oof.items()}
                            for g in np.unique(groups)},
            "relaxed": ({k: metrics(v[relaxed], y[relaxed]) for k, v in oof.items()} if relaxed.any() else None),
        }
        # final fit: C chosen by LODO over all rows
        C = choose_c(X, y, groups)
        fit = _fit_scaled(X, y, C)
        payload[target] = {"coef": [float(c) for c in fit["coef"]], "intercept": float(fit["intercept"]),
                           "C": C, "score_coef_dropped": bool(fit["dropped_score"])}
        payload["means"], payload["stds"] = [float(v) for v in fit["mean"]], [float(v) for v in fit["std"]]
        if fit["dropped_score"]:
            payload["notes"].append(f"{target}: raw-score coefficient was negative; feature dropped and refit.")
        print(f"\n[{target}] final C={C}; coefficients (standardized):")
        for n, c in zip(FEATURE_NAMES, fit["coef"]):
            print(f"  {n:<16}{c:+.3f}")
        print(f"  intercept       {fit['intercept']:+.3f}")

    worse = [t for t, ok in verdicts.items() if not ok]
    if worse:
        print(f"\nWARNING: model is WORSE than raw confianza out-of-fold on: {worse}; do not ship.")
        payload["notes"].append(f"model not better than raw confianza OOF on {worse}")
    else:
        print("\nModel is at least as good as raw confianza out-of-fold (logloss <=, brier within tolerance) on both targets.")
    payload["notes"].append(
        "Labels are GPS point-in-polygon on a small validation set; neighbouring parcels ~6 m apart make "
        "predio labels noisy, so P(predio) is a conservative estimate.")
    if worse:
        return 2
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
