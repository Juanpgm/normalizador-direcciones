"""Calibrated reliability of an OK row (serve time: numpy/stdlib only).

Three artifact kinds share ``artifacts/reliability.json`` (chosen by its ``kind``
field; a file without ``kind`` is the old logistic format):

* ``hybrid`` - isotonic ``P(manzana | confianza)`` (monotone knots, linearly
  interpolated) and ``P(predio) = P(manzana) * q_kind`` with ``q_kind`` an
  empirical-Bayes rate of ``P(predio | manzana)`` per row kind, see :class:`HybridReliability`;

* ``cells`` - a table of empirical-Bayes shrunk hit rates per (kind, margin band)
  cell, see :func:`reliability_cell` and :class:`CellReliability`;
* ``logistic`` - the earlier pair of logistic regressions described below.

Logistic kind: two tiny logistic regressions, fitted offline by ``scripts/fit_reliability.py``,
turn the features captured in ``service.reliability_features`` into

* ``P(predio)``  - probability the delivered ``numero_predial_nacional`` is right,
* ``P(manzana)`` - probability the delivered manzana is right, forced to be
  ``>= P(predio)`` (a wrong manzana implies a wrong predio).

Information only: the ``estado`` decision never depends on it. When the artifact
is absent or malformed :func:`load_reliability` returns ``None`` and the columns
stay ``None`` on OK rows; the raw score is never silently substituted.

Caveat: labels come from GPS point-in-polygon ground truth on a small validation
set, and neighbouring parcels are ~6 m apart, so the measured P(predio) is a
noisy, probably conservative estimate.
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass

__all__ = [
    "BANDS", "FEATURE_NAMES", "HYBRID_CLIP", "KINDS", "RELIABILITY_FILENAME", "CellReliability",
    "HybridReliability", "Reliability", "interpolate_knots", "load_reliability", "reliability_cell",
]

RELIABILITY_FILENAME = "reliability.json"

#: Feature order shared by the fit script, the artifact and the server.
FEATURE_NAMES = (
    "confidence", "struct", "relaxed_letter", "relaxed_plate", "log_n_predios",
    "nivel_esquina", "nivel_via", "nivel_manzana", "nivel_direccion", "gate_applied",
    "margen_manzana", "margen_predio", "n_competidores",
)

#: Standardized features are clipped to +-CLIP sigmas so an outlier cannot blow up the logit.
CLIP = 5.0
#: Output probabilities stay strictly inside (0, 1).
P_MIN, P_MAX = 0.001, 0.999


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@dataclass
class Reliability:
    """Two logistic heads over the same standardized feature vector."""

    feature_names: list[str]
    means: list[float]
    stds: list[float]
    coef_predio: list[float]
    intercept_predio: float
    coef_manzana: list[float]
    intercept_manzana: float
    version: str = ""

    def _standardize(self, features: dict) -> list[float]:
        out = []
        for name, mean, std in zip(self.feature_names, self.means, self.stds):
            try:
                value = float(features.get(name, mean))
            except (TypeError, ValueError):
                value = mean
            if math.isnan(value):
                value = mean
            z = (value - mean) / std if std > 0 else 0.0
            out.append(max(-CLIP, min(CLIP, z)))
        return out

    def predict(self, features: dict) -> tuple[float, float]:
        """``(P(predio), P(manzana))`` in ``[0.001, 0.999]`` with ``manzana >= predio``.

        A missing, ``None`` or NaN feature is treated as the training mean
        (neutral); infinities are clipped like any other outlier.
        """
        z = self._standardize(features)
        lp = self.intercept_predio + sum(c * v for c, v in zip(self.coef_predio, z))
        lm = self.intercept_manzana + sum(c * v for c, v in zip(self.coef_manzana, z))
        pp = min(max(_sigmoid(lp), P_MIN), P_MAX)
        pm = min(max(_sigmoid(lm), P_MIN), P_MAX)
        return pp, max(pm, pp)


#: Row kinds of the cell model, in the order they are tabulated.
KINDS = ("exacto", "relajada", "compartida", "parcial")
#: Margin bands, ascending; reliability is non-decreasing along this order within a kind.
BANDS = ("<0.02", "0.02-0.05", "0.05-0.10", "holgado")
#: Band edges (lower bounds of ``0.02-0.05``, ``0.05-0.10`` and ``holgado``).
BAND_EDGES = (0.02, 0.05, 0.10)


def _flag(features: dict, name: str) -> bool:
    try:
        return float(features.get(name, 0.0)) > 0.5
    except (TypeError, ValueError):
        return False


def reliability_cell(features: dict) -> tuple[str, str]:
    """``(kind, band)`` of an OK row from its :func:`service.reliability_features` dict.

    Kind, first match wins: ``relajada`` (a relaxed letter/plate), ``parcial``
    (nivel ``esquina``/``via``), ``compartida`` (nivel ``manzana``/``direccion``,
    i.e. a shared address), otherwise ``exacto``. Band from ``margen_manzana``:
    ``>= 0.10`` (and a missing, ``None``, NaN, non-numeric or infinite margin, i.e.
    "no competitor") is ``holgado``; ``[0.05, 0.10)``, ``[0.02, 0.05)`` and ``< 0.02``
    (also negatives) are the tighter bands.
    """
    if _flag(features, "relaxed_letter") or _flag(features, "relaxed_plate"):
        kind = "relajada"
    elif _flag(features, "nivel_esquina") or _flag(features, "nivel_via"):
        kind = "parcial"
    elif _flag(features, "nivel_manzana") or _flag(features, "nivel_direccion"):
        kind = "compartida"
    else:
        kind = "exacto"
    try:
        margin = float(features.get("margen_manzana"))
    except (TypeError, ValueError):
        margin = math.nan
    if math.isnan(margin) or margin >= BAND_EDGES[2]:
        band = "holgado"
    elif margin >= BAND_EDGES[1]:
        band = "0.05-0.10"
    elif margin >= BAND_EDGES[0]:
        band = "0.02-0.05"
    else:
        band = "<0.02"
    return kind, band


@dataclass
class CellReliability:
    """Lookup table of shrunk ``(P(predio), P(manzana))`` per (kind, band) cell.

    A cell missing from the table falls back to its kind marginal, then to the
    global rate. Rates are precomputed offline (shrinkage and monotonicity are
    applied by ``scripts/fit_reliability.py``); serving only looks them up.
    """

    cells: dict[tuple[str, str], tuple[float, float]]
    kinds: dict[str, tuple[float, float]]
    global_rate: tuple[float, float]
    version: str = ""

    def predict(self, features: dict) -> tuple[float, float]:
        """``(P(predio), P(manzana))`` in ``[0.001, 0.999]`` with ``manzana >= predio``."""
        kind, band = reliability_cell(features)
        pp, pm = self.cells.get((kind, band)) or self.kinds.get(kind) or self.global_rate
        pp = min(max(pp, P_MIN), P_MAX)
        pm = min(max(pm, P_MIN), P_MAX)
        return pp, max(pm, pp)


def _rate_pair(block) -> tuple[float, float]:
    pair = (float(block["predio"]), float(block["manzana"]))
    if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in pair):
        raise ValueError("rate outside [0, 1]")
    return pair


def _load_cells(data: dict) -> CellReliability:
    cells: dict[tuple[str, str], tuple[float, float]] = {}
    if not isinstance(data["cells"], list):
        raise ValueError("cells must be a list")
    for cell in data["cells"]:
        key = (cell["kind"], cell["band"])
        if key[0] not in KINDS or key[1] not in BANDS:
            raise ValueError("unknown cell")
        cells[key] = _rate_pair(cell)
    kinds: dict[str, tuple[float, float]] = {}
    raw_kinds = data.get("kinds", {})
    if not isinstance(raw_kinds, dict):
        raise ValueError("kinds must be an object")
    for kind, block in raw_kinds.items():
        if kind not in KINDS:
            raise ValueError("unknown kind")
        kinds[kind] = _rate_pair(block)
    return CellReliability(
        cells=cells, kinds=kinds, global_rate=_rate_pair(data["global"]),
        version=str(data.get("version", "")),
    )


#: Hybrid output bounds: the isotonic score never reaches certainty in either direction.
HYBRID_CLIP = (0.02, 0.98)


def interpolate_knots(xs: list[float], ys: list[float], x: float) -> float:
    """Piecewise-linear value at ``x`` through knots ``(xs, ys)``, clamped at both ends.

    ``xs`` must be ascending. Infinities clamp like any out-of-range value; a
    single knot is a constant. Non-decreasing ``ys`` give a non-decreasing result.
    """
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    hi = bisect.bisect_right(xs, x)
    x0, x1, y0, y1 = xs[hi - 1], xs[hi], ys[hi - 1], ys[hi]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@dataclass
class HybridReliability:
    """Isotonic ``P(manzana | confianza)`` times a per-kind ``q = P(predio | manzana)``.

    ``manzana`` is a non-decreasing function of the raw score, ``predio`` is
    ``manzana * q_kind`` with ``q <= 1`` so it never exceeds it. An undefined
    score (missing, ``None``, NaN, non-numeric) gets the lowest step, the
    conservative choice; infinities clamp to the end steps.
    """

    xs: list[float]
    ys: list[float]
    q: dict[str, float]
    q_global: float
    version: str = ""

    def predict(self, features: dict) -> tuple[float, float]:
        """``(P(predio), P(manzana))``; manzana in ``HYBRID_CLIP``, ``0 < predio <= manzana``."""
        try:
            score = float(features.get("confidence"))
        except (TypeError, ValueError):
            score = math.nan
        value = self.ys[0] if math.isnan(score) else interpolate_knots(self.xs, self.ys, score)
        pm = min(max(value, HYBRID_CLIP[0]), HYBRID_CLIP[1])
        q = min(max(self.q.get(reliability_cell(features)[0], self.q_global), 0.0), 1.0)
        return min(max(pm * q, P_MIN), pm), pm


def _unit(value) -> float:
    v = float(value)
    if not (math.isfinite(v) and 0.0 <= v <= 1.0):
        raise ValueError("value outside [0, 1]")
    return v


def _load_hybrid(data: dict) -> HybridReliability:
    iso = data["isotonic"]
    if not isinstance(iso, dict):
        raise ValueError("isotonic must be an object")
    xs = _floats(iso["x"], len(iso["x"]))
    ys = [_unit(v) for v in iso["y"]]
    if not xs or len(xs) != len(ys):
        raise ValueError("knots must be non-empty and aligned")
    if any(b <= a for a, b in zip(xs, xs[1:])) or any(b < a for a, b in zip(ys, ys[1:])):
        raise ValueError("knots must be increasing in x and non-decreasing in y")
    q: dict[str, float] = {}
    raw_q = data.get("q", {})
    if not isinstance(raw_q, dict):
        raise ValueError("q must be an object")
    for kind, block in raw_q.items():
        if kind not in KINDS:
            raise ValueError("unknown kind")
        q[kind] = _unit(block["q"])
    return HybridReliability(xs=xs, ys=ys, q=q, q_global=_unit(data["q_global"]),
                             version=str(data.get("version", "")))


def _floats(values, n: int) -> list[float]:
    if not isinstance(values, list) or len(values) != n:
        raise ValueError("wrong vector length")
    out = [float(v) for v in values]
    if not all(math.isfinite(v) for v in out):
        raise ValueError("non-finite value")
    return out


def load_reliability(path: str) -> Reliability | CellReliability | HybridReliability | None:
    """Read ``reliability.json``; ``None`` if missing or malformed (never raises).

    ``kind == "hybrid"`` gives a :class:`HybridReliability`, ``"cells"`` a
    :class:`CellReliability`; ``"logistic"`` or no
    ``kind`` (the pre-cells format) a :class:`Reliability`; anything else ``None``.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        kind = data.get("kind", "logistic")
        if kind == "hybrid":
            return _load_hybrid(data)
        if kind == "cells":
            return _load_cells(data)
        if kind != "logistic":
            raise ValueError("unknown artifact kind")
        # The artifact defines its own feature list, so an older artifact (fitted
        # before the margin features existed) keeps loading: the server only reads
        # the names it lists. Unknown or duplicated names are rejected.
        names = list(data["feature_names"])
        if not names or len(set(names)) != len(names) or not set(names) <= set(FEATURE_NAMES):
            raise ValueError("feature names differ from the serving contract")
        n = len(names)
        pred, manz = data["predio"], data["manzana"]
        intercepts = [float(pred["intercept"]), float(manz["intercept"])]
        if not all(math.isfinite(v) for v in intercepts):
            raise ValueError("non-finite intercept")
        return Reliability(
            feature_names=names,
            means=_floats(data["means"], n),
            stds=_floats(data["stds"], n),
            coef_predio=_floats(pred["coef"], n),
            intercept_predio=intercepts[0],
            coef_manzana=_floats(manz["coef"], n),
            intercept_manzana=intercepts[1],
            version=str(data.get("version", "")),
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
