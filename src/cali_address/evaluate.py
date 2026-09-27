"""Evaluation of the normalizer against the five external datasets and IDESC."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import haversine_m
from .parser import match_key, parse_address

__all__ = [
    "attach_ground_truth",
    "reference_quality",
    "score_predictions",
    "summarize",
    "idesc_comparison",
    "structural_agreement_table",
    "calibration_table",
    "via_type_confusion",
    "strict_summary",
    "motivo_breakdown",
    "nivel_precision_breakdown",
    "DIST_BUCKETS",
]

DIST_BUCKETS = (25, 50, 100, 250)

#: Columns produced by :func:`strict_summary`, one row per (dataset, estado)
#: plus a pooled "ALL" dataset row per estado.
STRICT_SUMMARY_COLUMNS = [
    "dataset", "estado", "n", "n_with_gt",
    "manzana_precision", "predial_precision", "idesc_text_agreement",
    "lost_correct", "lost_correct_share",
    "n_ok_sin_predial", "n_ok_sin_manzana",
]

#: Columns produced by :func:`nivel_precision_breakdown`.
NIVEL_BREAKDOWN_COLUMNS = ["nivel_precision", "n"]

#: Columns produced by :func:`motivo_breakdown`.
MOTIVO_BREAKDOWN_COLUMNS = [
    "motivo_prefix", "n",
    "raw_top1_correct_true", "raw_top1_correct_false", "raw_top1_correct_nan",
]

#: ``motivo`` templates introduce their kind ("barrio X", "comuna 15", ...)
#: right after these connector words; grouping stops there so every gate
#: rejection collapses onto one prefix regardless of the place name.
_MOTIVO_STOP_KEYWORDS = {"barrio", "comuna", "corregimiento"}


def _motivo_prefix(motivo) -> str:
    """Collapse a ``motivo`` string to its template stem for grouping.

    Keeps leading words while they are plain lowercase connectors, and stops
    at the first token that looks like the variable part of the message: one
    containing a digit, ``:`` or ``<``, an all-uppercase code (via type,
    single letter), or a known gate-kind keyword (``barrio`` / ``comuna`` /
    ``corregimiento``) that is always followed by a place name. Blank input
    (including NaN) becomes ``"(vacio)"`` rather than an empty string, so it
    is never silently dropped from a groupby.
    """
    if motivo is None or (isinstance(motivo, float) and pd.isna(motivo)):
        text = ""
    else:
        text = str(motivo)
    text = text.strip()
    if not text:
        return "(vacio)"
    kept: list[str] = []
    for token in text.split(" "):
        bare = token.strip(".,;")
        if not bare:
            kept.append(token)
            continue
        if ":" in token or "<" in token or any(ch.isdigit() for ch in token):
            break
        if bare.lower() in _MOTIVO_STOP_KEYWORDS:
            break
        if bare.isupper() and bare.isalpha():
            break
        kept.append(token)
    prefix = " ".join(kept).strip()
    return prefix or "(vacio)"


def _share(values: pd.Series) -> float:
    """Mean of a bool/NaN series, ignoring NaN; ``nan`` when nothing is left."""
    present = values.dropna()
    if not len(present):
        return float("nan")
    return float(np.mean([bool(v) for v in present]))


def _bool_or_false(values: pd.Series) -> np.ndarray:
    """Elementwise ``bool(v)``, treating NaN/None as ``False`` (no downcast warnings)."""
    return np.array([bool(v) if pd.notna(v) else False for v in values], dtype=bool)


def _count_missing(group: pd.DataFrame, column: str) -> int:
    """Rows whose ``column`` is None/NaN/blank; 0 when the column is absent."""
    if column not in group.columns:
        return 0
    return int(sum(
        1 for v in group[column]
        if v is None or pd.isna(v) or not str(v).strip()
    ))


def _strict_summary_group(dataset: str, estado: str, group: pd.DataFrame) -> dict:
    n = len(group)
    has_gt = pd.Series(_bool_or_false(group["has_gt"]), index=group.index)
    n_with_gt = int(has_gt.sum())
    is_ok = estado == "OK"
    if is_ok:
        manzana_precision = _share(group.loc[has_gt, "correct_manzana"])
        predial_precision = _share(group.loc[has_gt, "correct_predial"])
        idesc_text_agreement = _share(group["text_agree_idesc"])
        lost_correct = 0
    else:
        manzana_precision = float("nan")
        predial_precision = float("nan")
        idesc_text_agreement = float("nan")
        lost_correct = int(_bool_or_false(group["raw_top1_correct_manzana"]).sum())
    lost_correct_share = (lost_correct / n_with_gt) if n_with_gt else float("nan")
    return {
        "dataset": dataset, "estado": estado, "n": n, "n_with_gt": n_with_gt,
        "manzana_precision": manzana_precision, "predial_precision": predial_precision,
        "idesc_text_agreement": idesc_text_agreement,
        "lost_correct": lost_correct, "lost_correct_share": lost_correct_share,
        # Shared addresses answer OK without a predial (and sometimes without a
        # manzana); those rows are excluded from the precision denominators
        # (their correct_* is NaN) and counted here instead.
        "n_ok_sin_predial": _count_missing(group, "numero_predial_nacional") if is_ok else 0,
        "n_ok_sin_manzana": _count_missing(group, "manzana") if is_ok else 0,
    }


def strict_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(dataset, estado) accuracy of the production path (``normalize_strict``).

    ``df`` has one row per input address, already joined with ground truth:
    ``dataset``, ``estado``, ``has_gt`` (bool), ``correct_manzana`` (bool or NaN
    when there is no ground truth), ``correct_predial`` (same shape),
    ``raw_top1_correct_manzana`` (whether the model's raw top-1 candidate, before
    any rule/gate/threshold, had the right manzana) and ``text_agree_idesc``
    (bool or NaN when there is no IDESC answer to compare against).

    Coordinates are intentionally NOT a correctness criterion here: only
    ``manzana_precision``, ``predial_precision`` and ``idesc_text_agreement``
    (all coordinate-free) are reported. A caller that wants raw distances can
    still compute them per row; this summary never buckets by distance.

    ``manzana_precision`` / ``predial_precision`` / ``idesc_text_agreement``
    are only meaningful for the ``OK`` rows (a non-OK row never carries a
    cadastral match), so they are ``NaN`` on every other estado. ``lost_correct``
    is the count of non-OK rows whose raw top-1 candidate did have the right
    manzana - i.e. the model *found* the parcel but a rule, the geo gate or the
    threshold discarded it. ``lost_correct_share`` divides that by ``n_with_gt``
    for the same estado, and is ``NaN`` when ``n_with_gt`` is 0.

    Returns one row per ``(dataset, estado)`` actually present, plus the same
    breakdown pooled across every dataset under ``dataset == "ALL"``. Never
    raises: an empty (or column-less) ``df`` returns an empty result.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=STRICT_SUMMARY_COLUMNS)

    rows = []
    for dataset, group in df.groupby("dataset", sort=True):
        for estado, sub in group.groupby("estado", sort=True):
            rows.append(_strict_summary_group(str(dataset), str(estado), sub))
    for estado, sub in df.groupby("estado", sort=True):
        rows.append(_strict_summary_group("ALL", str(estado), sub))
    return pd.DataFrame(rows, columns=STRICT_SUMMARY_COLUMNS)


def nivel_precision_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Counts per ``nivel_precision`` among the ``OK`` rows.

    Blank/NaN levels are reported as ``"(vacio)"``. Never raises: a frame
    without ``estado``/``nivel_precision`` (or empty) returns an empty result.
    """
    if df is None or len(df) == 0 or not {"estado", "nivel_precision"} <= set(df.columns):
        return pd.DataFrame(columns=NIVEL_BREAKDOWN_COLUMNS)
    ok = df.loc[df["estado"] == "OK", "nivel_precision"]
    labels = ok.map(lambda v: "(vacio)" if v is None or pd.isna(v) or not str(v).strip() else str(v))
    counts = labels.value_counts()
    return pd.DataFrame(
        {"nivel_precision": counts.index.astype(str), "n": counts.to_numpy(dtype="int64")},
        columns=NIVEL_BREAKDOWN_COLUMNS,
    )


def motivo_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Counts per ``motivo`` template, crossed with ``raw_top1_correct_manzana``.

    ``df`` needs ``motivo`` (free text, may be blank/NaN) and
    ``raw_top1_correct_manzana`` (bool or NaN). Each motivo is collapsed to its
    template stem by :func:`_motivo_prefix` before counting, so
    ``"confianza 0.61 < umbral 0.73"`` and ``"confianza 0.40 < umbral 0.73"``
    land in the same ``"confianza"`` row - which rule discards a correct match
    is then visible from how often ``raw_top1_correct_true`` shows up under it.
    Never raises: an empty ``df`` returns an empty result.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=MOTIVO_BREAKDOWN_COLUMNS)

    prefixes = df["motivo"].map(_motivo_prefix)
    flags = df["raw_top1_correct_manzana"]
    rows = []
    for prefix, idx in prefixes.groupby(prefixes).groups.items():
        col = flags.loc[idx]
        rows.append({
            "motivo_prefix": prefix,
            "n": len(idx),
            "raw_top1_correct_true": int((col == True).sum()),  # noqa: E712
            "raw_top1_correct_false": int((col == False).sum()),  # noqa: E712
            "raw_top1_correct_nan": int(col.isna().sum()),
        })
    result = pd.DataFrame(rows, columns=MOTIVO_BREAKDOWN_COLUMNS)
    return result.sort_values("n", ascending=False).reset_index(drop=True)


def attach_ground_truth(
    samples: pd.DataFrame, catastro: pd.DataFrame, parcel_index, doc_id_by_address: dict
) -> pd.DataFrame:
    """Resolve the true parcel for every sampled row that carries coordinates.

    A point-in-polygon query against the cadastral parcels gives the true predio;
    its cadastral address gives the true retrieval target (``gt_doc_id``).
    """
    out = samples.copy()
    lat = pd.to_numeric(out["gt_lat"], errors="coerce").to_numpy(dtype="float64")
    lon = pd.to_numeric(out["gt_lon"], errors="coerce").to_numpy(dtype="float64")
    rows = parcel_index.query(lat, lon)
    out["gt_parcel_row"] = rows
    pred = catastro["numero_predial_nacional"].astype(str).to_numpy()
    mz = catastro["numero_predial_manzana"].astype(str).to_numpy()
    addr = catastro["direccion"].astype(str).to_numpy()
    ok = rows >= 0
    out["gt_predial"] = np.where(ok, pred[np.clip(rows, 0, None)], None)
    out["gt_manzana"] = np.where(ok, mz[np.clip(rows, 0, None)], None)
    gt_addr = np.where(ok, addr[np.clip(rows, 0, None)], None)
    out["gt_cadastral_address"] = gt_addr
    out["gt_comuna"] = [p[9:11] if isinstance(p, str) and len(p) >= 13 else None
                        for p in out["gt_predial"]]
    out["gt_doc_id"] = [
        doc_id_by_address.get(" ".join(str(a).split())) if a is not None else None for a in gt_addr
    ]
    # A handful of resolved parcels carry an address string that is not in the
    # retrieval index (blank or degenerate `direccion`, or a centroid that was
    # dropped when the documents were built). They have no reachable target, so
    # they cannot be scored; the count is surfaced rather than silently absorbed.
    resolved = out["gt_parcel_row"] >= 0
    scorable = pd.Series(out["gt_doc_id"]).notna().to_numpy()
    out.attrs["gt_parcels_resolved"] = int(resolved.sum())
    out.attrs["gt_parcels_scorable"] = int((resolved & scorable).sum())
    out.attrs["gt_parcels_unreachable"] = int((resolved & ~scorable).sum())
    return out


def score_predictions(
    evaluated: pd.DataFrame, preds: pd.DataFrame, docs: pd.DataFrame
) -> pd.DataFrame:
    """Join predictions with ground truth and compute per-row correctness."""
    df = pd.concat([evaluated.reset_index(drop=True), preds.reset_index(drop=True).drop(
        columns=["raw_address"], errors="ignore")], axis=1)
    doc_manzana = docs["manzana"].astype(str).to_numpy()

    def _top1_doc(ids):
        return int(ids[0]) if isinstance(ids, (list, np.ndarray)) and len(ids) else -1

    df["pred_doc_id"] = df["top5_doc_ids"].apply(_top1_doc)
    gt = pd.to_numeric(df["gt_doc_id"], errors="coerce")
    df["has_gt"] = gt.notna()
    df["correct_top1"] = df["has_gt"] & (df["pred_doc_id"] == gt.fillna(-1).astype("int64"))
    df["correct_top5"] = [
        bool(has and g == g and int(g) in [int(x) for x in (ids if isinstance(ids, (list, np.ndarray)) else [])])
        for has, g, ids in zip(df["has_gt"], gt, df["top5_doc_ids"])
    ]
    df["correct_topk"] = [
        bool(has and g == g and int(g) in [int(x) for x in (ids if isinstance(ids, (list, np.ndarray)) else [])])
        for has, g, ids in zip(df["has_gt"], gt, df["topk_doc_ids"])
    ]
    pred_mz = np.where(
        df["pred_doc_id"].to_numpy() >= 0,
        doc_manzana[np.clip(df["pred_doc_id"].to_numpy(), 0, None)],
        None,
    )
    df["pred_manzana"] = pred_mz
    df["correct_manzana"] = df["has_gt"] & (df["pred_manzana"] == df["gt_manzana"])
    df["dist_m"] = haversine_m(
        pd.to_numeric(df["gt_lat"], errors="coerce"),
        pd.to_numeric(df["gt_lon"], errors="coerce"),
        pd.to_numeric(df["lat"], errors="coerce"),
        pd.to_numeric(df["lon"], errors="coerce"),
    )
    df.loc[~df["has_gt"], "dist_m"] = np.nan
    return df


def summarize(scored: pd.DataFrame, label: str = "model") -> pd.DataFrame:
    """Per-dataset metric table."""
    rows = []
    for key, g in scored.groupby("dataset", sort=True):
        gt = g[g["has_gt"]]
        d = pd.to_numeric(gt["dist_m"], errors="coerce").dropna()
        row = {
            "variant": label,
            "dataset": key,
            "n": len(g),
            "n_with_gt_parcel": len(gt),
            "parse_rate": float(g["parse_ok"].mean()),
            "match_rate_conf": float(g["matched"].mean()),
            "predio_top1": float(gt["correct_top1"].mean()) if len(gt) else np.nan,
            "predio_top5": float(gt["correct_top5"].mean()) if len(gt) else np.nan,
            "predio_top20": float(gt["correct_topk"].mean()) if len(gt) else np.nan,
            "manzana_top1": float(gt["correct_manzana"].mean()) if len(gt) else np.nan,
            "median_dist_m": float(d.median()) if len(d) else np.nan,
            "mean_dist_m": float(d.mean()) if len(d) else np.nan,
        }
        for b in DIST_BUCKETS:
            row[f"within_{b}m"] = float((d <= b).mean()) if len(d) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def idesc_comparison(scored: pd.DataFrame) -> pd.DataFrame:
    """Canonical-string agreement and georeferencing comparison against IDESC."""
    rows = []
    for key, g in scored.groupby("dataset", sort=True):
        has_idesc = g["idesc_dir_ajusta"].notna()
        ours = g["rule_canonical_glued"].fillna("")
        theirs = g["idesc_dir_ajusta"].fillna("")
        exact = (ours.str.strip() == theirs.str.strip()) & (ours.str.strip() != "")
        key_eq = [
            bool(a) and match_key(a) == match_key(b)
            for a, b in zip(ours.tolist(), theirs.tolist())
        ]
        both = g["idesc_lat"].notna() & g["lat"].notna()
        dist = haversine_m(
            pd.to_numeric(g["idesc_lat"], errors="coerce"),
            pd.to_numeric(g["idesc_lon"], errors="coerce"),
            pd.to_numeric(g["lat"], errors="coerce"),
            pd.to_numeric(g["lon"], errors="coerce"),
        )
        dist = pd.Series(dist, index=g.index).where(both)
        rows.append(
            {
                "dataset": key,
                "n": len(g),
                "idesc_answered": float(has_idesc.mean()),
                "idesc_normalized_nonempty": float((theirs.str.strip() != "").mean()),
                "idesc_georef_rate": float(g["idesc_lat"].notna().mean()),
                "our_highconf_rate": float(g["matched"].mean()),
                "canonical_exact_match": float(exact.mean()),
                "canonical_key_match": float(np.mean(key_eq)),
                "both_have_coords": int(both.sum()),
                "median_dist_to_idesc_m": float(dist.dropna().median()) if both.any() else np.nan,
                "within_50m_of_idesc": float((dist.dropna() <= 50).mean()) if both.any() else np.nan,
                "within_100m_of_idesc": float((dist.dropna() <= 100).mean()) if both.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def accuracy_by_idesc_estado(scored: pd.DataFrame) -> pd.DataFrame:
    g = scored[scored["has_gt"]]
    if not len(g):
        return pd.DataFrame()
    keys = g["idesc_estado_class"].fillna("(no answer)").rename("estado_class")
    out = g.groupby(keys).agg(
        n=("correct_top1", "size"),
        predio_top1=("correct_top1", "mean"),
        predio_top5=("correct_top5", "mean"),
        manzana_top1=("correct_manzana", "mean"),
        median_dist_m=("dist_m", "median"),
        our_confidence=("confidence", "mean"),
    )
    return out.rename_axis("idesc_estado_class").reset_index()


def structural_agreement_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Compare parsed structural fields with the dataset's own normalized column."""
    rows = []
    fields = ["via_type", "via_number", "via_letters", "cross_number", "plate"]
    for key, g in scored.groupby("dataset", sort=True):
        sub = g[g["their_normalized"].notna() & (g["their_normalized"].astype(str).str.strip() != "")]
        if not len(sub):
            continue
        ours = [parse_address(v) for v in sub["raw_address"]]
        theirs = [parse_address(v) for v in sub["their_normalized"]]
        row = {"dataset": key, "n": len(sub),
               "both_parse_ok": float(np.mean([a.parse_ok and b.parse_ok for a, b in zip(ours, theirs)]))}
        for f in fields:
            pairs = [
                (getattr(a, f), getattr(b, f))
                for a, b in zip(ours, theirs) if a.parse_ok and b.parse_ok
            ]
            row[f] = float(np.mean([x == y for x, y in pairs])) if pairs else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_table(scored: pd.DataFrame, bins=(0, 0.4, 0.55, 0.65, 0.72, 0.8, 0.88, 1.01)) -> pd.DataFrame:
    g = scored[scored["has_gt"]].copy()
    if not len(g):
        return pd.DataFrame()
    g["bin"] = pd.cut(g["confidence"], bins=list(bins), right=False)
    out = g.groupby("bin", observed=True).agg(
        n=("correct_top1", "size"),
        accuracy_top1=("correct_top1", "mean"),
        mean_confidence=("confidence", "mean"),
        median_dist_m=("dist_m", "median"),
    ).reset_index()
    return out


def via_type_confusion(scored: pd.DataFrame) -> pd.DataFrame:
    g = scored[scored["has_gt"] & scored["parse_ok"]].copy()
    if not len(g):
        return pd.DataFrame()
    truth = [parse_address(a).via_type for a in g["gt_cadastral_address"].astype(str)]
    pred = [parse_address(a).via_type for a in g["matched_cadastral_address"].astype(str)]
    return pd.crosstab(
        pd.Series(truth, name="true_via_type"), pd.Series(pred, name="predicted_via_type")
    )


#: Complement markers that indicate horizontal property / lot addressing.
_COMPLEMENT_MARKERS = (" AP ", " TO ", " BLQ ", " CA ", " LT ", " MZ ", " INT ",
                       " CONJ ", " ED ", " PH ", " OF ")
#: Tokens that indicate a rural / corregimiento address, absent from urbano_terreno.
_RURAL_MARKERS = ("PANCE", "NAVARRO", "PICHINDE", "ELVIRA", "LEONERA", "GOLONDRINAS",
                  "SALADITO", "FELIDIA", "MONTEBELLO", "VILLACARMELO", "HORMIGUERO",
                  "CORREGIMIENTO", "CGTO", "VEREDA", "VDA")


def failure_family(row) -> str:
    """Coarse diagnosis for an incorrect top-1 prediction."""
    raw = str(row.get("raw_address", "")).upper()
    if not row["parse_ok"]:
        notes = str(row.get("parse_notes", ""))
        if any(t in raw for t in _RURAL_MARKERS):
            return "unparseable: rural / corregimiento"
        if "no_alphabetic_content" in notes or "empty" in notes:
            return "unparseable: no address content"
        return "unparseable: free text / non-address"
    canon = " " + str(row.get("rule_canonical", "")) + " "
    if any(t in raw for t in _RURAL_MARKERS):
        return "rural / corregimiento (not in urbano_terreno)"
    if any(t in canon for t in _COMPLEMENT_MARKERS):
        return "condominium / lote complement"
    if " # " not in canon:
        return "missing cross number"
    gt_comuna = row.get("gt_comuna")
    if gt_comuna is not None and str(gt_comuna) != "None" and str(row.get("comuna")) != str(gt_comuna):
        return "wrong comuna (geographic confusion)"
    dist = row.get("dist_m")
    if dist is not None and pd.notna(dist) and float(dist) <= 150:
        return "near miss: neighbouring parcel"
    return "other: letter/number mismatch"


def reference_quality(scored: pd.DataFrame) -> pd.DataFrame:
    """Measure how noisy the coordinate-derived ground truth itself is.

    IDESC is an independent normalizer. When it reports ``estado A`` ("normalized
    and georeferenced *exact*") its ``dir_ajusta`` is a high-quality text answer
    and its coordinate is a high-quality geographic answer. If the coordinate
    stored in the source workbook fell on the parcel that IDESC names, the two
    would agree. The share where they do **is an upper bound on the exact-predio
    accuracy any text normalizer can score against this ground truth**.
    """
    g = scored[scored["gt_parcel_row"] >= 0].copy()
    if not len(g):
        return pd.DataFrame()
    g["gt_key"] = [match_key(str(a)) for a in g["gt_cadastral_address"]]
    g["idesc_key"] = [match_key(a) if isinstance(a, str) else "" for a in g["idesc_dir_ajusta"]]
    g["offset_m"] = haversine_m(
        pd.to_numeric(g["gt_lat"], errors="coerce"), pd.to_numeric(g["gt_lon"], errors="coerce"),
        pd.to_numeric(g["idesc_lat"], errors="coerce"), pd.to_numeric(g["idesc_lon"], errors="coerce"),
    )
    rows = []
    for cls, sub in g.groupby(g["idesc_estado_class"].fillna("(no answer)")):
        off = pd.to_numeric(sub["offset_m"], errors="coerce").dropna()
        rows.append(
            {
                "idesc_estado_class": cls,
                "n": len(sub),
                "gt_parcel_address_equals_idesc": float((sub["gt_key"] == sub["idesc_key"]).mean()),
                "n_with_idesc_coords": int(len(off)),
                "median_offset_file_vs_idesc_m": float(off.median()) if len(off) else np.nan,
                "p90_offset_m": float(off.quantile(0.9)) if len(off) else np.nan,
                "offset_within_25m": float((off <= 25).mean()) if len(off) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def parcel_spacing(docs: pd.DataFrame, n_probe: int = 20_000, seed: int = 42) -> dict:
    """Median distance between a parcel centroid and its nearest neighbour.

    This is the physical resolution limit of the task: a reference coordinate with
    an error larger than this distance lands on the wrong parcel by construction.
    """
    from sklearn.neighbors import NearestNeighbors

    xy = docs[["x_m", "y_m"]].to_numpy()
    rng = np.random.default_rng(seed)
    probe = rng.choice(len(xy), min(n_probe, len(xy)), replace=False)
    nn = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(xy)
    dist, _ = nn.kneighbors(xy[probe])
    d = dist[:, 1]
    return {
        "median_nn_distance_m": float(np.median(d)),
        "p25_nn_distance_m": float(np.percentile(d, 25)),
        "p75_nn_distance_m": float(np.percentile(d, 75)),
    }
