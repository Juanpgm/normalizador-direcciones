"""Evaluate the PRODUCTION path (``normalize_strict``) against ground truth.

Every other eval script in this repo measures
``AddressNormalizer.normalize_batch``'s raw ``matched`` flag. That is NOT what
the CLI or the HTTP service return: both call
``cali_address.service.normalize_strict``, which adds the structural rules,
the geographic gate and the confidence threshold on top of the raw model
match, and is what actually produces ``estado == "OK"``. This script measures
that exact path.

Coordinates are secondary here and are never used as a correctness
criterion: a row is judged right or wrong by cadastral identity (does the
matched ``manzana`` / ``numero_predial_nacional`` equal the ground truth?)
and by text agreement with IDESC's own normalization. ``dist_m`` is kept in
the rows output purely as an informational column.

Usage
-----
    PYTHONPATH=src python scripts/eval_strict.py --tag baseline
    PYTHONPATH=src python scripts/eval_strict.py --limit 50 --tag smoke
    PYTHONPATH=src python scripts/eval_strict.py --datasets fasecolda,stickers
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address import geo  # noqa: E402
from cali_address.evaluate import motivo_breakdown, nivel_precision_breakdown, strict_summary  # noqa: E402
from cali_address.inference import AddressNormalizer  # noqa: E402
from cali_address.labels import select_quality  # noqa: E402
from cali_address.experiment import is_frozen_path  # noqa: E402
from cali_address.parser import match_key  # noqa: E402
from cali_address.paths import resolve_artifacts_dir  # noqa: E402
from cali_address.reliability import FEATURE_NAMES  # noqa: E402
from cali_address.inference import normalize_soft_rules, validate_max_soft  # noqa: E402
from cali_address.service import DEFAULT_AMBIGUITY_DELTA, load_gazetteer, normalize_strict  # noqa: E402

DEFAULT_ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
DEFAULT_BASEMAPS_DIR = os.path.join(ROOT, "basemaps")


# ---------------------------------------------------------------------------
# Cadastral-code / text comparison (coordinate-free correctness)
# ---------------------------------------------------------------------------
def _norm_code(value) -> str | None:
    """String key for a manzana / predial code; ``None`` for blank/NaN/"nan"."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _codes_equal(a, b) -> bool | float:
    """``True``/``False`` when both codes are present, ``NaN`` when either is missing.

    Both codes are fixed-width digit strings in this project's tables, but a
    parquet/Excel round trip can silently turn a leading-zero code into an
    int and drop the padding, so purely numeric codes are also compared
    zero-padded to the longer width before giving up.
    """
    a, b = _norm_code(a), _norm_code(b)
    if a is None or b is None:
        return float("nan")
    if a == b:
        return True
    if a.isdigit() and b.isdigit():
        width = max(len(a), len(b))
        return a.zfill(width) == b.zfill(width)
    return False


def _text_agrees_with_idesc(direccion_normalizada, idesc_dir_ajusta) -> bool | float:
    """``match_key`` agreement between our output and IDESC's; NaN with no IDESC answer."""
    if not isinstance(idesc_dir_ajusta, str) or not idesc_dir_ajusta.strip():
        return float("nan")
    if direccion_normalizada is None or (
        isinstance(direccion_normalizada, float) and np.isnan(direccion_normalizada)
    ):
        return False
    return match_key(str(direccion_normalizada)) == match_key(idesc_dir_ajusta)


# ---------------------------------------------------------------------------
# Raw (unfiltered) top-1: what the model would have picked before any rule,
# geo gate or threshold - this is what `lost_correct` is measured against.
# ---------------------------------------------------------------------------
def _raw_top1_manzana(normalizer: AddressNormalizer, raws: list, k: int) -> list:
    """Manzana of the model's raw top-1 candidate for every row.

    Fuses the same components with the same weights ``normalize_strict``
    fuses internally (service.py, ~line 350), but is computed directly from
    ``score_components`` on every raw string - parseable or not - which is
    what lets a rejected/unparseable row still be checked for a "lost" match.
    """
    comps = normalizer.score_components(raws, k=k)
    weights = normalizer.weights
    fused = (
        weights["sim"] * comps["sim"] + weights["fuzz"] * comps["fuzz"]
        + weights["struct"] * comps["struct"] + weights["geo"] * comps["geo"]
    )
    docs = comps["doc"]
    out: list[str | None] = []
    for i in range(len(raws)):
        slots = np.flatnonzero(docs[i] >= 0)
        if len(slots) == 0:
            out.append(None)
            continue
        best = slots[np.argmax(fused[i, slots])]
        out.append(str(normalizer._manzana[docs[i, best]]))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_headline(summary: pd.DataFrame) -> None:
    all_rows = summary[summary["dataset"] == "ALL"]
    pooled_ok = all_rows[all_rows["estado"] == "OK"]
    total = int(all_rows["n"].sum())
    lost = int(all_rows["lost_correct"].sum())
    print("--- headline (pooled ALL) ---")
    if pooled_ok.empty:
        print(f"OK rows:              0 (of {total})")
        print(f"lost_correct (total): {lost}")
        return
    row = pooled_ok.iloc[0]
    print(f"OK rows:              {int(row['n'])} (of {total})")
    print(f"manzana_precision:    {row['manzana_precision']:.4f}")
    print(f"predial_precision:    {row['predial_precision']:.4f}")
    print(f"idesc_text_agreement: {row['idesc_text_agreement']:.4f}")
    print(f"n_ok_sin_predial:     {int(row['n_ok_sin_predial'])}")
    print(f"n_ok_sin_manzana:     {int(row['n_ok_sin_manzana'])}")
    print(f"lost_correct (total): {lost}")


def decision_kwargs(args) -> dict:
    """``normalize_strict`` kwargs for the Stage 3 decision options.

    ``None`` (flag not given) lets the artifact's ``tuning.json`` ``decision`` block decide, so
    ``--artifacts-dir <experiment>`` scores that experiment's own configuration.
    """
    soft_rules = None if args.soft_rules is None else normalize_soft_rules(args.soft_rules)
    max_soft = None if args.max_soft is None else validate_max_soft(args.max_soft)
    gate_fallback = None if args.gate_fallback is None else bool(args.gate_fallback)
    return {"soft_rules": soft_rules, "max_soft": max_soft, "gate_fallback": gate_fallback}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", default=os.path.join(DEFAULT_ARTIFACTS_DIR, "eval_scored_full.parquet"),
                    help="scored parquet with raw_address/dataset/gt_*/idesc_* columns")
    ap.add_argument("--datasets", default=None, help="comma-separated subset (default: all present)")
    ap.add_argument("--gate-escalate", type=int, choices=(0, 1), default=1,
                    help="matches the API/CLI default of on; normalize_strict's own default is False")
    ap.add_argument("--barrio-buffer", type=float, default=1000.0)
    ap.add_argument("--zone-buffer", type=float, default=300.0)
    ap.add_argument("--min-struct", type=float, default=0.6)
    ap.add_argument("--plate-tolerance", type=int, default=0)
    ap.add_argument("--ambiguity-delta", type=float, default=DEFAULT_AMBIGUITY_DELTA,
                    help="ambiguity abstention, mirrors production (default 0.02; 0 disables)")
    ap.add_argument("--soft-rules", default=None,
                    help="comma-separated rule classes downgraded to soft (default: the artifact's decision block)")
    ap.add_argument("--max-soft", type=int, default=None,
                    help="soft violations an OK row may carry (default: the artifact's decision block, else 1)")
    ap.add_argument("--gate-fallback", type=int, choices=(0, 1), default=None,
                    help="keep text-ranked candidates when the geographic gate empties the list "
                         "(default: the artifact's decision block, else off)")
    ap.add_argument("--gold-only", action="store_true",
                    help="keep only rows with gt_quality == 'gold' (needs that column, see build_splits.py)")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--artifacts", "--artifacts-dir", dest="artifacts", default=None,
                    help="artifacts directory holding model.pt etc. (env CALI_ARTIFACTS_DIR; default artifacts/)")
    ap.add_argument("--out-dir", default=None,
                    help="where the strict_eval_<tag>_* files are written (default: the artifacts directory)")
    ap.add_argument("--final", action="store_true",
                    help="required to score artifacts/splits/frozen_test.parquet (final measurement only)")
    ap.add_argument("--basemaps", default=DEFAULT_BASEMAPS_DIR)
    ap.add_argument("--limit", type=int, default=None, help="subsample rows per dataset for a quick run")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    args = ap.parse_args(argv)
    args.artifacts = resolve_artifacts_dir(args.artifacts)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else args.artifacts
    if is_frozen_path(args.scored) and not args.final:
        print("refusing to score frozen_test without --final (it is never used for selection)", file=sys.stderr)
        return 2

    df = pd.read_parquet(args.scored)
    if args.datasets:
        wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}
        df = df[df["dataset"].isin(wanted)].reset_index(drop=True)
    if args.gold_only:
        try:
            df = select_quality(df, gold_only=True)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    if args.limit:
        df = df.groupby("dataset", group_keys=False).head(args.limit).reset_index(drop=True)
    if not len(df):
        print("no rows selected", file=sys.stderr)
        return 1
    print(f"rows: {len(df)} across datasets {sorted(df['dataset'].unique())}", file=sys.stderr)

    gazetteer = load_gazetteer(args.basemaps, warn=lambda m: print(m, file=sys.stderr))
    normalizer = AddressNormalizer(args.artifacts, device=args.device)

    raws = df["raw_address"].astype(str).tolist()
    feature_sink: list = []
    result = normalize_strict(
        normalizer, raws, feature_sink=feature_sink, min_struct=args.min_struct, plate_tolerance=args.plate_tolerance,
        gazetteer=gazetteer, barrio_buffer_m=args.barrio_buffer, zone_buffer_m=args.zone_buffer,
        gate_escalate=bool(args.gate_escalate), ambiguity_delta=args.ambiguity_delta,
        **decision_kwargs(args),
    ).reset_index(drop=True)

    raw_top1_manzana = _raw_top1_manzana(normalizer, raws, k=20)
    has_idesc_col = "idesc_dir_ajusta" in df.columns
    has_gt = df["has_gt"].astype(bool).to_numpy()

    rows = pd.DataFrame({
        "dataset": df["dataset"].to_numpy(),
        "raw_address": df["raw_address"].to_numpy(),
        "estado": result["estado"].to_numpy(),
        "motivo": result["motivo"].fillna("").to_numpy(),
        "nivel_precision": result["nivel_precision"].to_numpy(),
        "confianza": result["confianza"].to_numpy(),
        "confiabilidad": result["confiabilidad"].to_numpy(),
        "confiabilidad_manzana": result["confiabilidad_manzana"].to_numpy(),
        "margen": pd.to_numeric(result["margen"], errors="coerce").to_numpy(),
        "has_gt": has_gt,
        "gt_manzana": df["gt_manzana"].to_numpy(),
        "gt_predial": df["gt_predial"].to_numpy(),
        "manzana": result["manzana"].to_numpy(),
        "numero_predial_nacional": result["numero_predial_nacional"].to_numpy(),
        "direccion_normalizada": result["direccion_normalizada"].to_numpy(),
        "idesc_dir_ajusta": df["idesc_dir_ajusta"].to_numpy() if has_idesc_col else None,
        "lat": result["lat"].to_numpy(),
        "lon": result["lon"].to_numpy(),
        "gt_lat": df["gt_lat"].to_numpy(),
        "gt_lon": df["gt_lon"].to_numpy(),
        "raw_top1_manzana": raw_top1_manzana,
    })

    # Reliability-model inputs (None -> NaN for non-OK rows), read by fit_reliability.py.
    for name in FEATURE_NAMES:
        rows[f"feat_{name}"] = [np.nan if f is None else f[name] for f in feature_sink]

    # Informational only: never fed into strict_summary's correctness metrics.
    rows["dist_m"] = geo.haversine_m(
        pd.to_numeric(rows["gt_lat"], errors="coerce"), pd.to_numeric(rows["gt_lon"], errors="coerce"),
        pd.to_numeric(rows["lat"], errors="coerce"), pd.to_numeric(rows["lon"], errors="coerce"),
    )
    rows.loc[~rows["has_gt"], "dist_m"] = np.nan

    rows["correct_manzana"] = [
        _codes_equal(m, gt) if hg else float("nan")
        for m, gt, hg in zip(rows["manzana"], rows["gt_manzana"], rows["has_gt"])
    ]
    rows["correct_predial"] = [
        _codes_equal(p, gt) if hg else float("nan")
        for p, gt, hg in zip(rows["numero_predial_nacional"], rows["gt_predial"], rows["has_gt"])
    ]
    rows["raw_top1_correct_manzana"] = [
        _codes_equal(m, gt) if hg else float("nan")
        for m, gt, hg in zip(rows["raw_top1_manzana"], rows["gt_manzana"], rows["has_gt"])
    ]
    rows["text_agree_idesc"] = [
        _text_agrees_with_idesc(d, i)
        for d, i in zip(rows["direccion_normalizada"], rows["idesc_dir_ajusta"])
    ]

    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, f"strict_eval_{args.tag}_rows.parquet")
    summary_path = os.path.join(out_dir, f"strict_eval_{args.tag}_summary.csv")
    motivos_path = os.path.join(out_dir, f"strict_eval_{args.tag}_motivos.csv")

    rows.to_parquet(rows_path, index=False)
    summary = strict_summary(rows)
    summary.to_csv(summary_path, index=False)
    motivos = motivo_breakdown(rows)
    motivos.to_csv(motivos_path, index=False)
    niveles = nivel_precision_breakdown(rows)
    niveles.to_csv(os.path.join(out_dir, f"strict_eval_{args.tag}_niveles.csv"), index=False)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.to_string(index=False))
    _print_headline(summary)
    print("--- nivel_precision (OK rows) ---")
    print(niveles.to_string(index=False))
    print(f"wrote {rows_path}", file=sys.stderr)
    print(f"wrote {summary_path}", file=sys.stderr)
    print(f"wrote {motivos_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
