"""Label-quality flags for ground truth obtained by GPS point-in-polygon.

The ground truth (``gt_doc_id`` / ``gt_predial`` / ``gt_manzana``) comes from
locating a GPS point inside a cadastral polygon. Neighbouring parcels are only
a few metres apart, so the polygon hit is frequently the *neighbour* of the
parcel the address actually names. This module flags which rows can be trusted
as supervision by cross-checking the input text against the GT parcel's own
cadastral address.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .parser import canonical, match_key, parse_address

__all__ = ["plate_agrees", "label_quality", "address_key", "GT_QUALITY_VALUES"]

GT_QUALITY_VALUES = ("gold", "noisy", "no_gt")

# Fields that identify the "via" (the street the address hangs off).
_VIA_FIELDS = (
    "via_type", "via_number", "via_letters", "via_suffix", "via_suffix_letters",
    "via_bis", "via_quadrant",
)


def _blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    text = str(value).strip()
    return not text or text.lower() in ("nan", "none")


def _norm_plate(plate: str | None) -> str | None:
    if plate is None:
        return None
    text = str(plate).strip().lstrip("0")
    return text or "0"


def _parse(value: object):
    return parse_address(None if _blank(value) else str(value))


def _plate_of(parsed) -> str | None:
    return _norm_plate(parsed.plate) if parsed.parse_ok else None


def plate_agrees(input_address: object, doc_address: object) -> bool | None:
    """Whether the parsed plate numbers of two addresses are equal.

    Zero padding is ignored (``05`` == ``5``); complements (apartment, local...)
    are irrelevant. Returns ``None`` when either address is blank, unparseable
    or carries no plate, so the caller can treat "cannot tell" separately from
    "differs".
    """
    a, b = _plate_of(_parse(input_address)), _plate_of(_parse(doc_address))
    if a is None or b is None:
        return None
    return a == b


def _is_gold(input_address: object, doc_address: object) -> bool:
    a, b = _parse(input_address), _parse(doc_address)
    if not (a.parse_ok and b.parse_ok):
        return False
    if any(getattr(a, f) != getattr(b, f) for f in _VIA_FIELDS):
        return False
    if a.cross_number is None or a.cross_number != b.cross_number:
        return False
    pa, pb = _plate_of(a), _plate_of(b)
    return pa is not None and pa == pb


def label_quality(df: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``gt_quality`` column.

    * ``no_gt``  - no ground-truth doc: ``gt_doc_id`` missing/NaN, the column
      absent, or an id that is not in ``docs``.
    * ``gold``   - has GT AND the GT doc's cadastral address agrees with the
      input on the via (type, number, letters, suffix, BIS, quadrant), on the
      cross-street number AND on the plate. The GT parcel is then, by text, the
      parcel the address names.
    * ``noisy``  - has GT but the text disagrees on any of the above (neighbour
      lot on the same street pair, other face of a corner with via/cross
      swapped, ...) or a plate/cross cannot be determined (blank or unparseable
      input, cadastre record without plate).

    ``docs`` needs a ``direccion`` column; it is indexed by its ``doc_id``
    column when present, otherwise by position. ``df`` needs ``raw_address``.
    """
    out = df.copy()
    if len(out) == 0:
        out["gt_quality"] = pd.Series(dtype="object")
        return out
    if "doc_id" in docs.columns:
        addr_by_id = dict(zip(docs["doc_id"].astype("int64"), docs["direccion"].astype(str)))
    else:
        addr_by_id = dict(enumerate(docs["direccion"].astype(str)))

    gt_ids = out["gt_doc_id"] if "gt_doc_id" in out.columns else pd.Series(np.nan, index=out.index)
    ids = pd.to_numeric(gt_ids, errors="coerce")
    quality: list[str] = []
    for raw, gid in zip(out["raw_address"], ids):
        doc_addr = None if pd.isna(gid) else addr_by_id.get(int(gid))
        if doc_addr is None:
            quality.append("no_gt")
        else:
            quality.append("gold" if _is_gold(raw, doc_addr) else "noisy")
    out["gt_quality"] = quality
    return out


def address_key(text: object) -> str:
    """Stable key identifying an address for split disjointness.

    Built on the project's own parser so spelling variants collapse
    (``Carrera 26 H 1 No 73-10`` == ``KR 26 H 1 # 73 - 10``): the key is
    ``match_key`` (accent-folded, upper-cased, alphanumerics only) of the
    parser's canonical form INCLUDING the complement, so two units in the same
    building stay distinct. Plain ``match_key`` on the raw string is used when
    the address does not parse, and an empty string for blank/NaN input.
    Plain ``match_key`` alone was not enough because it does not unify via-type
    spellings or ``No``/``#``.
    """
    if _blank(text):
        return ""
    raw = str(text)
    parsed = parse_address(raw)
    if parsed.parse_ok:
        canon = canonical(parsed)
        if canon:
            return match_key(canon)
    return match_key(raw)


def select_quality(df: pd.DataFrame, gold_only: bool) -> pd.DataFrame:
    """Return ``df`` restricted to ``gt_quality == 'gold'`` when ``gold_only``.

    Raises ``KeyError`` if ``gold_only`` is requested but the frame has no
    ``gt_quality`` column (so a metric is never silently reported on all rows).
    """
    if not gold_only:
        return df
    if "gt_quality" not in df.columns:
        raise KeyError("gt_quality column missing; build the file with scripts/build_splits.py")
    return df[df["gt_quality"] == "gold"].reset_index(drop=True)
