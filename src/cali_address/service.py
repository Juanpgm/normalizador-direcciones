"""Shared normalization engine: batching, scoring, gating, table IO.

This module holds everything that used to live inside ``scripts/normalizar.py``
so the CLI and the HTTP service in ``cali_address.api`` run the exact same code
path. The CLI is now only argument parsing plus stdout/stderr formatting.

Strictness contract
-------------------
Unlike ``AddressNormalizer.normalize_batch`` (which always returns the best
candidate, even for junk input), :func:`normalize_strict` is strict: a row only
receives a cadastral match when every rule passes. Everything else is reported
with an explicit status and NO candidate fields, so a low-quality guess can
never be mistaken for a real match.

Statuses
--------
OK                  parsed, matched above the tuned confidence cutoff, inside the
                    detected place (when one was detected), and the matched
                    cadastral address agrees structurally with the input
NO_PARSEABLE        the address grammar could not parse the text (the detected
                    place columns are still filled)
SIN_MATCH           no cadastral record can be assigned. ``motivo`` says why:
                    blank/placeholder input, confidence below the cutoff, a
                    structural rule violated by the best candidate (via type /
                    via number / cross number / plate / low agreement), or every
                    candidate outside the detected barrio / comuna / corregimiento

Table reading
-------------
:func:`read_table` accepts xlsx/xls, csv, GeoJSON and a zipped shapefile and
returns a plain ``pandas.DataFrame``. Format, warnings and any geometry columns
it derived are attached to ``df.attrs`` (see :data:`ATTRS_KEYS`).

Column ambiguity
----------------
:func:`detect_address_column` scores every column against a Spanish/English
synonym set; :func:`resolve_address_column` turns that into either a single
answer or an :class:`AddressColumnError` carrying the ranked candidates, which
is how both the CLI and the API ask the caller to pick a field.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import tempfile
import unicodedata
import zipfile

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .gazetteer import (
    BARRIO_BUFFER_M,
    ZONE_BUFFER_M,
    Candidate,
    Detection,
    Gazetteer,
    gate_candidates,
)
from .inference import SOFT_RULE_NAMES, AddressNormalizer, normalize_soft_rules, validate_max_soft
from .parser import QUADRANT_SHORT, _QUADRANT_WORDS, canonical, filter_complement, parse_address

__all__ = [
    "AddressColumnError",
    "KeepColumnsError",
    "TableFormatError",
    "ADDRESS_SYNONYMS",
    "ATTRS_KEYS",
    "CLI_BARRIO_BUFFER_M",
    "DEFAULT_MIN_STRUCT",
    "OUTPUT_COLUMNS",
    "reliability_features",
    "PLACEHOLDERS",
    "STATUS_MAP",
    "SUPPORTED_EXTENSIONS",
    "ZONE_COLUMNS",
    "attach_source_columns",
    "detect_address_column",
    "guess_header_row",
    "is_unspecified",
    "load_gazetteer",
    "normalize_strict",
    "read_input_file",
    "read_table",
    "resolve_address_column",
    "results_to_geojson",
    "results_to_records",
    "summarize",
    "write_csv_bytes",
    "write_output",
    "write_xlsx_bytes",
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
DEFAULT_BASEMAPS_DIR = os.path.join(PROJECT_ROOT, "basemaps")

#: CLI/API default for the barrio gate. Measured on stickers (inspector GPS as
#: reference): 200 m withheld 138 rows of which 86 % sat within 100 m of the GPS
#: point, while 1000 m + escalation keeps them at the same 3.1 % share of >500 m
#: errors.
CLI_BARRIO_BUFFER_M = 1000.0

#: Minimum weighted structural agreement between the input parse and the matched
#: cadastral parse. Tuned default, shared by the CLI flag and the API form field.
DEFAULT_MIN_STRUCT = 0.6

PLACEHOLDERS = {
    "", "-", "--", ".", "0", "00", "N/A", "NA", "N.A", "N.A.", "NAN", "NONE", "NULL",
    "NO APLICA", "NO REGISTRA", "NO REPORTA", "NO INFORMA", "NINGUNA", "NINGUNO",
    "SIN ESPECIFICAR", "SIN DATO", "SIN DATOS", "SIN DIRECCION", "SIN INFORMACION",
    "S/D", "S.D", "SD", "DESCONOCIDA", "DESCONOCIDO", "NO TIENE", "NO HAY", "X", "XX", "XXX",
}

#: Normalized zone, one value each: the official barrio (or vereda) name and the
#: official comuna (or corregimiento) name. On OK rows they come from the polygon
#: the matched predio sits in; on every other row from the place detected in the
#: text, so a rural or unparseable address still gets a normalized zone.
ZONE_COLUMNS = ["barrio_vereda", "comuna_corregimiento"]

OUTPUT_COLUMNS = [
    "direccion_entrada",
    "estado",
    "motivo",
    "direccion_normalizada",
    "fuente_normalizacion",
    "numero_predial_nacional",
    "manzana",
    "lat",
    "lon",
    "confianza",
    "confiabilidad",
    "confiabilidad_manzana",
    "margen",
    "nivel_precision",
    *ZONE_COLUMNS,
]

#: Every rejection except an unparseable address is reported as SIN_MATCH; the
#: specific reason stays in ``motivo``.
STATUS_MAP = {"SIN_ESPECIFICAR": "SIN_MATCH", "SIN_MATCH_REGLAS": "SIN_MATCH"}

#: The three statuses a caller can ever see in ``estado``.
PUBLIC_STATUSES = ("OK", "SIN_MATCH", "NO_PARSEABLE")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class AddressColumnError(ValueError):
    """The address column could not be resolved without asking the caller.

    Carries the structured data both front ends need: ``available`` is every
    column in the table, ``candidates`` is the ranked guess list (possibly
    empty when nothing looked like an address at all), and ``requested`` is set
    when the caller named a column that does not exist.
    """

    def __init__(
        self,
        message: str,
        available: list[str] | None = None,
        candidates: list[dict] | None = None,
        requested: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.available = list(available or [])
        self.candidates = list(candidates or [])
        self.requested = requested

    def as_dict(self) -> dict:
        return {
            "error": "address_column_required",
            "message": self.message,
            "requested_column": self.requested,
            "available_columns": self.available,
            "candidates": self.candidates,
        }


class KeepColumnsError(ValueError):
    """``keep`` named columns the source table does not have (CLI only)."""


class TableFormatError(ValueError):
    """The uploaded bytes are not a table this service can read."""


# ---------------------------------------------------------------------------
# Placeholder / rule helpers (moved verbatim from scripts/normalizar.py)
# ---------------------------------------------------------------------------
def _clean_key(value) -> str:
    """Uppercase, accent-free, whitespace-collapsed key used for placeholder detection."""
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip().upper()
    return text


def is_unspecified(value) -> bool:
    key = _clean_key(value)
    if key in PLACEHOLDERS:
        return True
    # only punctuation / digits with no letters and fewer than 3 digits
    if not re.search(r"[A-Z]", key) and len(re.sub(r"\D", "", key)) < 3:
        return True
    return False


#: Field name -> Spanish label used in the ``letra de ...`` violation messages.
_LETTER_FIELD_LABELS = {
    "via_letters": "letra de via",
    "cross_letters": "letra de cruce",
    "via_suffix_letters": "letra de sufijo de via",
    "cross_suffix_letters": "letra de sufijo de cruce",
}

#: Field name -> Spanish label used in the ``bis de ...`` violation messages.
_BIS_FIELD_LABELS = {
    "via_bis": "bis de via",
    "cross_bis": "bis de cruce",
}

#: Field name -> Spanish label used in the ``cuadrante de ...`` violation messages.
_QUADRANT_FIELD_LABELS = {
    "via_quadrant": "cuadrante de via",
    "cross_quadrant": "cuadrante de cruce",
}

#: Standalone complement tokens the parser missed as a quadrant (see the module
#: docstring background on ``CL 5 # 38-10 NORTE``): the full quadrant words and
#: the multi-letter abbreviations from :data:`QUADRANT_SHORT`. Single cardinal
#: letters (``N``, ``S``, ``E``, ``O``/``W``) are deliberately excluded, because
#: in complement position they are unit letters (``BLQ E``), not quadrants.
_COMPLEMENT_QUADRANT_WORDS = frozenset(_QUADRANT_WORDS) | frozenset(QUADRANT_SHORT)


def _norm_letters(value) -> str:
    """Comparison key for a letters field: upper/stripped, ``None``/``""`` -> ``""``."""
    if value is None:
        return ""
    return str(value).strip().upper()


#: Field name -> its quadrant field on the same side, for the abbreviated
#: quadrant-letter equivalence below (via_letters/via_quadrant, cross_letters/
#: cross_quadrant). Suffix-letter fields are not covered: users do not type
#: ``NORTE`` glued after a secondary number.
_LETTER_QUADRANT_SIDES = {
    "via_letters": "via_quadrant",
    "cross_letters": "cross_quadrant",
}

#: A trailing cardinal LETTER the parser reads as a street letter (``CALLE 6N``,
#: ``AV 12 O``, ``KR 46 W``) but that is equivalent to a spelled-out quadrant the
#: cadastre carries instead (``CL 6 NORTE``, ``AV 12 OESTE``). ``E``/``S`` are
#: deliberately excluded: the cadastre spells those out too, but a bare ``E``
#: or ``S`` letter is a real, distinct street (``KR 41 E``) - ESTE never occurs
#: in the cadastral layer and SUR occurs once, so treating them as equivalent
#: would erase real streets, not typos.
_QUADRANT_LETTER_EQUIVALENTS = {"N": "NORTE", "O": "OESTE", "W": "OESTE"}


def _relax_letters_for_quadrant(query_letters: str, cadastral_quadrant: str | None) -> str:
    """Strip a trailing cardinal-letter suffix from ``query_letters`` when it is
    equivalent to ``cadastral_quadrant`` (see :data:`_QUADRANT_LETTER_EQUIVALENTS`).

    ``CALLE 6N`` -> via_letters ``"N"`` compared against a cadastral via_quadrant
    ``"NORTE"`` becomes an empty letters value, same as the cadastre's own
    (letter-less) via_letters. A multi-letter run (``"D W"``) only loses its
    trailing cardinal letter (-> ``"D"``); everything else is left untouched.
    """
    if not query_letters or cadastral_quadrant not in ("NORTE", "OESTE"):
        return query_letters
    if _QUADRANT_LETTER_EQUIVALENTS.get(query_letters[-1]) != cadastral_quadrant:
        return query_letters
    return query_letters[:-1].strip()


def _complement_quadrant_violation(query_parse, cadastral_parse) -> str | None:
    """A quadrant word the query's parser dropped into ``complement`` that the
    matched cadastral address does not carry in either quadrant field, nor as
    the same leftover word in its own complement (some real cadastral strings -
    ``AV 4 2 2 OESTE # 13 B - 19``, ``KR 72 CL 13 ZONA VERDE OCCIDENTAL SUR`` -
    push a quadrant word into the complement on both the query and the matched
    record itself, so comparing an address against its own parse must stay
    clean)."""
    complement = query_parse.complement or ""
    cadastral_quadrants = {cadastral_parse.via_quadrant, cadastral_parse.cross_quadrant}
    cadastral_complement_tokens = set((cadastral_parse.complement or "").split())
    for token in complement.split():
        if token in _QUADRANT_WORDS:
            word = token
        elif token in QUADRANT_SHORT:
            word = QUADRANT_SHORT[token]
        else:
            continue
        if word in cadastral_quadrants or word in cadastral_complement_tokens:
            continue
        return f"cuadrante en complemento {word} no esta en catastro"
    return None


#: Plate differences up to this many units (both plates numeric) are a *soft*
#: violation: the row is kept but only guaranteed at manzana level. The
#: ``plate_tolerance`` tunable is the hard tolerance inside it (no violation).
PLATE_SOFT_WINDOW = 2


def _rule_violations(query_parse, cadastral_parse, min_struct: float, struct_score: float, plate_tolerance: int = 0) -> list[str]:
    """Structural rules a match must satisfy; returns every violated rule (hard and soft)."""
    return [msg for msg, _soft in _tagged_violations(
        query_parse, cadastral_parse, min_struct, struct_score, plate_tolerance)]


def _classify_violations(
    query_parse, cadastral_parse, min_struct: float, struct_score: float, plate_tolerance: int = 0,
    soft_rules: frozenset = frozenset(),
) -> tuple[list[str], list[str]]:
    """Split the violations into ``(hard, soft)`` message lists.

    Soft: a one-sided ``letra de via`` (exactly one side has a via letter), a
    numeric plate difference within :data:`PLATE_SOFT_WINDOW`, and every class
    named in ``soft_rules`` (see :data:`SOFT_RULE_NAMES`). Everything else is hard.
    """
    tagged = _tagged_violations(
        query_parse, cadastral_parse, min_struct, struct_score, plate_tolerance, soft_rules)
    return [m for m, soft in tagged if not soft], [m for m, soft in tagged if soft]


def _letter_is_soft(field: str, qv: str, cv: str, soft_rules: frozenset) -> bool:
    """Whether a letters mismatch on ``field`` is downgraded to soft."""
    one_sided = not qv or not cv
    if field == "via_letters":
        return one_sided or "letra_via" in soft_rules
    if field == "cross_letters":
        if "letra_cruce" in soft_rules:
            return True
        if one_sided and "letra_cruce_una_cara" in soft_rules:
            return True
        return (
            one_sided and "letra_cruce_cardinal" in soft_rules
            and (qv or cv) in _QUADRANT_LETTER_EQUIVALENTS
        )
    return False


def _tagged_violations(
    query_parse, cadastral_parse, min_struct: float, struct_score: float, plate_tolerance: int = 0,
    soft_rules: frozenset = frozenset(),
) -> list[tuple[str, bool]]:
    """Every violation as ``(message, is_soft)`` in a stable order."""
    violations: list[tuple[str, bool]] = []
    if not cadastral_parse.parse_ok:
        return [("candidato catastral no parseable", False)]
    if query_parse.via_type and cadastral_parse.via_type and query_parse.via_type != cadastral_parse.via_type:
        violations.append((f"tipo de via {query_parse.via_type} != {cadastral_parse.via_type}", False))
    if query_parse.via_number and cadastral_parse.via_number:
        if query_parse.via_number.lstrip("0") != str(cadastral_parse.via_number).lstrip("0"):
            violations.append((f"numero de via {query_parse.via_number} != {cadastral_parse.via_number}", False))
    if query_parse.cross_number and cadastral_parse.cross_number:
        if query_parse.cross_number.lstrip("0") != str(cadastral_parse.cross_number).lstrip("0"):
            violations.append((f"numero de cruce {query_parse.cross_number} != {cadastral_parse.cross_number}", False))
    if query_parse.plate and cadastral_parse.plate:
        qp, cp = str(query_parse.plate).lstrip("0"), str(cadastral_parse.plate).lstrip("0")
        if qp != cp:
            try:
                delta = abs(int(qp or 0) - int(cp or 0))
            except ValueError:
                delta = None
            if delta is None or delta > plate_tolerance:
                soft = delta is not None and delta <= PLATE_SOFT_WINDOW
                violations.append((f"placa {query_parse.plate} != {cadastral_parse.plate}", soft))
    if (
        query_parse.cross_type and cadastral_parse.cross_type
        and query_parse.cross_type != cadastral_parse.cross_type
    ):
        violations.append((f"tipo de cruce {query_parse.cross_type} != {cadastral_parse.cross_type}", False))

    for field, label in _LETTER_FIELD_LABELS.items():
        qv = _norm_letters(getattr(query_parse, field))
        cv = _norm_letters(getattr(cadastral_parse, field))
        quadrant_field = _LETTER_QUADRANT_SIDES.get(field)
        if quadrant_field is not None and getattr(query_parse, quadrant_field) is None:
            qv = _relax_letters_for_quadrant(qv, getattr(cadastral_parse, quadrant_field))
        if qv != cv:
            soft = _letter_is_soft(field, qv, cv, soft_rules)
            violations.append((f"{label} {qv or '-'} != {cv or '-'}", soft))

    for field, label in _BIS_FIELD_LABELS.items():
        qv, cv = bool(getattr(query_parse, field)), bool(getattr(cadastral_parse, field))
        if qv != cv:
            soft = ("bis_via" if field == "via_bis" else "bis_cruce") in soft_rules
            violations.append((f"{label} {'si' if qv else 'no'} != {'si' if cv else 'no'}", soft))

    # Symmetric-lenient: a one-sided quadrant (either side) is allowed - it is
    # the more common, correct case (17 correct vs 3 wrong on the strict-eval
    # ground truth). Only a genuine both-sides disagreement is a violation.
    for field, label in _QUADRANT_FIELD_LABELS.items():
        qv = getattr(query_parse, field)
        cv = getattr(cadastral_parse, field)
        if qv is not None and cv is not None and qv != cv:
            violations.append((f"{label} {qv} != {cv}", False))

    complement_violation = _complement_quadrant_violation(query_parse, cadastral_parse)
    if complement_violation is not None:
        violations.append((complement_violation, "cuadrante_compl" in soft_rules))

    if struct_score < min_struct:
        violations.append((f"coincidencia estructural {struct_score:.2f} < {min_struct:.2f}", False))
    return violations


def _comuna_label(value) -> str | None:
    """`19`, `"19"`, `"019"` -> `"Comuna 19"`; other names pass through."""
    if value is None or value == "":
        return None
    text = str(value).strip()
    if text.isdigit():
        return f"Comuna {int(text)}"
    return text


def _zone_columns(detection: Detection | None, located: dict | None = None,
                  comuna_fallback=None) -> dict:
    """One normalized barrio/vereda and one comuna/corregimiento value.

    Priority: the polygon the matched predio falls in (OK rows) > the place
    detected in the text > the comuna encoded in the predial number.
    """
    located = located or {}
    barrio = located.get("barrio") or located.get("vereda")
    zone = located.get("comuna")
    zone = _comuna_label(zone) if zone is not None else located.get("corregimiento")
    if detection is not None and detection.detected:
        barrio = barrio or detection.barrio or detection.vereda
        if not zone:
            zone = _comuna_label(detection.comuna) if detection.comuna is not None else detection.corregimiento
    if not zone and comuna_fallback is not None:
        zone = _comuna_label(comuna_fallback)
    return {"barrio_vereda": barrio or None, "comuna_corregimiento": zone or None}


def _precision_level(parsed) -> str:
    """How much of the via/cross/plate triple the query itself confirmed.

    An OK row can still bind to a real predio when the query left a field
    unconfirmed (see the module docstring background on ``Calle 5 - 38`` and
    ``Calle 5 # 38``): ``_rule_violations`` skips a rule whenever the query's
    side of it is empty, so the match comes from the *other* fields plus the
    model, not from that field. This says which piece was actually pinned down
    by the input, single place to grow when a shared match needs a coarser
    level (``"manzana"``, ``"direccion"``).
    """
    if parsed.plate is None:
        return "esquina"
    if parsed.cross_number is None:
        return "via"
    return "predio"


#: Levels from least to most precise; the combined level of a row is the least
#: precise one that applies to it.
_PRECISION_ORDER = ("direccion", "via", "esquina", "manzana", "predio")


def _least_precise(*levels: str) -> str:
    """Combine precision levels: the least precise wins.

    Order (least to most precise): ``direccion`` < ``via`` < ``esquina`` <
    ``manzana`` < ``predio``. Used to merge the incomplete-input level of
    :func:`_precision_level` with the shared-address level.
    """
    return min(levels, key=_PRECISION_ORDER.index)


def reliability_features(
    confidence: float, struct: float, relaxed_letter: bool, relaxed_plate: bool,
    n_predios: int, nivel: str, gate_applied: bool,
    margen_manzana: float | None = None, margen_predio: float | None = None,
    n_competidores: int = 0,
) -> dict[str, float]:
    """Features of one OK row for the calibrated reliability model.

    Pure and shared by serving and the offline fit (via ``feature_sink``), so the
    two can never drift. ``nivel`` is one-hot encoded against the base level
    ``predio`` (an unknown level also maps to the base). ``n_predios <= 1`` gives
    ``log_n_predios == 0``. The margins are clipped to ``[0, MARGIN_CLIP]`` with
    ``None``/NaN meaning "no competition" (== ``MARGIN_CLIP``); ``n_competidores``
    is clipped to ``0..COMPETITOR_CLIP``.
    """
    def _margin(value) -> float:
        if value is None or math.isnan(value):
            return MARGIN_CLIP
        return min(max(float(value), 0.0), MARGIN_CLIP)


    return {
        "confidence": float(confidence),
        "struct": float(struct),
        "relaxed_letter": 1.0 if relaxed_letter else 0.0,
        "relaxed_plate": 1.0 if relaxed_plate else 0.0,
        "log_n_predios": math.log(max(int(n_predios), 1)),
        "nivel_esquina": 1.0 if nivel == "esquina" else 0.0,
        "nivel_via": 1.0 if nivel == "via" else 0.0,
        "nivel_manzana": 1.0 if nivel == "manzana" else 0.0,
        "nivel_direccion": 1.0 if nivel == "direccion" else 0.0,
        "gate_applied": 1.0 if gate_applied else 0.0,
        "margen_manzana": _margin(margen_manzana),
        "margen_predio": _margin(margen_predio),
        "n_competidores": float(min(max(int(n_competidores), 0), COMPETITOR_CLIP)),
    }


#: Production default of the ambiguity abstention (0 disables it).
DEFAULT_AMBIGUITY_DELTA = 0.02
#: Fused-score gap above which a runner-up is treated as "no competition" by the model.
MARGIN_CLIP = 0.3
#: ``n_competidores`` counts rule-passing rivals in other manzanas within this fused-score gap.
COMPETITOR_WINDOW = 0.05
#: Clip of ``n_competidores`` in the model features.
COMPETITOR_CLIP = 5
_EPS = 1e-9


def _competition(
    normalizer, kept: list, best, parsed, struct_by_doc: dict, min_struct: float, plate_tolerance: int,
    soft_rules: frozenset = frozenset(), max_soft: int = 1,
) -> tuple[float | None, float | None, int]:
    """``(margen_manzana, margen_predio, n_competidores)`` for the chosen candidate.

    Only rivals that would themselves be OK-able compete: fused score at or above
    the threshold, no hard rule violation and at most ``max_soft`` soft ones. NaN scores and
    the chosen doc itself are skipped. ``margen_manzana`` is the gap to the best
    rival in a DIFFERENT manzana, ``margen_predio`` the gap to the best rival with
    a different doc (any manzana); ``None`` when there is none. ``n_competidores``
    counts rivals in other manzanas within :data:`COMPETITOR_WINDOW`.
    """
    best_manzana = str(normalizer._manzana[best.doc])
    margen_manzana = margen_predio = None
    n_competidores = 0
    rivals = sorted(
        (c for c in kept if c.doc != best.doc and not math.isnan(c.score) and c.score >= normalizer.threshold),
        key=lambda c: -c.score,
    )
    for cand in rivals:
        gap = max(best.score - cand.score, 0.0)
        # Sorted by score: once both margins are known and we left the window, nothing changes.
        if margen_manzana is not None and margen_predio is not None and gap > COMPETITOR_WINDOW + _EPS:
            break
        hard, soft = _classify_violations(
            parsed, parse_address(str(normalizer._direccion[cand.doc])), min_struct,
            struct_by_doc[cand.doc], plate_tolerance, soft_rules,
        )
        if hard or len(soft) > max_soft:
            continue
        if margen_predio is None:
            margen_predio = gap
        if str(normalizer._manzana[cand.doc]) != best_manzana:
            if margen_manzana is None:
                margen_manzana = gap
            if gap <= COMPETITOR_WINDOW + _EPS:
                n_competidores += 1
    return margen_manzana, margen_predio, n_competidores


def _empty_result(raw, estado: str, motivo: str, rule_canonical: str | None = None,
                  detection: Detection | None = None) -> dict:
    """A non-OK row: place columns may be filled, cadastral columns never are."""
    return {
        "direccion_entrada": raw,
        "estado": STATUS_MAP.get(estado, estado),
        "motivo": motivo,
        "direccion_normalizada": rule_canonical or None,
        "fuente_normalizacion": "reglas" if rule_canonical else None,
        "numero_predial_nacional": None,
        "manzana": None,
        "lat": None,
        "lon": None,
        "confianza": None,
        "confiabilidad": None,
        "confiabilidad_manzana": None,
        "margen": None,
        "nivel_precision": None,
        **_zone_columns(detection),
    }


def normalize_strict(
    normalizer: AddressNormalizer,
    raws: list,
    min_struct: float = DEFAULT_MIN_STRUCT,
    plate_tolerance: int = 0,
    k: int = 20,
    gazetteer: Gazetteer | None = None,
    stats: dict | None = None,
    barrio_buffer_m: float = BARRIO_BUFFER_M,
    zone_buffer_m: float = ZONE_BUFFER_M,
    gate_escalate: bool = False,
    feature_sink: list | None = None,
    ambiguity_delta: float = DEFAULT_AMBIGUITY_DELTA,
    soft_rules=None,
    max_soft: int | None = None,
    gate_fallback: bool | None = None,
) -> pd.DataFrame:
    """Normalize a list of raw strings; only rows passing every rule get cadastral fields.

    ``stats`` is filled in place with gate counters when a dict is passed, which is
    how the CLI reports what the geographic gate actually changed.

    ``feature_sink``, when a list, receives one entry per input row: the
    :func:`reliability_features` dict for OK rows and ``None`` otherwise. The
    reliability columns come from ``normalizer.reliability`` (``None`` when the
    artifact is missing, which leaves them ``None`` on OK rows).

    ``ambiguity_delta`` (default :data:`DEFAULT_AMBIGUITY_DELTA`, 0 = off) is a soft
    abstention: when positive, an OK row whose ``margen`` (fused-score gap to the
    best rule-passing candidate in another manzana) is strictly below it becomes
    ``SIN_MATCH``. A row with no competitor (``margen`` None) never abstains.

    Stage 3 options (``None`` = "use the artifact's ``decision`` block, else the historical
    behaviour"; an explicit value always wins):

    ``soft_rules``  names from :data:`SOFT_RULE_NAMES` whose violations become *soft* (row stays OK,
                    ``nivel_precision`` capped at ``manzana``) instead of rejecting the row.
    ``max_soft``    how many soft violations an OK row may carry (default 1).
    ``gate_fallback`` when the geographic gate would leave no candidate, keep the text-ranked
                    candidates instead of abstaining (note added, ``nivel_precision`` capped at
                    ``manzana``). The confidence threshold and every rule still apply.
    """
    decision = getattr(normalizer, "decision", None) or {}
    soft_rules = normalize_soft_rules(decision.get("soft_rules") if soft_rules is None else soft_rules)
    max_soft = validate_max_soft(decision.get("max_soft", 1) if max_soft is None else max_soft)
    gate_fallback = bool(decision.get("gate_fallback", False) if gate_fallback is None else gate_fallback)
    counters = {
        "filas_con_lugar": 0, "filas_con_reja": 0, "candidatos_descartados": 0,
        "mejor_candidato_cambiado": 0, "filas_sin_candidato_tras_reja": 0,
        "texto_limpiado": 0, "parseable_tras_limpiar": 0,
    }
    rows: list[dict | None] = [None] * len(raws)
    detections: list[Detection | None] = [None] * len(raws)
    queries: list[str | None] = [None] * len(raws)
    parses: list[object] = [None] * len(raws)
    to_model: list[int] = []
    feats: list[dict | None] = [None] * len(raws)
    reliability = getattr(normalizer, "reliability", None)

    for i, raw in enumerate(raws):
        if is_unspecified(raw):
            rows[i] = _empty_result(raw, "SIN_ESPECIFICAR", "dato vacio o marcador sin informacion")
            continue
        detection = gazetteer.detect(raw) if gazetteer is not None else None
        detections[i] = detection
        if detection is not None and detection.detected:
            counters["filas_con_lugar"] += 1
        text = str(raw) if detection is None else detection.cleaned_text
        if text != str(raw):
            counters["texto_limpiado"] += 1
        parsed = parse_address(text)
        if not parsed.parse_ok and text != str(raw):
            # The cleaned text must never lose a row the raw text could parse.
            fallback = parse_address(str(raw))
            if fallback.parse_ok:
                text, parsed = str(raw), fallback
        elif parsed.parse_ok and text != str(raw) and not parse_address(str(raw)).parse_ok:
            counters["parseable_tras_limpiar"] += 1
        if not parsed.parse_ok:
            rows[i] = _empty_result(
                raw, "NO_PARSEABLE",
                "; ".join(parsed.notes) or "gramatica de direccion no reconocida",
                detection=detection,
            )
            continue
        queries[i] = text
        parses[i] = parsed
        to_model.append(i)

    if to_model:
        comps = normalizer.score_components([queries[i] for i in to_model], k=k)
        weights = normalizer.weights
        fused = (
            weights["sim"] * comps["sim"] + weights["fuzz"] * comps["fuzz"]
            + weights["struct"] * comps["struct"] + weights["geo"] * comps["geo"]
        )
        for pos, i in enumerate(to_model):
            rows[i] = _build_row(
                normalizer, gazetteer, raws[i], parses[i], detections[i],
                comps["doc"][pos], comps["struct"][pos], fused[pos],
                min_struct, plate_tolerance, counters,
                barrio_buffer_m, zone_buffer_m, gate_escalate,
                reliability, feats, i, ambiguity_delta,
                soft_rules, max_soft, gate_fallback,
            )

    if feature_sink is not None:
        feature_sink.extend(feats)
    if stats is not None:
        stats.update(counters)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _build_row(
    normalizer, gazetteer, raw, parsed, detection,
    docs, struct_row, fused_row, min_struct, plate_tolerance, counters,
    barrio_buffer_m, zone_buffer_m, gate_escalate,
    reliability=None, feats: list | None = None, index: int = 0,
    ambiguity_delta: float = DEFAULT_AMBIGUITY_DELTA,
    soft_rules: frozenset = frozenset(), max_soft: int = 1, gate_fallback: bool = False,
) -> dict:
    """Score one already-parsed row: fuse, gate, threshold, then apply the rules."""
    rule_canonical = canonical(parsed, style="spaced", with_complement=True) or None
    slots = np.flatnonzero(docs >= 0)
    candidates = [
        Candidate(
            score=float(fused_row[s]), doc=int(docs[s]),
            lon=float(normalizer._lon[docs[s]]), lat=float(normalizer._lat[docs[s]]),
        )
        for s in slots
    ]
    struct_by_doc = {int(docs[s]): float(struct_row[s]) for s in slots}
    if not candidates:
        return _empty_result(
            raw, "SIN_MATCH", f"confianza 0.00 < umbral {normalizer.threshold:.2f}",
            rule_canonical, detection,
        )

    best_before = max(candidates, key=lambda c: c.score)
    kept, gate_label = (
        gate_candidates(
            gazetteer, detection, candidates,
            barrio_buffer_m=barrio_buffer_m, zone_buffer_m=zone_buffer_m,
            escalate=gate_escalate,
        )
        if (gazetteer is not None and detection is not None) else (candidates, None)
    )
    if gate_label is not None:
        counters["filas_con_reja"] += 1
        counters["candidatos_descartados"] += len(candidates) - len(kept)
    gate_overridden = None
    if not kept and gate_fallback and gate_label is not None:
        kept, gate_overridden = list(candidates), gate_label
    if not kept:
        counters["filas_sin_candidato_tras_reja"] += 1
        return _empty_result(
            raw, "SIN_MATCH_REGLAS", f"candidatos fuera de {gate_label}",
            rule_canonical, detection,
        )

    best = max(kept, key=lambda c: c.score)
    if gate_label is not None and best.doc != best_before.doc:
        counters["mejor_candidato_cambiado"] += 1

    confidence = float(min(max(best.score, 0.0), 1.0))
    if confidence < normalizer.threshold:
        return _empty_result(
            raw, "SIN_MATCH",
            f"confianza {confidence:.2f} < umbral {normalizer.threshold:.2f}",
            rule_canonical, detection,
        )

    cadastral_address = str(normalizer._direccion[best.doc])
    cadastral_parsed = parse_address(cadastral_address)
    hard, soft = _classify_violations(
        parsed, cadastral_parsed, min_struct,
        struct_by_doc[best.doc], plate_tolerance, soft_rules,
    )
    if hard or len(soft) > max_soft:
        return _empty_result(
            raw, "SIN_MATCH_REGLAS", "; ".join(hard + soft), rule_canonical, detection
        )

    margen_manzana, margen_predio, n_competidores = _competition(
        normalizer, kept, best, parsed, struct_by_doc, min_struct, plate_tolerance, soft_rules, max_soft,
    )
    if ambiguity_delta > 0 and margen_manzana is not None and margen_manzana < ambiguity_delta:
        return _empty_result(
            raw, "SIN_MATCH",
            f"ambiguo: {max(n_competidores, 1)} candidatos cercanos en otras manzanas",
            rule_canonical, detection,
        )

    located = gazetteer.locate(best.lon, best.lat) if gazetteer is not None else {}
    predial = str(normalizer._predial[best.doc])
    manzana = str(normalizer._manzana[best.doc])
    nivel = _precision_level(parsed)
    notes: list[str] = []
    if soft:
        # Soft violation(s): keep the predio but only vouch for the manzana.
        notes.append("aproximado: " + "; ".join(soft))
        nivel = _least_precise(nivel, "manzana")
    if gate_overridden is not None:
        # The detected place excluded every candidate: keep the text match, vouch for the manzana only.
        notes.append(f"fuera de {gate_overridden}")
        nivel = _least_precise(nivel, "manzana")
    n_predios = int(normalizer._n_predios[best.doc])
    if n_predios > 1:
        # The cadastre cannot tell the predios of a shared address apart: the
        # first predio (by OBJECTID) is reported, flagged and downgraded.
        notes.append(f"direccion compartida por {n_predios} predios")
        if int(normalizer._manzana_nunique[best.doc]) > 1:
            nivel = _least_precise(nivel, "direccion")
        else:
            nivel = _least_precise(nivel, "manzana")
    motivo = "; ".join(notes)
    features = reliability_features(
        confidence, struct_by_doc[best.doc],
        relaxed_letter=any(m.startswith(("letra de", "bis de", "cuadrante")) for m in soft),
        relaxed_plate=any(m.startswith("placa") for m in soft),
        n_predios=n_predios, nivel=nivel, gate_applied=gate_label is not None,
        margen_manzana=margen_manzana, margen_predio=margen_predio, n_competidores=n_competidores,
    )
    if feats is not None:
        feats[index] = features
    conf_predio = conf_manzana = None
    if reliability is not None:
        pp, pm = reliability.predict(features)
        conf_predio, conf_manzana = round(pp, 3), round(pm, 3)
    return {
        "direccion_entrada": raw,
        "estado": "OK",
        "motivo": motivo,
        # Cadastral units (AP, TO, ...) only survive when the input mentioned that kind.
        "direccion_normalizada": filter_complement(cadastral_address, parsed.complement),
        "fuente_normalizacion": "catastro",
        "numero_predial_nacional": predial,
        "manzana": manzana,
        "lat": best.lat,
        "lon": best.lon,
        "confianza": round(confidence, 4),
        "confiabilidad": conf_predio,
        "confiabilidad_manzana": conf_manzana,
        "margen": None if margen_manzana is None else round(margen_manzana, 4),
        "nivel_precision": nivel,
        **_zone_columns(detection, located, normalizer._comuna[best.doc]),
    }


def load_gazetteer(basemaps_dir: str, warn=None) -> Gazetteer | None:
    """Load the gazetteer, degrading to plain normalization if the basemaps are gone.

    ``warn`` receives one human-readable line when the gazetteer is unavailable;
    the CLI routes it to stderr, the API to the startup log.
    """
    try:
        return Gazetteer.load(basemaps_dir)
    except (FileNotFoundError, ValueError) as exc:
        if warn is not None:
            warn(f"warning: gazetteer disabled ({exc})")
        return None


# ---------------------------------------------------------------------------
# Address column detection
# ---------------------------------------------------------------------------
#: Column-name synonyms, already accent-folded and lowercased. "direccion"
#: covers "dirección", "ubicacion" covers "ubicación": the scorer normalizes the
#: candidate name the same way before comparing.
ADDRESS_SYNONYMS = (
    "direccion",
    "address",
    "addr",
    "street",
    "calle",
    "ubicacion",
    "domicilio",
    "location",
    "dir",
    "domicile",
    "localizacion",
)

#: Synonyms short enough that a bare substring hit would be mostly noise
#: ("dir" inside "director", "directorio"). They still match as a whole token.
_MIN_SUBSTRING_LEN = 4

#: A guess below this never reaches the caller as a candidate.
CANDIDATE_SCORE_FLOOR = 70.0

#: Auto-selection needs this score ...
AUTO_SCORE_FLOOR = 85.0
#: ... and this much daylight over the runner-up, unless it is an exact synonym.
#: Neither relaxation applies to a tie at the top score: see
#: :func:`resolve_address_column`, where a tie always raises.
AUTO_MARGIN = 15.0


def _normalize_column_name(name) -> str:
    """`"DIRECCIÓN_NORMALIZADA"` -> `"direccion normalizada"`."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^0-9a-zA-Z]+", " ", text).strip().lower()
    return re.sub(r"\s+", " ", text)


def _score_column(name) -> tuple[float, bool]:
    """Score one column name against the synonym set.

    Returns ``(score, exact)``. Tiers, highest first: the normalized name *is* a
    synonym (100, exact); one whole token is a synonym (88-95); a synonym is a
    substring (78-87); otherwise a discounted rapidfuzz ``token_set_ratio`` so a
    pure fuzzy resemblance can only clear the candidate floor for near-typos
    ("direcion"), never for unrelated words ("descripcion", "directorio").
    """
    key = _normalize_column_name(name)
    if not key:
        return 0.0, False
    best_fuzz = max(fuzz.token_set_ratio(key, syn) for syn in ADDRESS_SYNONYMS)
    if key in ADDRESS_SYNONYMS:
        return 100.0, True
    tokens = key.split()
    if any(token in ADDRESS_SYNONYMS for token in tokens):
        return round(min(95.0, 88.0 + best_fuzz / 100.0 * 7.0), 2), False
    if any(syn in key for syn in ADDRESS_SYNONYMS if len(syn) >= _MIN_SUBSTRING_LEN):
        return round(min(87.0, 78.0 + best_fuzz / 100.0 * 9.0), 2), False
    return round(best_fuzz * 0.75, 2), False


def detect_address_column(columns) -> tuple[str | None, list[dict]]:
    """Guess which column holds the address.

    Returns ``(best, ranked)``. ``best`` is the top-scoring column when it clears
    :data:`CANDIDATE_SCORE_FLOOR`, else ``None``. ``ranked`` covers *every* column,
    highest score first, so a caller can render a dropdown: each entry is
    ``{"column": str, "score": float}``. Ties keep the original column order,
    which matters because real exports put the authoritative address column
    before its derived twin ("DIRECCIÓN" then "DIRECCIÓN_NORMALIZADA").
    """
    scored = [
        {"column": str(col), "score": _score_column(col)[0], "_pos": pos}
        for pos, col in enumerate(columns)
    ]
    scored.sort(key=lambda item: (-item["score"], item["_pos"]))
    ranked = [{"column": item["column"], "score": item["score"]} for item in scored]
    best = ranked[0]["column"] if ranked and ranked[0]["score"] >= CANDIDATE_SCORE_FLOOR else None
    return best, ranked


def resolve_address_column(df: pd.DataFrame, requested: str | None) -> str:
    """Return the address column to use, or raise :class:`AddressColumnError`.

    With ``requested`` the column only has to exist (exact match first, then
    case-insensitive). Without it the guess must be unambiguous, in this order:

    1. Two or more columns sharing the top score are ALWAYS ambiguous and raise.
       That includes a tie between two exact synonym matches - a table with both
       ``direccion`` and ``domicilio`` gives no basis to prefer either, and
       picking the first by column order would silently normalize whichever field
       the export happened to put first.
    2. A single top scorer is accepted when it is an exact synonym match, or when
       it scores at least :data:`AUTO_SCORE_FLOOR` and leads the runner-up by
       :data:`AUTO_MARGIN`.

    Anything else is the "the caller must pick a field" case and raises, carrying
    the ranked candidates.
    """
    columns = [str(c) for c in df.columns]
    if requested is not None and str(requested).strip() != "":
        requested = str(requested)
        if requested in columns:
            return requested
        lowered = {c.lower(): c for c in reversed(columns)}
        hit = lowered.get(requested.lower())
        if hit is not None:
            return hit
        raise AddressColumnError(
            f"la columna {requested!r} no existe en el archivo",
            available=columns, requested=requested,
        )

    best, ranked = detect_address_column(columns)
    candidates = [item for item in ranked if item["score"] >= CANDIDATE_SCORE_FLOOR]
    if best is None:
        raise AddressColumnError(
            "no se pudo detectar la columna de direccion; indique cual usar",
            available=columns, candidates=candidates,
        )
    top = ranked[0]
    runner_up = ranked[1]["score"] if len(ranked) > 1 else 0.0
    # The tie check comes FIRST and the exactness shortcut below can never
    # override it: `direccion` and `domicilio` are both exact synonyms scoring
    # 100, and returning whichever the file listed first would normalize the wrong
    # field without telling anyone.
    tied_at_top = [item["column"] for item in ranked if item["score"] == top["score"]]
    if len(tied_at_top) == 1:
        exact = _score_column(top["column"])[1]
        if exact or (top["score"] >= AUTO_SCORE_FLOOR
                     and top["score"] - runner_up >= AUTO_MARGIN):
            return top["column"]
    raise AddressColumnError(
        "varias columnas podrian ser la direccion; indique cual usar",
        available=columns, candidates=candidates,
    )


# ---------------------------------------------------------------------------
# Header row detection
# ---------------------------------------------------------------------------
#: A header cell is a short label, not a value. Longer than this and it is prose.
_MAX_LABEL_LEN = 60


def _looks_like_label(value) -> bool:
    """True for a short, letter-bearing, non-numeric string cell."""
    if value is None or not isinstance(value, str):
        return False
    text = value.strip()
    if not text or len(text) > _MAX_LABEL_LEN or "\n" in text:
        return False
    if not re.search(r"[A-Za-zÀ-ɏ]", text):
        return False
    try:
        float(text.replace(",", "."))
    except ValueError:
        return True
    return False


def guess_header_row(raw_df_no_header: pd.DataFrame, max_scan_rows: int = 15) -> int:
    """Index of the row that most likely holds the column names.

    The heuristic is the one this project already applied by hand to the ArcGIS
    Excel exports (inspecciones / stickers / acciones prefix the real header with
    2-5 metadata rows): take the row with the most non-null cells, and break ties
    toward the row whose values look most like short labels rather than data.
    Counting non-nulls is what separates the header from the data in these files,
    because every column is named while most data rows are sparse (inspecciones:
    145 named columns, 92-96 filled cells per data row). Returns 0 for an empty
    or single-row frame, so a well-formed table is unaffected.
    """
    if raw_df_no_header is None or len(raw_df_no_header) == 0:
        return 0
    scan = min(int(max_scan_rows), len(raw_df_no_header))
    best_index, best_key = 0, None
    for i in range(scan):
        row = raw_df_no_header.iloc[i]
        values = [v for v in row.tolist() if not (v is None or (isinstance(v, float) and np.isnan(v)))]
        non_null = len(values)
        if non_null == 0:
            continue
        label_ratio = sum(1 for v in values if _looks_like_label(v)) / non_null
        key = (non_null, label_ratio)
        if best_key is None or key > best_key:
            best_index, best_key = i, key
    return best_index


# ---------------------------------------------------------------------------
# Multi-format table reading
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = (".xlsx", ".xls", ".xlsm", ".csv", ".geojson", ".json", ".zip")

#: Keys :func:`read_table` sets on ``df.attrs``.
ATTRS_KEYS = ("source_format", "warnings", "geometry_columns", "header_row")

_HEADER_SCAN_ROWS = 15
_GEOMETRY_COLUMNS = ("_lon", "_lat")


def _finalize(df: pd.DataFrame, source_format: str, warnings: list[str],
              header_row: int | None) -> pd.DataFrame:
    """Attach the side-channel metadata and normalize the index."""
    df = df.reset_index(drop=True)
    df.attrs["source_format"] = source_format
    df.attrs["warnings"] = list(warnings)
    df.attrs["geometry_columns"] = [c for c in _GEOMETRY_COLUMNS if c in df.columns]
    df.attrs["header_row"] = header_row
    return df


def _read_excel_bytes(data: bytes, sheet, header: int | None,
                      warnings: list[str]) -> tuple[pd.DataFrame, int]:
    sheet_name = sheet if sheet not in (None, "") else 0
    if header is None:
        probe = pd.read_excel(io.BytesIO(data), sheet_name=sheet_name, header=None,
                              nrows=_HEADER_SCAN_ROWS)
        header = guess_header_row(probe, _HEADER_SCAN_ROWS)
        if header != 0:
            warnings.append(f"fila de encabezado detectada automaticamente: {header}")
    df = pd.read_excel(io.BytesIO(data), sheet_name=sheet_name, header=header)
    return df, header


def _sniff_delimiter(data: bytes) -> str:
    """`,` vs `;`, decided on the first non-empty line."""
    text = data[:65536].decode("utf-8", errors="replace")
    for line in text.splitlines():
        if line.strip():
            return ";" if line.count(";") > line.count(",") else ","
    return ","


def _decode_csv(data: bytes) -> tuple[str, str]:
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("latin-1"), "latin-1"


def _csv_rows(text: str, sep: str, max_rows: int) -> list[list]:
    """First rows of a CSV at their own widths, blanks normalized to ``None``.

    Deliberately NOT ``pd.read_csv``: pandas fixes the column count from the first
    physical line, so a 2-cell metadata row above a 31-column header either raises
    or (with ``on_bad_lines="skip"``) silently drops the real header - which is
    exactly the row being looked for. ``csv.reader`` keeps every row at its own
    width.
    """
    import csv

    rows: list[list] = []
    for index, row in enumerate(csv.reader(io.StringIO(text), delimiter=sep)):
        if index >= max_rows:
            break
        rows.append([cell if str(cell).strip() != "" else None for cell in row])
    return rows


def _pad(rows: list[list]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    width = max(len(row) for row in rows)
    return pd.DataFrame([row + [None] * (width - len(row)) for row in rows])


def _width_stable_from(rows: list[list], start: int) -> bool:
    """True when the table is rectangular from ``start`` down, with data under it.

    That is the signature of a real header: the metadata lines above it may be
    narrow or uneven, everything from the header down has the same width, and at
    least one data row follows. Requiring a following row is what rejects a guess
    that landed on the last row of a ragged file - a CSV with unquoted commas
    inside a field, where the caller should get a parse error rather than columns
    silently renamed to a data value.
    """
    widths = [
        len(row) for index, row in enumerate(rows)
        if index >= start and any(cell is not None for cell in row)
    ]
    return len(widths) > 1 and len(set(widths)) == 1


def _first_stable_width_row(rows: list[list]) -> int | None:
    """Earliest index that :func:`_width_stable_from` accepts, or ``None``."""
    for index, row in enumerate(rows):
        if any(cell is not None for cell in row) and _width_stable_from(rows, index):
            return index
    return None


def _read_csv_bytes(data: bytes, header: int | None,
                    warnings: list[str]) -> tuple[pd.DataFrame, int]:
    text, encoding = _decode_csv(data)
    if encoding != "utf-8":
        warnings.append(f"archivo decodificado como {encoding} (no es UTF-8 valido)")
    sep = _sniff_delimiter(data)
    if header is None:
        rows = _csv_rows(text, sep, _HEADER_SCAN_ROWS)
        header = guess_header_row(_pad(rows), _HEADER_SCAN_ROWS)
        if header > 0 and not _width_stable_from(rows, header):
            # The densest row is not a row the table stays rectangular from, so it
            # is a data row, not a header. Fall back rather than rename columns
            # after a value that happened to contain the delimiter.
            stable = _first_stable_width_row(rows)
            if stable is None:
                warnings.append(
                    "no se pudo identificar una fila de encabezado consistente; se uso "
                    "la fila 0. Revise que el archivo no tenga comas sin entrecomillar "
                    "dentro de un campo."
                )
            header = stable if stable is not None else 0
        if header != 0:
            warnings.append(f"fila de encabezado detectada automaticamente: {header}")
    # `skiprows` rather than `header=N`: the C parser fixes the field count from the
    # FIRST physical line, so a 2-cell metadata row above a 31-column header makes it
    # fail with "Expected 2 fields". Skipping makes the header the first line it sees.
    df = pd.read_csv(io.StringIO(text), skiprows=header, header=0, sep=sep)
    return df, header


def _centroid(geometry) -> tuple[float, float]:
    """Centroid of any GeoJSON-like mapping, as ``(lon, lat)`` or ``(nan, nan)``."""
    from shapely.geometry import shape as shapely_shape

    if not geometry:
        return float("nan"), float("nan")
    try:
        point = shapely_shape(geometry).centroid
        if point.is_empty:
            return float("nan"), float("nan")
        return float(point.x), float(point.y)
    except Exception:  # malformed or unsupported geometry: keep the attributes
        return float("nan"), float("nan")


def _read_geojson_bytes(data: bytes, warnings: list[str]) -> pd.DataFrame:
    text, encoding = _decode_csv(data)
    if encoding != "utf-8":
        warnings.append(f"archivo decodificado como {encoding} (no es UTF-8 valido)")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TableFormatError(f"JSON invalido: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise TableFormatError(
            "el archivo GeoJSON debe ser un FeatureCollection "
            "(se recibio "
            f"{payload.get('type') if isinstance(payload, dict) else type(payload).__name__})"
        )
    features = payload.get("features")
    if not isinstance(features, list):
        raise TableFormatError("el FeatureCollection no tiene una lista 'features'")
    rows, any_geometry = [], False
    for feature in features:
        if not isinstance(feature, dict):
            raise TableFormatError("cada elemento de 'features' debe ser un objeto")
        properties = feature.get("properties") or {}
        if not isinstance(properties, dict):
            properties = {"properties": properties}
        row = dict(properties)
        geometry = feature.get("geometry")
        lon, lat = _centroid(geometry)
        if geometry:
            any_geometry = True
        row["_lon"], row["_lat"] = lon, lat
        rows.append(row)
    if not any_geometry:
        warnings.append("ninguna feature trae geometria; _lon/_lat quedan vacios")
    return pd.DataFrame(rows)


_SHAPEFILE_REQUIRED = (".shp", ".shx", ".dbf")


def _read_shapefile_zip(data: bytes, warnings: list[str]) -> pd.DataFrame:
    import shapefile

    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise TableFormatError(f"el .zip no se puede abrir: {exc}") from exc
    members = {
        os.path.splitext(name)[1].lower(): name
        for name in archive.namelist()
        if not name.endswith("/")
    }
    missing = [ext for ext in _SHAPEFILE_REQUIRED if ext not in members]
    if missing:
        raise TableFormatError(
            "el .zip no contiene un shapefile completo; falta "
            + ", ".join(missing)
            + ". Comprima juntos .shp, .shx y .dbf (y .prj si lo tiene)."
        )
    with tempfile.TemporaryDirectory() as tmp:
        archive.extractall(tmp)
        shp_path = os.path.join(tmp, members[".shp"])
        transformer = None
        if ".prj" in members:
            transformer = _prj_transformer(os.path.join(tmp, members[".prj"]), warnings)
        else:
            warnings.append(
                "el shapefile no trae .prj; se asume EPSG:4326 para los centroides"
            )
        reader = shapefile.Reader(shp_path)
        try:
            rows, any_geometry = [], False
            for shape_record in reader.iterShapeRecords():
                row = dict(shape_record.record.as_dict())
                geometry = None
                try:
                    geometry = shape_record.shape.__geo_interface__
                except Exception:
                    geometry = None
                lon, lat = _centroid(geometry)
                if geometry and np.isfinite(lon) and np.isfinite(lat):
                    any_geometry = True
                    if transformer is not None:
                        lon, lat = transformer.transform(lon, lat)
                row["_lon"], row["_lat"] = float(lon), float(lat)
                rows.append(row)
        finally:
            reader.close()
    if not any_geometry:
        warnings.append("ninguna geometria utilizable; _lon/_lat quedan vacios")
    return pd.DataFrame(rows)


def _prj_transformer(prj_path: str, warnings: list[str]):
    """Transformer from the shapefile CRS to EPSG:4326, or ``None`` if unneeded."""
    import pyproj

    with open(prj_path, "r", encoding="utf-8", errors="replace") as fh:
        wkt = fh.read().strip()
    if not wkt:
        warnings.append("el .prj esta vacio; se asume EPSG:4326")
        return None
    crs = None
    try:
        crs = pyproj.CRS.from_wkt(wkt)
    except Exception:
        try:
            crs = pyproj.CRS.from_user_input(wkt)
        except Exception:
            warnings.append("no se pudo interpretar el .prj; se asume EPSG:4326")
            return None
    target = pyproj.CRS.from_epsg(4326)
    if crs.equals(target):
        return None
    warnings.append(f"centroides reproyectados de {crs.name} a EPSG:4326")
    return pyproj.Transformer.from_crs(crs, target, always_xy=True)


def read_table(data: bytes, filename: str, sheet: str | None = None,
               header: int | None = None) -> pd.DataFrame:
    """Read tabular bytes into a DataFrame, dispatching on the file extension.

    Supported: ``.xlsx`` / ``.xlsm`` / ``.xls`` (``sheet``, header guessed when
    ``header`` is None), ``.csv`` (UTF-8 then latin-1, ``,`` / ``;`` sniffed),
    ``.geojson`` / ``.json`` (a FeatureCollection; properties become columns and
    the feature centroid becomes ``_lon`` / ``_lat``) and a ``.zip`` holding a
    shapefile (``.shp`` + ``.shx`` + ``.dbf``, optional ``.prj`` used to
    reproject the centroids to EPSG:4326).

    Format, any warnings, the geometry columns it derived and the header row it
    used are attached to ``df.attrs`` (:data:`ATTRS_KEYS`), so nothing is lost.
    Raises :class:`TableFormatError` (a ``ValueError``) with an actionable
    message for anything else.
    """
    ext = os.path.splitext(str(filename))[1].lower()
    warnings: list[str] = []
    if not data:
        raise TableFormatError("el archivo esta vacio (0 bytes)")

    if ext in (".xlsx", ".xlsm", ".xls"):
        try:
            df, used_header = _read_excel_bytes(data, sheet, header, warnings)
        except TableFormatError:
            raise
        except ValueError as exc:
            raise TableFormatError(f"no se pudo leer la hoja de calculo: {exc}") from exc
        except Exception as exc:
            raise TableFormatError(f"no se pudo leer la hoja de calculo: {exc}") from exc
        return _finalize(df, "xlsx", warnings, used_header)

    if ext == ".csv":
        try:
            df, used_header = _read_csv_bytes(data, header, warnings)
        except TableFormatError:
            raise
        except Exception as exc:
            raise TableFormatError(f"no se pudo leer el CSV: {exc}") from exc
        return _finalize(df, "csv", warnings, used_header)

    if ext in (".geojson", ".json"):
        return _finalize(_read_geojson_bytes(data, warnings), "geojson", warnings, None)

    if ext == ".zip":
        return _finalize(_read_shapefile_zip(data, warnings), "shp", warnings, None)

    if ext in (".shp", ".shx", ".dbf", ".prj"):
        raise TableFormatError(
            f"un archivo {ext} suelto no se puede leer: comprima juntos .shp, .shx y .dbf "
            "(y .prj si lo tiene) en un unico .zip y suba ese .zip."
        )

    raise TableFormatError(
        f"extension no soportada: {ext or '(sin extension)'}. "
        f"Formatos aceptados: {', '.join(SUPPORTED_EXTENSIONS)}"
    )


# ---------------------------------------------------------------------------
# CLI-facing input reading
# ---------------------------------------------------------------------------
def read_input_file(path: str, column: str | None, sheet, header: int | None
                    ) -> tuple[list, pd.DataFrame | None, str | None]:
    """Return (raw addresses, source dataframe or None for plain text files, address column).

    ``header`` of ``None`` means "guess with :func:`guess_header_row`"; an explicit
    integer always wins. Raises :class:`AddressColumnError` when the address
    column cannot be resolved.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls", ".csv"):
        with open(path, "rb") as fh:
            data = fh.read()
        df = read_table(data, path, sheet=sheet, header=header)
    elif ext in (".geojson", ".json", ".zip"):
        with open(path, "rb") as fh:
            data = fh.read()
        df = read_table(data, path)
    else:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return [line.rstrip("\r\n") for line in fh], None, None
    resolved = resolve_address_column(df, column)
    return df[resolved].tolist(), df.reset_index(drop=True), resolved


def attach_source_columns(result: pd.DataFrame, source: pd.DataFrame | None, keep: str | None,
                          address_column: str | None = None) -> pd.DataFrame:
    """Prepend selected source columns.

    Output column names are the documented contract, so a source column that
    collides with one of them (`comuna`, `lat`, `barrio_vereda`...) is carried as
    `<name>_original`. Source coordinates are suffixed as a pair so `lat`/`lng`
    never end up half renamed.
    """
    if source is None or not keep:
        return result
    if keep.strip().lower() == "all":
        cols = list(source.columns)
    else:
        cols = [c.strip() for c in keep.split(",") if c.strip()]
        missing = [c for c in cols if c not in source.columns]
        if missing:
            raise KeepColumnsError(
                f"--keep-columns not found in input: {missing}. Columns: {list(source.columns)}"
            )
    if address_column in cols:
        # the input address is already carried by the source column: keep exactly
        # two address columns, the input and the normalized output
        result = result.drop(columns=["direccion_entrada"])
    coordinate_names = {"lat", "lon", "lng", "latitud", "longitud"}
    rename = {c: f"{c}_original" for c in cols if c in result.columns}
    if coordinate_names & {c.lower() for c in cols}:
        rename.update({c: f"{c}_original" for c in cols if c.lower() in coordinate_names})
    kept = source[cols].rename(columns=rename).reset_index(drop=True)
    return pd.concat([kept, result.reset_index(drop=True)], axis=1)


# ---------------------------------------------------------------------------
# Output serialization
# ---------------------------------------------------------------------------
def _write_xlsx(result: pd.DataFrame, target) -> None:
    """Three sheets: every row, the ones to review, and the status counts."""
    summary = result["estado"].value_counts().rename_axis("estado").reset_index(name="filas")
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="normalizado", index=False)
        result[result["estado"] != "OK"].to_excel(writer, sheet_name="revisar", index=False)
        summary.to_excel(writer, sheet_name="resumen", index=False)


def write_output(result: pd.DataFrame, path: str) -> None:
    if path.lower().endswith(".xlsx"):
        _write_xlsx(result, path)
    else:
        result.to_csv(path, index=False, encoding="utf-8-sig")


def write_xlsx_bytes(result: pd.DataFrame) -> bytes:
    """Same workbook :func:`write_output` produces, in memory."""
    buffer = io.BytesIO()
    _write_xlsx(result, buffer)
    return buffer.getvalue()


def write_csv_bytes(result: pd.DataFrame) -> bytes:
    return result.to_csv(index=False).encode("utf-8-sig")


def results_to_records(result: pd.DataFrame) -> list[dict]:
    """JSON-safe records: every NaN/NaT becomes ``None``."""
    return result.astype(object).where(pd.notna(result), None).to_dict(orient="records")


def summarize(result: pd.DataFrame) -> dict:
    """Status counts, always with all three public statuses present."""
    counts = result["estado"].value_counts().to_dict() if len(result) else {}
    out = {status: int(counts.get(status, 0)) for status in PUBLIC_STATUSES}
    out["total"] = int(len(result))
    return out


def results_to_geojson(result: pd.DataFrame) -> dict:
    """FeatureCollection with a Point per resolved row.

    Rows without a coordinate keep ``"geometry": null`` rather than being dropped,
    so the feature count always matches the row count and a caller can see what
    could not be located.
    """
    features = []
    for record in results_to_records(result):
        lat, lon = record.get("lat"), record.get("lon")
        geometry = None
        try:
            if lat is not None and lon is not None:
                flon, flat = float(lon), float(lat)
                if np.isfinite(flon) and np.isfinite(flat):
                    geometry = {"type": "Point", "coordinates": [flon, flat]}
        except (TypeError, ValueError):
            geometry = None
        properties = {k: v for k, v in record.items() if k not in ("lat", "lon")}
        features.append({"type": "Feature", "geometry": geometry, "properties": properties})
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
