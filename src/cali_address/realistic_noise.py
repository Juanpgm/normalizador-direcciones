"""Measure how real address strings are written and reweight the augmenter.

Only *structural* frequencies are measured (does the string contain a ``#``, a
glued ``5BN``, a spelled-out street type, ...). The measured table holds numbers
only; no address text is stored. The frequencies of the real strings are compared
with the frequencies of the strings the current augmenter produces, and the ratio
becomes a multiplier on the probability of the matching corruption operator (see
``augment.corrupt(op_scale=...)``). No operator is removed.
"""

from __future__ import annotations

import random
import re

from .augment import OPERATIONS, corrupt_many

__all__ = ["PATTERNS", "PATTERN_TO_OP", "measure_patterns", "compute_op_scale", "estimate_op_scale"]

_VIA_WORDS = re.compile(r"\b(CALLE|CARRERA|AVENIDA|DIAGONAL|TRANSVERSAL|CIRCULAR|AUTOPISTA)\b", re.I)
_VIA_ABBREV = re.compile(r"^\W*(CL|CLL|CLLE|CR|CRA|KR|KRA|K|C|AV|AVDA|AC|AK|DG|DIAG|TV|TRV)\b", re.I)
_QUAD_SPELLED = re.compile(r"\b(NORTE|SUR|ESTE|OESTE|ORIENTE|OCCIDENTE)\b", re.I)
_QUAD_ABBREV = re.compile(r"(?<=\d)\s*(N|S|E|O|NTE|OE)\b|\b(NTE|OE)\b", re.I)
_NO_TOKEN = re.compile(r"\b(No|Nro|Num|Numero|N)\b\.?|N°|Nº", re.I)
_ATTACHED = re.compile(r"\d[A-Za-z]")
_MISSING_SEP = re.compile(r"^\W*[A-Za-z]{1,4}\.?\s*\d+\s*[A-Za-z]{0,3}\s+\d+\s*[A-Za-z]{0,3}\s+\d+")
_DOUBLE_PLATE = re.compile(r"\d+\s*[-#]\s*\d+.*\d+\s*[-#]\s*\d+")
_COMPLEMENT = re.compile(
    r"\b(APTO?|APT|AP|TORRE|TO|CASA|CS|LOCAL|LC|PISO|BLOQUE|BQ|INT|INTERIOR|CONJUNTO|CONJ|EDIF|EDIFICIO|"
    r"OFICINA|OF|BODEGA|BG|MANZANA|MZ|LOTE|LT|URB|UNIDAD)\b", re.I)
_CITY = re.compile(r"\bCALI\b", re.I)

#: pattern name -> predicate on a single string.
PATTERNS = {
    "has_hash": lambda s: "#" in s,
    "has_dash": lambda s: "-" in s,
    "has_no_token": lambda s: bool(_NO_TOKEN.search(s)),
    "digit_letter_attached": lambda s: bool(_ATTACHED.search(s)),
    "missing_separators": lambda s: "#" not in s and "-" not in s and bool(_MISSING_SEP.search(s)),
    "via_type_spelled": lambda s: bool(_VIA_WORDS.search(s)),
    "via_type_abbrev": lambda s: bool(_VIA_ABBREV.search(s)),
    "quadrant_spelled": lambda s: bool(_QUAD_SPELLED.search(s)),
    "quadrant_abbrev": lambda s: bool(_QUAD_ABBREV.search(s)),
    "double_plate": lambda s: bool(_DOUBLE_PLATE.search(s)),
    "trailing_complement": lambda s: bool(_COMPLEMENT.search(s)),
    "has_lowercase": lambda s: any(c.islower() for c in s),
    "city_suffix": lambda s: bool(_CITY.search(s)),
}

#: pattern -> (augment operation, "same" | "inverse").
#: "same": more of the pattern in real data means the operator should fire more.
#: "inverse": the operator *removes* the pattern from the canonical form.
PATTERN_TO_OP = {
    "has_hash": ("hash_variant", "inverse"),
    "has_dash": ("plate_separator", "inverse"),
    "digit_letter_attached": ("glue_letters", "same"),
    "missing_separators": ("glue_all", "same"),
    "via_type_spelled": ("expand_via_type", "same"),
    "quadrant_abbrev": ("quadrant_abbrev", "same"),
    "double_plate": ("duplicate_address", "same"),
    "trailing_complement": ("add_complement", "same"),
    "has_lowercase": ("lowercase", "same"),
    "city_suffix": ("city_suffix", "same"),
}

_EPS = 0.01
SCALE_MIN, SCALE_MAX = 0.25, 4.0


def measure_patterns(strings) -> dict:
    """Share of strings exhibiting each pattern; empty/None strings are ignored."""
    clean = [str(s) for s in strings if s is not None and str(s).strip()]
    n = len(clean)
    out = {name: 0.0 for name in PATTERNS}
    if n == 0:
        return out
    for name, fn in PATTERNS.items():
        out[name] = sum(1 for s in clean if fn(s)) / n
    return out


def compute_op_scale(real: dict, synth: dict) -> dict:
    """Operator multipliers from real-vs-synthetic pattern frequencies.

    Ratios are smoothed with a small epsilon and clamped to
    ``[SCALE_MIN, SCALE_MAX]`` so one noisy pattern can never switch an operator
    off or make it dominate. Only operators from ``OPERATIONS`` are emitted.
    """
    scale: dict = {}
    for pattern, (op, mode) in PATTERN_TO_OP.items():
        if op not in OPERATIONS or pattern not in real or pattern not in synth:
            continue
        r, s = real[pattern], synth[pattern]
        if mode == "inverse":
            r, s = 1.0 - r, 1.0 - s
        ratio = (r + _EPS) / (s + _EPS)
        scale[op] = float(min(max(ratio, SCALE_MIN), SCALE_MAX))
    # complements: the drop operator moves opposite to the add operator.
    if "add_complement" in scale:
        scale["drop_complement"] = float(min(max(1.0 / scale["add_complement"], SCALE_MIN), SCALE_MAX))
    return scale


def estimate_op_scale(real_strings, canon_docs, n_docs: int = 3000, seed: int = 42) -> dict:
    """Measure real vs current-augmenter frequencies and derive the operator scale.

    Returns ``{"real": {...}, "synthetic": {...}, "op_scale": {...}, "n_real": int, "n_synthetic": int}``.
    """
    real_strings = [s for s in real_strings if s is not None and str(s).strip()]
    rng = random.Random(seed)
    canon_docs = [c for c in canon_docs if c]
    picks = rng.sample(canon_docs, min(n_docs, len(canon_docs))) if canon_docs else []
    synth = [v for c in picks for v in corrupt_many(c, 3, rng)]
    real_f, synth_f = measure_patterns(real_strings), measure_patterns(synth)
    return {
        "real": real_f,
        "synthetic": synth_f,
        "op_scale": compute_op_scale(real_f, synth_f) if real_strings and synth else {},
        "n_real": len(real_strings),
        "n_synthetic": len(synth),
    }
