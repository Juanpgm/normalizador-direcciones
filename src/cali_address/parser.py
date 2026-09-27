"""Rule-based canonicalizer/parser for Colombian (Cali) street addresses.

The target canonical form is the one used by the Cali cadastral layer
(`urbano_terreno.direccion`) and, up to letter spacing, by the IDESC geocoder
(`dir_ajusta`)::

    KR 26 H 1 # 73 - 10
    CL 12 # 48 BIS - 31
    AV 15 OESTE # 9 OESTE - 137
    CL 25 NORTE # AV 6 - 30

Grammar (informal)::

    address    := via "#" cross "-" plate [complement]
    via        := via_type number [letters] [number] [letters] [BIS] [quadrant]
    cross      := [via_type] number [letters] [number] [letters] [BIS] [quadrant]
    plate      := digits                      # zero padded to 2 in canonical form
    complement := free text (AP 501, LC 1, TO 3, ED PAMPLONA, GASS 144, ...)

Design notes
------------
* The `#` and the plate `-` are located **before** any token gluing/splitting so
  the complement text can be preserved verbatim (``14C`` stays ``14C``).
* Legacy one-letter via types used by the cadastre (``K``=KR, ``C``=CL,
  ``A``=AV, ``D``=DG, ``T``=TV) are normalized.
* The parser never raises; unparseable input yields ``parse_ok=False``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Iterable

__all__ = [
    "ParsedAddress",
    "parse_address",
    "parse_many",
    "canonical",
    "canonicalize",
    "filter_complement",
    "match_key",
    "strip_accents",
    "VIA_TYPE_CANON",
    "COMPLEMENT_CANON",
    "QUADRANT_CANON",
]

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

#: Every spelling we have observed -> canonical cadastral abbreviation.
VIA_TYPE_CANON: dict[str, str] = {}


def _register(canon: str, *aliases: str) -> None:
    VIA_TYPE_CANON[canon] = canon
    for alias in aliases:
        VIA_TYPE_CANON[alias] = canon


_register("CL", "CALLE", "CALLES", "CLL", "CLLE", "CLE", "CALL", "CAL", "C", "CLA")
_register("KR", "CARRERA", "CARRERAS", "CRA", "CR", "CRR", "CRRA", "KRA", "KRR",
          "KAR", "CARR", "CARERA", "CARREA", "K")
_register("AV", "AVENIDA", "AVDA", "AVE", "AVN", "AVEN", "A")
_register("AC", "AVENIDA CALLE", "AVCL", "AV CALLE", "AVC", "AV CL")
_register("AK", "AVENIDA CARRERA", "AVKR", "AV CARRERA", "AVK", "AV KR", "AV CRA")
_register("DG", "DIAGONAL", "DIAG", "DIGONAL", "D")
_register("TV", "TRANSVERSAL", "TRANSV", "TRANV", "TRV", "TRAV", "TRANSVESAL", "T")
_register("PJ", "PASAJE", "PSJ", "PJE")
_register("PS", "PASEO")
_register("CV", "CIRCUNVALAR", "CIRCUNV", "CIRC")
_register("AU", "AUTOPISTA", "AUTOP")
_register("CT", "CALLEJON")

#: Quadrant / cardinal suffixes. Only spellings of length >= 3 are accepted as a
#: quadrant when they sit right after a street number, because single letters in
#: that position are street letters in cadastral practice (``KR 26 N``).
QUADRANT_CANON: dict[str, str] = {
    "NORTE": "NORTE", "NTE": "NORTE", "NOR": "NORTE", "NORT": "NORTE",
    "SUR": "SUR",
    "ESTE": "ESTE", "EST": "ESTE",
    "OESTE": "OESTE", "OES": "OESTE", "OESTE.": "OESTE", "WEST": "OESTE",
}
#: Two-letter quadrant abbreviations. Single letters (``N``, ``S``, ``E``, ``O``,
#: ``W``) are deliberately NOT treated as quadrants: the cadastre writes
#: quadrants out in full (NORTE 30k / OESTE 21k occurrences, ESTE 0) and uses
#: single letters as street letters (``KR 41 E``, ``C 60 N``, ``K 73 B # 2 W``).
QUADRANT_SHORT: dict[str, str] = {"OE": "OESTE", "NTE": "NORTE"}
_QUADRANT_WORDS = ("NORTE", "SUR", "ESTE", "OESTE")

#: Complement (unit) kinds. Canonical spellings follow the cadastral layer, so
#: e.g. ``BLQ`` (not ``BL``) and ``PQ``/``GA`` are kept as they appear there.
COMPLEMENT_CANON: dict[str, str] = {}


def _register_comp(canon: str, *aliases: str) -> None:
    COMPLEMENT_CANON[canon] = canon
    for alias in aliases:
        COMPLEMENT_CANON[alias] = canon


_register_comp("AP", "APTO", "APARTAMENTO", "APT", "APA", "APARTAMENT", "APTO.")
_register_comp("TO", "TORRE", "TOR", "TORR")
_register_comp("LC", "LOCAL", "LOC", "LCL")
_register_comp("CA", "CASA", "CS")
_register_comp("ET", "ETAPA", "ETP")
_register_comp("BLQ", "BLOQUE", "BL", "BLK", "BQ")
_register_comp("MZ", "MANZANA", "MZN", "MZA")
_register_comp("LT", "LOTE", "LOT")
_register_comp("PH", "PENTHOUSE")
_register_comp("OF", "OFICINA", "OFC", "OFI")
_register_comp("ED", "EDIFICIO", "EDIF", "EDF")
_register_comp("CONJ", "CONJUNTO", "CJTO", "CONJT")
_register_comp("UR", "URBANIZACION", "URB")
_register_comp("INT", "INTERIOR", "IN")
_register_comp("PQ", "PARQUEADERO", "PARQ")
_register_comp("GA", "GARAJE", "GAR")
_register_comp("BG", "BODEGA", "BOD")
_register_comp("BR", "BARRIO", "BRR")
_register_comp("CGTO", "CORREGIMIENTO", "CORR")
_register_comp("VDA", "VEREDA")

#: Trailing administrative noise that carries no address information.
CITY_NOISE = (
    "CALI", "SANTIAGO DE CALI", "VALLE DEL CAUCA", "VALLE", "COLOMBIA",
    "V/CAUCA", "VALLE DEL CAUCA COLOMBIA", "CAUCA", "VALLE CAUCA",
)

_BIS_WORDS = ("BIS", "BS", "BISS")

_LETTER_RE = re.compile(r"^[A-Z]{1,4}$")
_NUM_RE = re.compile(r"^\d{1,4}$")
_HAS_LETTER_RE = re.compile(r"[A-Z]")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ParsedAddress:
    """Structured representation of a Colombian address."""

    raw: str = ""
    parse_ok: bool = False
    via_type: str | None = None
    via_number: str | None = None
    via_letters: str | None = None
    via_suffix: str | None = None
    via_suffix_letters: str | None = None
    via_bis: bool = False
    via_quadrant: str | None = None
    cross_type: str | None = None
    cross_number: str | None = None
    cross_letters: str | None = None
    cross_suffix: str | None = None
    cross_suffix_letters: str | None = None
    cross_bis: bool = False
    cross_quadrant: str | None = None
    plate: str | None = None
    complement: str | None = None
    had_hash: bool = False
    had_plate_sep: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def canonical(self, style: str = "spaced", with_complement: bool = True) -> str:
        return canonical(self, style=style, with_complement=with_complement)

    @property
    def has_cross(self) -> bool:
        return self.cross_number is not None

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return self.canonical()


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−_"), "-")


def strip_accents(text: str) -> str:
    """ASCII-fold a string (``Ñ`` -> ``N``), tolerating any unicode input."""
    nfkd = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _pre_normalize(raw: str) -> str:
    """Uppercase, ASCII-fold, unify separators and `number` words."""
    text = strip_accents(raw).upper()
    text = text.translate(_DASHES)
    # `N° 12` / `Nº 12` must be handled before the degree signs are stripped.
    text = re.sub(r"\bN\s*[°º]\s*\.?\s*(?=\d)", " # ", text)
    text = text.replace("º", " ").replace("°", " ").replace("ª", " ")
    # `No.12`, `No 12`, `NRO 12`, `NUMERO 12` -> `# 12`
    text = re.sub(r"\bNRO?S?\.?\s*(?=[\d#])", " # ", text)
    text = re.sub(r"\bNUMERO?S?\.?\s*(?=[\d#])", " # ", text)
    text = re.sub(r"\bNUM\.?\s*(?=[\d#])", " # ", text)
    text = re.sub(r"\bNO?\s*\.\s*(?=\d)", " # ", text)     # "No." / "N."
    text = re.sub(r"\bNO\s+(?=\d)", " # ", text)            # "No 12"
    # Ordinal noise: 3RA, 5TA, 1RO, 2DO, 4TO, 7MA, 9NA ...
    text = re.sub(r"(?<=\d)(RA|RO|TA|TO|DA|DO|MA|MO|NA|ER|ERA|VA|VO)\b", " ", text)
    text = re.sub(r"#+", " # ", text)
    text = re.sub(r"-{2,}", "-", text)
    text = text.replace("/", " ")
    text = re.sub(r"[^\w#\-;,. ]+", " ", text)
    text = re.sub(r"\s*-\s*", " - ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _drop_city_noise(text: str) -> tuple[str, list[str]]:
    """Remove trailing `, Cali, Valle del Cauca`-style administrative tails."""
    notes: list[str] = []
    parts = [p.strip() for p in text.split(",")]
    kept: list[str] = []
    for i, part in enumerate(parts):
        norm = re.sub(r"\s+", " ", part).strip(" .")
        if i > 0 and norm in CITY_NOISE:
            notes.append("dropped_city_noise")
            continue
        kept.append(part)
    text = ", ".join(p for p in kept if p)
    for noise in sorted(CITY_NOISE, key=len, reverse=True):
        new = re.sub(r"(?:[\s,.-]+)" + re.escape(noise) + r"\s*$", "", text)
        if new != text:
            notes.append("dropped_city_noise")
            text = new
    # A trailing '-' is meaningful (the cadastre writes `KR 27 #  -` for records
    # with no plate), so it is preserved.
    return text.lstrip(" ,.-").rstrip(" ,."), notes


def _split_glued(text: str) -> list[str]:
    """Tokenize a via/cross segment, splitting letter<->digit boundaries.

    ``CL12A`` -> ``['CL', '12', 'A']``; ``3E-1`` -> ``['3', 'E', '1']``.
    """
    text = re.sub(r"(?<=[A-Z])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[A-Z])", " ", text)
    for ch in "-.,#;:":
        text = text.replace(ch, " ")
    return [t for t in text.split() if t]


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


_VIA_ALIASES_LONG = None


def _fuzzy_via_type(token: str) -> str | None:
    """Repair via-type typos (``CATRERA`` -> ``KR``, ``CLLLE`` -> ``CL``)."""
    global _VIA_ALIASES_LONG
    if len(token) < 4 or not token.isalpha():
        return None
    if _VIA_ALIASES_LONG is None:
        _VIA_ALIASES_LONG = [a for a in VIA_TYPE_CANON if len(a) >= 4 and a.isalpha()]
    best, best_d = None, 99
    for alias in _VIA_ALIASES_LONG:
        d = _levenshtein(token, alias)
        if d < best_d:
            best, best_d = alias, d
    if best is not None and best_d <= max(1, len(best) // 4):
        return VIA_TYPE_CANON[best]
    return None


def _fuzzy_quadrant(token: str) -> str | None:
    """Repair quadrant typos such as ``OEATE`` -> ``OESTE`` by edit distance."""
    if len(token) < 4:
        return None
    best, best_d = None, 99
    for word in _QUADRANT_WORDS:
        d = _levenshtein(token, word)
        if d < best_d:
            best, best_d = word, d
    if best is not None and best_d <= max(1, len(best) // 4):
        return best
    return None


# ---------------------------------------------------------------------------
# Segment parsing
# ---------------------------------------------------------------------------


def _parse_segment(tokens: list[str]) -> dict:
    """Parse ``KR 26 H 1 B BIS OESTE`` style token runs."""
    out = {
        "type": None, "number": None, "letters": None, "suffix": None,
        "suffix_letters": None, "bis": False, "quadrant": None, "leftover": [],
    }
    i, n = 0, len(tokens)
    if i + 1 < n and f"{tokens[i]} {tokens[i + 1]}" in VIA_TYPE_CANON:
        out["type"] = VIA_TYPE_CANON[f"{tokens[i]} {tokens[i + 1]}"]
        i += 2
    elif i < n and tokens[i] in VIA_TYPE_CANON:
        out["type"] = VIA_TYPE_CANON[tokens[i]]
        i += 1
    elif i < n and _fuzzy_via_type(tokens[i]) is not None:
        out["type"] = _fuzzy_via_type(tokens[i])
        i += 1
    # Leading BIS / spelled-out quadrant before the number is rare but legal.
    while i < n and not _NUM_RE.match(tokens[i]):
        tok = tokens[i]
        if tok in _BIS_WORDS:
            out["bis"] = True
            i += 1
            continue
        q = QUADRANT_CANON.get(tok)
        if q is not None:
            out["quadrant"] = q
            i += 1
            continue
        break
    if i < n and _NUM_RE.match(tokens[i]):
        out["number"] = str(int(tokens[i]))
        i += 1
    else:
        out["leftover"] = tokens[i:]
        return out
    # Letter run right after the number (street letter, e.g. `26 H`). The
    # cadastre occasionally stacks two of them (`A 2 A N`, `K 24 D W`).
    letters: list[str] = []
    while i < n and len(letters) < 2 and _LETTER_RE.match(tokens[i])             and tokens[i] not in _BIS_WORDS:
        tok = tokens[i]
        q = QUADRANT_CANON.get(tok) or QUADRANT_SHORT.get(tok)
        if q is not None:
            out["quadrant"] = q
            i += 1
            break
        if (
            len(tok) == 2
            and tok[1] in "NOS"
            and tok not in VIA_TYPE_CANON
            and tok not in QUADRANT_SHORT
            and not letters
        ):
            # `AV 5 AN` / `# 23 DN` -> street letter + glued cardinal suffix.
            letters.append(tok[0])
            out["quadrant"] = {"N": "NORTE", "O": "OESTE", "S": "SUR"}[tok[1]]
            i += 1
            break
        letters.append(tok)
        i += 1
    if letters:
        out["letters"] = " ".join(letters)
    # Optional secondary number (`KR 26 H 1`) and its letter (`KR 1 A 5 B`).
    if i < n and _NUM_RE.match(tokens[i]):
        out["suffix"] = str(int(tokens[i]))
        i += 1
        if i < n and _LETTER_RE.match(tokens[i]) and tokens[i] not in _BIS_WORDS \
                and tokens[i] not in QUADRANT_CANON:
            out["suffix_letters"] = tokens[i]
            i += 1
    # BIS / quadrant tail in any order.
    while i < n:
        tok = tokens[i]
        if tok in _BIS_WORDS:
            out["bis"] = True
            i += 1
            continue
        q = QUADRANT_CANON.get(tok) or QUADRANT_SHORT.get(tok)
        if q is not None and out["quadrant"] is None:
            out["quadrant"] = q
            i += 1
            continue
        fq = _fuzzy_quadrant(tok)
        if fq is not None and out["quadrant"] is None:
            out["quadrant"] = fq
            i += 1
            continue
        break
    out["leftover"] = tokens[i:]
    return out


def _canon_complement(text: str) -> str | None:
    """Canonicalize complement words but keep unknown tokens verbatim."""
    text = re.sub(r"\s+", " ", text.replace(",", " ")).strip(" .-")
    if not text:
        return None
    out = [COMPLEMENT_CANON.get(tok, tok) for tok in text.split()]
    return " ".join(out) or None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_JUNK_RE = re.compile(r"^[\d\s.,;:#/-]*$")


def parse_address(raw: str | None) -> ParsedAddress:
    """Parse a free-form Cali address into structured fields."""
    res = ParsedAddress(raw="" if raw is None else str(raw))
    if raw is None:
        res.notes.append("null_input")
        return res
    text0 = str(raw)
    if not text0.strip():
        res.notes.append("empty_input")
        return res
    if ";" in text0:
        res.notes.append("multiple_addresses_first_taken")
        text0 = text0.split(";")[0]

    text = _pre_normalize(text0)
    text, noise_notes = _drop_city_noise(text)
    res.notes.extend(noise_notes)
    if not text:
        res.notes.append("empty_after_cleanup")
        return res
    if _JUNK_RE.match(text):
        res.notes.append("no_alphabetic_content")
        return res

    # Drop a trailing neighbourhood segment after a comma when the head already
    # looks like an address (`Cl. 55b # 47-55, Navarro` -> `Cl. 55b # 47-55`).
    if "," in text:
        head, _, tail = text.partition(",")
        head = head.strip()
        if head and ("#" in head or any(t in VIA_TYPE_CANON for t in _split_glued(head))):
            if tail.strip():
                res.notes.append("dropped_trailing_segment")
            text = head
    text = text.replace(",", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # --- locate the '#' -------------------------------------------------
    if "#" in text:
        via_text, _, rest = text.partition("#")
        res.had_hash = True
        res.notes.append("explicit_hash")
    elif " - " in text:
        # No `#` but an explicit dash. Two shapes share this branch:
        # `Catrera 36 5b3-65` (a cross candidate `5B3` already sits between the
        # via and the dash, so the right side of the dash is the plate) and
        # `Calle 5 - 38` (nothing sits between the via and the dash, so there is
        # no cross candidate on the left: the number on the right IS the cross,
        # not a plate - a bare dash without `#` never introduces a plate on its
        # own).
        left, _, right = text.partition(" - ")
        via_text, cross_text_pre = _segment_without_hash(left, res, allow_plate=False)
        rest = f"{cross_text_pre} - {right}" if cross_text_pre else right
    else:
        via_text, rest = _segment_without_hash(text, res)

    # --- locate the plate separator ------------------------------------
    rest = rest.strip()
    if " - " in rest:
        cross_text, _, plate_text = rest.partition(" - ")
        res.had_plate_sep = True
    elif rest.endswith("-"):
        cross_text, plate_text = rest[:-1], ""
        res.had_plate_sep = True
    elif rest.startswith("- "):
        cross_text, plate_text = "", rest[2:]
        res.had_plate_sep = True
    else:
        cross_text, plate_text = _split_cross_plate_no_dash(rest, res)

    via_tokens = _split_glued(via_text)
    if not via_tokens:
        res.notes.append("empty_via_segment")
        return res
    via = _parse_segment(via_tokens)
    if via["type"] is None or via["number"] is None:
        res.notes.append("via_unparsed")
        return res
    res.via_type = via["type"]
    res.via_number = via["number"]
    res.via_letters = via["letters"]
    res.via_suffix = via["suffix"]
    res.via_suffix_letters = via["suffix_letters"]
    res.via_bis = via["bis"]
    res.via_quadrant = via["quadrant"]

    cross = _parse_segment(_split_glued(cross_text))
    res.cross_type = cross["type"]
    res.cross_number = cross["number"]
    res.cross_letters = cross["letters"]
    res.cross_suffix = cross["suffix"]
    res.cross_suffix_letters = cross["suffix_letters"]
    res.cross_bis = cross["bis"]
    res.cross_quadrant = cross["quadrant"]

    plate_text = plate_text.strip()
    plate_rest = ""
    m = re.match(r"^(\d{1,6})(?![\dA-Z])\s*(.*)$", plate_text)
    if m:
        res.plate = m.group(1)
        plate_rest = m.group(2)
    else:
        plate_rest = plate_text

    comp = " ".join(
        part for part in (
            " ".join(via["leftover"]), " ".join(cross["leftover"]), plate_rest,
        ) if part.strip()
    )
    res.complement = _canon_complement(comp)
    res.parse_ok = True
    return res


def _segment_without_hash(text: str, res: ParsedAddress, allow_plate: bool = True) -> tuple[str, str]:
    """Recover the via/cross boundary when the `#` is missing.

    ``CL 9 C 49 141``    -> via ``CL 9 C``,  rest ``49 - 141``
    ``KR 13 60 34 38``   -> via ``KR 13``,   rest ``60 - 34 38``
    """
    res.notes.append("implicit_hash")
    tokens = _split_glued(text)
    i, n = 0, len(tokens)
    via: list[str] = []
    if i + 1 < n and f"{tokens[i]} {tokens[i+1]}" in VIA_TYPE_CANON:
        via += tokens[i:i + 2]
        i += 2
    elif i < n and (tokens[i] in VIA_TYPE_CANON or _fuzzy_via_type(tokens[i])):
        via.append(tokens[i])
        i += 1
    if i < n and _NUM_RE.match(tokens[i]):
        via.append(tokens[i])
        i += 1
    # One optional letter run and one optional secondary number stay with the via.
    if i < n and _LETTER_RE.match(tokens[i]):
        via.append(tokens[i])
        i += 1
    while i < n and (tokens[i] in _BIS_WORDS or tokens[i] in QUADRANT_CANON):
        via.append(tokens[i])
        i += 1
    rest = tokens[i:]
    if not allow_plate:
        return " ".join(via), " ".join(rest)
    if rest and _NUM_RE.match(rest[0]):
        cut = _plate_cut(rest)
        return " ".join(via), " ".join(rest[:cut]) + " - " + " ".join(rest[cut:])
    return " ".join(via), " ".join(rest)


def _plate_cut(tokens: list[str]) -> int:
    """Index at which the plate starts inside a `cross + plate` token run.

    ``['5','B','2','09']``          -> 3   (cross `5 B 2`, plate `09`)
    ``['49','141']``                -> 1   (cross `49`,    plate `141`)
    ``['60','34','38','LC','1']``   -> 1   (cross `60`,    plate `34`, rest complement)
    """
    numbers = [k for k, t in enumerate(tokens) if _NUM_RE.match(t)]
    if len(numbers) < 2:
        return len(tokens)
    cut = numbers[1]
    # `5 B 2 09`: when the cross carries a street letter, a secondary number can
    # follow it and the plate is then the very last token, so shift the cut one
    # number to the right. Bare number runs (`60 34 38`) are left alone.
    cross_has_letter = any(
        _LETTER_RE.match(t) and t not in _BIS_WORDS and t not in QUADRANT_CANON
        for t in tokens[:numbers[1]]
    )
    if (
        cross_has_letter
        and len(numbers) >= 3
        and numbers[2] == numbers[1] + 1
        and numbers[2] == len(tokens) - 1
    ):
        cut = numbers[2]
    return cut


def _split_cross_plate_no_dash(rest: str, res: ParsedAddress) -> tuple[str, str]:
    """Split post-`#` text into cross and plate when no `-` separator is present."""
    tokens = _split_glued(rest)
    if not tokens:
        return "", ""
    if sum(1 for t in tokens if _NUM_RE.match(t)) >= 2:
        res.notes.append("implicit_plate_separator")
        cut = _plate_cut(tokens)
        return " ".join(tokens[:cut]), " ".join(tokens[cut:])
    return " ".join(tokens), ""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_segment(
    vtype: str | None, number: str | None, letters: str | None, suffix: str | None,
    suffix_letters: str | None, bis: bool, quadrant: str | None, style: str,
) -> str:
    if number is None:
        return ""
    parts: list[str] = []
    if vtype:
        parts.append(vtype)
    if style == "glued" and letters:
        parts.append(f"{number}{letters.replace(' ', '')}")
    else:
        parts.append(number)
        if letters:
            parts.append(letters)
    if suffix:
        if style == "glued" and suffix_letters:
            parts.append(f"{suffix}{suffix_letters}")
        else:
            parts.append(suffix)
            if suffix_letters:
                parts.append(suffix_letters)
    if bis:
        parts.append("BIS")
    if quadrant:
        parts.append(quadrant)
    return " ".join(parts)


def canonical(p: ParsedAddress, style: str = "spaced", with_complement: bool = True) -> str:
    """Render a parsed address in the cadastral canonical form.

    ``style='spaced'`` emits ``KR 98 F # 98 - 66`` (the cadastral layer style);
    ``style='glued'`` emits ``KR 98F # 98 - 66`` (the IDESC ``dir_ajusta`` style).
    The plate is zero-padded to two digits, as the cadastre does.
    """
    if not p.parse_ok:
        return ""
    out = _render_segment(
        p.via_type, p.via_number, p.via_letters, p.via_suffix, p.via_suffix_letters,
        p.via_bis, p.via_quadrant, style,
    )
    cross = _render_segment(
        p.cross_type, p.cross_number, p.cross_letters, p.cross_suffix,
        p.cross_suffix_letters, p.cross_bis, p.cross_quadrant, style,
    )
    if p.had_hash or cross or p.plate is not None:
        out += " #"
        if cross:
            out += f" {cross}"
        plate = p.plate
        if plate is not None:
            if plate.isdigit():
                plate = f"{int(plate):02d}"
            out += f" - {plate}"
        elif p.had_plate_sep:
            out += " -"
    if with_complement and p.complement:
        out += f" {p.complement}"
    return re.sub(r"\s+", " ", out).strip()


_COMPLEMENT_KINDS = frozenset(COMPLEMENT_CANON.values())


def _kind_key(token: str) -> str:
    """Comparable key of a complement token: accent-free, upper, dot-free, alias-resolved."""
    tok = strip_accents(token).upper().strip(".")
    return COMPLEMENT_CANON.get(tok, tok)


def _value_key(token: str) -> str:
    """Comparable key of a complement value token: upper, punctuation-free, no leading zeros.

    Punctuation splits sub-tokens (``101-A`` -> ``101`` + ``A``); each digit-only
    sub-token loses its leading zeros. Sub-tokens are then concatenated, so
    ``101-A``, ``101 A`` and ``101A`` all compare equal.
    """
    parts = re.findall(r"[A-Z0-9]+", strip_accents(token).upper())
    return "".join((p.lstrip("0") or "0") if p.isdigit() else p for p in parts)


def _complement_chunks(text: str) -> list[tuple[list[str], str, str]]:
    """Split complement text into ``(original tokens, kind key, value key)`` chunks.

    A chunk starts at every registered kind token; a leading chunk before any
    registered kind (``BO 002003``) is keyed by its first token.
    """
    chunks: list[list[str]] = []
    for tok in text.split():
        if not chunks or _kind_key(tok) in _COMPLEMENT_KINDS:
            chunks.append([tok])
        else:
            chunks[-1].append(tok)
    return [
        (c, _kind_key(c[0]), "".join(_value_key(t) for t in c[1:]))
        for c in chunks
    ]


def filter_complement(cadastral_address: str, input_complement: str | None) -> str:
    """Drop cadastral complement chunks (AP 101, TO 2, ...) the input did not mention.

    The cadastral complement is split into chunks that start at each registered kind
    (``BLQ G CA 7`` -> ``BLQ G`` + ``CA 7``); a leading chunk before any registered
    kind (``BO 002003`` in ``BO 002003 LT 0031``) is keyed by its first token. A chunk
    is kept iff the input has a chunk of the same kind (aliases resolved through
    ``COMPLEMENT_CANON``) with the same non-empty value: values are compared
    as one key after uppercasing, dropping accents/punctuation and leading zeros
    (``0031`` == ``31``). A kind mentioned without a value never matches. The kept
    chunk is spelled as the cadastre has it. Never raises; an unparseable cadastral
    address or one without a complement is returned unchanged.
    """
    if cadastral_address is None:
        return ""
    p = parse_address(cadastral_address)
    if not p.parse_ok or not p.complement:
        return cadastral_address
    base = canonical(p, with_complement=False)
    if cadastral_address.startswith(base):
        comp_text = cadastral_address[len(base):]
    else:
        comp_text = p.complement

    mentioned = {
        (kind, values)
        for _, kind, values in _complement_chunks(input_complement or "")
        if values
    }
    kept = [
        " ".join(tokens)
        for tokens, kind, values in _complement_chunks(comp_text)
        if values and (kind, values) in mentioned
    ]
    return re.sub(r"\s+", " ", " ".join([base, *kept])).strip()


def match_key(text: str) -> str:
    """Whitespace/punctuation-insensitive key used for exact-match comparisons."""
    return re.sub(r"[^A-Z0-9]", "", strip_accents(text).upper())


def canonicalize(raw: str | None, style: str = "spaced", with_complement: bool = True) -> str:
    """Convenience wrapper: free-form text -> canonical string (``''`` on failure)."""
    return canonical(parse_address(raw), style=style, with_complement=with_complement)


def parse_many(values: Iterable[str | None]) -> list[ParsedAddress]:
    return [parse_address(v) for v in values]
