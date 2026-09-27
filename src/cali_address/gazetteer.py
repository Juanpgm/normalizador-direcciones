"""Place gazetteer: barrio / vereda / comuna / corregimiento from the IDESC basemaps.

Why this exists
---------------
The cadastral matcher only sees nomenclature (``KR 77 # 1 C - 140``). Two things
are lost that humans write all the time:

1. **Place names** (``Barrio Siloé``, ``vereda la reforma``, ``Comuna 15``). They
   are noise for the address grammar - ``Barrio Siloé calle 1 # 2-3`` does not
   parse at all - but they are strong *spatial* evidence.
2. **Where the answer must be.** Cali repeats its nomenclature across the city,
   so ``CL 5 # 38-25`` has several plausible predios. A detected place turns that
   ambiguity into a hard geographic constraint.

This module therefore does two jobs: it *detects* place mentions and strips them
so the parser sees clean nomenclature (:meth:`Gazetteer.detect`), and it answers
point-in-polygon questions (:meth:`Gazetteer.locate`, :meth:`Gazetteer.contains`)
so the CLI can gate candidates geographically (:func:`gate_candidates`).

Data
----
``basemaps/barrios_veredas.geojson`` (440 features, EPSG:4326 lon/lat). Urban rows
carry ``COMUNA`` / ``ID_BARRIO`` / ``BARRIO``; rural rows have an empty ``COMUNA``
and carry ``ID_VEREDA`` / ``VEREDA`` / ``ID_CORREG`` / ``CORREGIMIE``. Urban rows
*also* carry ``VEREDA`` / ``CORREGIMIE`` values left over from a spatial join -
those are ignored. ``basemaps/comunas_corregimientos.geojson`` (37 features) gives
the 22 comuna polygons and the 15 corregimiento polygons.

Entries are keyed by a folded name (accent-free, upper case, punctuation to
space, ``(Cabecera)`` stripped), so duplicate rows and accent-only duplicates
collapse into a single entry whose geometry is the union of the originals and
which remembers every code and parent.

Detection rules (in order)
--------------------------
a. **Explicit labels** anywhere in the text: ``COMUNA <n>``,
   ``BARRIO|BRR|B/|URB|URBANIZACION|CONJUNTO|CONJ|SECTOR <name>``,
   ``VEREDA|VDA <name>``, ``CORREGIMIENTO|CORREG|CGTO <name>``. The name is the
   best-scoring window of the following 1-4 tokens.
b. **Segments.** The text is split on ``,`` ``;`` ``|`` and newlines; a piece that
   does not parse as an address is further split on `` - `` and `` / `` (a piece
   that *does* parse is never split, so a spaced plate dash cannot cut an address
   in half). The *core* is the first segment that :func:`parse_address` accepts,
   or the first segment when none does. Every other segment is matched against
   the gazetteer - exact first, then ``token_set_ratio >= 90`` when the segment
   has at least 5 letters. Pure city/department noise is dropped first.
c. **Core tail.** Only the tokens *after* the last number of the core segment are
   inspected, and only multi-word names, single words of 7+ letters, or an exact
   single-word match of 5-6 letters that is the whole tail may fire. This is what
   makes ``Calle 3ra A oeste # 44-06 SILOE`` work while keeping ``Sucre`` /
   ``Lili`` / ``Popular`` from firing inside ``Calle 5 # 38-25``.

Precedence decisions (deliberate, exercised by ``tests/test_gazetteer.py``)
--------------------------------------------------------------------------
* A barrio fills the comuna from its parent; a vereda fills its corregimiento.
  An explicit comuna that disagrees with the barrio's comuna is kept (explicit
  wins) and the ``comuna_barrio_conflict`` note is added.
* For an **unlabeled** match that is ambiguous across kinds, the order is
  corregimiento > vereda > barrio. ``Navarro`` is a corregimiento, its cabecera
  vereda and part of the barrio ``Navarro - La Chanca``; picking the widest
  polygon keeps the geographic gate from being tighter than the evidence.
* Labelled matches beat segment matches, which beat core-tail matches.
* A tie between same-named places is broken by the parent the rest of the text
  implies, which is how ``corregimiento los andes, vereda la reforma`` resolves to
  ``El Mango - La Reforma`` and ``corregimiento la buitrera, vereda la reforma``
  to ``Acueducto de la Reforma``.
* Only rules (a) and (c) may remove text from inside the address core; rule (b)
  removes whole segments. When nothing is detected, ``cleaned_text`` is the raw
  text verbatim, so rows without a place keep their previous behaviour.
"""

from __future__ import annotations

import math
import os
import pickle
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, NamedTuple, Sequence

from rapidfuzz import fuzz, process
from shapely import STRtree, prepare, union_all
from shapely.geometry import Point, shape

from .parser import parse_address

__all__ = [
    "Gazetteer",
    "GazEntry",
    "Detection",
    "Candidate",
    "gate_candidates",
    "normalize_key",
    "BARRIO_BUFFER_M",
    "ZONE_BUFFER_M",
]

BARRIOS_FILENAME = "barrios_veredas.geojson"
ZONES_FILENAME = "comunas_corregimientos.geojson"
CACHE_FILENAME = "gazetteer.pkl"
CACHE_VERSION = 1

#: Slack applied to a detected barrio polygon before rejecting a candidate. Barrio
#: borders run down the middle of streets and the matcher scores *predio
#: centroids*, so a metre-exact test would reject legitimate matches on the far
#: kerb. 200 m is roughly two blocks.
BARRIO_BUFFER_M = 200.0

#: Slack for the coarser comuna / corregimiento gate.
ZONE_BUFFER_M = 300.0

#: Degrees -> metres. Cali sits at latitude ~3.4 deg, where a degree of longitude
#: is 110.8 km and one of latitude 110.6 km, so a single constant is within 0.3 % -
#: far below the buffer sizes this module works with.
DEG_TO_M = 111_000.0

#: Minimum rapidfuzz score for a labelled name (rule a) and for a segment (rule b).
LABEL_SCORE_CUTOFF = 88
SEGMENT_SCORE_CUTOFF = 90

#: A fuzzy (non-exact) match needs this many letters to be trustworthy.
MIN_FUZZY_LETTERS = 5

#: Longest name window considered after a label or inside the core tail.
MAX_NAME_TOKENS = 4

#: Comuna numbers that exist in Cali.
MIN_COMUNA, MAX_COMUNA = 1, 22

BARRIO_LABELS = frozenset(
    {"BARRIO", "BARRIOS", "BRR", "BR", "URB", "URBANIZACION", "CONJUNTO", "CONJ",
     "CJTO", "SECTOR", "SECT"}
)
VEREDA_LABELS = frozenset({"VEREDA", "VEREDAS", "VDA", "VRD"})
CORREGIMIENTO_LABELS = frozenset({"CORREGIMIENTO", "CORREGIMIENTOS", "CORREG", "CGTO", "CORR"})

#: City / department tails that must never be read as a place name.
NOISE_KEYS = frozenset(
    {"CALI", "SANTIAGO DE CALI", "VALLE", "VALLE DEL CAUCA", "VALLE CAUCA", "CAUCA",
     "COLOMBIA", "V CAUCA", "VALLE DEL CAUCA COLOMBIA", "MUNICIPIO DE CALI",
     "SANTIAGO DE CALI VALLE DEL CAUCA", "CALI VALLE", "CALI VALLE DEL CAUCA"}
)

#: Kinds a name can resolve to, widest polygon first. Used to break a tie for an
#: unlabeled mention.
UNLABELED_KINDS = ("corregimiento", "vereda", "barrio")

_STRONG_SEP_RE = re.compile(r"[,;|\n\r]")
_WEAK_SEP_RE = re.compile(r"\s+-\s+|\s+/\s+")
_TOKEN_RE = re.compile(r"[A-Z0-9]+")
_CABECERA_RE = re.compile(r"\(?\s*CABECERA\s*\)?")
_DIGITS_RE = re.compile(r"^\d+$")
_SEP_CHARS = " \t,;|-/"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _fold(text: str) -> tuple[str, list[int]]:
    """Accent-fold and upper-case ``text``, keeping a char -> source index map.

    The folded string has one ``index_map`` entry per character, so a span found
    in it maps straight back onto the original string. Every character that is
    neither ``A-Z0-9`` nor a separator becomes a space, which is also what makes
    mojibake (``U+FFFD``) harmless: it degrades into a token boundary instead of
    corrupting a name.
    """
    out: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(text):
        for part in unicodedata.normalize("NFKD", ch):
            if unicodedata.combining(part):
                continue
            up = part.upper()
            if up.isascii() and up.isalnum():
                out.append(up)
            elif up in ",;|\n\r-/":
                out.append(up)
            else:
                out.append(" ")
            index_map.append(i)
    return "".join(out), index_map


def normalize_key(text: Any) -> str:
    """Folded lookup key: accent-free, upper case, ``(Cabecera)`` stripped."""
    if text is None:
        return ""
    folded, _ = _fold(str(text))
    folded = _CABECERA_RE.sub(" ", folded)
    return re.sub(r"[^A-Z0-9]+", " ", folded).strip()


def _letters(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def _finite_point(lon, lat) -> Point | None:
    try:
        x, y = float(lon), float(lat)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return Point(x, y)


def _coerce_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, float) and math.isnan(raw):
        return ""
    return str(raw)


def _to_int(value) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    return int(text) if _DIGITS_RE.match(text) else None


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@dataclass
class GazEntry:
    """One place, with every code/parent of the rows that folded into it."""

    kind: str
    name: str
    key: str
    codes: tuple[str, ...]
    parents: tuple[str, ...]
    geometry: Any

    @property
    def parent(self) -> str | None:
        """The single parent (comuna number / corregimiento name), if unambiguous."""
        return self.parents[0] if len(self.parents) == 1 else None


@dataclass
class Detection:
    """Places found in one raw string plus the text left for the address parser."""

    barrio: str | None = None
    comuna: int | None = None
    corregimiento: str | None = None
    vereda: str | None = None
    cleaned_text: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def detected(self) -> bool:
        return any(
            v is not None for v in (self.barrio, self.comuna, self.corregimiento, self.vereda)
        )

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)


class Candidate(NamedTuple):
    """A reranked cadastral candidate, as far as the geographic gate cares."""

    score: float
    doc: int
    lon: float
    lat: float


class _Row(NamedTuple):
    """One basemap feature, kept verbatim for point-in-polygon lookups."""

    kind: str
    name: str
    code: str
    parent: str | None
    geometry: Any


class _Match(NamedTuple):
    """A place mention found in the text.

    ``alternatives`` holds the equally-scoring entries of the same kind, so the
    resolver can still apply a parent hint that was only discovered later.
    """

    kind: str
    entry: GazEntry
    alternatives: tuple[GazEntry, ...]
    score: float
    tier: int                       # 0 = label, 1 = segment, 2 = core tail
    position: int                   # offset in the folded text, for stable ordering
    span: tuple[int, int] | None    # span to strip from the raw text


# ---------------------------------------------------------------------------
# Gazetteer
# ---------------------------------------------------------------------------


class Gazetteer:
    """Name index + spatial index over Cali's administrative basemaps."""

    def __init__(self, rows: Sequence[_Row], from_cache: bool = False) -> None:
        self.rows: list[_Row] = list(rows)
        self.from_cache = bool(from_cache)
        self.entries: list[GazEntry] = _build_entries(self.rows)
        self._by_kind: dict[str, list[GazEntry]] = {}
        self._index: dict[tuple[str, str], GazEntry] = {}
        for entry in self.entries:
            self._by_kind.setdefault(entry.kind, []).append(entry)
            self._index[(entry.kind, entry.key)] = entry
            prepare(entry.geometry)
        # Comuna names are all `Comuna <n>`; the digit rule handles them, so they
        # stay out of the fuzzy name pool where they would only add noise.
        self._pool: dict[str, list[GazEntry]] = {}
        for kind in UNLABELED_KINDS:
            for entry in self._by_kind.get(kind, []):
                self._pool.setdefault(entry.key, []).append(entry)
        self._keys: list[str] = list(self._pool)
        self._trees: dict[str, tuple[STRtree, list[_Row]]] = {}
        for kind in ("barrio", "vereda", "comuna", "corregimiento"):
            kind_rows = [r for r in self.rows if r.kind == kind]
            if kind_rows:
                self._trees[kind] = (STRtree([r.geometry for r in kind_rows]), kind_rows)

    # -- loading ----------------------------------------------------------
    @classmethod
    def load(
        cls,
        basemaps_dir: str,
        cache_path: str | None = None,
        use_cache: bool = True,
    ) -> "Gazetteer":
        """Parse the basemaps, reusing ``artifacts/gazetteer.pkl`` when it is fresh.

        The two GeoJSON files are 12 MB of text and cost seconds to parse, which is
        unacceptable per CLI run. The cache is keyed by both files' mtimes, so
        replacing a basemap invalidates it automatically; a corrupt or stale cache
        is silently rebuilt rather than trusted.
        """
        paths = {
            name: os.path.join(basemaps_dir, name)
            for name in (BARRIOS_FILENAME, ZONES_FILENAME)
        }
        for path in paths.values():
            if not os.path.exists(path):
                raise FileNotFoundError(f"basemap not found: {path}")
        mtimes = {name: os.path.getmtime(path) for name, path in paths.items()}
        if cache_path is None:
            cache_path = os.path.join(
                os.path.dirname(os.path.abspath(basemaps_dir.rstrip("\\/"))),
                "artifacts", CACHE_FILENAME,
            )
        if use_cache:
            cached = _read_cache(cache_path, mtimes)
            if cached is not None:
                return cls(cached, from_cache=True)
        rows = _read_basemaps(paths[BARRIOS_FILENAME], paths[ZONES_FILENAME])
        if use_cache:
            _write_cache(cache_path, mtimes, rows)
        return cls(rows, from_cache=False)

    # -- name access ------------------------------------------------------
    def by_kind(self, kind: str) -> list[GazEntry]:
        return list(self._by_kind.get(kind, []))

    def lookup(self, kind: str, name: Any) -> GazEntry | None:
        return self._index.get((kind, self._as_key(kind, name)))

    def comuna_entry(self, number: Any) -> GazEntry | None:
        return self.lookup("comuna", number)

    @staticmethod
    def _as_key(kind: str, name: Any) -> str:
        if kind == "comuna":
            text = str(name).strip()
            if _DIGITS_RE.match(text):
                return f"COMUNA {int(text)}"
        return normalize_key(name)

    # -- geometry ---------------------------------------------------------
    def locate(self, lon, lat) -> dict:
        """Which barrio/comuna (or vereda/corregimiento) a coordinate falls in."""
        out = {
            "barrio": None, "id_barrio": None, "comuna": None,
            "vereda": None, "id_vereda": None,
            "corregimiento": None, "id_corregimiento": None,
        }
        point = _finite_point(lon, lat)
        if point is None:
            return out
        barrio = self._hit("barrio", point)
        if barrio is not None:
            out["barrio"] = barrio.name
            out["id_barrio"] = barrio.code
            out["comuna"] = _to_int(barrio.parent)
            return out
        vereda = self._hit("vereda", point)
        if vereda is not None:
            out["vereda"] = vereda.name
            out["id_vereda"] = vereda.code
            out["corregimiento"] = vereda.parent
            parent = self.lookup("corregimiento", vereda.parent)
            if parent is not None and parent.codes:
                out["id_corregimiento"] = parent.codes[0]
            return out
        comuna = self._hit("comuna", point)
        if comuna is not None:
            out["comuna"] = _to_int(comuna.code)
            return out
        correg = self._hit("corregimiento", point)
        if correg is not None:
            out["corregimiento"] = correg.name
            out["id_corregimiento"] = correg.code
        return out

    def _hit(self, kind: str, point: Point) -> _Row | None:
        found = self._trees.get(kind)
        if found is None:
            return None
        tree, kind_rows = found
        for idx in tree.query(point, predicate="intersects"):
            row = kind_rows[int(idx)]
            if row.geometry.covers(point):
                return row
        return None

    def contains(self, kind: str, name: Any, lon, lat, buffer_m: float = 0.0) -> bool:
        """Is ``(lon, lat)`` inside the named place, allowing ``buffer_m`` of slack?

        The slack is measured with the flat-earth approximation documented at
        :data:`DEG_TO_M`; buffering the polygon itself would cost far more for an
        answer that differs by well under a metre.
        """
        entry = self.lookup(kind, name)
        point = _finite_point(lon, lat)
        if entry is None or point is None:
            return False
        if entry.geometry.covers(point):
            return True
        if buffer_m <= 0:
            return False
        return entry.geometry.distance(point) * DEG_TO_M <= buffer_m

    # -- detection --------------------------------------------------------
    def detect(self, raw: Any) -> Detection:
        """Find place mentions in ``raw`` and strip them from ``cleaned_text``."""
        text = _coerce_text(raw)
        det = Detection(cleaned_text=text)
        if not text.strip():
            return det
        folded, index_map = _fold(text)
        segments = _segments(folded, text, index_map)
        if not segments:
            return det
        core = _core_segment(text, index_map, segments)

        comuna_spans: list[tuple[int, int]] = []
        explicit_comuna = self._detect_comuna(folded, segments, det, comuna_spans)
        matches = self._detect_labeled(folded, segments)
        matches += self._detect_segments(folded, segments, core, matches)
        matches += self._detect_core_tail(folded, segments[core])

        chosen = _resolve(matches, explicit_comuna)
        for kind, match in chosen.items():
            setattr(det, kind, match.entry.name)
        spans = comuna_spans + [m.span for m in chosen.values() if m.span is not None]
        self._fill_parents(det, chosen, explicit_comuna)
        det.cleaned_text = _clean_text(text, folded, index_map, spans, segments, core, det)
        return det

    # -- detection helpers ------------------------------------------------
    def _detect_comuna(
        self, folded: str, segments: list[tuple[int, int]], det: Detection,
        spans: list[tuple[int, int]],
    ) -> int | None:
        """Rule (a) for ``COMUNA <n>``; out-of-range numbers are reported, not used."""
        found: int | None = None
        for start, end in segments:
            tokens = _tokens(folded, start, end)
            for i, (tok, span) in enumerate(tokens):
                if tok != "COMUNA" or i + 1 >= len(tokens):
                    continue
                nxt, nxt_span = tokens[i + 1]
                if not _DIGITS_RE.match(nxt):
                    continue
                number = int(nxt) if len(nxt) <= 2 else None
                if number is None or not MIN_COMUNA <= number <= MAX_COMUNA:
                    det.note("comuna_out_of_range")
                    continue
                spans.append((span[0], nxt_span[1]))
                if found is None:
                    found = number
        return found

    def _detect_labeled(self, folded: str, segments: list[tuple[int, int]]) -> list[_Match]:
        """Rule (a): a place label followed by the best window of 1-4 tokens."""
        out: list[_Match] = []
        for start, end in segments:
            tokens = _tokens(folded, start, end)
            for i, (tok, span) in enumerate(tokens):
                kinds = _label_kinds(tok, folded, span)
                if not kinds:
                    continue
                best = self._best_window(tokens, i + 1, kinds, LABEL_SCORE_CUTOFF)
                if best is None:
                    continue
                entries, score, last = best
                out.append(
                    _Match(entries[0].kind, entries[0], entries, score, 0, span[0], (span[0], last))
                )
        return out

    def _detect_segments(
        self, folded: str, segments: list[tuple[int, int]], core: int, already: list[_Match],
    ) -> list[_Match]:
        """Rule (b): whole non-core segments matched against the gazetteer."""
        out: list[_Match] = []
        consumed = [m.span for m in already if m.span is not None]
        for idx, (start, end) in enumerate(segments):
            if idx == core or any(s <= start and end <= e for s, e in consumed):
                continue
            key = normalize_key(folded[start:end])
            if not key or key in NOISE_KEYS:
                continue
            entries = self._pool.get(key)
            if entries:
                ordered = _order(entries, UNLABELED_KINDS)
                out.append(
                    _Match(ordered[0].kind, ordered[0], ordered, 101.0, 1, start, (start, end))
                )
                continue
            if _letters(key) < MIN_FUZZY_LETTERS:
                continue
            best = self._best_fuzzy(key, UNLABELED_KINDS, SEGMENT_SCORE_CUTOFF)
            if best is not None:
                entries, score = best
                out.append(
                    _Match(entries[0].kind, entries[0], entries, score, 1, start, (start, end))
                )
        return out

    def _detect_core_tail(self, folded: str, core_span: tuple[int, int]) -> list[_Match]:
        """Rule (c): only the tokens after the last number of the address core."""
        tokens = _tokens(folded, *core_span)
        last_number = max(
            (i for i, (tok, _) in enumerate(tokens) if _DIGITS_RE.match(tok)), default=-1
        )
        tail = tokens[last_number + 1:]
        out: list[_Match] = []
        for i in range(len(tail)):
            for size in range(min(MAX_NAME_TOKENS, len(tail) - i), 0, -1):
                window = tail[i:i + size]
                key = " ".join(tok for tok, _ in window)
                entries = self._pool.get(key)
                if not entries or key in NOISE_KEYS:
                    continue
                if not _tail_allowed(key, size, i == 0 and size == len(tail)):
                    continue
                ordered = _order(entries, UNLABELED_KINDS)
                span = (window[0][1][0], window[-1][1][1])
                out.append(
                    _Match(ordered[0].kind, ordered[0], ordered, 101.0, 2, span[0], span)
                )
        return out

    def _best_window(
        self, tokens: list[tuple[str, tuple[int, int]]], start: int,
        kinds: Sequence[str], cutoff: float,
    ) -> tuple[tuple[GazEntry, ...], float, int] | None:
        """Best 1-4 token name window starting at ``start``, restricted to ``kinds``."""
        best: tuple[tuple[float, int], tuple[GazEntry, ...], int] | None = None
        for size in range(1, MAX_NAME_TOKENS + 1):
            window = tokens[start:start + size]
            if len(window) < size:
                break
            key = " ".join(tok for tok, _ in window)
            if key in NOISE_KEYS:
                continue
            last = window[-1][1][1]
            exact = [e for e in self._pool.get(key, []) if e.kind in kinds]
            if exact:
                cand = ((101.0, size), _order(exact, kinds), last)
            else:
                if _letters(key) < MIN_FUZZY_LETTERS:
                    continue
                found = self._best_fuzzy(key, kinds, cutoff)
                if found is None:
                    continue
                entries, score = found
                cand = ((score, size), entries, last)
            if best is None or cand[0] > best[0]:
                best = cand
        if best is None:
            return None
        return best[1], best[0][0], best[2]

    def _best_fuzzy(
        self, key: str, kinds: Sequence[str], cutoff: float
    ) -> tuple[tuple[GazEntry, ...], float] | None:
        """Top-scoring gazetteer entries for ``key``, all ties kept.

        ``process.extract`` runs the whole comparison in C; scoring the ~450 keys
        one by one in Python showed up as the dominant cost on a 3.7k-row file.

        ``token_set_ratio`` gives 100 to *any* candidate whose tokens are a subset
        of the mention, so ``Nueva Granada`` ties the barrio ``Granada`` (comuna 2)
        with ``Urbanización Nueva Granada`` (comuna 19). Coverage - the share of
        the mention's tokens the candidate accounts for - breaks that tie towards
        the name that actually explains the whole mention.
        """
        hits = process.extract(
            key, self._keys, scorer=fuzz.token_set_ratio, score_cutoff=cutoff, limit=None
        )
        best: tuple[float, float] | None = None
        entries: list[GazEntry] = []
        for candidate, score, _ in hits:
            matching = [e for e in self._pool[candidate] if e.kind in kinds]
            if not matching or not _plausible(key, candidate):
                continue
            rank = (float(score), _coverage(key, candidate))
            if best is None or rank > best:
                best, entries = rank, list(matching)
            elif rank == best:
                entries.extend(matching)
        if best is None:
            return None
        return _order(entries, kinds), best[0]

    def _fill_parents(
        self, det: Detection, chosen: dict[str, _Match], explicit_comuna: int | None
    ) -> None:
        barrio = chosen.get("barrio")
        vereda = chosen.get("vereda")
        barrio_comuna = _to_int(barrio.entry.parent) if barrio is not None else None
        if explicit_comuna is not None:
            det.comuna = explicit_comuna
            if barrio_comuna is not None and barrio_comuna != explicit_comuna:
                det.note("comuna_barrio_conflict")
        elif barrio_comuna is not None:
            det.comuna = barrio_comuna
            det.note("comuna_from_barrio")
        if det.corregimiento is None and vereda is not None and vereda.entry.parent:
            det.corregimiento = vereda.entry.parent
            det.note("corregimiento_from_vereda")


# ---------------------------------------------------------------------------
# Detection plumbing (module level: no state, straightforward to test)
# ---------------------------------------------------------------------------


def _tokens(folded: str, start: int, end: int) -> list[tuple[str, tuple[int, int]]]:
    return [
        (m.group(0), (start + m.start(), start + m.end()))
        for m in _TOKEN_RE.finditer(folded[start:end])
    ]


def _label_kinds(token: str, folded: str, span: tuple[int, int]) -> tuple[str, ...]:
    """Which kinds a label token introduces. Bare ``B`` only counts as ``B/``.

    ``B`` on its own is a street letter (``KR 26 B``), so treating it as a barrio
    label without the slash would shred ordinary addresses.
    """
    if token == "B":
        return ("barrio",) if folded[span[1]:span[1] + 2].lstrip().startswith("/") else ()
    if token in BARRIO_LABELS:
        return ("barrio",)
    if token in VEREDA_LABELS:
        return ("vereda",)
    if token in CORREGIMIENTO_LABELS:
        return ("corregimiento",)
    return ()


def _segments(folded: str, text: str, index_map: list[int]) -> list[tuple[int, int]]:
    """Split on strong separators; subdivide only the pieces that do not parse."""
    out: list[tuple[int, int]] = []
    for start, end in _split(folded, 0, len(folded), _STRONG_SEP_RE):
        if parse_address(_slice_raw(text, index_map, start, end)).parse_ok:
            out.append((start, end))
        else:
            out.extend(_split(folded, start, end, _WEAK_SEP_RE))
    return [(s, e) for s, e in out if folded[s:e].strip(_SEP_CHARS)]


def _split(folded: str, start: int, end: int, pattern: re.Pattern) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    cursor = start
    for m in pattern.finditer(folded, start, end):
        out.append((cursor, m.start()))
        cursor = m.end()
    out.append((cursor, end))
    return [(s, e) for s, e in out if s < e]


def _core_segment(text: str, index_map: list[int], segments: list[tuple[int, int]]) -> int:
    """Index of the first segment the address grammar accepts, else 0."""
    for i, (start, end) in enumerate(segments):
        if parse_address(_slice_raw(text, index_map, start, end)).parse_ok:
            return i
    return 0


def _slice_raw(text: str, index_map: list[int], start: int, end: int) -> str:
    if start >= end or start >= len(index_map):
        return ""
    first = index_map[start]
    last = index_map[min(end, len(index_map)) - 1]
    return text[first:last + 1]


def _tail_allowed(key: str, size: int, whole_tail: bool) -> bool:
    """Guard that keeps generic single words from firing inside an address core."""
    letters = _letters(key)
    if size > 1:
        return letters >= MIN_FUZZY_LETTERS
    if letters >= 7:
        return True
    return whole_tail and letters >= MIN_FUZZY_LETTERS


def _coverage(key: str, candidate: str) -> float:
    """Share of the mention's tokens that ``candidate`` accounts for."""
    wanted = set(key.split())
    if not wanted:
        return 0.0
    return len(wanted & set(candidate.split())) / len(wanted)


def _plausible(key: str, candidate: str) -> bool:
    """Reject fuzzy hits that only share a generic word with a much longer name.

    ``token_set_ratio`` returns 100 whenever one side's tokens are a subset of the
    other's, so the lone word ``RESIDENCIAL`` in ``Conjunto Residencial Zoila``
    scores a perfect match against the barrio ``Unidad Residencial El Coliseo``.
    A hit is only plausible when the mention explains at least half of the official
    name's tokens, or when the two have the same shape and therefore differ only in
    spelling (``BELALCAZER`` / ``BELALCAZAR``, ``SANTA ELEN`` / ``SANTA ELENA``).
    """
    mention = set(key.split())
    official = set(candidate.split())
    if not mention or not official:
        return False
    if len(mention) == len(official):
        return True
    return len(mention & official) / len(official) >= 0.5


def _order(entries: Iterable[GazEntry], kinds: Sequence[str]) -> tuple[GazEntry, ...]:
    """Entries ordered by kind width, then shortest name, restricted to ``kinds``."""
    rank = {kind: i for i, kind in enumerate(kinds)}
    keep = [e for e in entries if e.kind in rank]
    return tuple(sorted(keep, key=lambda e: (rank[e.kind], len(e.key), e.name)))


def _resolve(matches: list[_Match], explicit_comuna: int | None) -> dict[str, _Match]:
    """One winner per kind, then re-pick ambiguous names using the parent hints."""
    chosen: dict[str, _Match] = {}
    for match in sorted(matches, key=lambda m: (m.tier, -m.score, m.position)):
        chosen.setdefault(match.kind, match)
    # An unlabeled mention may only claim one kind: if a wider kind won the same
    # span, the narrower reading of that span is dropped.
    for kind in ("vereda", "barrio"):
        match = chosen.get(kind)
        if match is None or match.tier == 0:
            continue
        if any(
            other.span == match.span and other.tier == match.tier
            for k, other in chosen.items()
            if k != kind and k in UNLABELED_KINDS
            and UNLABELED_KINDS.index(k) < UNLABELED_KINDS.index(kind)
        ):
            del chosen[kind]
    correg = chosen.get("corregimiento")
    if correg is not None:
        chosen["corregimiento"] = _prefer(correg, lambda e: True)
    vereda = chosen.get("vereda")
    if vereda is not None and correg is not None:
        chosen["vereda"] = _prefer(vereda, lambda e: correg.entry.name in e.parents)
    barrio = chosen.get("barrio")
    if barrio is not None and explicit_comuna is not None:
        chosen["barrio"] = _prefer(
            barrio, lambda e: any(_to_int(p) == explicit_comuna for p in e.parents)
        )
    return chosen


def _prefer(match: _Match, wanted) -> _Match:
    """Swap in the first equally-scoring alternative satisfying ``wanted``."""
    for entry in match.alternatives:
        if entry.kind == match.kind and wanted(entry):
            return match._replace(entry=entry)
    return match


def _clean_text(
    text: str, folded: str, index_map: list[int], spans: list[tuple[int, int]],
    segments: list[tuple[int, int]], core: int, det: Detection,
) -> str:
    """Raw text minus the detected mentions, falling back to the address core.

    When nothing was detected and the raw text already parses, the text is
    returned verbatim so rows without a place behave exactly as before.
    """
    if not spans and parse_address(text).parse_ok:
        return text
    core_only = _strip_spans(text, index_map, spans, [segments[core]])
    if not spans:
        # Nothing detected and the raw text does not parse: isolating the address
        # core is strictly an improvement, because such a row is NO_PARSEABLE today.
        if len(segments) > 1 and core_only and parse_address(core_only).parse_ok:
            det.note("core_segment_isolated")
            return core_only
        return text
    keep = [seg for i, seg in enumerate(segments) if i == core or not _is_noise(folded, seg)]
    cleaned = _strip_spans(text, index_map, spans, keep)
    if parse_address(cleaned).parse_ok or not core_only:
        return cleaned
    if parse_address(core_only).parse_ok:
        det.note("core_segment_isolated")
        return core_only
    return cleaned


def _is_noise(folded: str, segment: tuple[int, int]) -> bool:
    return normalize_key(folded[segment[0]:segment[1]]) in NOISE_KEYS


def _strip_spans(
    text: str, index_map: list[int], spans: Iterable[tuple[int, int]],
    segments: Sequence[tuple[int, int]],
) -> str:
    """Rebuild the raw text from ``segments`` with ``spans`` blanked out."""
    blanked: set[int] = set()
    for start, end in spans:
        for pos in range(max(start, 0), min(end, len(index_map))):
            blanked.add(index_map[pos])
    pieces: list[str] = []
    for start, end in segments:
        chars = [
            text[index_map[pos]]
            for pos in range(start, min(end, len(index_map)))
            if index_map[pos] not in blanked
        ]
        piece = re.sub(r"\s+", " ", "".join(chars)).strip(_SEP_CHARS)
        if piece:
            pieces.append(piece)
    return ", ".join(pieces)


# ---------------------------------------------------------------------------
# Basemap reading / caching
# ---------------------------------------------------------------------------


def _read_basemaps(barrios_path: str, zones_path: str) -> list[_Row]:
    import json

    rows: list[_Row] = []
    with open(barrios_path, encoding="utf-8") as fh:
        barrios = json.load(fh)
    for feature in barrios.get("features", []):
        props = feature.get("properties") or {}
        geom = feature.get("geometry")
        if geom is None:
            continue
        comuna = str(props.get("COMUNA") or "").strip()
        if comuna:
            name = str(props.get("BARRIO") or "").strip()
            if name:
                code = str(props.get("ID_BARRIO") or "").strip()
                rows.append(_Row("barrio", name, code, comuna, shape(geom)))
        else:
            # Rural row: only here are VEREDA / CORREGIMIE trustworthy, because
            # urban rows carry leftovers from a spatial join.
            name = str(props.get("VEREDA") or "").strip()
            if name:
                code = props.get("ID_VEREDA")
                parent = str(props.get("CORREGIMIE") or "").strip() or None
                rows.append(
                    _Row("vereda", name, "" if code is None else str(code).strip(),
                         parent, shape(geom))
                )
    with open(zones_path, encoding="utf-8") as fh:
        zones = json.load(fh)
    for feature in zones.get("features", []):
        props = feature.get("properties") or {}
        geom = feature.get("geometry")
        if geom is None:
            continue
        comuna = props.get("COMUNA")
        if comuna is not None and str(comuna).strip() != "":
            number = int(comuna)
            rows.append(_Row("comuna", f"Comuna {number}", str(number), None, shape(geom)))
        else:
            name = str(props.get("CORREGIMIE") or "").strip()
            if name:
                code = props.get("ID_CORREG")
                rows.append(
                    _Row("corregimiento", name, "" if code is None else str(code).strip(),
                         None, shape(geom))
                )
    return rows


def _build_entries(rows: Sequence[_Row]) -> list[GazEntry]:
    """Fold rows into one entry per (kind, key), unioning duplicate geometries."""
    grouped: dict[tuple[str, str], list[_Row]] = {}
    for row in rows:
        key = f"COMUNA {row.code}" if row.kind == "comuna" else normalize_key(row.name)
        if key:
            grouped.setdefault((row.kind, key), []).append(row)
    out: list[GazEntry] = []
    for (kind, key), members in grouped.items():
        geometry = (
            members[0].geometry if len(members) == 1
            else union_all([m.geometry for m in members])
        )
        out.append(
            GazEntry(
                kind=kind,
                name=members[0].name,
                key=key,
                codes=tuple(dict.fromkeys(m.code for m in members if m.code)),
                parents=tuple(dict.fromkeys(m.parent for m in members if m.parent)),
                geometry=geometry,
            )
        )
    return out


def _read_cache(cache_path: str, mtimes: dict[str, float]) -> list[_Row] | None:
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as fh:
            blob = pickle.load(fh)
    except Exception:
        return None  # a damaged cache must never be fatal
    if not isinstance(blob, dict) or blob.get("version") != CACHE_VERSION:
        return None
    if blob.get("mtimes") != mtimes or not blob.get("rows"):
        return None
    return [_Row(*row) for row in blob["rows"]]


def _write_cache(cache_path: str, mtimes: dict[str, float], rows: Sequence[_Row]) -> None:
    try:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        tmp = f"{cache_path}.tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(
                {"version": CACHE_VERSION, "mtimes": mtimes, "rows": [tuple(r) for r in rows]},
                fh, protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(tmp, cache_path)
    except OSError:
        pass  # a read-only artifacts dir must not break normalization


# ---------------------------------------------------------------------------
# Geographic gate
# ---------------------------------------------------------------------------


def gate_candidates(
    gaz: Gazetteer,
    detection: Detection,
    candidates: Sequence[Candidate],
    barrio_buffer_m: float = BARRIO_BUFFER_M,
    zone_buffer_m: float = ZONE_BUFFER_M,
    escalate: bool = False,
) -> tuple[list[Candidate], str | None]:
    """Drop candidates whose centroid cannot be in the detected place.

    Returns ``(kept, label)``. ``label`` is ``None`` when no place was detected -
    the gate is then a no-op and the CLI behaves exactly as before. Candidates
    without a usable centroid are kept: absence of evidence is not evidence of
    absence. A vereda is gated through its corregimiento, whose polygon is the
    superset, so a wrong vereda inside the right corregimiento cannot over-reject.

    ``escalate`` widens the gate instead of rejecting: if the barrio leaves nothing,
    the comuna (then the corregimiento) is tried before giving up. It is off by
    default because the strict reading is the safer one, but it is the better
    trade-off on address text written by reverse geocoders, whose colloquial barrio
    extents do not line up with the official polygons - see the CLI's
    ``--gate-escalate`` flag.
    """
    tiers: list[tuple[str, Any, float]] = []
    if detection.barrio and gaz.lookup("barrio", detection.barrio) is not None:
        tiers.append(("barrio", detection.barrio, barrio_buffer_m))
    if detection.comuna is not None and gaz.comuna_entry(detection.comuna) is not None:
        tiers.append(("comuna", detection.comuna, zone_buffer_m))
    if detection.corregimiento and gaz.lookup("corregimiento", detection.corregimiento):
        tiers.append(("corregimiento", detection.corregimiento, zone_buffer_m))
    if not tiers:
        return list(candidates), None
    if not escalate:
        tiers = tiers[:1]
    label = None
    for kind, name, buffer_m in tiers:
        label = f"{kind} {name}"
        kept = [
            c for c in candidates
            if _finite_point(c.lon, c.lat) is None
            or gaz.contains(kind, name, c.lon, c.lat, buffer_m)
        ]
        if kept:
            return kept, label
    return [], label
