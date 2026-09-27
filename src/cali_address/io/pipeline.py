"""The dataset pipeline: source chunks -> strict normalization -> sink, one bad row never aborts.

Guarantees
----------
* Matching is untouched: every chunk goes through ``cali_address.service.normalize_strict``
  with the caller's tunables, so identical input rows give identical results.
* Input row order and count are preserved for any ``chunk_size``.
* A row that makes the normalizer raise becomes ``estado='ERROR'`` with
  ``motivo='<ExceptionType>: <short message>'`` and every result column empty. The
  failing chunk is bisected until the offending rows are isolated, so the other
  rows are still scored normally (``on_error='raise'`` re-raises instead).
* Optional out-of-area guard: with a ``municipality_col``, a non-empty municipality
  that is not Cali is reported as ``estado='FUERA_DE_AREA'`` without matching.

Output schema
-------------
Source columns (all of them by default), then ``cali_address.service.OUTPUT_COLUMNS``
unchanged. When the address is a single kept source column, ``direccion_entrada``
is omitted (it would duplicate that column), exactly as the legacy CLI does. A
source column whose name collides with an output column is carried as
``<name>_original`` (coordinate-like names are renamed as a pair).
"""

from __future__ import annotations

import math
import re
import time
import unicodedata
from dataclasses import dataclass, field
from os import PathLike
from typing import Callable, Iterable, Iterator

import pandas as pd

from ..gazetteer import ZONE_BUFFER_M
from ..service import (
    CLI_BARRIO_BUFFER_M,
    DEFAULT_MIN_STRUCT,
    OUTPUT_COLUMNS,
    _empty_result,
    is_unspecified,
    normalize_strict,
)
from .errors import SourceReadError, UsageError
from .mapping import ColumnMapping, ResolvedMapping
from .readers import DEFAULT_CHUNK_SIZE, TableChunks, read_table
from .writers import Sink

#: ``estado`` values of this pipeline: the three public ones plus the two it adds.
ESTADOS = ("OK", "SIN_MATCH", "NO_PARSEABLE", "FUERA_DE_AREA", "ERROR")
_COORDINATE_NAMES = {"lat", "lon", "lng", "latitud", "longitud"}
_MAX_ERROR_SAMPLES = 5


@dataclass(frozen=True)
class Tunables:
    """``normalize_strict`` options. Defaults equal the legacy CLI / API defaults."""

    min_struct: float = DEFAULT_MIN_STRUCT
    plate_tolerance: int = 0
    ambiguity_delta: float = 0.02
    barrio_buffer_m: float = CLI_BARRIO_BUFFER_M
    zone_buffer_m: float = ZONE_BUFFER_M
    gate_escalate: bool = True
    soft_rules: tuple | None = None
    max_soft: int | None = None
    gate_fallback: bool | None = None
    k: int = 20
    #: Overrides ``normalizer.threshold`` when set (applied by the CLI, not by ``normalize_strict``).
    threshold: float | None = None

    def as_kwargs(self) -> dict:
        return {
            "min_struct": self.min_struct, "plate_tolerance": self.plate_tolerance,
            "ambiguity_delta": self.ambiguity_delta, "barrio_buffer_m": self.barrio_buffer_m,
            "zone_buffer_m": self.zone_buffer_m, "gate_escalate": self.gate_escalate,
            "soft_rules": self.soft_rules, "max_soft": self.max_soft,
            "gate_fallback": self.gate_fallback, "k": self.k,
        }


# ---------------------------------------------------------------------------
# value coercion
# ---------------------------------------------------------------------------
def _is_missing(value) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float) and math.isnan(value)


def coerce_raw(value):
    """Address cell -> ``str`` (or ``None`` when missing). Strings are returned untouched."""
    if isinstance(value, str):
        return value
    if _is_missing(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


# ---------------------------------------------------------------------------
# municipality guard
# ---------------------------------------------------------------------------
_CALI_FORMS = {"cali", "santiago de cali", "76001"}
_SUFFIX = re.compile(r"(?: (?:valle del cauca|valle|colombia))+$")


def _municipality_key(value) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    text = re.sub(r"^municipio de ", "", text)
    return _SUFFIX.sub("", text)


def is_outside_cali(value) -> bool:
    """True only for a non-empty municipality that is clearly not Cali."""
    if _is_missing(value) or is_unspecified(value):
        return False
    return _municipality_key(value) not in _CALI_FORMS


# ---------------------------------------------------------------------------
# chunk processing
# ---------------------------------------------------------------------------
def _short_message(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    text = text[:200] + ("..." if len(text) > 200 else "")
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _error_row(raw, exc: BaseException) -> dict:
    return _empty_result(raw, "ERROR", _short_message(exc))


class _Runner:
    """Holds the normalizer and the counters shared by every chunk."""

    def __init__(self, normalizer, gazetteer, tunables: Tunables, on_error: str) -> None:
        self.normalizer = normalizer
        self.gazetteer = gazetteer
        self.kwargs = tunables.as_kwargs()
        self.on_error = on_error
        self.gate_stats: dict = {}

    def _call(self, raws: list) -> pd.DataFrame:
        stats: dict = {}
        frame = normalize_strict(self.normalizer, raws, gazetteer=self.gazetteer, stats=stats, **self.kwargs)
        for key, value in stats.items():
            self.gate_stats[key] = self.gate_stats.get(key, 0) + value
        return frame

    def isolate(self, raws: list) -> list[dict]:
        """Records for ``raws``; failing rows become ERROR rows (bisecting the failing batch)."""
        if not raws:
            return []
        try:
            return self._call(raws).to_dict(orient="records")
        except Exception as exc:
            if self.on_error == "raise":
                raise
            if len(raws) == 1:
                return [_error_row(raws[0], exc)]
            mid = len(raws) // 2
            return self.isolate(raws[:mid]) + self.isolate(raws[mid:])

    def normalize(self, raws: list, skip: dict[int, dict]) -> pd.DataFrame:
        """Result frame for one chunk; ``skip`` maps row position -> a prebuilt result record."""
        if not skip:
            try:
                return self._call(raws)
            except Exception:
                if self.on_error == "raise":
                    raise
        positions = [i for i in range(len(raws)) if i not in skip]
        records = iter(self.isolate([raws[i] for i in positions]))
        rows = [skip[i] if i in skip else next(records) for i in range(len(raws))]
        return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _output_layout(resolved: ResolvedMapping) -> tuple[list[str], list[str], dict[str, str], bool]:
    """(source columns, result columns, rename map, direccion_entrada dropped?)."""
    source_cols = list(resolved.passthrough_cols)
    drop_input = resolved.address_col is not None and resolved.address_col in source_cols
    result_cols = [c for c in OUTPUT_COLUMNS if not (drop_input and c == "direccion_entrada")]
    rename = {c: f"{c}_original" for c in source_cols if c in result_cols}
    if _COORDINATE_NAMES & {c.lower() for c in source_cols}:
        rename.update({c: f"{c}_original" for c in source_cols if c.lower() in _COORDINATE_NAMES})
    taken = set(source_cols) - set(rename) | set(result_cols)
    for old, new in list(rename.items()):
        while new in taken:
            new += "_"
        rename[old] = new
        taken.add(new)
    return source_cols, result_cols, rename, drop_input


def _process_chunk(chunk: pd.DataFrame, resolved: ResolvedMapping, runner: _Runner, layout) -> pd.DataFrame:
    source_cols, result_cols, rename, drop_input = layout
    chunk = chunk.reset_index(drop=True)
    raws = [coerce_raw(v) for v in resolved.build_addresses(chunk)]
    skip: dict[int, dict] = {}
    if resolved.municipality_col is not None:
        for i, value in enumerate(chunk[resolved.municipality_col].tolist()):
            if is_outside_cali(value):
                skip[i] = _empty_result(raws[i], "FUERA_DE_AREA", f"municipio distinto de Cali: {str(value).strip()!r}")
    result = runner.normalize(raws, skip)
    if drop_input:
        result = result.drop(columns=["direccion_entrada"])
    kept = chunk[source_cols].rename(columns=rename).reset_index(drop=True)
    return pd.concat([kept, result.reset_index(drop=True)], axis=1)


# ---------------------------------------------------------------------------
# source handling
# ---------------------------------------------------------------------------
def _open_source(source, chunk_size: int, read_options: dict) -> tuple[list[str], Iterator[pd.DataFrame], list[str], dict]:
    """(columns, chunk iterator, warnings, meta) for a path/URL, a TableChunks, a DataFrame or an iterable."""
    if isinstance(source, (str, PathLike)):
        source = read_table(source, chunk_size=chunk_size, **read_options)
    if isinstance(source, TableChunks):
        return source.columns, source, source.warnings, {"source_format": source.format, "header_row": source.header_row}
    if isinstance(source, pd.DataFrame):
        chunks: Iterator[pd.DataFrame] = (
            source.iloc[i:i + chunk_size] for i in range(0, len(source), chunk_size)
        )
        return [str(c) for c in source.columns], chunks, [], {"source_format": "dataframe", "header_row": None}
    iterator = iter(source)
    first = next(iterator, None)
    if first is None:
        return [], iter(()), [], {"source_format": "chunks", "header_row": None}
    import itertools

    return [str(c) for c in first.columns], itertools.chain([first], iterator), [], {
        "source_format": "chunks", "header_row": None,
    }


@dataclass
class _Totals:
    rows: int = 0
    chunks: int = 0
    by_estado: dict = field(default_factory=lambda: {e: 0 for e in ESTADOS})
    by_nivel: dict = field(default_factory=dict)
    error_samples: list = field(default_factory=list)

    def add(self, result: pd.DataFrame, offset: int) -> None:
        self.rows += len(result)
        self.chunks += 1
        for estado, count in result["estado"].value_counts().items():
            self.by_estado[estado] = self.by_estado.get(estado, 0) + int(count)
        for nivel, count in result["nivel_precision"].dropna().value_counts().items():
            self.by_nivel[nivel] = self.by_nivel.get(nivel, 0) + int(count)
        if len(self.error_samples) < _MAX_ERROR_SAMPLES:
            bad = result[result["estado"] == "ERROR"]
            for pos, motivo in zip(bad.index, bad["motivo"]):
                if len(self.error_samples) < _MAX_ERROR_SAMPLES:
                    self.error_samples.append(f"row {offset + int(pos)}: {motivo}")


def normalize_dataset(
    source,
    mapping: ColumnMapping,
    sink: Sink,
    *,
    normalizer,
    gazetteer=None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    on_error: str = "mark",
    progress: Callable[[int, int], None] | None = None,
    tunables: Tunables | None = None,
    limit: int | None = None,
    **read_options,
) -> dict:
    """Normalize every row of ``source`` into ``sink`` and return the run summary.

    ``source``: a path / http(s) URL / SQLAlchemy URL (``read_options`` are forwarded to
    :func:`read_table`: ``fmt``, ``encoding``, ``delimiter``, ``sheet``, ``header_row``,
    ``table``, ``query``), an open :class:`TableChunks`, a ``DataFrame`` or an iterable of
    ``DataFrame`` chunks.

    ``progress(rows_done, chunk_number)`` is called after each chunk is written;
    ``limit`` stops after that many rows (dry runs). Raises the IO layer's
    :class:`DatasetError` subclasses for unusable input, before touching the normalizer.
    """
    if on_error not in ("mark", "raise"):
        raise ValueError("on_error must be 'mark' or 'raise'")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise UsageError(f"chunk_size must be a positive integer (got {chunk_size!r})")
    started = time.perf_counter()
    columns, chunks, warnings, meta = _open_source(source, chunk_size, read_options)
    resolved = mapping.resolve(columns)  # fail fast, before any normalization work
    layout = _output_layout(resolved)
    source_cols, result_cols, rename, _ = layout
    out_columns = [rename.get(c, c) for c in source_cols] + result_cols
    runner = _Runner(normalizer, gazetteer, tunables or Tunables(), on_error)
    totals = _Totals()
    offset = 0
    wrote_any = False
    for chunk in chunks:
        if [str(c) for c in chunk.columns] != columns:
            raise SourceReadError(
                f"the columns changed between chunks (expected {columns}, got {list(chunk.columns)})"
            )
        if limit is not None:
            chunk = chunk.iloc[: max(limit - offset, 0)]
            if not len(chunk):
                break
        result = _process_chunk(chunk, resolved, runner, layout)
        sink.write(result)
        wrote_any = True
        totals.add(result, offset)
        offset += len(result)
        if progress is not None:
            progress(offset, totals.chunks)
        if limit is not None and offset >= limit:
            break
    if not wrote_any:
        sink.write(pd.DataFrame(columns=out_columns))
    if hasattr(chunks, "close"):
        chunks.close()

    seconds = time.perf_counter() - started
    by = totals.by_estado
    return {
        "rows": totals.rows,
        "ok": by["OK"], "sin_match": by["SIN_MATCH"], "no_parseable": by["NO_PARSEABLE"],
        "fuera_de_area": by["FUERA_DE_AREA"], "error": by["ERROR"],
        "by_estado": dict(by),
        "by_nivel_precision": dict(sorted(totals.by_nivel.items())),
        "chunks": totals.chunks,
        "seconds": round(seconds, 3),
        "rows_per_sec": round(totals.rows / seconds, 1) if totals.rows and seconds > 0 else 0,
        "error_samples": list(totals.error_samples),
        "gazetteer": dict(runner.gate_stats),
        "warnings": list(warnings),
        **resolved.describe(),
        **meta,
    }
