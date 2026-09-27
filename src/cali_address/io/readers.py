"""Reader registry: one entry point, many formats, always chunked.

``read_table(path_or_url, fmt=None, **opts)`` returns a :class:`TableChunks`, an
iterator of ``pandas.DataFrame`` chunks (at most ``chunk_size`` rows each, in file
order) that also carries the schema (``.columns``), warnings and detection
results (``.header_row``, ``.delimiter``, ``.encoding``, ``.sheet_names``).
The first chunk is read eagerly, so unreadable inputs fail at call time, and a
header-only input still reports its columns while yielding no chunks.

Streaming
---------
CSV/TSV/TXT, XLSX (openpyxl read-only mode), Parquet, JSONL, shapefile (``.shp``)
and SQL stream, so inputs larger than memory work. ``.xls``, ``.json``,
GeoJSON, zipped shapefiles and GPKG are read whole and then sliced; they are
inherently in-memory formats.

Values
------
Text formats are read as text (``dtype=str``, no NA coercion): an id such as
``007`` or an address such as ``NA`` is never rewritten. Typed formats (Excel,
Parquet, SQL, JSON) keep their native Python/NumPy values.

Optional dependencies (``sqlalchemy``, ``pyogrio``, ``xlrd``) are imported lazily
and raise :class:`MissingDependencyError` with an install hint.
"""

from __future__ import annotations

import codecs
import csv
import io as _stdio
import itertools
import json
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import pandas as pd

from .errors import (
    MissingDependencyError,
    SourceReadError,
    UnsupportedFormatError,
    UsageError,
)

DEFAULT_CHUNK_SIZE = 20_000

#: Rows inspected to auto-detect a header row (same window as ``cali_address.tables``).
_HEADER_SCAN_ROWS = 15
_SNIFF_BYTES = 256 * 1024
csv.field_size_limit(2**30)  # cells above the 128 KB default must not abort a read
_DELIMITERS = (",", ";", "\t", "|")
_URL_SCHEMES = ("http", "https")


# ---------------------------------------------------------------------------
# result type
# ---------------------------------------------------------------------------
class TableChunks:
    """Iterator of DataFrame chunks plus what was learned while opening the source."""

    def __init__(
        self,
        fmt: str,
        columns: list[str],
        chunks: Iterator[pd.DataFrame],
        *,
        warnings: list[str] | None = None,
        header_row: int | None = None,
        sheet_names: list[str] | None = None,
        delimiter: str | None = None,
        encoding: str | None = None,
        closers: Iterable[Callable[[], None]] = (),
    ) -> None:
        self.format = fmt
        self.columns = list(columns)
        self.warnings = list(warnings or [])
        self.header_row = header_row
        self.sheet_names = sheet_names
        self.delimiter = delimiter
        self.encoding = encoding
        self._chunks = chunks
        self._closers = list(closers)

    def __iter__(self) -> "TableChunks":
        return self

    def __next__(self) -> pd.DataFrame:
        try:
            return next(self._chunks)
        except StopIteration:
            self.close()
            raise
        except SourceReadError:
            self.close()
            raise

    def close(self) -> None:
        closers, self._closers = self._closers, []
        for fn in closers:
            try:
                fn()
            except Exception:  # cleanup must never mask the real result
                pass

    def __enter__(self) -> "TableChunks":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass


@dataclass(frozen=True)
class ReadOptions:
    chunk_size: int = DEFAULT_CHUNK_SIZE
    encoding: str | None = None
    delimiter: str | None = None
    sheet: str | None = None
    header_row: int | None = None
    table: str | None = None
    query: str | None = None


@dataclass
class _Opened:
    columns: list[str]
    chunks: Iterator[pd.DataFrame]
    warnings: list[str]
    header_row: int | None = None
    sheet_names: list[str] | None = None
    delimiter: str | None = None
    encoding: str | None = None
    closers: tuple = ()


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
_READERS: dict[str, Callable[[str, ReadOptions, str], _Opened]] = {}
_EXTENSIONS: dict[str, str] = {}
_ALIASES = {
    "excel": "xlsx", "xlsm": "xlsx", "shapefile": "shp", "ndjson": "jsonl", "geopackage": "gpkg",
    "database": "sql", "db": "sql", "text": "txt",
}
_NOTES: dict[str, str] = {}


def register_reader(fmt: str, func: Callable, extensions: Iterable[str] = (), note: str = "") -> None:
    """Add (or replace) a reader. ``func(path_or_url, ReadOptions, fmt) -> _Opened``."""
    _READERS[fmt] = func
    for ext in extensions:
        _EXTENSIONS[ext.lower()] = fmt
    _NOTES[fmt] = note


def list_reader_formats() -> list[str]:
    return sorted(_READERS)


def describe_reader_formats() -> list[dict]:
    """``[{"format", "extensions", "note"}]`` for the ``formats`` command."""
    return [
        {"format": fmt, "extensions": sorted(e for e, f in _EXTENSIONS.items() if f == fmt),
         "note": _NOTES.get(fmt, "")}
        for fmt in sorted(_READERS)
    ]


def _canonical_format(fmt: str) -> str:
    key = str(fmt).strip().lower().lstrip(".")
    key = _ALIASES.get(key, key)
    if key not in _READERS:
        raise UnsupportedFormatError(
            f"unsupported input format {fmt!r}. Supported formats: {', '.join(sorted(_READERS))}"
        )
    return key


def _is_url(source: str) -> bool:
    return urllib.parse.urlparse(source).scheme.lower() in _URL_SCHEMES and "://" in source


def _is_db_url(source: str) -> bool:
    return "://" in source and not _is_url(source)


def infer_format(source: str) -> str:
    """Format from a URL scheme or a file extension; raises when neither says."""
    source = os.fspath(source)
    if _is_db_url(source):
        return "sql"
    path = urllib.parse.urlparse(source).path if _is_url(source) else source
    ext = os.path.splitext(path)[1].lower()
    if ext in _EXTENSIONS:
        return _EXTENSIONS[ext]
    supported = ", ".join(sorted(_EXTENSIONS))
    raise UnsupportedFormatError(
        f"cannot infer the format of {source!r} (extension {ext or '(none)'}). "
        f"Supported extensions: {supported}. Use --format to force one, or a SQLAlchemy URL for a database."
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def read_table(
    source,
    fmt: str | None = None,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    encoding: str | None = None,
    delimiter: str | None = None,
    sheet: str | None = None,
    header_row: int | None = None,
    table: str | None = None,
    query: str | None = None,
) -> TableChunks:
    """Open ``source`` (path, http(s) URL or SQLAlchemy URL) as a chunk iterator."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise UsageError(f"chunk_size must be a positive integer (got {chunk_size!r})")
    if header_row is not None and (isinstance(header_row, bool) or not isinstance(header_row, int) or header_row < 0):
        raise UsageError(f"header_row must be a non-negative integer (got {header_row!r})")
    source = os.fspath(source)
    resolved = _canonical_format(fmt) if fmt else infer_format(source)
    opts = ReadOptions(chunk_size, encoding, delimiter, sheet, header_row, table, query)

    cleanup: list[Callable[[], None]] = []
    path = source
    try:
        if resolved != "sql":
            if _is_url(source):
                path = _download(source, cleanup)
            else:
                _check_local_file(path)
        opened = _READERS[resolved](path, opts, resolved)
    except BaseException:
        for fn in cleanup:
            try:
                fn()
            except Exception:
                pass
        raise
    return TableChunks(
        resolved, opened.columns, opened.chunks, warnings=opened.warnings,
        header_row=opened.header_row, sheet_names=opened.sheet_names,
        delimiter=opened.delimiter, encoding=opened.encoding,
        closers=[*opened.closers, *cleanup],
    )


def _check_local_file(path: str) -> None:
    if not os.path.exists(path):
        raise SourceReadError(f"input file not found: {path}")
    if os.path.isdir(path):
        raise SourceReadError(f"input path is a directory, not a file: {path}")
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise SourceReadError(f"cannot read {path}: {exc}") from exc
    if size == 0:
        raise SourceReadError(f"input file is empty (0 bytes): {path}")


def _download(url: str, cleanup: list) -> str:
    suffix = os.path.splitext(urllib.parse.urlparse(url).path)[1]
    fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="cali_address_")
    cleanup.append(lambda: os.path.exists(tmp) and os.remove(tmp))
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
            shutil.copyfileobj(response, out, length=1024 * 1024)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise SourceReadError(f"cannot download {url}: {exc}") from exc
    _check_local_file(tmp)
    return tmp


def _batched(rows: Iterable, size: int) -> Iterator[list]:
    it = iter(rows)
    while True:
        batch = list(itertools.islice(it, size))
        if not batch:
            return
        yield batch


def _dedupe_columns(names: list) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for i, raw in enumerate(names):
        name = "" if raw is None else str(raw).strip()
        if not name:
            name = f"Unnamed: {i}"
        if name in seen:
            seen[name] += 1
            candidate = f"{name}.{seen[name]}"
            while candidate in seen:
                seen[name] += 1
                candidate = f"{name}.{seen[name]}"
            name = candidate
        seen.setdefault(name, 0)
        out.append(name)
    return out


def _is_blank_row(row) -> bool:
    return all(v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, float) and v != v)
               for v in row)


def _has_real_header(columns: list[str]) -> bool:
    return any(not c.startswith("Unnamed:") and c.strip() for c in columns)


# ---------------------------------------------------------------------------
# header detection (reuses the tables heuristic)
# ---------------------------------------------------------------------------
def _guess_header(rows: list[list], warnings: list[str], *, check_width: bool) -> int:
    from ..tables import (  # lazy: keeps the import graph light
        _first_stable_width_row, _pad, _width_stable_from, guess_header_row,
    )

    if not rows:
        return 0
    header = guess_header_row(_pad(rows), _HEADER_SCAN_ROWS)
    if check_width and header > 0 and not _width_stable_from(rows, header):
        stable = _first_stable_width_row(rows)
        header = stable if stable is not None else 0
    if header != 0:
        warnings.append(f"header row auto-detected: {header} (use --header-row to override)")
    return header


# ---------------------------------------------------------------------------
# text: encoding + delimiter sniffing
# ---------------------------------------------------------------------------
def detect_encoding(path: str, warnings: list[str] | None = None) -> str:
    """BOM first, then a full streaming UTF-8 validity scan, then cp1252, then latin-1."""
    with open(path, "rb") as fh:
        head = fh.read(4)
    if head.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"

    def valid(enc: str) -> bool:
        decoder = codecs.getincrementaldecoder(enc)()
        try:
            with open(path, "rb") as fh:
                for block in iter(lambda: fh.read(1024 * 1024), b""):
                    decoder.decode(block, final=False)
            decoder.decode(b"", final=True)
            return True
        except UnicodeDecodeError:
            return False

    if valid("utf-8"):
        return "utf-8"
    chosen = "cp1252" if valid("cp1252") else "latin-1"
    if warnings is not None:
        warnings.append(
            f"encoding: file is not valid UTF-8; decoded as {chosen} (use --encoding to override)"
        )
    return chosen


def _sample_text(path: str, encoding: str) -> tuple[str, bool]:
    with open(path, "rb") as fh:
        raw = fh.read(_SNIFF_BYTES + 1)
    truncated = len(raw) > _SNIFF_BYTES
    raw = raw[:_SNIFF_BYTES]
    return raw.decode(encoding, errors="replace"), truncated


def _sniff_delimiter(text: str, truncated: bool) -> str | None:
    """Best of ``, ; TAB |`` by width consistency over the first records; ``None`` if none splits."""
    best, best_score = None, (False, 0.0, 0)
    for cand in _DELIMITERS:
        widths: list[int] = []
        try:
            for row in csv.reader(_stdio.StringIO(text), delimiter=cand):
                if any(cell.strip() for cell in row):
                    widths.append(len(row))
                if len(widths) >= 40:
                    break
        except csv.Error:
            continue
        if truncated and len(widths) > 1:
            widths.pop()  # the last record may have been cut mid-field
        if not widths:
            continue
        modal = max(set(widths), key=lambda w: (widths.count(w), w))
        score = (modal > 1, widths.count(modal) / len(widths), modal)
        if score > best_score:
            best, best_score = cand, score
    return best if best_score[0] else None


def _read_csv_like(path: str, o: ReadOptions, fmt: str) -> _Opened:
    from ..tables import _csv_rows  # lazy

    warnings: list[str] = []
    encoding = o.encoding or detect_encoding(path, warnings)
    try:
        codecs.lookup(encoding)
    except LookupError as exc:
        raise UsageError(f"unknown encoding {encoding!r}") from exc
    text, truncated = _sample_text(path, encoding)

    delimiter = o.delimiter
    if delimiter is None:
        delimiter = "\t" if fmt == "tsv" else _sniff_delimiter(text, truncated)
    if delimiter is None:
        if fmt == "txt":
            return _read_lines(path, o, warnings, encoding)
        delimiter = ","
    if len(delimiter) != 1:
        raise UsageError(f"the delimiter must be a single character (got {delimiter!r})")

    header = o.header_row
    if header is None:
        rows = _csv_rows(text.lstrip("﻿"), delimiter, _HEADER_SCAN_ROWS)
        header = _guess_header(rows, warnings, check_width=True)

    # The stdlib csv module (not pandas' chunked C parser): pandas silently DROPS the extra fields of
    # an over-long row when chunking, which would lose data without a word. Here every row is checked.
    try:
        handle = open(path, encoding=encoding, newline="")
    except OSError as exc:
        raise SourceReadError(f"cannot read {path}: {exc}") from exc
    reader = csv.reader(handle, delimiter=delimiter)
    try:
        for _ in range(header):
            if next(reader, None) is None:
                raise SourceReadError(f"header row {header} is beyond the end of {path}")
        names_row = next(reader, None)
    except UnicodeDecodeError as exc:
        handle.close()
        raise SourceReadError(f"cannot decode {path} as {encoding}: {exc}. Try --encoding.") from exc
    except (csv.Error, SourceReadError) as exc:
        handle.close()
        if isinstance(exc, SourceReadError):
            raise
        raise SourceReadError(f"malformed table in {path}: {exc}") from exc
    if names_row and names_row[0].startswith("﻿"):
        names_row[0] = names_row[0].lstrip("﻿")
    columns = _dedupe_columns(names_row or [])
    if not names_row or not _has_real_header(columns):
        handle.close()
        raise SourceReadError(f"no header row found in {path}: row {header} is blank (is the file empty?)")
    width = len(columns)

    def rows() -> Iterator[list]:
        try:
            for row in reader:
                if not row:  # a blank line
                    if width > 1:
                        continue
                    row = [""]
                if len(row) > width:
                    extra = row[width:]
                    if any(cell.strip() for cell in extra):
                        raise SourceReadError(
                            f"malformed table in {path}, line {reader.line_num}: expected {width} fields "
                            f"but found {len(row)}. Check the delimiter ({delimiter!r}) and the quoting."
                        )
                    row = row[:width]  # trailing separators only
                elif len(row) < width:
                    row = row + [None] * (width - len(row))
                yield row
        except UnicodeDecodeError as exc:
            raise SourceReadError(f"cannot decode {path} as {encoding}: {exc}. Try --encoding.") from exc
        except csv.Error as exc:
            raise SourceReadError(f"malformed table in {path}, line {reader.line_num}: {exc}") from exc

    def gen() -> Iterator[pd.DataFrame]:
        for batch in _batched(rows(), o.chunk_size):
            yield pd.DataFrame(batch, columns=columns, dtype=object)

    return _prime(_Opened(columns, gen(), warnings, header, None, delimiter, encoding, (handle.close,)))


def _read_lines(path: str, o: ReadOptions, warnings: list[str], encoding: str) -> _Opened:
    """A ``.txt`` without any delimiter: one address per line, no header."""

    def lines() -> Iterator[list]:
        try:
            with open(path, encoding=encoding, newline=None) as fh:
                for line in fh:
                    yield [line.rstrip("\r\n")]
        except UnicodeDecodeError as exc:
            raise SourceReadError(f"cannot decode {path} as {encoding}: {exc}. Try --encoding.") from exc

    def gen() -> Iterator[pd.DataFrame]:
        for batch in _batched(lines(), o.chunk_size):
            yield pd.DataFrame(batch, columns=["direccion"], dtype=object)

    warnings.append("no delimiter found: reading one address per line into column 'direccion'")
    return _prime(_Opened(["direccion"], gen(), warnings, None, None, None, encoding))


def _read_lines_format(path: str, o: ReadOptions, fmt: str) -> _Opened:
    warnings: list[str] = []
    encoding = o.encoding or detect_encoding(path, warnings)
    return _read_lines(path, o, warnings, encoding)


def _prime(opened: _Opened) -> _Opened:
    """Pull the first chunk now so errors surface at call time; put it back in front."""
    try:
        first = next(opened.chunks)
    except StopIteration:
        opened.chunks = iter(())
        return opened
    opened.chunks = itertools.chain([first], opened.chunks)
    return opened


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------
def _rows_to_opened(row_iter: Iterator, o: ReadOptions, warnings: list[str],
                    sheet_names: list[str] | None, closers: tuple, what: str) -> _Opened:
    head = list(itertools.islice(row_iter, _HEADER_SCAN_ROWS))
    if not head:
        raise SourceReadError(f"{what} is empty (no rows)")
    header = o.header_row
    if header is None:
        header = _guess_header([list(r) for r in head], warnings, check_width=False)
    stream = itertools.chain(head, row_iter)
    for _ in range(header):
        if next(stream, None) is None:
            raise SourceReadError(f"header row {header} is beyond the end of {what}")
    names_row = next(stream, None)
    if names_row is None or _is_blank_row(names_row):
        raise SourceReadError(f"no header found in {what}: row {header} is blank")
    columns = _dedupe_columns(list(names_row))
    width = len(columns)

    def gen() -> Iterator[pd.DataFrame]:
        good = (
            (list(r[:width]) + [None] * (width - len(r))) for r in stream if not _is_blank_row(r)
        )
        for batch in _batched(good, o.chunk_size):
            yield pd.DataFrame(batch, columns=columns, dtype=object)

    return _prime(_Opened(columns, gen(), warnings, header, sheet_names, None, None, closers))


def _read_xlsx(path: str, o: ReadOptions, fmt: str) -> _Opened:
    import openpyxl

    try:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:
        raise SourceReadError(f"cannot open Excel file {path}: {exc}") from exc
    try:
        names = [str(n) for n in workbook.sheetnames]
        warnings: list[str] = []
        if o.sheet in (None, ""):
            chosen = names[0]
            if len(names) > 1:
                warnings.append(
                    f"workbook has {len(names)} sheets ({', '.join(names)}); reading {chosen!r}. "
                    "Use --sheet to choose another."
                )
        elif o.sheet in names:
            chosen = o.sheet
        elif str(o.sheet).isdigit() and int(o.sheet) < len(names):
            chosen = names[int(o.sheet)]
        else:
            raise SourceReadError(f"sheet {o.sheet!r} not found. Available sheets: {', '.join(names)}")
        rows = workbook[chosen].iter_rows(values_only=True)
        return _rows_to_opened(rows, o, warnings, names, (workbook.close,), f"sheet {chosen!r}")
    except BaseException:
        workbook.close()
        raise


def _read_xls(path: str, o: ReadOptions, fmt: str) -> _Opened:
    warnings: list[str] = []
    try:
        book = pd.ExcelFile(path)
    except ImportError as exc:
        raise MissingDependencyError(
            "reading legacy .xls files requires xlrd: pip install xlrd (or save the file as .xlsx)"
        ) from exc
    except Exception as exc:
        raise SourceReadError(f"cannot open Excel file {path}: {exc}") from exc
    names = [str(n) for n in book.sheet_names]
    if o.sheet in (None, ""):
        chosen = names[0]
        if len(names) > 1:
            warnings.append(f"workbook has {len(names)} sheets; reading {chosen!r}. Use --sheet.")
    elif o.sheet in names:
        chosen = o.sheet
    else:
        raise SourceReadError(f"sheet {o.sheet!r} not found. Available sheets: {', '.join(names)}")
    frame = book.parse(sheet_name=chosen, header=None, dtype=object)
    frame = frame.astype(object).where(frame.notna(), None)
    return _rows_to_opened(frame.itertuples(index=False, name=None), o, warnings, names, (book.close,),
                           f"sheet {chosen!r}")


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------
def _read_parquet(path: str, o: ReadOptions, fmt: str) -> _Opened:
    import pyarrow.parquet as pq

    try:
        pf = pq.ParquetFile(path)
    except Exception as exc:
        raise SourceReadError(f"cannot read Parquet file {path}: {exc}") from exc
    columns = [str(n) for n in pf.schema_arrow.names]

    def gen() -> Iterator[pd.DataFrame]:
        try:
            for batch in pf.iter_batches(batch_size=o.chunk_size):
                yield batch.to_pandas().reset_index(drop=True)
        except Exception as exc:
            raise SourceReadError(f"cannot read Parquet file {path}: {exc}") from exc

    return _prime(_Opened(columns, gen(), [], None, None, None, None, (getattr(pf, "close", lambda: None),)))


# ---------------------------------------------------------------------------
# JSON family
# ---------------------------------------------------------------------------
def _read_text(path: str, o: ReadOptions, warnings: list[str]) -> tuple[str, str]:
    encoding = o.encoding or detect_encoding(path, warnings)
    try:
        with open(path, encoding=encoding) as fh:
            return fh.read(), encoding
    except UnicodeDecodeError as exc:
        raise SourceReadError(f"cannot decode {path} as {encoding}: {exc}. Try --encoding.") from exc


def _slice_frame(frame: pd.DataFrame, chunk_size: int) -> Iterator[pd.DataFrame]:
    for start in range(0, len(frame), chunk_size):
        yield frame.iloc[start:start + chunk_size].reset_index(drop=True)


def _frame_opened(frame: pd.DataFrame, o: ReadOptions, warnings: list[str], encoding: str | None = None) -> _Opened:
    return _Opened([str(c) for c in frame.columns], _slice_frame(frame, o.chunk_size), warnings,
                   None, None, None, encoding)


def _parse_jsonl_line(line: str, number: int, path: str) -> dict:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SourceReadError(f"invalid JSON on line {number} of {path}: {exc.msg}") from exc
    if not isinstance(obj, dict):
        raise SourceReadError(f"line {number} of {path} is not a JSON object")
    return obj


def _read_jsonl(path: str, o: ReadOptions, fmt: str) -> _Opened:
    warnings: list[str] = []
    encoding = o.encoding or detect_encoding(path, warnings)

    def records() -> Iterator[dict]:
        try:
            with open(path, encoding=encoding) as fh:
                for number, line in enumerate(fh, start=1):
                    if line.strip():
                        yield _parse_jsonl_line(line, number, path)
        except UnicodeDecodeError as exc:
            raise SourceReadError(f"cannot decode {path} as {encoding}: {exc}. Try --encoding.") from exc

    columns: dict[str, None] = {}
    count = 0
    for obj in records():  # first pass: the union of keys, so every chunk has the same schema
        count += 1
        for key in obj:
            columns.setdefault(str(key), None)
    if not count:
        raise SourceReadError(f"no JSON records found in {path}")
    cols = list(columns)

    def gen() -> Iterator[pd.DataFrame]:
        for batch in _batched(records(), o.chunk_size):
            yield pd.DataFrame(batch, columns=cols)

    return _prime(_Opened(cols, gen(), warnings, None, None, None, encoding))


_WRAPPER_KEYS = ("records", "data", "rows", "items", "results")


def _table_from_json(payload, path: str) -> pd.DataFrame | None:
    """A list of objects, or an object wrapping one; ``None`` when it is not a table."""
    if isinstance(payload, dict):
        for key in _WRAPPER_KEYS:
            if isinstance(payload.get(key), list):
                return _table_from_json(payload[key], path)
        lists = [v for v in payload.values() if isinstance(v, list)]
        if len(lists) == 1:
            return _table_from_json(lists[0], path)
        return None
    if isinstance(payload, list):
        if not payload:
            raise SourceReadError(f"no records found in {path}")
        if not all(isinstance(r, dict) for r in payload):
            raise SourceReadError(f"{path} must be a list of JSON objects (one per row)")
        return pd.DataFrame(payload)
    return None


def _read_json(path: str, o: ReadOptions, fmt: str) -> _Opened:
    warnings: list[str] = []
    text, encoding = _read_text(path, o, warnings)
    if not text.strip():
        raise SourceReadError(f"{path} is empty")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        if "Extra data" in exc.msg:
            warnings.append("file holds one JSON object per line; read as JSON Lines")
            opened = _read_jsonl(path, o, "jsonl")
            opened.warnings = warnings + opened.warnings
            return opened
        raise SourceReadError(f"invalid JSON in {path}: {exc}") from exc
    if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
        return _read_geojson(path, o, "geojson")
    frame = _table_from_json(payload, path)
    if frame is None:
        raise SourceReadError(
            f"{path} is valid JSON but not a table: expected a list of objects, "
            f"an object wrapping one (keys like {', '.join(_WRAPPER_KEYS)}) or a GeoJSON FeatureCollection"
        )
    return _frame_opened(frame, o, warnings, encoding)


def _read_geojson(path: str, o: ReadOptions, fmt: str) -> _Opened:
    from ..tables import TableFormatError, _read_geojson_bytes  # lazy

    warnings: list[str] = []
    with open(path, "rb") as fh:
        data = fh.read()
    try:
        frame = _read_geojson_bytes(data, warnings)
    except TableFormatError as exc:
        raise SourceReadError(f"cannot read GeoJSON {path}: {exc}") from exc
    return _frame_opened(frame, o, warnings)


# ---------------------------------------------------------------------------
# Shapefile / GPKG
# ---------------------------------------------------------------------------
def _read_shapefile(path: str, o: ReadOptions, fmt: str) -> _Opened:
    from ..tables import TableFormatError, _centroid, _prj_transformer, _read_shapefile_zip  # lazy

    warnings: list[str] = []
    if path.lower().endswith(".zip"):
        with open(path, "rb") as fh:
            data = fh.read()
        try:
            frame = _read_shapefile_zip(data, warnings)
        except TableFormatError as exc:
            raise SourceReadError(f"cannot read shapefile {path}: {exc}") from exc
        return _frame_opened(frame, o, warnings)

    import shapefile

    base = os.path.splitext(path)[0]
    encoding = o.encoding
    if encoding is None and os.path.exists(base + ".cpg"):
        with open(base + ".cpg", encoding="ascii", errors="ignore") as fh:
            encoding = fh.read().strip() or None
    try:
        reader = shapefile.Reader(path, encoding=encoding or "utf-8", encodingErrors="replace")
    except Exception as exc:
        raise SourceReadError(
            f"cannot read shapefile {path}: {exc}. A shapefile needs its .shp, .shx and .dbf side by side."
        ) from exc
    transformer = None
    if os.path.exists(base + ".prj"):
        transformer = _prj_transformer(base + ".prj", warnings)
    else:
        warnings.append("the shapefile has no .prj; EPSG:4326 is assumed for the centroids")
    fields = [f[0] for f in reader.fields if f[0] != "DeletionFlag"]
    columns = _dedupe_columns(fields + ["_lon", "_lat"])

    def rows() -> Iterator[list]:
        import numpy as np

        for shape_record in reader.iterShapeRecords():
            values = list(shape_record.record)
            try:
                geometry = shape_record.shape.__geo_interface__
            except Exception:
                geometry = None
            lon, lat = _centroid(geometry)
            if geometry and np.isfinite(lon) and np.isfinite(lat) and transformer is not None:
                lon, lat = transformer.transform(lon, lat)
            yield values + [float(lon), float(lat)]

    def gen() -> Iterator[pd.DataFrame]:
        for batch in _batched(rows(), o.chunk_size):
            yield pd.DataFrame(batch, columns=columns)

    return _prime(_Opened(columns, gen(), warnings, None, None, None, encoding, (reader.close,)))


def _read_gpkg(path: str, o: ReadOptions, fmt: str) -> _Opened:
    try:
        import pyogrio
        import geopandas  # noqa: F401  (pyogrio.read_dataframe needs it)
    except ImportError as exc:
        raise MissingDependencyError(
            "reading GeoPackage files requires pyogrio and geopandas: pip install pyogrio geopandas"
        ) from exc
    warnings: list[str] = []
    layer = o.table
    try:
        info = pyogrio.read_info(path, layer=layer)
        total = int(info["features"])
        fields = [str(f) for f in info["fields"]]
    except Exception as exc:
        raise SourceReadError(f"cannot read GeoPackage {path}: {exc}") from exc
    columns = fields + ["_lon", "_lat"]

    def gen() -> Iterator[pd.DataFrame]:
        for start in range(0, total, o.chunk_size):
            try:
                gdf = pyogrio.read_dataframe(path, layer=layer, skip_features=start, max_features=o.chunk_size)
            except Exception as exc:
                raise SourceReadError(f"cannot read GeoPackage {path}: {exc}") from exc
            if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(4326)
            centroids = gdf.geometry.centroid
            frame = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
            frame["_lon"], frame["_lat"] = centroids.x.values, centroids.y.values
            yield frame.reset_index(drop=True)

    return _prime(_Opened(columns, gen(), warnings))


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_LEADING_COMMENTS = re.compile(r"^(?:\s+|--[^\n]*(?:\n|$)|/\*.*?\*/)+", re.S)
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")


def redact_url(url: str) -> str:
    """The SQLAlchemy URL with its password masked (safe to print)."""
    try:
        from sqlalchemy.engine import make_url

        return make_url(url).render_as_string(hide_password=True)
    except Exception:
        return "<unparseable url>"


def _validate_table(name: str) -> tuple[str | None, str]:
    parts = name.split(".")
    if len(parts) > 2 or not all(_IDENTIFIER.match(p) for p in parts):
        raise UsageError(
            f"invalid table name {name!r}: use [schema.]table with letters, digits and underscores only "
            "(use --sql-query for anything else)"
        )
    return (parts[0], parts[1]) if len(parts) == 2 else (None, parts[0])


def _validate_query(query: str) -> str:
    text = _LEADING_COMMENTS.sub("", query).strip().rstrip(";").strip()
    if not re.match(r"(?is)^(select|with)\b", text):
        raise UsageError("--sql-query must be a single read-only SELECT (or WITH ... SELECT) statement")
    if ";" in _STRING_LITERAL.sub("''", text):
        raise UsageError("--sql-query must hold a single statement (found ';')")
    return text


def _short(exc: Exception, secret: str | None) -> str:
    text = " ".join(str(exc).split())[:300]
    return text.replace(secret, "***") if secret else text


def _read_sql(url: str, o: ReadOptions, fmt: str) -> _Opened:
    if bool(o.table) == bool(o.query):
        raise UsageError("a SQL source needs exactly one of --table or --sql-query")
    try:
        import sqlalchemy as sa
        from sqlalchemy import exc as sa_exc
    except ImportError as exc:
        raise MissingDependencyError("reading from a database requires SQLAlchemy: pip install sqlalchemy") from exc

    if o.table:
        schema, name = _validate_table(o.table)
        statement = sa.select(sa.literal_column("*")).select_from(sa.table(name, schema=schema))
    else:
        statement = sa.text(_validate_query(o.query))

    safe_url = redact_url(url)
    try:
        secret = sa.engine.make_url(url).password
    except Exception:
        secret = None
    try:
        engine = sa.create_engine(url)
    except ImportError as exc:  # the DBAPI driver (psycopg2, pymysql, ...) is missing
        raise MissingDependencyError(
            f"the database driver for {safe_url} is not installed ({_short(exc, secret)}); "
            "install it, e.g. pip install psycopg2-binary"
        ) from exc
    except (sa_exc.ArgumentError, sa_exc.NoSuchModuleError) as exc:
        raise UsageError(f"invalid database URL {safe_url}: {_short(exc, secret)}") from exc

    connection = None
    try:
        connection = engine.connect()
        result = connection.execution_options(stream_results=True).execute(statement)
        columns = _dedupe_columns(list(result.keys()))
    except (sa_exc.SQLAlchemyError, OSError) as exc:
        if connection is not None:
            connection.close()
        engine.dispose()
        raise SourceReadError(f"database query failed for {safe_url}: {_short(exc, secret)}") from exc

    def close() -> None:
        result.close()
        connection.close()
        engine.dispose()

    def gen() -> Iterator[pd.DataFrame]:
        try:
            while True:
                rows = result.fetchmany(o.chunk_size)
                if not rows:
                    return
                yield pd.DataFrame([tuple(r) for r in rows], columns=columns, dtype=object)
        except sa_exc.SQLAlchemyError as exc:
            raise SourceReadError(f"database read failed for {safe_url}: {_short(exc, secret)}") from exc

    return _Opened(columns, gen(), [], closers=(close,))


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------
register_reader("csv", _read_csv_like, [".csv"], "delimiter and encoding sniffed; streaming")
register_reader("tsv", _read_csv_like, [".tsv"], "tab separated; streaming")
register_reader("txt", _read_csv_like, [".txt"], "delimited table, or one address per line; streaming")
register_reader("lines", _read_lines_format, [], "one address per line, no header; streaming")
register_reader("xlsx", _read_xlsx, [".xlsx", ".xlsm"], "--sheet, --header-row; streaming (openpyxl read-only)")
register_reader("xls", _read_xls, [".xls"], "legacy Excel; needs xlrd; read whole")
register_reader("parquet", _read_parquet, [".parquet", ".pq"], "streaming by batches")
register_reader("json", _read_json, [".json"], "list of objects, wrapped list or FeatureCollection; read whole")
register_reader("jsonl", _read_jsonl, [".jsonl", ".ndjson"], "one JSON object per line; streaming")
register_reader("geojson", _read_geojson, [".geojson"], "FeatureCollection; centroid -> _lon/_lat; read whole")
register_reader("shp", _read_shapefile, [".shp", ".zip"], ".shp streams; .zip (shapefile) read whole")
register_reader("gpkg", _read_gpkg, [".gpkg"], "needs pyogrio + geopandas; --table selects the layer")
register_reader("sql", _read_sql, [], "SQLAlchemy URL + --table or --sql-query; server-side chunks")
