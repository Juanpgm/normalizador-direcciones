"""Writer registry: streaming sinks with atomic output.

``open_sink(path, fmt=None)`` returns a :class:`Sink` (a context manager). Chunks
are appended with ``write(df)``; the destination only appears (atomically, via a
temporary file in the same directory) when the sink closes cleanly, so a crashed
run never leaves a truncated result and never clobbers a previous good one.

CSV, TSV, JSON, JSONL, GeoJSON and Parquet stream chunk by chunk. XLSX buffers
(the workbook format cannot be appended to) and is limited by Excel's row cap.
"""

from __future__ import annotations

import json
import math
import os
import uuid
import warnings
from typing import Callable

import numpy as np
import pandas as pd

from .errors import SinkWriteError, UnsupportedFormatError

_EXCEL_MAX_ROWS = 1_048_575
_EXCEL_MAX_CELL = 32_767
_FLOAT_OUTPUT_COLUMNS = ("lat", "lon", "confianza", "confiabilidad", "confiabilidad_manzana", "margen")


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
_WRITERS: dict[str, Callable[..., "Sink"]] = {}
_WRITER_EXTENSIONS: dict[str, str] = {}
_WRITER_NOTES: dict[str, str] = {}


def register_writer(fmt: str, factory: Callable[..., "Sink"], extensions=(), note: str = "") -> None:
    _WRITERS[fmt] = factory
    for ext in extensions:
        _WRITER_EXTENSIONS[ext.lower()] = fmt
    _WRITER_NOTES[fmt] = note


def list_writer_formats() -> list[str]:
    return sorted(_WRITERS)


def describe_writer_formats() -> list[dict]:
    return [
        {"format": fmt, "extensions": sorted(e for e, f in _WRITER_EXTENSIONS.items() if f == fmt),
         "note": _WRITER_NOTES.get(fmt, "")}
        for fmt in sorted(_WRITERS)
    ]


def infer_sink_format(path: str) -> str:
    ext = os.path.splitext(os.fspath(path))[1].lower()
    if ext in _WRITER_EXTENSIONS:
        return _WRITER_EXTENSIONS[ext]
    raise UnsupportedFormatError(
        f"cannot infer the output format of {path!r} (extension {ext or '(none)'}). "
        f"Supported output extensions: {', '.join(sorted(_WRITER_EXTENSIONS))}. Use --output-format to force one."
    )


def open_sink(path: str, fmt: str | None = None, **opts) -> "Sink":
    path = os.fspath(path)
    if fmt:
        key = str(fmt).strip().lower().lstrip(".")
        key = {"ndjson": "jsonl", "excel": "xlsx"}.get(key, key)
        if key not in _WRITERS:
            raise UnsupportedFormatError(
                f"unsupported output format {fmt!r}. Supported: {', '.join(sorted(_WRITERS))}"
            )
    else:
        key = infer_sink_format(path)
    return _WRITERS[key](path, **opts)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _is_missing(value) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float) and math.isnan(value)


def _json_default(value):
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (set, frozenset)):
        return sorted(map(str, value))
    return str(value)


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=_json_default)


def _records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    clean = df.replace([np.inf, -np.inf], np.nan).astype(object)
    return clean.where(pd.notna(clean), None).to_dict(orient="records")


# ---------------------------------------------------------------------------
# sinks
# ---------------------------------------------------------------------------
class Sink:
    """Base class: chunked writes, atomic finish, context-manager protocol."""

    def write(self, df: pd.DataFrame) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        """Finish and publish the output."""

    def abort(self) -> None:
        """Discard any partial output."""

    def __enter__(self) -> "Sink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


class MemorySink(Sink):
    """Collects chunks in memory (dry runs, tests, the API)."""

    def __init__(self) -> None:
        self._frames: list[pd.DataFrame] = []
        self._columns: list[str] = []

    def write(self, df: pd.DataFrame) -> None:
        self._columns = list(df.columns)
        if len(df):
            self._frames.append(df.reset_index(drop=True))

    @property
    def frame(self) -> pd.DataFrame:
        if not self._frames:
            return pd.DataFrame(columns=self._columns)
        with warnings.catch_warnings():  # all-NA columns in some chunks: pandas' dtype-inference FutureWarning
            warnings.simplefilter("ignore", FutureWarning)
            return pd.concat(self._frames, ignore_index=True)


class _FileSink(Sink):
    """Writes to a hidden temp file next to the target, then ``os.replace``s it."""

    def __init__(self, path: str) -> None:
        self.path = os.fspath(path)
        self._done = False
        if os.path.isdir(self.path):
            raise SinkWriteError(f"output path is a directory: {self.path}")
        parent = os.path.dirname(os.path.abspath(self.path))
        self._tmp = os.path.join(
            parent, f".{os.path.basename(self.path)}.{os.getpid()}.{uuid.uuid4().hex[:8]}.partial"
        )
        try:
            os.makedirs(parent, exist_ok=True)
            with open(self._tmp, "wb"):  # fail fast (before any expensive work) and keep default permissions
                pass
        except OSError as exc:
            raise SinkWriteError(f"cannot write to {self.path}: {exc}") from exc

    def _publish(self) -> None:
        try:
            os.replace(self._tmp, self.path)
        except OSError as exc:
            self._discard()
            raise SinkWriteError(f"cannot write to {self.path}: {exc}") from exc

    def _discard(self) -> None:
        try:
            if os.path.exists(self._tmp):
                os.remove(self._tmp)
        except OSError:
            pass

    def _finish(self) -> None:
        """Flush and close the underlying handle(s); overridden by subclasses."""

    def close(self) -> None:
        if self._done:
            return
        self._done = True
        try:
            self._finish()
        except SinkWriteError:
            self._discard()
            raise
        except OSError as exc:
            self._discard()
            raise SinkWriteError(f"cannot write to {self.path}: {exc}") from exc
        self._publish()

    def abort(self) -> None:
        if self._done:
            return
        self._done = True
        try:
            self._abort_handles()
        finally:
            self._discard()

    def _abort_handles(self) -> None:
        self._finish()

    def write(self, df: pd.DataFrame) -> None:
        try:
            self._write(df)
        except SinkWriteError:
            raise
        except OSError as exc:
            raise SinkWriteError(f"cannot write to {self.path}: {exc}") from exc

    def _write(self, df: pd.DataFrame) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class _TextSink(_FileSink):
    """Shared handle management for the text formats."""

    def __init__(self, path: str, encoding: str = "utf-8") -> None:
        super().__init__(path)
        try:
            self._fh = open(self._tmp, "w", encoding=encoding, newline="")
        except OSError as exc:
            self._discard()
            raise SinkWriteError(f"cannot write to {self.path}: {exc}") from exc

    def _finish(self) -> None:
        if not self._fh.closed:
            self._fh.close()


class CsvSink(_TextSink):
    def __init__(self, path: str, encoding: str = "utf-8-sig", delimiter: str = ",") -> None:
        super().__init__(path, encoding)
        self._delimiter = delimiter
        self._columns: list[str] | None = None

    def _write(self, df: pd.DataFrame) -> None:
        first = self._columns is None
        if first:
            self._columns = list(df.columns)
        elif list(df.columns) != self._columns:
            raise SinkWriteError("the columns changed between chunks; cannot append to a CSV")
        df.to_csv(self._fh, index=False, header=first, sep=self._delimiter, lineterminator="\n")


class JsonlSink(_TextSink):
    def _write(self, df: pd.DataFrame) -> None:
        for record in _records(df):
            self._fh.write(_dumps(record) + "\n")


class JsonSink(_TextSink):
    """A JSON array of row objects, streamed."""

    def __init__(self, path: str, encoding: str = "utf-8") -> None:
        super().__init__(path, encoding)
        self._count = 0
        self._fh.write("[")

    def _write(self, df: pd.DataFrame) -> None:
        for record in _records(df):
            self._fh.write(("," if self._count else "") + "\n" + _dumps(record))
            self._count += 1

    def _finish(self) -> None:
        if not self._fh.closed:
            self._fh.write("\n]\n" if self._count else "]\n")
            self._fh.close()

    def _abort_handles(self) -> None:
        if not self._fh.closed:
            self._fh.close()


class GeoJsonSink(_TextSink):
    """A FeatureCollection with a Point per located row and ``null`` geometry otherwise."""

    def __init__(self, path: str, encoding: str = "utf-8") -> None:
        super().__init__(path, encoding)
        self._count = 0
        self._fh.write(
            '{"type": "FeatureCollection", "crs": {"type": "name", "properties": '
            '{"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}, "features": ['
        )

    def _write(self, df: pd.DataFrame) -> None:
        from ..service import results_to_geojson  # lazy

        if df.empty:
            return
        for feature in results_to_geojson(df)["features"]:
            self._fh.write(("," if self._count else "") + "\n" + _dumps(feature))
            self._count += 1

    def _finish(self) -> None:
        if not self._fh.closed:
            self._fh.write("\n]}\n" if self._count else "]}\n")
            self._fh.close()

    def _abort_handles(self) -> None:
        if not self._fh.closed:
            self._fh.close()


class ParquetSink(_FileSink):
    """Streams row groups. Object columns are stored as strings so chunks always agree."""

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self._writer = None
        self._schema = None
        self._pa = None

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.reset_index(drop=True).copy()
        for col in out.columns:
            if col in _FLOAT_OUTPUT_COLUMNS:
                out[col] = pd.to_numeric(out[col], errors="coerce")
            elif out[col].dtype == object:
                out[col] = out[col].map(lambda v: None if _is_missing(v) else (v if isinstance(v, str) else str(v)))
        out.columns = [str(c) for c in out.columns]
        return out

    def _write(self, df: pd.DataFrame) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        frame = self._prepare(df)
        if self._writer is None:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            fields = [
                pa.field(f.name, pa.string() if pa.types.is_null(f.type) else f.type) for f in table.schema
            ]
            self._schema = pa.schema(fields)
            self._writer = pq.ParquetWriter(self._tmp, self._schema)
        try:
            table = pa.Table.from_pandas(frame, schema=self._schema, preserve_index=False)
        except (pa.ArrowInvalid, pa.ArrowTypeError, KeyError, ValueError) as exc:
            raise SinkWriteError(
                f"a chunk does not fit the Parquet schema of the first chunk ({exc}); use csv or jsonl instead"
            ) from exc
        self._writer.write_table(table)

    def _finish(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def close(self) -> None:
        if self._writer is None and not self._done:
            # never written: still emit a valid (empty) file
            import pyarrow as pa
            import pyarrow.parquet as pq

            pq.write_table(pa.table({}), self._tmp)
        super().close()


class XlsxSink(_FileSink):
    """Buffers the whole result (three sheets, as the legacy writer): normalizado / revisar / resumen."""

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self._frames: list[pd.DataFrame] = []
        self._columns: list[str] = []
        self._rows = 0

    def _write(self, df: pd.DataFrame) -> None:
        self._columns = list(df.columns)
        self._rows += len(df)
        if self._rows > _EXCEL_MAX_ROWS:
            raise SinkWriteError(
                f"too many rows for an Excel sheet ({self._rows} > {_EXCEL_MAX_ROWS}); write csv or parquet instead"
            )
        if len(df):
            self._frames.append(df.reset_index(drop=True))

    def _finish(self) -> None:
        from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            frame = pd.concat(self._frames, ignore_index=True) if self._frames else pd.DataFrame(columns=self._columns)

        def clean(value):
            if isinstance(value, str):
                return ILLEGAL_CHARACTERS_RE.sub("", value)[:_EXCEL_MAX_CELL]
            return value

        for col in frame.columns:
            if frame[col].dtype == object:
                frame[col] = frame[col].map(clean)
        try:
            with open(self._tmp, "wb") as fh, pd.ExcelWriter(fh, engine="openpyxl") as writer:
                frame.to_excel(writer, sheet_name="normalizado", index=False)
                if "estado" in frame.columns:
                    frame[frame["estado"] != "OK"].to_excel(writer, sheet_name="revisar", index=False)
                    summary = frame["estado"].value_counts().rename_axis("estado").reset_index(name="filas")
                    summary.to_excel(writer, sheet_name="resumen", index=False)
        except (ValueError, TypeError) as exc:
            raise SinkWriteError(f"cannot write the Excel workbook: {exc}") from exc

    def _abort_handles(self) -> None:
        self._frames.clear()


register_writer("csv", lambda path, **o: CsvSink(path, **o), [".csv"], "UTF-8 with BOM (Excel friendly); streaming")
register_writer("tsv", lambda path, **o: CsvSink(path, delimiter="\t", **o), [".tsv"], "tab separated; streaming")
register_writer("xlsx", lambda path, **o: XlsxSink(path), [".xlsx"], "sheets normalizado/revisar/resumen; buffered")
register_writer("parquet", lambda path, **o: ParquetSink(path), [".parquet"], "streaming row groups")
register_writer("json", lambda path, **o: JsonSink(path, **o), [".json"], "array of row objects; streaming")
register_writer("jsonl", lambda path, **o: JsonlSink(path, **o), [".jsonl", ".ndjson"], "one JSON object per line; streaming")
register_writer("geojson", lambda path, **o: GeoJsonSink(path, **o), [".geojson"], "Point per located row; streaming")
