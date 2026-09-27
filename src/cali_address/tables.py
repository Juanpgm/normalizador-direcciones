"""Torch-free table helpers: address-column detection, header-row guessing and multi-format reading.

Everything here is importable without the model stack (torch), so ``cali-address --help``,
``inspect`` and ``formats`` start fast. :mod:`cali_address.service` re-exports every name, so
``from cali_address.service import read_table`` keeps working.
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
