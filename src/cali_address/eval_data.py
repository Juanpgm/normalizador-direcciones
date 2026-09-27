"""Loading and sampling of the five external evaluation datasets in ``context/``.

Column names in these workbooks carry accents and stray whitespace, so every
column is matched on an ASCII-folded, lowercased, stripped form (prefix match).
"""

from __future__ import annotations

import os
import re
import unicodedata

import pandas as pd

__all__ = ["DATASETS", "DatasetSpec", "norm_col", "pick_col", "load_dataset", "sample_dataset"]

SEED = 42
SAMPLE_N = 500


def norm_col(name: object) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip().lower()


def pick_col(df: pd.DataFrame, *prefixes: str) -> str | None:
    """Return the first column whose normalized name starts with any prefix."""
    norm = {c: norm_col(c) for c in df.columns}
    for prefix in prefixes:
        target = norm_col(prefix)
        for col, n in norm.items():
            if n == target:
                return col
        for col, n in norm.items():
            if n.startswith(target):
                return col
    return None


class DatasetSpec:
    def __init__(
        self,
        key: str,
        filename: str,
        sheet: str,
        header: int,
        address_col: str,
        normalized_col: str | None = None,
        lat_col: str | None = None,
        lon_col: str | None = None,
        coords_col: str | None = None,
        accuracy_col: str | None = None,
        sample_n: int = SAMPLE_N,
        note: str = "",
    ) -> None:
        self.key = key
        self.filename = filename
        self.sheet = sheet
        self.header = header
        self.address_col = address_col
        self.normalized_col = normalized_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.coords_col = coords_col
        self.accuracy_col = accuracy_col
        self.sample_n = sample_n
        self.note = note


DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        "fasecolda",
        "20260821 BD Inmuebles asegurados Cali - Fasecolda.xlsx",
        "EXPUESTOS", 0,
        address_col="direccion",
        normalized_col="direccion_normalizada",
        lat_col="latitud (y) wgs84",
        lon_col="longitud (x) wgs 84",
        note="Insured properties (Fasecolda). Has its own normalized column and WGS84 coordinates.",
    ),
    DatasetSpec(
        "rud",
        "RUD VALLE DEL CAUCA - CALI.xlsx",
        "RUD MUNICIPIO", 0,
        address_col="direccion_bien",
        note="Single disaster victims registry (RUD). No coordinates; several addresses per cell.",
    ),
    DatasetSpec(
        "inspecciones",
        "inspecciones_2026-09-23_17-27.xlsx",
        "inspecciones", 2,
        address_col="direccion",
        normalized_col="direccion_norm",
        coords_col="coords",
        accuracy_col="gps_error_m",
        note=("Field structural inspections. `coords` is a 'lat, lon' string filled for ~93% of "
              "rows; `gps_error_m` carries the reported GPS error."),
    ),
    DatasetSpec(
        "stickers",
        "stickers_2026-09-23_20-23.xlsx",
        "stickers", 5,
        address_col="direccion",
        lat_col="lat",
        lon_col="lng",
        accuracy_col="accuracy",
        note=("Habitability stickers captured on a mobile app; lat/lng from device GPS. The "
              "`accuracy` column exists but is empty in the delivered file."),
    ),
    DatasetSpec(
        "acciones",
        "acciones_candidatos_demolicion_2026-09-23_20-25.xlsx",
        "acciones", 2,
        address_col="direccion",
        sample_n=10_000,  # only 78 rows exist -> all of them are used
        note="Demolition candidates. Only 78 rows, so the whole dataset is evaluated.",
    ),
]


def _coerce_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False).str.strip(),
        errors="coerce",
    )


def load_dataset(spec: DatasetSpec, context_dir: str) -> pd.DataFrame:
    """Read one workbook and return a tidy frame: raw / their_norm / lat / lon / accuracy."""
    raw = pd.read_excel(
        os.path.join(context_dir, spec.filename), sheet_name=spec.sheet, header=spec.header
    )
    addr = pick_col(raw, spec.address_col)
    if addr is None:
        raise KeyError(f"{spec.key}: address column {spec.address_col!r} not found")
    out = pd.DataFrame({"raw_address": raw[addr].astype("object")})
    out["dataset"] = spec.key
    out["source_row"] = raw.index

    if spec.normalized_col:
        col = pick_col(raw, spec.normalized_col)
        out["their_normalized"] = raw[col].astype("object") if col else None
    else:
        out["their_normalized"] = None

    lat = lon = None
    if spec.coords_col:
        col = pick_col(raw, spec.coords_col)
        if col is not None:
            pieces = raw[col].astype(str).str.split(",", n=1, expand=True)
            if pieces.shape[1] == 2:
                lat = _coerce_float(pieces[0])
                lon = _coerce_float(pieces[1])
    if lat is None and spec.lat_col:
        cl, co = pick_col(raw, spec.lat_col), pick_col(raw, spec.lon_col or "")
        if cl is not None and co is not None:
            lat, lon = _coerce_float(raw[cl]), _coerce_float(raw[co])
    out["gt_lat"] = lat if lat is not None else pd.NA
    out["gt_lon"] = lon if lon is not None else pd.NA

    if spec.accuracy_col:
        col = pick_col(raw, spec.accuracy_col)
        out["gps_accuracy_m"] = _coerce_float(raw[col]) if col else pd.NA
    else:
        out["gps_accuracy_m"] = pd.NA

    # Cali bounding box sanity check: anything outside is treated as missing.
    lat_n = pd.to_numeric(out["gt_lat"], errors="coerce")
    lon_n = pd.to_numeric(out["gt_lon"], errors="coerce")
    inside = lat_n.between(3.2, 3.65) & lon_n.between(-76.75, -76.4)
    out["gt_lat"] = lat_n.where(inside)
    out["gt_lon"] = lon_n.where(inside)
    return out


def sample_dataset(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Drop null/blank addresses, then take up to ``n`` rows with a fixed seed."""
    s = df["raw_address"]
    keep = s.notna() & (s.astype(str).str.strip() != "") & (s.astype(str).str.strip().str.lower() != "nan")
    clean = df[keep]
    if len(clean) <= n:
        return clean.reset_index(drop=True)
    return clean.sample(n, random_state=seed).reset_index(drop=True)
