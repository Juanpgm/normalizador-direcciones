"""Geometry helpers: projection, distances and point-in-parcel lookup.

No geopandas is required: shapely 2 exposes ``STRtree`` directly and pyproj gives
the projection. EPSG:3116 (MAGNA-SIRGAS / Colombia Bogota zone) is used as the
metric CRS because it covers Cali with sub-metre distortion for our purposes.
"""

from __future__ import annotations

import numpy as np
from pyproj import CRS, Transformer
from shapely import STRtree, points
from shapely import wkb as shapely_wkb

__all__ = ["METRIC_CRS", "to_metric", "haversine_m", "ParcelIndex"]

METRIC_CRS = "EPSG:3116"

_FWD = Transformer.from_crs(CRS("EPSG:4326"), CRS(METRIC_CRS), always_xy=True)


def to_metric(lon, lat) -> tuple[np.ndarray, np.ndarray]:
    """Project WGS84 lon/lat arrays to metres in :data:`METRIC_CRS`."""
    x, y = _FWD.transform(np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64"))
    return np.asarray(x), np.asarray(y)


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in metres (vectorized, NaN-safe)."""
    lat1, lon1, lat2, lon2 = (np.asarray(v, dtype="float64") for v in (lat1, lon1, lat2, lon2))
    r = 6_371_008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class ParcelIndex:
    """Point-in-polygon lookup over the cadastral parcels using an STRtree.

    ``query(lat, lon)`` returns, for every input point, the row index of the
    containing parcel, or -1. Points that fall in no polygon (street, park,
    coordinate noise) optionally snap to the nearest parcel within
    ``snap_max_m`` metres.
    """

    def __init__(self, wkb_series, ids=None) -> None:
        blobs = np.asarray(wkb_series, dtype=object)
        present = np.flatnonzero(np.array([b is not None for b in blobs]))
        # shapely.from_wkb is vectorized and ~50x faster than a Python loop.
        loaded = shapely_wkb.loads(list(blobs[present]))
        valid = np.array([g is not None and not g.is_empty for g in loaded])
        self.geoms = np.asarray(list(np.asarray(loaded, dtype=object)[valid]), dtype=object)
        self.row_index = present[valid].astype(np.int64)
        self.ids = None if ids is None else np.asarray(ids)[self.row_index]
        self.tree = STRtree(self.geoms)

    def __len__(self) -> int:
        return len(self.geoms)

    def query(self, lat, lon, snap_max_m: float = 0.0) -> np.ndarray:
        lat = np.asarray(lat, dtype="float64")
        lon = np.asarray(lon, dtype="float64")
        pts = points(lon, lat)
        out = np.full(len(lat), -1, dtype=np.int64)
        valid = np.isfinite(lat) & np.isfinite(lon)
        if not valid.any():
            return out
        idx = np.flatnonzero(valid)
        hit_q, hit_t = self.tree.query(pts[idx], predicate="within")
        # keep the first containing polygon per point
        seen: dict[int, int] = {}
        for q, t in zip(hit_q.tolist(), hit_t.tolist()):
            seen.setdefault(q, t)
        for q, t in seen.items():
            out[idx[q]] = self.row_index[t]
        if snap_max_m > 0:
            missing = idx[[q for q in range(len(idx)) if q not in seen]]
            if len(missing):
                near = self.tree.nearest(pts[missing])
                dist_deg = np.array(
                    [pts[m].distance(self.geoms[t]) for m, t in zip(missing, near)]
                )
                # ~111 km per degree of latitude at the equator
                within = dist_deg * 111_000.0 <= snap_max_m
                out[missing[within]] = self.row_index[near[within]]
        return out
