"""Download the full Cali catastral layer (urbano_terreno) from the ArcGIS FeatureServer.

Stores attributes + centroid + polygon geometry (as WKB) in a parquet cache under artifacts/.
Resumable: each page is cached as its own parquet part; a final merge produces catastro.parquet.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from shapely.geometry import Polygon, MultiPolygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(ROOT, "artifacts")
PARTS = os.path.join(ARTIFACTS, "catastro_parts")
OUT = os.path.join(ARTIFACTS, "catastro.parquet")

SERVICE = (
    "https://services8.arcgis.com/ljfiJpg35HWgdtaC/arcgis/rest/services/"
    "Revison_Reportes_Ciudadanos_V2_WFL1/FeatureServer/0/query"
)
FIELDS = (
    "OBJECTID,numero_predial_nacional,numero_predial_manzana,direccion,"
    "direccion_py,id_predio_sis,area_terreno,MAX_numero_pisos,Shape__Area"
)
PAGE = 2000
MAX_WORKERS = 4


def count_records() -> int:
    r = requests.get(
        SERVICE,
        params={"where": "1=1", "returnCountOnly": "true", "f": "json"},
        timeout=120,
    )
    r.raise_for_status()
    return int(r.json()["count"])


def rings_to_wkb(rings) -> bytes | None:
    """Convert an ArcGIS polygon ring list into WKB.

    ArcGIS uses clockwise outer rings and counter-clockwise holes. We rely on
    shapely to build polygons and treat any ring contained by a previous ring as
    a hole; for this dataset (cadastral parcels) holes are rare, so a simple
    shell/hole heuristic based on signed area is sufficient.
    """
    if not rings:
        return None
    shells, holes = [], []
    for ring in rings:
        if len(ring) < 4:
            continue
        # Signed area (shoelace); ArcGIS: negative => counter-clockwise => hole.
        area = 0.0
        for i in range(len(ring) - 1):
            x1, y1 = ring[i][0], ring[i][1]
            x2, y2 = ring[i + 1][0], ring[i + 1][1]
            area += x1 * y2 - x2 * y1
        (shells if area < 0 else holes).append(ring)
    # ArcGIS clockwise rings are outer in screen coords; with shoelace on
    # lon/lat, clockwise gives negative area. Keep the larger set as shells.
    if not shells:
        shells, holes = holes, []
    polys = []
    for shell in shells:
        try:
            polys.append(Polygon(shell))
        except Exception:
            continue
    if not polys:
        return None
    if holes:
        assigned = {i: [] for i in range(len(polys))}
        for hole in holes:
            try:
                hp = Polygon(hole)
            except Exception:
                continue
            for i, p in enumerate(polys):
                if p.contains(hp.representative_point()):
                    assigned[i].append(hole)
                    break
        polys = [
            Polygon(shell, assigned[i]) if assigned[i] else polys[i]
            for i, shell in enumerate(shells)
        ]
    geom = polys[0] if len(polys) == 1 else MultiPolygon(polys)
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom.wkb


def fetch_page(offset: int, session: requests.Session) -> str:
    part = os.path.join(PARTS, f"part_{offset:07d}.parquet")
    if os.path.exists(part) and os.path.getsize(part) > 0:
        return part
    params = {
        "where": "1=1",
        "outFields": FIELDS,
        "returnGeometry": "true",
        "returnCentroid": "true",
        "outSR": "4326",
        "geometryPrecision": "6",
        "orderByFields": "OBJECTID",
        "resultOffset": str(offset),
        "resultRecordCount": str(PAGE),
        "f": "json",
    }
    last_err = None
    for attempt in range(6):
        try:
            r = session.get(SERVICE, params=params, timeout=180)
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(data["error"])
            feats = data.get("features", [])
            rows = []
            for f in feats:
                a = f.get("attributes", {})
                c = f.get("centroid") or {}
                g = f.get("geometry") or {}
                rows.append(
                    {
                        **a,
                        "centroid_lon": c.get("x"),
                        "centroid_lat": c.get("y"),
                        "geom_wkb": rings_to_wkb(g.get("rings")),
                    }
                )
            df = pd.DataFrame(rows)
            tmp = part + ".tmp"
            df.to_parquet(tmp, index=False)
            os.replace(tmp, part)
            print(f"offset={offset} rows={len(df)}", flush=True)
            return part
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"offset {offset} failed: {last_err}")


def main() -> None:
    os.makedirs(PARTS, exist_ok=True)
    total = count_records()
    print(f"total records = {total}", flush=True)
    offsets = list(range(0, total + PAGE, PAGE))
    session = requests.Session()
    session.headers["User-Agent"] = "cali-address-normalizer/1.0"
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_page, o, session): o for o in offsets}
        done = 0
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 10 == 0:
                print(f"[{done}/{len(offsets)}] {time.time()-t0:.0f}s", flush=True)
    parts = sorted(
        os.path.join(PARTS, f) for f in os.listdir(PARTS) if f.endswith(".parquet")
    )
    dfs = [pd.read_parquet(p) for p in parts]
    df = pd.concat([d for d in dfs if len(d)], ignore_index=True)
    df = df.drop_duplicates(subset=["OBJECTID"]).sort_values("OBJECTID").reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    print(f"WROTE {OUT} rows={len(df)} cols={list(df.columns)}", flush=True)
    meta = {
        "source": SERVICE,
        "total_reported": total,
        "rows_downloaded": int(len(df)),
        "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "outSR": 4326,
        "geometryPrecision": 6,
    }
    with open(os.path.join(ARTIFACTS, "catastro_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


if __name__ == "__main__":
    sys.exit(main())
