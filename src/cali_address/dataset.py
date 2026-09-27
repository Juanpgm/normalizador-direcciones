"""Builds the cadastral document index and the augmented training tensors.

Everything is cached under ``artifacts/`` so notebook runs never redo the work:

* ``catastro_docs.parquet``  - one row per unique cadastral address (the retrieval
  index), with predial code, manzana, centroid, comuna, barrio and metric x/y.
* ``train_data.npz``         - uint8 token matrices for queries and documents,
  the query -> doc mapping, the train/val split and the auxiliary targets.
"""

from __future__ import annotations

import os
import random
import re
import time

import numpy as np
import pandas as pd

from .augment import corrupt_many
from .geo import to_metric
from .model import MAX_LEN, encode_batch
from .parser import _pre_normalize, canonical, parse_address

__all__ = [
    "build_docs", "query_text_for", "build_training_arrays", "build_spatial_negatives",
    "build_plate_negatives", "N_VARIANTS", "SEED",
]

SEED = 42
N_VARIANTS = 3
VAL_DOC_FRACTION = 0.04


def build_docs(catastro: pd.DataFrame) -> pd.DataFrame:
    """Collapse the parcel table into one row per unique cadastral address."""
    df = catastro.copy()
    df["direccion"] = df["direccion"].astype("object")
    df = df[df["direccion"].notna()]
    df["direccion"] = df["direccion"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df = df[df["direccion"] != ""]
    pred = df["numero_predial_nacional"].astype(str)
    df["comuna"] = pred.str[9:11]
    df["barrio"] = pred.str[11:13]
    df["barrio_code"] = pred.str[9:13]
    df["manzana"] = df["numero_predial_manzana"].astype(str)

    df = df.sort_values("OBJECTID")
    agg = df.groupby("direccion", sort=False).agg(
        numero_predial_nacional=("numero_predial_nacional", "first"),
        manzana=("manzana", "first"),
        comuna=("comuna", "first"),
        barrio=("barrio", "first"),
        barrio_code=("barrio_code", "first"),
        centroid_lon=("centroid_lon", "median"),
        centroid_lat=("centroid_lat", "median"),
        n_predios=("OBJECTID", "size"),
        manzana_nunique=("manzana", "nunique"),
        objectid=("OBJECTID", "first"),
    ).reset_index()
    agg = agg[agg["centroid_lon"].notna() & agg["centroid_lat"].notna()].reset_index(drop=True)
    x, y = to_metric(agg["centroid_lon"].to_numpy(), agg["centroid_lat"].to_numpy())
    agg["x_m"], agg["y_m"] = x, y
    agg["doc_id"] = np.arange(len(agg), dtype=np.int64)
    # A parcel -> address map is useful when scoring point-in-polygon ground truth.
    return agg


def query_text_for(raw: str) -> tuple[str, str]:
    """Return ``(rule_canonical_or_prenorm, prenormalized_raw)`` for the encoder.

    The retrieval query is the rule-based canonical form when the parser
    succeeds; otherwise the pre-normalized raw text is used so that unparseable
    input still gets an embedding.
    """
    p = parse_address(raw)
    prenorm = _pre_normalize("" if raw is None else str(raw))[:MAX_LEN]
    canon = canonical(p, style="spaced", with_complement=True) if p.parse_ok else ""
    return (canon or prenorm), prenorm


def build_training_arrays(
    docs: pd.DataFrame, n_variants: int = N_VARIANTS, seed: int = SEED, op_scale: dict | None = None
):
    """Generate the noisy-variant training set (one query row per variant)."""
    rng = random.Random(seed)
    canon_list = docs["direccion"].tolist()
    n_docs = len(canon_list)

    t0 = time.time()
    q_canon: list[str] = []
    q_prenorm: list[str] = []
    q_doc: list[int] = []
    for doc_id, canon in enumerate(canon_list):
        for variant in corrupt_many(canon, n_variants, rng, op_scale=op_scale):
            a, b = query_text_for(variant)
            q_canon.append(a)
            q_prenorm.append(b)
            q_doc.append(doc_id)
        if doc_id % 50_000 == 0:
            print(f"  augment {doc_id}/{n_docs} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  augmentation done: {len(q_doc)} queries in {time.time()-t0:.0f}s", flush=True)

    q_tokens = encode_batch(q_canon)
    q_tokens_raw = encode_batch(q_prenorm)
    d_tokens = encode_batch([c.upper() for c in canon_list])

    q_doc_arr = np.asarray(q_doc, dtype=np.int64)

    # Split by document so no cadastral address (and therefore no predio) leaks
    # between train and validation.
    gen = np.random.default_rng(seed)
    perm = gen.permutation(n_docs)
    n_val = int(n_docs * VAL_DOC_FRACTION)
    val_docs = np.zeros(n_docs, dtype=bool)
    val_docs[perm[:n_val]] = True
    is_val = val_docs[q_doc_arr]

    comuna_codes = sorted(docs["comuna"].unique().tolist())
    barrio_codes = sorted(docs["barrio_code"].unique().tolist())
    comuna_map = {c: i for i, c in enumerate(comuna_codes)}
    barrio_map = {c: i for i, c in enumerate(barrio_codes)}
    doc_comuna = docs["comuna"].map(comuna_map).to_numpy(dtype=np.int64)
    doc_barrio = docs["barrio_code"].map(barrio_map).to_numpy(dtype=np.int64)

    xy = docs[["x_m", "y_m"]].to_numpy(dtype=np.float64)
    xy_mean = xy.mean(0)
    xy_std = xy.std(0)
    doc_xy = ((xy - xy_mean) / xy_std).astype(np.float32)

    return {
        "q_tokens": q_tokens,
        "q_tokens_raw": q_tokens_raw,
        "q_doc": q_doc_arr,
        "q_is_val": is_val,
        "d_tokens": d_tokens,
        "doc_comuna": doc_comuna,
        "doc_barrio": doc_barrio,
        "doc_xy": doc_xy,
        "xy_mean": xy_mean.astype(np.float64),
        "xy_std": xy_std.astype(np.float64),
        "comuna_codes": np.array(comuna_codes, dtype=object),
        "barrio_codes": np.array(barrio_codes, dtype=object),
    }


def build_spatial_negatives(docs: pd.DataFrame, k: int = 8, seed: int = SEED) -> np.ndarray:
    """For every document, ``k`` geographically confusable document ids.

    The first slots are filled with other addresses in the **same cadastral
    block** (``numero_predial_manzana``); the rest come from the nearest
    centroids, so every anchor always has real spatial hard negatives.
    """
    from sklearn.neighbors import NearestNeighbors

    n = len(docs)
    out = np.full((n, k), -1, dtype=np.int64)

    # nearest neighbours by projected centroid
    xy = docs[["x_m", "y_m"]].to_numpy()
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm="kd_tree").fit(xy)
    _, idx = nn.kneighbors(xy)
    # drop the self match (first column) where present
    neigh = np.where(idx[:, [0]] == np.arange(n)[:, None], idx[:, 1:], idx[:, :-1])
    out[:, : neigh.shape[1]] = neigh[:, :k]

    # overwrite the first two slots with same-block addresses when available
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for doc_id, mz in zip(docs["doc_id"].to_numpy(), docs["manzana"].astype(str).to_numpy()):
        groups.setdefault(mz, []).append(int(doc_id))
    for mz, members in groups.items():
        if len(members) < 2:
            continue
        arr = np.asarray(members)
        picks = rng.integers(0, len(arr), size=(len(arr), 2))
        chosen = arr[picks]
        same = chosen == arr[:, None]
        chosen[same] = arr[(picks[same] + 1) % len(arr)]
        out[arr, 0] = chosen[:, 0]
        out[arr, 1] = chosen[:, 1]
    return out


_PLATE_DIGITS = re.compile(r"^\s*(\d+)")


def _plate_int(plate) -> int | None:
    if plate is None:
        return None
    m = _PLATE_DIGITS.match(str(plate))
    return int(m.group(1)) if m else None


def build_plate_negatives(
    docs: pd.DataFrame,
    count: int = 8,
    max_delta: int = 20,
    seed: int = SEED,
    other_face_slots: int = 2,
) -> np.ndarray:
    """Plate-level hard negatives: the addresses a model confuses most easily.

    For every document (via V, cross-street C, plate P) the negatives are

    * documents on the **same block face** (same via and same cross street) whose
      plate differs by ``1..max_delta``, nearest plate first, and
    * documents with the **same plate on the swapped via/cross** (the other face
      of the corner), at most ``other_face_slots`` of them.

    Returns an ``(n_docs, count)`` int64 matrix, left-packed and padded with -1.
    Documents that cannot be parsed, have no plate/cross, or simply have no
    neighbour keep an all ``-1`` row, so callers must treat -1 as "no negative".
    """
    from bisect import bisect_left, bisect_right

    n = len(docs)
    out = np.full((n, max(count, 0)), -1, dtype=np.int64)
    if n == 0 or count <= 0:
        return out
    rng = np.random.default_rng(seed)

    def part(p, prefix):
        return (
            getattr(p, f"{prefix}_number"), getattr(p, f"{prefix}_letters"),
            getattr(p, f"{prefix}_suffix"), getattr(p, f"{prefix}_suffix_letters"),
            bool(getattr(p, f"{prefix}_bis")), getattr(p, f"{prefix}_quadrant"),
        )

    face_of: dict[int, tuple] = {}
    plate_of: dict[int, int] = {}
    corner_of: dict[int, tuple] = {}
    faces: dict[tuple, list[tuple[int, int]]] = {}
    corners: dict[tuple, list[int]] = {}
    for i, text in enumerate(docs["direccion"].tolist()):
        p = parse_address("" if text is None else str(text))
        plate = _plate_int(p.plate)
        if not p.parse_ok or p.via_number is None or p.cross_number is None or plate is None:
            continue
        v, c = part(p, "via"), part(p, "cross")
        face = (p.via_type, v, p.cross_type, c)
        face_of[i], plate_of[i] = face, plate
        corner = (tuple(sorted([v, c], key=repr)), plate)
        corner_of[i] = corner
        faces.setdefault(face, []).append((plate, i))
        corners.setdefault(corner, []).append(i)
    for members in faces.values():
        members.sort()
    face_plates = {f: [pl for pl, _ in m] for f, m in faces.items()}

    for i, face in face_of.items():
        chosen: list[int] = []
        others = [j for j in corners[corner_of[i]] if face_of[j] != face]
        if others and other_face_slots > 0:
            take = min(other_face_slots, count, len(others))
            chosen.extend(int(j) for j in rng.choice(others, size=take, replace=False))
        plate = plate_of[i]
        plates, members = face_plates[face], faces[face]
        lo, hi = bisect_left(plates, plate - max_delta), bisect_right(plates, plate + max_delta)
        near = [(abs(pl - plate), pl, j) for pl, j in members[lo:hi] if pl != plate]
        near.sort()
        for _, _, j in near:
            if len(chosen) >= count:
                break
            if j not in chosen:
                chosen.append(int(j))
        out[i, : len(chosen[:count])] = chosen[:count]
    return out
