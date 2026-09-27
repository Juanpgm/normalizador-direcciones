"""Tiny CPU fixtures for the training-infrastructure tests (no GPU, no real data)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cali_address.dataset import build_spatial_negatives, build_training_arrays  # noqa: E402
from cali_address.train import TrainConfig  # noqa: E402


def tiny_docs(n: int = 48) -> pd.DataFrame:
    rows = []
    for i in range(n):
        via, cross, plate = 1 + i // 8, 10 + (i // 4) % 2, 10 + 2 * (i % 4)
        rows.append({
            "direccion": f"CL {via} # {cross} - {plate:02d}",
            "numero_predial_nacional": f"7600101{i:014d}",
            "manzana": f"MZ{i // 4}",
            "comuna": f"{1 + i % 3:02d}",
            "barrio": f"{1 + i % 2:02d}",
            "barrio_code": f"{1 + i % 3:02d}{1 + i % 2:02d}",
            "centroid_lon": -76.5 + i * 1e-4,
            "centroid_lat": 3.4 + i * 1e-4,
            "n_predios": 1,
            "manzana_nunique": 1,
            "objectid": i,
            "x_m": 1000.0 + 15.0 * i,
            "y_m": 2000.0 + 7.0 * (i % 5),
            "doc_id": i,
        })
    return pd.DataFrame(rows)


def tiny_config(**kw) -> TrainConfig:
    base = dict(epochs=1, batch_size=16, d_model=16, n_layers=1, n_heads=2, d_out=16, dropout=0.0,
                max_len=40, max_minutes=2.0)
    base.update(kw)
    return TrainConfig(**base)


def tiny_data_dir(path: str, n_docs: int = 48) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """Write train_data.npz, spatial_negatives.npy and catastro_docs.parquet into ``path``."""
    os.makedirs(path, exist_ok=True)
    docs = tiny_docs(n_docs)
    data = build_training_arrays(docs, n_variants=3, seed=7)
    neg = build_spatial_negatives(docs, k=4)
    np.savez_compressed(os.path.join(path, "train_data.npz"), **data)
    np.save(os.path.join(path, "spatial_negatives.npy"), neg)
    docs.to_parquet(os.path.join(path, "catastro_docs.parquet"))
    return docs, data, neg


def write_pairs(path: str, docs: pd.DataFrame, n: int = 12) -> str:
    rows = []
    for i in range(n):
        d = i % len(docs)
        rows.append({
            "raw_address": docs["direccion"][d].replace("#", "No.").lower(),
            "address_key": f"K{i}",
            "dataset": "fasecolda",
            "gt_doc_id": d,
            "gt_manzana": None,
            "gt_predial": docs["numero_predial_nacional"][d],
            "gt_comuna": docs["comuna"][d],
            "source_row_id": f"r{i}",
        })
    pd.DataFrame(rows).to_parquet(path)
    return path
