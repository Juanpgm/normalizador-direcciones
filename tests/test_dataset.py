"""Unit tests for ``build_docs`` (address-level collapse of the parcel table)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cali_address.dataset import build_docs  # noqa: E402
from cali_address.inference import load_manzana_nunique  # noqa: E402


def _parcels(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "OBJECTID", "direccion", "numero_predial_nacional",
            "numero_predial_manzana", "centroid_lon", "centroid_lat",
        ],
    )


def _sample():
    pred = "760010100" + "0" * 12  # 21 chars is enough for the comuna/barrio slices
    return _parcels([
        (3, "CL 5 # 10 - 20", pred + "A", "MZ1", -76.50, 3.40),
        (1, "KR 9 # 1 - 1", pred + "B", "MZ9", -76.51, 3.41),
        (2, "CL 5 # 10 - 20", pred + "C", "MZ1", -76.52, 3.42),
        (4, "CL 5 # 10 - 20", pred + "D", "MZ2", -76.53, 3.43),
    ])


def test_build_docs_counts_manzanas():
    docs = build_docs(_sample())
    by_dir = docs.set_index("direccion")
    assert by_dir.loc["CL 5 # 10 - 20", "manzana_nunique"] == 2
    assert by_dir.loc["CL 5 # 10 - 20", "n_predios"] == 3
    assert by_dir.loc["KR 9 # 1 - 1", "manzana_nunique"] == 1
    assert docs["manzana_nunique"].dtype.kind == "i"


def test_docs_order_unchanged_after_rebuild():
    docs = build_docs(_sample())
    # Groups appear in order of their first predio by OBJECTID (1 -> KR 9, 2 -> CL 5).
    assert list(docs["direccion"]) == ["KR 9 # 1 - 1", "CL 5 # 10 - 20"]
    assert list(docs["doc_id"]) == [0, 1]
    # "first" values still come from the lowest OBJECTID of each group.
    cl5 = docs.set_index("direccion").loc["CL 5 # 10 - 20"]
    assert cl5["objectid"] == 2
    assert cl5["numero_predial_nacional"].endswith("C")
    assert cl5["manzana"] == "MZ1"


def test_manzana_nunique_is_independent_of_input_row_order():
    shuffled = _sample().sample(frac=1.0, random_state=0)
    docs = build_docs(shuffled)
    assert docs.set_index("direccion").loc["CL 5 # 10 - 20", "manzana_nunique"] == 2


def test_build_docs_drops_groups_without_centroid_and_keeps_counts_consistent():
    frame = _sample()
    frame.loc[frame["direccion"] == "KR 9 # 1 - 1", ["centroid_lon", "centroid_lat"]] = np.nan
    docs = build_docs(frame)
    assert list(docs["direccion"]) == ["CL 5 # 10 - 20"]
    assert list(docs["manzana_nunique"]) == [2]


def test_load_manzana_nunique_reads_column_when_present():
    docs = pd.DataFrame({"manzana_nunique": np.array([1, 2, 3], dtype=np.int64)})
    out = load_manzana_nunique(docs)
    assert list(out) == [1, 2, 3]


def test_load_manzana_nunique_falls_back_to_ones_when_column_missing():
    docs = pd.DataFrame({"direccion": ["a", "b", "c"]})
    out = load_manzana_nunique(docs)
    assert list(out) == [1, 1, 1]
    assert out.dtype.kind == "i"


def test_load_manzana_nunique_on_empty_frame():
    assert len(load_manzana_nunique(pd.DataFrame({"direccion": []}))) == 0
