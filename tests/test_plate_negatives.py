"""Plate-neighbour hard negatives (``build_plate_negatives``)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tiny import tiny_docs  # noqa: F401  (also puts src/ on sys.path)

from cali_address.dataset import build_plate_negatives


def _docs(addresses):
    return pd.DataFrame({"direccion": addresses, "doc_id": range(len(addresses))})


def test_same_face_neighbours_within_delta_nearest_first():
    docs = _docs(["CL 5 # 10 - 20", "CL 5 # 10 - 22", "CL 5 # 10 - 25", "CL 5 # 10 - 90"])
    out = build_plate_negatives(docs, count=4, max_delta=20)
    assert list(out[0][:2]) == [1, 2]            # |delta| 2 then 5
    assert 3 not in out[0]                       # delta 70 > max_delta


def test_max_delta_is_configurable():
    docs = _docs(["CL 5 # 10 - 20", "CL 5 # 10 - 90"])
    assert build_plate_negatives(docs, count=2, max_delta=5)[0, 0] == -1
    assert build_plate_negatives(docs, count=2, max_delta=100)[0, 0] == 1


def test_other_face_of_the_corner_same_plate():
    docs = _docs(["CL 5 # 10 - 20", "KR 10 # 5 - 20", "CL 7 # 10 - 20"])
    out = build_plate_negatives(docs, count=3, max_delta=20)
    assert 1 in out[0] and 0 in out[1]
    assert 2 not in out[0]                       # different corner


def test_doc_without_neighbours_gets_all_padding():
    docs = _docs(["CL 5 # 10 - 20", "KR 99 # 98 - 97"])
    out = build_plate_negatives(docs, count=3)
    assert (out == -1).all()
    assert out.shape == (2, 3)


def test_unparseable_missing_plate_and_none_are_skipped():
    docs = _docs(["", "no es una direccion", None, "CL 5 # 10", "CL 5 # 10 - 20", "CL 5 # 10 - 22"])
    out = build_plate_negatives(docs, count=2)
    assert (out[:4] == -1).all()
    assert out[4, 0] == 5 and out[5, 0] == 4


def test_never_returns_self_or_duplicates_and_is_left_packed():
    out = build_plate_negatives(tiny_docs(48), count=6, max_delta=30)
    for i, row in enumerate(out):
        valid = row[row >= 0]
        assert i not in valid
        assert len(set(valid.tolist())) == len(valid)
        assert (row[len(valid):] == -1).all()


def test_empty_frame_and_zero_count():
    assert build_plate_negatives(_docs([]), count=4).shape == (0, 4)
    assert build_plate_negatives(_docs(["CL 5 # 10 - 20"]), count=0).shape == (1, 0)


def test_deterministic_for_a_seed():
    a = build_plate_negatives(tiny_docs(48), count=4, seed=3)
    b = build_plate_negatives(tiny_docs(48), count=4, seed=3)
    assert np.array_equal(a, b)
