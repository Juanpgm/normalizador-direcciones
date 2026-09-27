"""Synthetic fixtures shared by the any-dataset IO tests.

Everything here is invented: no real cadastre, inspection or registry rows. The
stub cadastre holds ONE record (``CL 5 # 38 - 20``), so with the default
``StubNormalizer`` scoring:

* ``CL 5 # 38 - 20``       -> OK
* ``Carrera 100 # 15-30``   -> SIN_MATCH (structural rule violated vs the only doc)
* blank / placeholder      -> SIN_MATCH
* ``hola que tal``         -> NO_PARSEABLE
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from conftest import StubNormalizer, make_stub  # noqa: E402

ADDR_OK = "CL 5 # 38 - 20"
ADDR_OTHER = "Carrera 100 # 15-30 Apto 301"
ADDR_JUNK = "hola que tal"
ADDR_RAISES = "CL 66 # 66 - 66"

CADASTRE = [{"direccion": "CL 5 # 38 - 20", "predial": "PRED-1", "manzana": "MZ-1",
             "lat": 3.40, "lon": -76.50}]


class ExplodingStub(StubNormalizer):
    """Raises for any batch that contains ``ADDR_RAISES`` (a row 'poisoned' inside the model)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls = 0

    def score_components(self, queries, k=20):
        self.calls += 1
        if any(ADDR_RAISES in q for q in queries):
            raise RuntimeError("model exploded on a poisoned row")
        return super().score_components(queries, k=k)


def stub() -> StubNormalizer:
    return make_stub(CADASTRE)


def exploding_stub() -> ExplodingStub:
    return ExplodingStub(CADASTRE)


def synth_frame(n: int = 7) -> pd.DataFrame:
    """``n`` rows cycling through OK / SIN_MATCH / NO_PARSEABLE / blank addresses."""
    cycle = [ADDR_OK, ADDR_OTHER, ADDR_JUNK, "", ADDR_OK]
    return pd.DataFrame({
        "id": [f"{i:03d}" for i in range(n)],
        "direccion": [cycle[i % len(cycle)] for i in range(n)],
        "barrio": [f"barrio {i}" for i in range(n)],
        "extra_unknown": [f"x{i}" for i in range(n)],
    })


def write_csv(path, frame: pd.DataFrame, sep: str = ",", encoding: str = "utf-8", **kw) -> str:
    frame.to_csv(path, index=False, sep=sep, encoding=encoding, **kw)
    return str(path)


def collect(chunks) -> pd.DataFrame:
    frames = list(chunks)
    if not frames:
        return pd.DataFrame(columns=list(getattr(chunks, "columns", [])))
    return pd.concat(frames, ignore_index=True)


def assert_same_frames(a: pd.DataFrame, b: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        a.reset_index(drop=True).astype(object).where(pd.notna(a), None),
        b.reset_index(drop=True).astype(object).where(pd.notna(b), None),
        check_dtype=False,
    )


__all__ = [
    "ADDR_OK", "ADDR_OTHER", "ADDR_JUNK", "ADDR_RAISES", "CADASTRE", "ExplodingStub",
    "stub", "exploding_stub", "synth_frame", "write_csv", "collect", "assert_same_frames", "np",
]
