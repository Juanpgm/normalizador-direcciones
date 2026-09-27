"""Shared test doubles for the normalization engine.

``StubNormalizer`` stands in for ``cali_address.inference.AddressNormalizer``
wherever a test only needs the attributes ``normalize_strict``/``_build_row``
actually read (``_direccion``, ``_predial``, ``_manzana``, ``_lat``, ``_lon``,
``_comuna``, ``_n_predios``, ``threshold``, ``weights``) plus a
``score_components`` that returns arrays shaped like the real one. It never
loads a model or touches the GPU.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cali_address.inference import _structural_agreement  # noqa: E402
from cali_address.parser import parse_address  # noqa: E402
from cali_address.paths import DEFAULT_ARTIFACTS_DIR, check_artifacts  # noqa: E402


def pytest_collection_modifyitems(config, items):
    """Skip ``@pytest.mark.requires_artifacts`` tests when the real serving artifacts are absent.

    A fresh clone (or CI) has no ``catastro_emb.pt``: it is above GitHub's file-size limit and is
    regenerated locally (docs/model-artifacts.md).
    """
    missing, _ = check_artifacts(DEFAULT_ARTIFACTS_DIR)
    if not missing:
        return
    skip = pytest.mark.skip(reason="serving artifacts are not present: " + ", ".join(os.path.basename(p) for p in missing))
    for item in items:
        if "requires_artifacts" in item.keywords:
            item.add_marker(skip)


class StubNormalizer:
    """Minimal stand-in for ``AddressNormalizer``.

    ``docs`` is a list of dicts, one per cadastral candidate, with any of
    ``direccion``/``predial``/``manzana``/``lat``/``lon``/``comuna``/``n_predios``
    (each defaults to a harmless placeholder when omitted).

    ``scores`` is an optional ``list[list[float]]`` shaped ``(n_queries, n_docs)``
    used verbatim as the ``sim`` component for each (query, doc) pair - the
    same slot ``normalize_strict`` fuses into a confidence score. Without it,
    every query scores the first doc ``1.0`` and every other doc ``0.0``, which
    is enough for the common "single obvious candidate" test shape.

    The ``struct`` component is always computed for real, from the actual
    parse of the query and of ``doc["direccion"]``, because the rules under
    test (``_rule_violations``, ``_precision_level``) key off genuinely parsed
    fields, not a fabricated number.
    """

    def __init__(self, docs: list[dict] | None = None,
                scores: list[list[float]] | None = None,
                threshold: float = 0.5,
                weights: dict | None = None) -> None:
        docs = docs or [{}]
        self._direccion = np.array([d.get("direccion", "") for d in docs], dtype=object)
        self._predial = np.array([d.get("predial", "P1") for d in docs], dtype=object)
        self._manzana = np.array([d.get("manzana", "M1") for d in docs], dtype=object)
        self._comuna = np.array([d.get("comuna", "") for d in docs], dtype=object)
        self._lon = np.array([float(d.get("lon", -76.5)) for d in docs])
        self._lat = np.array([float(d.get("lat", 3.4)) for d in docs])
        self._n_predios = np.array([int(d.get("n_predios", 1)) for d in docs])
        self._manzana_nunique = np.array([int(d.get("manzana_nunique", 1)) for d in docs])
        self.threshold = threshold
        self.weights = dict(weights) if weights is not None else {
            "sim": 1.0, "fuzz": 0.0, "struct": 0.0, "geo": 0.0,
        }
        self._scores = scores

    def score_components(self, queries: list[str], k: int = 20) -> dict:
        n_docs = len(self._direccion)
        n_queries = len(queries)
        width = max(n_docs, 1)
        doc = np.full((n_queries, width), -1, dtype=np.int64)
        sim = np.zeros((n_queries, width), dtype=np.float64)
        struct = np.zeros((n_queries, width), dtype=np.float64)
        fuzz_arr = np.zeros((n_queries, width), dtype=np.float64)
        geo = np.zeros((n_queries, width), dtype=np.float64)
        for r, q in enumerate(queries):
            qp = parse_address(q)
            for c in range(n_docs):
                doc[r, c] = c
                if self._scores is not None:
                    sim[r, c] = self._scores[r][c]
                else:
                    sim[r, c] = 1.0 if c == 0 else 0.0
                cp = parse_address(str(self._direccion[c]))
                struct[r, c] = (
                    _structural_agreement(qp, cp) if qp.parse_ok and cp.parse_ok else 0.0
                )
        return {"doc": doc, "sim": sim, "fuzz": fuzz_arr, "struct": struct, "geo": geo}


def make_stub(docs: list[dict] | None = None, scores: list[list[float]] | None = None,
             **kwargs) -> StubNormalizer:
    """Factory: ``make_stub([{"direccion": "..."}], scores=[[0.9]])``."""
    return StubNormalizer(docs, scores=scores, **kwargs)


@pytest.fixture
def stub_normalizer():
    """Returns the ``make_stub`` factory so tests can build the stub they need."""
    return make_stub
