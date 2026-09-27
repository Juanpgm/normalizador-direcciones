"""Tests for the split builder (pure functions; no context/ workbooks needed)."""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest

from cali_address.labels import address_key, select_quality

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("build_splits", os.path.join(ROOT, "scripts", "build_splits.py"))
bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs)


def _labeled(addrs, dataset="fasecolda", quality="gold", start=0):
    n = len(addrs)
    return pd.DataFrame({
        "raw_address": addrs,
        "dataset": dataset,
        "source_row": range(start, start + n),
        "has_gt": True,
        "gt_doc_id": [float(i) for i in range(n)],
        "gt_predial": [f"P{i}" for i in range(n)],
        "gt_manzana": [f"M{i}" for i in range(n)],
        "gt_comuna": "10",
        "gt_lat": 3.4,
        "gt_lon": -76.5,
        "gt_quality": quality,
    })


def _addrs(n, plate0=10):
    return [f"KR 26 # 73 - {plate0 + i}" for i in range(n)]


class TestAddKeys:
    def test_adds_key_and_drops_blank(self):
        df = _labeled(["KR 26 # 73 - 10", None, "", "  ", np.nan])
        out = bs.add_keys(df)
        assert out["raw_address"].tolist() == ["KR 26 # 73 - 10"]
        assert out["address_key"].tolist() == [address_key("KR 26 # 73 - 10")]

    def test_empty(self):
        assert len(bs.add_keys(_labeled([]))) == 0


class TestSelectFrozen:
    def test_unique_keys_and_excludes_dev(self):
        pool = bs.add_keys(_labeled(_addrs(50) + ["kr 26 # 73 - 10", "Carrera 26 No 73-11"]))
        dev = {address_key("KR 26 # 73 - 12")}
        out = bs.select_frozen(pool, dev_keys=dev, size=100, seed=1)
        assert out["address_key"].is_unique
        assert not (set(out["address_key"]) & dev)
        # 50 unique keys - 2 keys with conflicting GT (duplicated spellings) - 1 dev key
        assert len(out) == 47

    def test_size_larger_than_pool_returns_pool(self):
        pool = bs.add_keys(_labeled(_addrs(5)))
        assert len(bs.select_frozen(pool, set(), size=2000, seed=1)) == 5

    def test_size_zero(self):
        pool = bs.add_keys(_labeled(_addrs(5)))
        assert len(bs.select_frozen(pool, set(), size=0, seed=1)) == 0

    def test_deterministic_per_seed_and_differs_across_seeds(self):
        pool = bs.add_keys(_labeled(_addrs(200)))
        a = bs.select_frozen(pool, set(), 20, seed=7)["address_key"].tolist()
        b = bs.select_frozen(pool, set(), 20, seed=7)["address_key"].tolist()
        c = bs.select_frozen(pool, set(), 20, seed=8)["address_key"].tolist()
        assert a == b and a != c

    def test_only_rows_with_gt(self):
        df = _labeled(_addrs(10))
        df.loc[:4, "gt_doc_id"] = np.nan
        df.loc[:4, "has_gt"] = False
        out = bs.select_frozen(bs.add_keys(df), set(), 100, seed=1)
        assert len(out) == 5 and out["has_gt"].all()

    def test_keeps_noisy_rows(self):
        df = _labeled(_addrs(10), quality="noisy")
        assert len(bs.select_frozen(bs.add_keys(df), set(), 100, seed=1)) == 10

    def test_fasecolda_only(self):
        df = pd.concat([_labeled(_addrs(5)), _labeled(_addrs(5, 500), dataset="stickers")])
        out = bs.select_frozen(bs.add_keys(df), set(), 100, seed=1)
        assert set(out["dataset"]) == {"fasecolda"}


class TestBuildTrainPairs:
    def test_only_gold(self):
        df = pd.concat([_labeled(_addrs(3)), _labeled(_addrs(3, 100), quality="noisy"),
                        _labeled(_addrs(3, 200), quality="no_gt")], ignore_index=True)
        out = bs.build_train_pairs(bs.add_keys(df), exclude_keys=set())
        assert len(out) == 3 and set(out.columns) == set(bs.TRAIN_COLUMNS)

    def test_excludes_dev_and_frozen_keys(self):
        df = bs.add_keys(_labeled(_addrs(10)))
        ex = set(df["address_key"].iloc[:4])
        out = bs.build_train_pairs(df, ex)
        assert len(out) == 6 and not (set(out["address_key"]) & ex)

    def test_dedup_across_dates_keeps_first(self):
        a = _labeled(["KR 26 # 73 - 10"], dataset="inspecciones", start=0)
        b = _labeled(["kr 26 #73-10"], dataset="inspecciones", start=5)
        out = bs.build_train_pairs(bs.add_keys(pd.concat([a, b], ignore_index=True)), set())
        assert len(out) == 1

    def test_conflicting_gt_for_same_key_dropped(self):
        a = _labeled(["KR 26 # 73 - 10"], dataset="stickers")
        b = _labeled(["KR 26 # 73 - 10"], dataset="stickers", start=5)
        b["gt_doc_id"] = 99.0
        out = bs.build_train_pairs(bs.add_keys(pd.concat([a, b], ignore_index=True)), set())
        assert len(out) == 0

    def test_source_row_id_and_types(self):
        out = bs.build_train_pairs(bs.add_keys(_labeled(["KR 26 # 73 - 10"], start=7)), set())
        assert out["source_row_id"].iloc[0] == "fasecolda:7"
        assert out["gt_doc_id"].dtype.kind == "i"

    def test_empty_and_missing_gt_columns(self):
        assert len(bs.build_train_pairs(bs.add_keys(_labeled([])), set())) == 0
        df = bs.add_keys(_labeled(_addrs(3))).drop(columns=["gt_quality"])
        assert len(bs.build_train_pairs(df, set())) == 0

    def test_no_gold_dataset_produces_nothing(self):
        df = bs.add_keys(_labeled(_addrs(3), quality="noisy"))
        assert len(bs.build_train_pairs(df, set())) == 0


class TestBuildUnlabeled:
    def test_exactly_two_columns_no_personal_data(self):
        rud = pd.DataFrame({"raw_address": ["KR 1 # 2 - 3", "KR 1 # 2 - 3 ", "CL 4 # 5 - 6"],
                            "dataset": "rud", "nombre": ["x", "y", "z"], "cedula": [1, 2, 3]})
        out = bs.build_unlabeled(bs.add_keys(rud), set())
        assert list(out.columns) == ["raw_address", "dataset"]
        assert len(out) == 2

    def test_excludes_dev_frozen_keys(self):
        df = pd.DataFrame({"raw_address": ["KR 1 # 2 - 3", "CL 4 # 5 - 6"], "dataset": "acciones"})
        ex = {address_key("KR 1 # 2 - 3")}
        assert bs.build_unlabeled(bs.add_keys(df), ex)["raw_address"].tolist() == ["CL 4 # 5 - 6"]

    def test_empty(self):
        out = bs.build_unlabeled(bs.add_keys(pd.DataFrame({"raw_address": [], "dataset": []})), set())
        assert list(out.columns) == ["raw_address", "dataset"] and len(out) == 0


class TestLeakage:
    def test_overlap_matrix_zero_when_disjoint(self):
        m = bs.overlap_matrix({"train_pairs": {"a"}, "dev": {"b"}, "frozen_test": {"c"}})
        assert all(m[a][b] == 0 for a in m for b in m if a != b)

    def test_detects_shared_keys(self):
        m = bs.overlap_matrix({"train_pairs": {"a", "b"}, "dev": {"b"}, "frozen_test": {"c"}})
        assert m["train_pairs"]["dev"] == 1 and m["dev"]["train_pairs"] == 1

    def test_guard_raises_on_leak(self):
        with pytest.raises(bs.LeakageError):
            bs.assert_no_leakage({"train_pairs": {"a"}, "dev": {"a"}, "frozen_test": set()})

    def test_guard_passes_when_clean(self):
        bs.assert_no_leakage({"train_pairs": {"a"}, "dev": {"b"}, "frozen_test": {"c"}})

    def test_end_to_end_pipeline_is_disjoint(self):
        fase = bs.add_keys(_labeled(_addrs(300)))
        dev = bs.add_keys(_labeled(_addrs(20, 10)))  # overlaps fase keys
        dev_keys = set(dev["address_key"])
        frozen = bs.select_frozen(fase, dev_keys, 50, seed=3)
        train = bs.build_train_pairs(fase, dev_keys | set(frozen["address_key"]))
        bs.assert_no_leakage({"train_pairs": set(train["address_key"]), "dev": dev_keys,
                              "frozen_test": set(frozen["address_key"])})
        assert len(train) == 300 - 20 - 50


class TestReport:
    def test_report_structure(self):
        frozen = bs.add_keys(_labeled(_addrs(4), quality="gold"))
        dev = bs.add_keys(_labeled(_addrs(2, 100), quality="noisy"))
        train = bs.add_keys(_labeled(_addrs(3, 200)))
        unl = pd.DataFrame({"raw_address": ["x"], "dataset": ["rud"]})
        universe = {"fasecolda": _labeled(_addrs(10), quality="gold")}
        rep = bs.build_report(frozen, dev, train, unl, universe, seed=42)
        json.dumps(rep)
        assert rep["counts"]["frozen_test"]["total"] == 4
        assert rep["counts"]["dev"]["by_gt_quality"] == {"noisy": 2}
        assert rep["gold_ratio_by_dataset"]["fasecolda"] == 1.0
        assert set(rep["overlap"]) == {"train_pairs", "dev", "frozen_test"}


class TestSelectQuality:
    def test_gold_only(self):
        df = _labeled(_addrs(3))
        df.loc[0, "gt_quality"] = "noisy"
        assert len(select_quality(df, gold_only=True)) == 2
        assert len(select_quality(df, gold_only=False)) == 3

    def test_gold_only_without_column_raises(self):
        with pytest.raises(KeyError):
            select_quality(_labeled(_addrs(2)).drop(columns="gt_quality"), gold_only=True)
