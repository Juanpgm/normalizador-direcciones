"""Tests for gold/noisy label quality and the split-disjointness address key."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cali_address.labels import address_key, label_quality, plate_agrees

DOCS = pd.DataFrame({
    "doc_id": [0, 1, 2, 3, 4],
    "direccion": [
        "KR 26 H 1 # 73 - 10",   # 0
        "KR 26 H 1 # 73 - 12",   # 1  neighbour lot, same street pair
        "CL 73 # 26 H 1 - 10",   # 2  other face of the corner (via/cross swapped)
        "CL 12 # 48 BIS - 31",   # 3
        "KR 27 # - ",            # 4  cadastre record with no plate
    ],
})


class TestPlateAgrees:
    def test_same_plate(self):
        assert plate_agrees("KR 26 H 1 # 73 - 10", "KR 26 H 1 # 73 - 10") is True

    def test_zero_padding_ignored(self):
        assert plate_agrees("KR 26 # 73 - 5", "KR 26 # 73 - 05") is True

    def test_different_plate(self):
        assert plate_agrees("KR 26 H 1 # 73 - 10", "KR 26 H 1 # 73 - 12") is False

    def test_complement_does_not_matter(self):
        assert plate_agrees("cra 26h1 no 73-10 apto 5", "KR 26 H 1 # 73 - 10") is True

    @pytest.mark.parametrize("bad", [None, "", "   ", float("nan"), "abc", "KR 27 # - "])
    def test_missing_or_unparseable_input_is_none(self, bad):
        assert plate_agrees(bad, "KR 26 H 1 # 73 - 10") is None
        assert plate_agrees("KR 26 H 1 # 73 - 10", bad) is None

    def test_both_missing_is_none(self):
        assert plate_agrees(None, None) is None


class TestLabelQuality:
    def _run(self, raws, gt_ids):
        df = pd.DataFrame({"raw_address": raws, "gt_doc_id": gt_ids})
        return label_quality(df, DOCS)["gt_quality"].tolist()

    def test_same_address_is_gold(self):
        assert self._run(["KR 26 H 1 # 73 - 10"], [0.0]) == ["gold"]

    def test_variant_spelling_is_gold(self):
        assert self._run(["Cra 26h1 No. 73-10 Apto 301, Cali"], [0]) == ["gold"]

    def test_neighbour_lot_is_noisy(self):
        assert self._run(["KR 26 H 1 # 73 - 10"], [1]) == ["noisy"]

    def test_other_face_of_corner_is_noisy(self):
        assert self._run(["KR 26 H 1 # 73 - 10"], [2]) == ["noisy"]

    def test_via_letters_mismatch_is_noisy(self):
        assert self._run(["KR 26 G 1 # 73 - 10"], [0]) == ["noisy"]

    def test_cross_number_mismatch_is_noisy(self):
        assert self._run(["KR 26 H 1 # 74 - 10"], [0]) == ["noisy"]

    def test_via_type_mismatch_is_noisy(self):
        assert self._run(["CL 26 H 1 # 73 - 10"], [0]) == ["noisy"]

    def test_no_gt_nan_and_none(self):
        assert self._run(["KR 26 H 1 # 73 - 10"] * 2, [np.nan, None]) == ["no_gt", "no_gt"]

    def test_gt_id_not_in_docs_is_no_gt(self):
        assert self._run(["KR 26 H 1 # 73 - 10"], [999]) == ["no_gt"]

    def test_missing_plate_in_input_is_noisy(self):
        assert self._run(["KR 26 H 1 # 73"], [0]) == ["noisy"]

    def test_missing_plate_in_gt_doc_is_noisy(self):
        assert self._run(["KR 27 # 5 - 3"], [4]) == ["noisy"]

    @pytest.mark.parametrize("bad", [None, "", "   ", np.nan, "???", 12345])
    def test_malformed_input_with_gt_is_noisy(self, bad):
        assert self._run([bad], [0]) == ["noisy"]

    def test_missing_gt_column_all_no_gt(self):
        df = pd.DataFrame({"raw_address": ["KR 26 H 1 # 73 - 10"]})
        assert label_quality(df, DOCS)["gt_quality"].tolist() == ["no_gt"]

    def test_empty_frame(self):
        out = label_quality(pd.DataFrame({"raw_address": [], "gt_doc_id": []}), DOCS)
        assert len(out) == 0 and "gt_quality" in out.columns

    def test_input_frame_not_mutated(self):
        df = pd.DataFrame({"raw_address": ["KR 26 H 1 # 73 - 10"], "gt_doc_id": [0.0]})
        label_quality(df, DOCS)
        assert "gt_quality" not in df.columns

    def test_preserves_index_and_order(self):
        df = pd.DataFrame({"raw_address": ["KR 26 H 1 # 73 - 10", "CL 12 # 48 BIS - 31"],
                           "gt_doc_id": [3, 3]}, index=[10, 20])
        out = label_quality(df, DOCS)
        assert out.index.tolist() == [10, 20]
        assert out["gt_quality"].tolist() == ["noisy", "gold"]

    def test_docs_without_doc_id_column_uses_position(self):
        docs = DOCS.drop(columns="doc_id")
        df = pd.DataFrame({"raw_address": ["KR 26 H 1 # 73 - 10"], "gt_doc_id": [0]})
        assert label_quality(df, docs)["gt_quality"].tolist() == ["gold"]


class TestAddressKey:
    def test_case_punctuation_spacing_insensitive(self):
        assert address_key("Kr 26 H 1 # 73-10") == address_key("KR  26H1  #73 - 10")

    def test_spelling_variants_collapse(self):
        assert address_key("Carrera 26 H 1 No 73 - 10") == address_key("KR 26 H 1 # 73 - 10")

    def test_accents_folded(self):
        assert address_key("Cra 26 # 73-10 Bogotá") == address_key("CRA 26 # 73-10 BOGOTA")

    def test_different_plate_different_key(self):
        assert address_key("KR 26 # 73 - 10") != address_key("KR 26 # 73 - 12")

    def test_unparseable_falls_back_to_raw_key(self):
        assert address_key("Barrio  El Peñón!") == address_key("barrio el penon")
        assert address_key("Barrio El Peñón") != ""

    @pytest.mark.parametrize("bad", [None, "", "   ", float("nan"), "nan", "None"])
    def test_blank_is_empty_key(self, bad):
        assert address_key(bad) == ""

    def test_stable_and_deterministic(self):
        assert address_key("KR 26 # 73 - 10") == address_key("KR 26 # 73 - 10")
