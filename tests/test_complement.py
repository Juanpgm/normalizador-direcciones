"""Tests for ``filter_complement``: cadastral units must not leak into the output.

The cadastral address is only allowed to keep a complement chunk (AP 101, TO 2,
...) when the *input* address mentions that complement kind.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cali_address.parser import COMPLEMENT_CANON, filter_complement, parse_address  # noqa: E402


def _input_complement(text: str | None) -> str | None:
    return parse_address(text).complement


# ---------------------------------------------------------------------------
# Helper: core behavior
# ---------------------------------------------------------------------------
def test_no_input_complement_drops_the_cadastral_unit():
    assert filter_complement("KR 29 A # 36 - 87 AP 101", None) == "KR 29 A # 36 - 87"


def test_input_mentions_the_kind_keeps_the_chunk():
    comp = _input_complement("Cra 29A # 36-87 apto 101")
    assert filter_complement("KR 29 A # 36 - 87 AP 101", comp) == "KR 29 A # 36 - 87 AP 101"


@pytest.mark.parametrize("alias", ["APTO", "APARTAMENTO", "APT", "apto.", "Apto.", "aparTamento"])
def test_aliases_of_the_kind_count_as_a_mention(alias):
    comp = _input_complement(f"Cra 29A # 36-87 {alias} 101")
    assert filter_complement("KR 29 A # 36 - 87 AP 101", comp) == "KR 29 A # 36 - 87 AP 101"


def test_partial_mention_keeps_only_the_mentioned_kind():
    comp = _input_complement("Cra 29A # 36-87 torre 2")
    assert filter_complement("KR 29 A # 36 - 87 TO 2 AP 101", comp) == "KR 29 A # 36 - 87 TO 2"


def test_multiple_chunks_keep_only_the_mentioned_ones():
    comp = _input_complement("Cra 66 # 33B-35 casa 7")
    assert filter_complement("KR 66 # 33 B - 35 BLQ G CA 7", comp) == "KR 66 # 33 B - 35 CA 7"


def test_multiple_chunks_all_dropped_without_mention():
    assert filter_complement("KR 66 # 33 B - 35 BLQ G CA 7", None) == "KR 66 # 33 B - 35"


def test_multiple_chunks_all_kept_when_all_mentioned():
    comp = _input_complement("Cra 66 # 33B-35 bloque g casa 7")
    assert filter_complement("KR 66 # 33 B - 35 BLQ G CA 7", comp) == "KR 66 # 33 B - 35 BLQ G CA 7"


# ---------------------------------------------------------------------------
# Leading chunk with an unregistered kind (BO 002003 LT 0031)
# ---------------------------------------------------------------------------
CADASTRAL_BO = "CL 6 H OESTE # KR 50 C - BO 002003 LT 0031"


def test_unregistered_leading_chunk_dropped_without_mention():
    assert filter_complement(CADASTRAL_BO, None) == "CL 6 H OESTE # KR 50 C -"


def test_unregistered_leading_chunk_kept_when_its_key_is_mentioned():
    out = filter_complement(CADASTRAL_BO, "BO 2003")
    assert out == "CL 6 H OESTE # KR 50 C - BO 002003"


def test_unregistered_leading_chunk_dropped_when_its_value_differs():
    assert filter_complement(CADASTRAL_BO, "BO 12") == "CL 6 H OESTE # KR 50 C -"


def test_unregistered_leading_chunk_and_registered_chunk_both_kept():
    out = filter_complement(CADASTRAL_BO, "BO 002003 LOTE 31")
    assert out == CADASTRAL_BO


def test_registered_chunk_kept_while_unregistered_leading_chunk_dropped():
    out = filter_complement(CADASTRAL_BO, "LOTE 31")
    assert out == "CL 6 H OESTE # KR 50 C - LT 0031"


# ---------------------------------------------------------------------------
# Empty plate / complement-only remainders
# ---------------------------------------------------------------------------
def test_empty_plate_with_complement_only_is_dropped():
    assert filter_complement("KR 72 # - LT 6", None) == "KR 72 # -"


def test_empty_plate_with_complement_kept_when_mentioned():
    assert filter_complement("KR 72 # - LT 6", "LOTE 6") == "KR 72 # - LT 6"


# ---------------------------------------------------------------------------
# Idempotence / untouched inputs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "address",
    ["KR 29 A # 36 - 87", "CL 13 B # 74 - 13", "KR 17 F # 30 - 02", "CL 6 H OESTE # KR 50 C - 11"],
)
def test_address_without_complement_is_unchanged(address):
    assert filter_complement(address, None) == address
    assert filter_complement(address, "AP 101") == address


def test_filter_is_idempotent():
    once = filter_complement("KR 29 A # 36 - 87 TO 2 AP 101", "TORRE 2")
    assert filter_complement(once, "TORRE 2") == once


def test_zero_padded_plate_is_preserved():
    assert filter_complement("KR 17 F # 30 - 02 GA 1", None) == "KR 17 F # 30 - 02"


def test_zero_padded_plate_preserved_with_kept_complement():
    assert filter_complement("KR 17 F # 30 - 02 GA 1", "GARAJE 1") == "KR 17 F # 30 - 02 GA 1"


# ---------------------------------------------------------------------------
# Odd inputs for the input complement
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("empty", [None, "", "   ", "\t\n"])
def test_empty_or_blank_input_complement_yields_base_only(empty):
    assert filter_complement("KR 29 A # 36 - 87 AP 101", empty) == "KR 29 A # 36 - 87"


@pytest.mark.parametrize("raw", ["apto 101", "APTO 101", "Apto. 101", "  apto   101 ", "APTO. 101"])
def test_case_and_dots_in_the_raw_input_complement_are_tolerated(raw):
    # Passed directly (not through parse_address): normalization happens in the helper.
    assert filter_complement("KR 29 A # 36 - 87 AP 101", raw) == "KR 29 A # 36 - 87 AP 101"


def test_accented_input_complement_is_tolerated():
    assert filter_complement("CL 5 # 4 - 03 UR VILLA", "urbanización villa") == "CL 5 # 4 - 03 UR VILLA"


def test_same_kind_with_a_different_value_drops_the_chunk():
    # Kind AND value must match: input `apto 505` does not license catastro `AP 101`.
    comp = _input_complement("Cra 29A # 36-87 apto 505")
    assert filter_complement("KR 29 A # 36 - 87 AP 101", comp) == "KR 29 A # 36 - 87"


def test_leading_zeros_are_equivalent_in_values():
    assert filter_complement("KR 72 # - LT 0031", "LOTE 31") == "KR 72 # - LT 0031"
    assert filter_complement("KR 72 # - LT 31", "LOTE 0031") == "KR 72 # - LT 31"
    assert filter_complement("KR 72 # - LT 0031", "LOTE 310") == "KR 72 # -"


def test_all_zero_value_is_not_erased_by_zero_stripping():
    assert filter_complement("KR 72 # - LT 0", "LOTE 00") == "KR 72 # - LT 0"
    assert filter_complement("KR 72 # - LT 0", "LOTE 1") == "KR 72 # -"


def test_kind_mentioned_without_a_value_drops_the_chunk():
    comp = _input_complement("Cra 29A # 36-87 apto")
    assert comp == "AP"
    assert filter_complement("KR 29 A # 36 - 87 AP 101", comp) == "KR 29 A # 36 - 87"


def test_cadastral_chunk_without_a_value_is_dropped_even_if_input_has_the_kind():
    assert filter_complement("KR 29 A # 36 - 87 AP", "AP 101") == "KR 29 A # 36 - 87"


def test_repeated_kind_in_the_input_matches_any_of_its_values():
    comp = _input_complement("Cra 29A # 36-87 apto 101 apto 202")
    assert filter_complement("KR 29 A # 36 - 87 AP 202", comp) == "KR 29 A # 36 - 87 AP 202"
    assert filter_complement("KR 29 A # 36 - 87 AP 101", comp) == "KR 29 A # 36 - 87 AP 101"
    assert filter_complement("KR 29 A # 36 - 87 AP 303", comp) == "KR 29 A # 36 - 87"


def test_multi_token_value_must_match_completely():
    assert filter_complement("KR 66 # 33 B - 35 BLQ G", "BLOQUE G") == "KR 66 # 33 B - 35 BLQ G"
    assert filter_complement("KR 66 # 33 B - 35 BLQ G", "BLOQUE H") == "KR 66 # 33 B - 35"
    assert filter_complement("KR 66 # 33 B - 35 ED LAS PALMAS", "EDIFICIO LAS") == "KR 66 # 33 B - 35"
    assert (
        filter_complement("KR 66 # 33 B - 35 ED LAS PALMAS", "edificio las palmas")
        == "KR 66 # 33 B - 35 ED LAS PALMAS"
    )


def test_value_punctuation_and_case_are_ignored():
    assert filter_complement("KR 66 # 33 B - 35 AP 101 A", "apto. 101-a") == "KR 66 # 33 B - 35 AP 101 A"


def test_input_mentioning_a_different_kind_drops_the_chunk():
    comp = _input_complement("Cra 29A # 36-87 oficina 3")
    assert filter_complement("KR 29 A # 36 - 87 AP 101", comp) == "KR 29 A # 36 - 87"


# ---------------------------------------------------------------------------
# Unparseable / degenerate cadastral input: never raise
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["", "   ", "???", "SIN DIRECCION", "12345"])
def test_unparseable_cadastral_address_is_returned_unchanged(bad):
    assert filter_complement(bad, None) == bad
    assert filter_complement(bad, "AP 1") == bad


def test_none_cadastral_address_does_not_raise():
    assert filter_complement(None, None) in ("", None)


# ---------------------------------------------------------------------------
# Registered kinds are the ones used to split chunks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", sorted(set(COMPLEMENT_CANON.values())))
def test_every_registered_kind_is_dropped_and_kept_symmetrically(kind):
    cadastral = f"KR 10 # 20 - 30 {kind} 5"
    assert filter_complement(cadastral, None) == "KR 10 # 20 - 30"
    assert filter_complement(cadastral, f"{kind} 5") == cadastral
    assert filter_complement(cadastral, kind) == "KR 10 # 20 - 30"


# ---------------------------------------------------------------------------
# Integration: _build_row (OK rows)
# ---------------------------------------------------------------------------
def _build(stub_normalizer, cadastral: str, raw: str) -> dict:
    from cali_address.service import _build_row

    normalizer = stub_normalizer([{"direccion": cadastral}])
    parsed = parse_address(raw)
    assert parsed.parse_ok
    return _build_row(
        normalizer, None, raw, parsed, None,
        np.array([0]), np.array([1.0]), np.array([0.99]),
        0.0, 0, {}, 0.0, 0.0, False,
    )


def test_build_row_drops_unit_not_mentioned_by_the_input(stub_normalizer):
    row = _build(stub_normalizer, "KR 29 A # 36 - 87 AP 101", "Cra 29A # 36-87")
    assert row["estado"] == "OK"
    assert row["direccion_normalizada"] == "KR 29 A # 36 - 87"


def test_build_row_keeps_unit_mentioned_by_the_input(stub_normalizer):
    row = _build(stub_normalizer, "KR 29 A # 36 - 87 AP 101", "Cra 29A # 36-87 apto 101")
    assert row["estado"] == "OK"
    assert row["direccion_normalizada"] == "KR 29 A # 36 - 87 AP 101"


def test_build_row_drops_unit_with_a_different_value(stub_normalizer):
    row = _build(stub_normalizer, "KR 29 A # 36 - 87 AP 101", "Cra 29A # 36-87 apto 505")
    assert row["estado"] == "OK"
    assert row["direccion_normalizada"] == "KR 29 A # 36 - 87"


def test_build_row_partial_mention_keeps_only_the_tower(stub_normalizer):
    row = _build(stub_normalizer, "KR 29 A # 36 - 87 TO 2 AP 101", "Cra 29A # 36-87 torre 2")
    assert row["direccion_normalizada"] == "KR 29 A # 36 - 87 TO 2"
