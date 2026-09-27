"""Unit tests for the structural-agreement scoring used by the reranker.

None of these load the model or the cadastral embeddings: :func:`_structural_agreement`
is pure and only needs two :class:`ParsedAddress` instances.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cali_address.inference import _structural_agreement  # noqa: E402
from cali_address.parser import ParsedAddress, parse_address  # noqa: E402


def test_identical_plain_parses_score_one():
    a = parse_address("KR 41 # 5-20")
    b = parse_address("KR 41 # 5-20")
    assert _structural_agreement(a, b) == 1.0


def test_bis_mismatch_lowers_the_score():
    a = parse_address("CL 13 BIS # 4-10")
    b = parse_address("CL 13 # 4-10")
    assert _structural_agreement(a, b) < 1.0


def test_cross_type_mismatch_lowers_the_score():
    a = parse_address("CL 5 # KR 38-10")
    b = parse_address("CL 5 # TV 38-10")
    assert _structural_agreement(a, b) < 1.0


def test_bis_false_false_does_not_change_score_for_plain_pairs():
    """Both sides plain (via_bis=cross_bis=False) must not inflate the denominator."""
    a = ParsedAddress(via_type="KR", via_number="41", cross_number="5", plate="20")
    b = ParsedAddress(via_type="KR", via_number="41", cross_number="5", plate="20")
    assert _structural_agreement(a, b) == 1.0


def test_bis_false_false_does_not_dilute_a_real_mismatch():
    """A via_type mismatch must keep its full weight; matching bis=False on both
    sides must not be counted as a free extra agreement that dilutes the score."""
    a = parse_address("CL 5 # 38-10")
    b = parse_address("KR 5 # 38-10")
    # via_type (1.0) mismatches; via_number (2.0), cross_number (2.0) and plate
    # (1.5) match. via_letters/via_quadrant/cross_letters/cross_quadrant/cross_type
    # are None on both sides and are skipped, same as via_bis/cross_bis (False on
    # both sides).
    assert _structural_agreement(a, b) == (2.0 + 2.0 + 1.5) / (1.0 + 2.0 + 2.0 + 1.5)
