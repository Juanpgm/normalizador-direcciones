"""Unit tests for the ``normalizar.py`` CLI argument parser.

Only ``build_parser``/``parse_args`` behaviour is covered here: no model
loading, no gazetteer, no I/O.
"""

from __future__ import annotations

import os
import sys

import pytest

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from normalizar import build_parser  # noqa: E402


def _parse(argv):
    return build_parser().parse_args(argv)


def test_plate_tolerance_above_2_is_rejected():
    with pytest.raises(SystemExit):
        _parse(["addr", "--plate-tolerance", "3"])


def test_plate_tolerance_of_2_is_accepted():
    args = _parse(["addr", "--plate-tolerance", "2"])
    assert args.plate_tolerance == 2


def test_plate_tolerance_negative_is_rejected():
    with pytest.raises(SystemExit):
        _parse(["addr", "--plate-tolerance", "-1"])


def test_plate_tolerance_default_is_zero():
    args = _parse(["addr"])
    assert args.plate_tolerance == 0


def test_ambiguity_delta_default_is_0_02():
    assert _parse(["addr"]).ambiguity_delta == 0.02


def test_ambiguity_delta_zero_disables():
    assert _parse(["addr", "--ambiguity-delta", "0"]).ambiguity_delta == 0.0


def test_ambiguity_delta_accepts_0_02_and_rejects_out_of_range():
    assert _parse(["addr", "--ambiguity-delta", "0.02"]).ambiguity_delta == 0.02
    for bad in ("0.5", "-0.01", "abc"):
        with pytest.raises(SystemExit):
            _parse(["addr", "--ambiguity-delta", bad])
