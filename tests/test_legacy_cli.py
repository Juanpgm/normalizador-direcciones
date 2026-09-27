"""The pre-existing ``scripts/normalizar.py`` CLI must keep working unchanged."""

from __future__ import annotations

import importlib.util
import json
import os

import pandas as pd
import pytest

from dataset_helpers import ADDR_OK, stub, synth_frame, write_csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def legacy(monkeypatch):
    spec = importlib.util.spec_from_file_location("legacy_normalizar", os.path.join(ROOT, "scripts", "normalizar.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "AddressNormalizer", lambda *a, **k: stub())
    monkeypatch.setattr(module, "load_gazetteer", lambda *a, **k: None)
    return module


def test_legacy_parser_still_exposes_every_flag(legacy):
    args = legacy.build_parser().parse_args([
        "CL 5 # 38-20", "--input", "x.csv", "--column", "direccion", "--sheet", "s", "--header", "3",
        "--output", "o.csv", "--keep-columns", "all", "--json", "--min-struct", "0.7",
        "--plate-tolerance", "1", "--ambiguity-delta", "0.05", "--no-gazetteer", "--barrio-buffer", "5",
        "--zone-buffer", "6", "--no-gate-escalate", "--device", "cpu",
    ])
    assert args.column == "direccion" and args.header == 3 and args.no_gazetteer


def test_legacy_positional_addresses_json(legacy, capsys):
    assert legacy.main([ADDR_OK, "hola que tal", "--json", "--no-gazetteer"]) == 0
    records = json.loads(capsys.readouterr().out)
    assert [r["estado"] for r in records] == ["OK", "NO_PARSEABLE"]


def test_legacy_csv_roundtrip_with_keep_columns(legacy, tmp_path):
    src = write_csv(tmp_path / "in.csv", synth_frame(4))
    out = tmp_path / "out.csv"
    assert legacy.main(["--input", src, "--output", str(out), "--keep-columns", "id,barrio"]) == 0
    back = pd.read_csv(out, dtype=str, encoding="utf-8-sig")
    assert list(back.columns)[:2] == ["id", "barrio"] and len(back) == 4


def test_legacy_ambiguous_column_still_exits_2(legacy, tmp_path, capsys):
    src = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_OK], "domicilio": [ADDR_OK]}))
    assert legacy.main(["--input", src]) == 2
    assert "--column" in capsys.readouterr().err
