"""`normalize INPUT` without -o writes <stem>_normalizado<ext> next to the input."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from dataset_helpers import stub, synth_frame, write_csv

from cali_address.cli import default_output_path, main


def run(argv):
    return main(argv, normalizer_factory=lambda a, d: stub(), gazetteer_loader=lambda *a, **k: None)


# ------------------------------------------------------------------ pure naming
@pytest.mark.parametrize(
    "name, fmt, expected",
    [
        ("data.csv", "csv", "data_normalizado.csv"),
        ("my file (1).csv", "csv", "my file (1)_normalizado.csv"),
        ("a.b.c.csv", "csv", "a.b.c_normalizado.csv"),
        ("noext", "csv", "noext_normalizado.csv"),
        ("DATA.XLSX", "xlsx", "DATA_normalizado.xlsx"),
        ("x_normalizado.csv", "csv", "x_normalizado_normalizado.csv"),
        ("old.xls", "xls", "old_normalizado.xlsx"),
        ("book.xlsm", "xlsx", "book_normalizado.xlsx"),
        ("list.txt", "txt", "list_normalizado.csv"),
        ("t.tsv", "tsv", "t_normalizado.tsv"),
        ("p.pq", "parquet", "p_normalizado.parquet"),
        ("j.json", "json", "j_normalizado.json"),
        ("l.ndjson", "jsonl", "l_normalizado.jsonl"),
        ("g.geojson", "geojson", "g_normalizado.geojson"),
        ("s.shp", "shp", "s_normalizado.geojson"),
        ("s.zip", "shp", "s_normalizado.geojson"),
        ("s.gpkg", "gpkg", "s_normalizado.geojson"),
    ],
)
def test_default_output_path(tmp_path, name, fmt, expected):
    got = default_output_path(str(tmp_path / name), fmt)
    assert got == str(tmp_path / expected)
    assert os.path.normcase(got) != os.path.normcase(str(tmp_path / name))


def test_dotted_directory_is_not_mistaken_for_an_extension(tmp_path):
    assert default_output_path(str(tmp_path / "v1.2" / "data"), "csv") == str(tmp_path / "v1.2" / "data_normalizado.csv")


# ------------------------------------------------------------------ through the CLI
def test_zero_flag_csv(tmp_path, capsys):
    src = write_csv(tmp_path / "direcciones.csv", synth_frame(6))
    before = open(src, "rb").read()
    assert run(["normalize", src]) == 0
    out = tmp_path / "direcciones_normalizado.csv"
    assert out.exists() and len(pd.read_csv(out, encoding="utf-8-sig")) == 6
    assert open(src, "rb").read() == before  # the input is untouched
    err = capsys.readouterr().err
    assert str(out) in err and "6 rows" in err and "OK=" in err


def test_zero_flag_xlsx_and_uppercase_extension(tmp_path):
    frame = synth_frame(5)
    src = tmp_path / "DATOS.XLSX"
    frame.to_excel(src, index=False)
    assert run(["normalize", str(src)]) == 0
    out = tmp_path / "DATOS_normalizado.xlsx"
    assert out.exists() and len(pd.read_excel(out)) == 5


def test_txt_input_produces_csv(tmp_path):
    src = tmp_path / "lista.txt"
    src.write_text("CALLE 5 # 10-20\nCARRERA 4 # 3-1\n", encoding="utf-8")
    assert run(["normalize", str(src)]) == 0
    assert (tmp_path / "lista_normalizado.csv").exists()


def test_no_extension_needs_format_flag(tmp_path, capsys):
    src = write_csv(tmp_path / "sinext", synth_frame(3))
    assert run(["normalize", src]) == 2
    assert not list(tmp_path.glob("*_normalizado*"))
    assert run(["normalize", src, "--format", "csv"]) == 0
    assert (tmp_path / "sinext_normalizado.csv").exists()


def test_already_normalized_name_never_overwrites_the_input(tmp_path):
    src = write_csv(tmp_path / "x_normalizado.csv", synth_frame(3))
    before = open(src, "rb").read()
    assert run(["normalize", src]) == 0
    assert open(src, "rb").read() == before
    assert (tmp_path / "x_normalizado_normalizado.csv").exists()


def test_existing_output_is_overwritten_atomically(tmp_path):
    src = write_csv(tmp_path / "d.csv", synth_frame(4))
    out = tmp_path / "d_normalizado.csv"
    out.write_text("stale", encoding="utf-8")
    assert run(["normalize", src]) == 0
    assert len(pd.read_csv(out, encoding="utf-8-sig")) == 4
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".partial")] == []


def test_explicit_output_still_wins(tmp_path):
    src = write_csv(tmp_path / "d.csv", synth_frame(3))
    assert run(["normalize", src, "-o", str(tmp_path / "custom.csv")]) == 0
    assert (tmp_path / "custom.csv").exists() and not (tmp_path / "d_normalizado.csv").exists()


@pytest.mark.parametrize("source", ["https://example.org/data.csv", "sqlite:///x.db", "postgresql://u:p@h/db"])
def test_url_and_sql_inputs_require_output(tmp_path, capsys, monkeypatch, source):
    monkeypatch.chdir(tmp_path)
    assert run(["normalize", source, "--table", "t"]) == 2
    err = capsys.readouterr().err
    assert "-o" in err and err.strip().count("\n") == 0 and "u:p" not in err
    assert list(tmp_path.iterdir()) == []


def test_output_path_is_a_directory_exit_3(tmp_path, capsys):
    src = write_csv(tmp_path / "d.csv", synth_frame(3))
    (tmp_path / "d_normalizado.csv").mkdir()
    assert run(["normalize", src]) == 3
    assert "Traceback" not in capsys.readouterr().err


def test_unwritable_output_location_exit_3(tmp_path, monkeypatch, capsys):
    from cali_address.io import SinkWriteError
    import cali_address.cli as cli

    src = write_csv(tmp_path / "d.csv", synth_frame(3))

    def deny(path, fmt=None, **opts):
        raise SinkWriteError(f"cannot write to {path}: [Errno 13] Permission denied")

    monkeypatch.setattr(cli, "open_sink", deny)
    assert run(["normalize", src]) == 3
    assert "Permission denied" in capsys.readouterr().err


def test_dry_run_still_writes_nothing(tmp_path):
    src = write_csv(tmp_path / "d.csv", synth_frame(4))
    assert run(["normalize", src, "--dry-run", "2"]) == 0
    assert not (tmp_path / "d_normalizado.csv").exists()


def test_bad_extension_error_is_a_usage_error(tmp_path, capsys):
    src = tmp_path / "data.weird"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    assert run(["normalize", str(src)]) == 2
