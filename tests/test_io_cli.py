"""`python -m cali_address` CLI: end-to-end with the stub normalizer, exit codes, error UX."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys

import openpyxl
import pandas as pd
import pytest

from dataset_helpers import ADDR_OK, stub, synth_frame, write_csv

from cali_address.cli import main
from cali_address.service import OUTPUT_COLUMNS

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def run(argv, *, normalizer=None):
    """Run main() with an injected normalizer factory and no gazetteer."""
    made = []

    def factory(artifacts_dir, device):
        made.append((artifacts_dir, device))
        return normalizer or stub()

    code = main(argv, normalizer_factory=factory, gazetteer_loader=lambda *a, **k: None)
    return code, made


def _read_out(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------
def test_normalize_csv_end_to_end(tmp_path, capsys):
    src = write_csv(tmp_path / "in.csv", synth_frame(7))
    out = tmp_path / "out.csv"
    code, made = run(["normalize", src, "-o", str(out)])
    assert code == 0 and made
    back = _read_out(out)
    assert len(back) == 7
    assert list(back.columns) == ["id", "direccion", "barrio", "extra_unknown"] + [c for c in OUTPUT_COLUMNS if c != "direccion_entrada"]
    assert back["id"].tolist() == [f"{i:03d}" for i in range(7)]
    err = capsys.readouterr().err
    assert "7 rows" in err or "rows: 7" in err


@pytest.mark.parametrize("sep", [";", "\t", "|"])
def test_odd_delimiters_autodetected(tmp_path, sep):
    src = write_csv(tmp_path / "in.csv", synth_frame(4), sep=sep)
    out = tmp_path / "out.csv"
    assert run(["normalize", src, "-o", str(out)])[0] == 0
    assert len(_read_out(out)) == 4


def test_explicit_format_delimiter_encoding(tmp_path):
    p = tmp_path / "in.dat"
    p.write_bytes("id;direccion\n1;CL 5 # 38 - 20\n2;Peñón\n".encode("cp1252"))
    out = tmp_path / "out.csv"
    code, _ = run(["normalize", str(p), "-o", str(out), "--format", "csv", "--delimiter", ";",
                   "--encoding", "cp1252"])
    assert code == 0 and len(_read_out(out)) == 2


def test_excel_sheet_and_header_row(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "datos"
    ws.append(["Reporte"]); ws.append([None]); ws.append(["id", "direccion"])
    ws.append(["1", ADDR_OK]); ws.append(["2", "hola que tal"])
    wb.create_sheet("otra").append(["x"])
    p = tmp_path / "in.xlsx"
    wb.save(p)
    out = tmp_path / "out.csv"
    code, _ = run(["normalize", str(p), "-o", str(out), "--sheet", "datos", "--header-row", "2"])
    back = _read_out(out)
    assert code == 0 and back["estado"].tolist() == ["OK", "NO_PARSEABLE"]


def test_address_parts_and_parts_sep(tmp_path):
    frame = pd.DataFrame({"via": ["CL 5", "CL 6"], "num": ["# 38 - 20", "# 1 - 1"]})
    src = write_csv(tmp_path / "in.csv", frame)
    out = tmp_path / "out.csv"
    code, _ = run(["normalize", src, "-o", str(out), "--address-parts", "via,num", "--parts-sep", " "])
    back = _read_out(out)
    assert code == 0 and back["direccion_entrada"].tolist() == ["CL 5 # 38 - 20", "CL 6 # 1 - 1"]


def test_id_and_municipality_columns(tmp_path):
    frame = pd.DataFrame({"id": ["a", "b"], "direccion": [ADDR_OK, ADDR_OK], "mun": ["Cali", "Bogotá"]})
    src = write_csv(tmp_path / "in.csv", frame)
    out = tmp_path / "out.csv"
    code, _ = run(["normalize", src, "-o", str(out), "--address-col", "direccion", "--id-col", "id",
                   "--municipality-col", "mun"])
    assert code == 0 and _read_out(out)["estado"].tolist() == ["OK", "FUERA_DE_AREA"]


def test_sql_source_with_table_and_view(tmp_path):
    db = tmp_path / "a.db"
    con = sqlite3.connect(db)
    synth_frame(5).to_sql("dir", con, index=False)
    con.close()
    url = f"sqlite:///{db.as_posix()}"
    out = tmp_path / "out.csv"
    assert run(["normalize", url, "-o", str(out), "--table", "dir"])[0] == 0
    assert len(_read_out(out)) == 5
    con = sqlite3.connect(db)
    con.execute("CREATE VIEW dir_late AS SELECT id, direccion FROM dir WHERE id > '001'")
    con.commit()
    con.close()
    out2 = tmp_path / "out2.csv"
    assert run(["normalize", url, "-o", str(out2), "--table", "dir_late"])[0] == 0
    assert len(_read_out(out2)) == 3


@pytest.mark.parametrize("ext", ["xlsx", "parquet", "jsonl", "geojson", "json", "tsv"])
def test_output_formats(tmp_path, ext):
    src = write_csv(tmp_path / "in.csv", synth_frame(4))
    out = tmp_path / f"out.{ext}"
    assert run(["normalize", src, "-o", str(out)])[0] == 0 and out.stat().st_size > 0


def test_chunk_size_flag_does_not_change_output(tmp_path):
    src = write_csv(tmp_path / "in.csv", synth_frame(9))
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    run(["normalize", src, "-o", str(a), "--chunk-size", "1"])
    run(["normalize", src, "-o", str(b), "--chunk-size", "1000"])
    assert a.read_bytes() == b.read_bytes()


def test_existing_tunable_flags_are_accepted(tmp_path):
    src = write_csv(tmp_path / "in.csv", synth_frame(3))
    out = tmp_path / "out.csv"
    code, _ = run(["normalize", src, "-o", str(out), "--min-struct", "0.7", "--plate-tolerance", "1",
                   "--ambiguity-delta", "0.05", "--barrio-buffer", "500", "--zone-buffer", "100",
                   "--no-gate-escalate", "--no-gazetteer", "--device", "cpu", "--threshold", "0.4",
                   "--max-soft", "2", "--gate-fallback"])
    assert code == 0


def test_threshold_overrides_normalizer_threshold(tmp_path):
    n = stub()
    src = write_csv(tmp_path / "in.csv", synth_frame(3))
    run(["normalize", src, "-o", str(tmp_path / "o.csv"), "--threshold", "0.123"], normalizer=n)
    assert n.threshold == pytest.approx(0.123)


def test_artifacts_dir_is_forwarded(tmp_path):
    src = write_csv(tmp_path / "in.csv", synth_frame(3))
    _, made = run(["normalize", src, "-o", str(tmp_path / "o.csv"), "--artifacts-dir", "somewhere"])
    assert made[0][0] == "somewhere"


def test_summary_json_written(tmp_path):
    src = write_csv(tmp_path / "in.csv", synth_frame(7))
    sj = tmp_path / "sub" / "summary.json"
    run(["normalize", src, "-o", str(tmp_path / "o.csv"), "--summary-json", str(sj)])
    data = json.loads(sj.read_text(encoding="utf-8"))
    assert data["rows"] == 7 and "by_estado" in data and "rows_per_sec" in data


def test_dry_run_prints_rows_and_writes_nothing(tmp_path, capsys):
    src = write_csv(tmp_path / "in.csv", synth_frame(9))
    out = tmp_path / "out.csv"
    code, _ = run(["normalize", src, "-o", str(out), "--dry-run", "3"])
    captured = capsys.readouterr()
    assert code == 0 and not out.exists()
    assert "dry run" in captured.err.lower()
    assert "estado" in captured.out and "000" in captured.out and "002" in captured.out
    assert "003" not in captured.out  # only the first 3 rows were processed


def test_dry_run_without_output_flag(tmp_path, capsys):
    src = write_csv(tmp_path / "in.csv", synth_frame(4))
    assert run(["normalize", src, "--dry-run", "2"])[0] == 0
    assert "estado" in capsys.readouterr().out


def test_output_required_unless_dry_run(tmp_path, capsys):
    src = write_csv(tmp_path / "in.csv", synth_frame(4))
    assert run(["normalize", src])[0] == 2
    assert "-o" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# error UX and exit codes
# ---------------------------------------------------------------------------
def test_missing_address_column_lists_detected_columns_exit_2(tmp_path, capsys):
    src = write_csv(tmp_path / "in.csv", pd.DataFrame({"a": ["x"], "b": ["y"]}))
    code, made = run(["normalize", src, "-o", str(tmp_path / "o.csv")])
    err = capsys.readouterr().err
    assert code == 2 and "Traceback" not in err
    assert "a" in err and "b" in err and "--address-col" in err
    assert made == []  # failed fast: the heavy normalizer was never built


def test_unknown_address_col_exit_2(tmp_path, capsys):
    src = write_csv(tmp_path / "in.csv", synth_frame(2))
    code, _ = run(["normalize", src, "-o", str(tmp_path / "o.csv"), "--address-col", "calle_zzz"])
    err = capsys.readouterr().err
    assert code == 2 and "calle_zzz" in err and "direccion" in err


def test_ambiguous_column_prints_candidates_exit_2(tmp_path, capsys):
    src = write_csv(tmp_path / "in.csv", pd.DataFrame({"id": ["1"], "direccion": [ADDR_OK], "domicilio": [ADDR_OK]}))
    code, made = run(["normalize", src, "-o", str(tmp_path / "o.csv")])
    err = capsys.readouterr().err
    assert code == 2 and "--address-col 'direccion'" in err and "--address-col 'domicilio'" in err
    assert made == []


def test_address_col_and_parts_together_exit_2(tmp_path, capsys):
    src = write_csv(tmp_path / "in.csv", synth_frame(2))
    code, _ = run(["normalize", src, "-o", str(tmp_path / "o.csv"), "--address-col", "direccion",
                   "--address-parts", "id,barrio"])
    assert code == 2 and "both" in capsys.readouterr().err


def test_missing_input_file_exit_3(tmp_path, capsys):
    code, _ = run(["normalize", str(tmp_path / "nope.csv"), "-o", str(tmp_path / "o.csv")])
    err = capsys.readouterr().err
    assert code == 3 and "Traceback" not in err and "nope.csv" in err


def test_empty_input_file_exit_3(tmp_path, capsys):
    p = tmp_path / "e.csv"
    p.write_bytes(b"")
    assert run(["normalize", str(p), "-o", str(tmp_path / "o.csv")])[0] == 3


def test_unsupported_input_extension_exit_2(tmp_path, capsys):
    p = tmp_path / "x.docx"
    p.write_bytes(b"x")
    code, _ = run(["normalize", str(p), "-o", str(tmp_path / "o.csv")])
    assert code == 2 and ".docx" in capsys.readouterr().err


def test_unsupported_output_extension_exit_2(tmp_path, capsys):
    src = write_csv(tmp_path / "in.csv", synth_frame(2))
    assert run(["normalize", src, "-o", str(tmp_path / "o.docx")])[0] == 2


def test_unwritable_output_path_exit_3(tmp_path, capsys):
    src = write_csv(tmp_path / "in.csv", synth_frame(2))
    d = tmp_path / "out.csv"
    d.mkdir()
    code, made = run(["normalize", src, "-o", str(d)])
    assert code == 3 and "Traceback" not in capsys.readouterr().err
    assert made == []  # output checked before loading the model


def test_unreachable_database_exit_3(tmp_path, capsys):
    url = f"sqlite:///{(tmp_path / 'no' / 'dir' / 'x.db').as_posix()}"
    code, _ = run(["normalize", url, "-o", str(tmp_path / "o.csv"), "--table", "t"])
    assert code == 3 and "Traceback" not in capsys.readouterr().err


def test_sql_without_table_or_query_exit_2(tmp_path, capsys):
    db = tmp_path / "a.db"
    sqlite3.connect(db).close()
    assert run(["normalize", f"sqlite:///{db.as_posix()}", "-o", str(tmp_path / "o.csv")])[0] == 2


def test_out_of_range_tunable_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        main(["normalize", "x.csv", "-o", "y.csv", "--plate-tolerance", "9"])
    assert exc.value.code == 2


def test_no_subcommand_is_usage_error():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_row_that_raises_in_the_model_does_not_abort_and_exit_is_0(tmp_path, capsys):
    from dataset_helpers import ADDR_RAISES, exploding_stub

    frame = pd.DataFrame({"direccion": [ADDR_OK, ADDR_RAISES, ADDR_OK]})
    src = write_csv(tmp_path / "in.csv", frame)
    out = tmp_path / "o.csv"
    code, _ = run(["normalize", src, "-o", str(out)], normalizer=exploding_stub())
    assert code == 0
    assert _read_out(out)["estado"].tolist() == ["OK", "ERROR", "OK"]
    assert "ERROR" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# --config
# ---------------------------------------------------------------------------
def test_config_file_supplies_options_and_cli_overrides(tmp_path):
    frame = pd.DataFrame({"id": ["1", "2"], "calle": [ADDR_OK, ADDR_OK]})
    src = write_csv(tmp_path / "in.csv", frame)
    cfg = tmp_path / "c.toml"
    cfg.write_text(f'address_col = "calle"\nchunk_size = 1\nid_col = "id"\ninput = "{src.replace(chr(92), "/")}"\n'
                   f'output = "{(tmp_path / "from_cfg.csv").as_posix()}"\n', encoding="utf-8")
    assert run(["normalize", "--config", str(cfg)])[0] == 0
    assert (tmp_path / "from_cfg.csv").exists()
    override = tmp_path / "override.csv"
    assert run(["normalize", "--config", str(cfg), "-o", str(override)])[0] == 0
    assert override.exists()


def test_config_cli_flag_beats_file_value(tmp_path, capsys):
    frame = pd.DataFrame({"a": [ADDR_OK], "b": ["hola que tal"]})
    src = write_csv(tmp_path / "in.csv", frame)
    cfg = tmp_path / "c.toml"
    cfg.write_text('address_col = "b"\n', encoding="utf-8")
    out = tmp_path / "o.csv"
    assert run(["normalize", src, "-o", str(out), "--config", str(cfg), "--address-col", "a"])[0] == 0
    assert _read_out(out)["estado"].tolist() == ["OK"]


def test_malformed_config_exit_2(tmp_path, capsys):
    cfg = tmp_path / "c.toml"
    cfg.write_text("not = = toml", encoding="utf-8")
    src = write_csv(tmp_path / "in.csv", synth_frame(2))
    code, _ = run(["normalize", src, "-o", str(tmp_path / "o.csv"), "--config", str(cfg)])
    assert code == 2 and "TOML" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# inspect / formats
# ---------------------------------------------------------------------------
def test_inspect_prints_format_columns_guess_and_preview(tmp_path, capsys):
    src = write_csv(tmp_path / "in.csv", synth_frame(8), sep=";")
    assert main(["inspect", src]) == 0
    out = capsys.readouterr().out
    for needle in ("csv", "direccion", "guessed address column", "barrio", "rows", "CL 5 # 38 - 20"):
        assert needle in out
    assert out.count("extra_unknown") >= 1


def test_inspect_reports_ambiguity_without_failing(tmp_path, capsys):
    src = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": ["a"], "domicilio": ["b"]}))
    assert main(["inspect", src]) == 0
    assert "ambiguous" in capsys.readouterr().out.lower()


def test_inspect_missing_file_exit_3(tmp_path, capsys):
    assert main(["inspect", str(tmp_path / "nope.csv")]) == 3


def test_inspect_json(tmp_path, capsys):
    src = write_csv(tmp_path / "in.csv", synth_frame(3))
    assert main(["inspect", src, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["format"] == "csv" and data["columns"][:2] == ["id", "direccion"]
    assert data["guessed_address_column"] == "direccion" and len(data["preview"]) == 3


def test_formats_lists_readers_and_writers(capsys):
    assert main(["formats"]) == 0
    out = capsys.readouterr().out
    for fmt in ("csv", "xlsx", "parquet", "geojson", "shp", "gpkg", "sql", "jsonl"):
        assert fmt in out


def test_python_dash_m_entry_point_formats():
    proc = subprocess.run(
        [sys.executable, "-m", "cali_address", "formats"], capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": SRC}, timeout=180,
    )
    assert proc.returncode == 0 and "csv" in proc.stdout


def test_python_dash_m_usage_error_is_exit_2_without_traceback(tmp_path):
    p = write_csv(tmp_path / "in.csv", pd.DataFrame({"a": ["x"]}))
    proc = subprocess.run(
        [sys.executable, "-m", "cali_address", "normalize", p, "-o", str(tmp_path / "o.csv")],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": SRC}, timeout=180,
    )
    assert proc.returncode == 2 and "Traceback" not in proc.stderr


# ---------------------------------------------------------------------------
# more failure modes
# ---------------------------------------------------------------------------
def test_on_error_raise_aborts_with_message_and_no_output(tmp_path, capsys):
    from dataset_helpers import ADDR_RAISES, exploding_stub

    src = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_OK, ADDR_RAISES]}))
    out = tmp_path / "o.csv"
    code, _ = run(["normalize", src, "-o", str(out), "--on-error", "raise"], normalizer=exploding_stub())
    err = capsys.readouterr().err
    assert code == 1 and "Traceback" not in err and "RuntimeError" in err
    assert not out.exists()


def test_read_failure_in_a_later_chunk_leaves_no_partial_output(tmp_path, capsys):
    p = tmp_path / "in.csv"
    p.write_text("id,direccion\n1,CL 5 # 38 - 20\n2,CL 5 # 38 - 20\n3,CL 5 # 38 - 20,extra,fields\n", encoding="utf-8")
    out = tmp_path / "o.csv"
    code, _ = run(["normalize", str(p), "-o", str(out), "--chunk-size", "1"])
    assert code == 3 and not out.exists()
    assert [f for f in os.listdir(tmp_path) if f.endswith(".partial")] == []


def test_legacy_flag_aliases(tmp_path):
    p = tmp_path / "in.csv"
    p.write_text("Reporte,\nid,dir\n1,CL 5 # 38 - 20\n", encoding="utf-8")
    out = tmp_path / "o.csv"
    code, _ = run(["normalize", str(p), "-o", str(out), "--header", "1", "--column", "dir"])
    assert code == 0 and _read_out(out)["estado"].tolist() == ["OK"]


def test_zero_rows_summary_and_header_only_output(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("id,direccion\n", encoding="utf-8")
    out = tmp_path / "o.csv"
    sj = tmp_path / "s.json"
    assert run(["normalize", str(p), "-o", str(out), "--summary-json", str(sj)])[0] == 0
    assert json.loads(sj.read_text(encoding="utf-8"))["rows"] == 0
    assert list(_read_out(out).columns)[:2] == ["id", "direccion"]
