"""Reader registry: format inference, streaming chunks, encodings, delimiters, edge cases."""

from __future__ import annotations

import functools
import http.server
import json
import sys
import threading
import zipfile

import openpyxl
import pandas as pd
import pytest

from dataset_helpers import (
    ADDR_OK, assert_same_frames, collect, synth_frame, write_csv,
)

from cali_address.io import (
    DEFAULT_CHUNK_SIZE,
    DatasetError,
    MissingDependencyError,
    SourceReadError,
    UnsupportedFormatError,
    UsageError,
    infer_format,
    list_reader_formats,
    read_table,
)

CHUNK_SIZES = [1, 2, 6, 7, 8, DEFAULT_CHUNK_SIZE]


def test_default_chunk_size_is_20000():
    assert DEFAULT_CHUNK_SIZE == 20_000


def test_error_hierarchy():
    assert issubclass(UsageError, DatasetError)
    assert issubclass(SourceReadError, DatasetError)
    assert not issubclass(SourceReadError, UsageError)


# ---------------------------------------------------------------------------
# format inference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,fmt", [
    ("a.csv", "csv"), ("a.CSV", "csv"), ("a.tsv", "tsv"), ("a.txt", "txt"),
    ("a.xlsx", "xlsx"), ("a.xlsm", "xlsx"), ("a.xls", "xls"),
    ("a.parquet", "parquet"), ("a.json", "json"), ("a.jsonl", "jsonl"), ("a.ndjson", "jsonl"),
    ("a.geojson", "geojson"), ("a.shp", "shp"), ("a.zip", "shp"), ("a.gpkg", "gpkg"),
    ("sqlite:///x.db", "sql"), ("postgresql://u:p@h/db", "sql"),
    ("postgresql+psycopg2://u:p@h/db", "sql"), ("https://host/data.csv", "csv"),
    ("https://host/data.xlsx?token=1", "xlsx"),
])
def test_infer_format(name, fmt):
    assert infer_format(name) == fmt


def test_infer_format_unknown_extension_lists_supported():
    with pytest.raises(UnsupportedFormatError) as exc:
        infer_format("data.docx")
    assert ".csv" in str(exc.value) and ".docx" in str(exc.value)


def test_infer_format_no_extension():
    with pytest.raises(UnsupportedFormatError):
        infer_format("data")


def test_list_reader_formats_covers_all_requested_formats():
    names = set(list_reader_formats())
    assert {"csv", "tsv", "txt", "xlsx", "xls", "parquet", "json", "jsonl",
            "geojson", "shp", "gpkg", "sql"} <= names


def test_unknown_explicit_format():
    with pytest.raises(UnsupportedFormatError):
        read_table("whatever.csv", fmt="docx")


def test_format_override_beats_extension(tmp_path):
    p = tmp_path / "data.dat"
    write_csv(p, synth_frame(3))
    out = collect(read_table(str(p), fmt="csv"))
    assert list(out.columns) == ["id", "direccion", "barrio", "extra_unknown"]


def test_missing_file_is_source_read_error(tmp_path):
    with pytest.raises(SourceReadError):
        read_table(str(tmp_path / "nope.csv"))


def test_directory_is_source_read_error(tmp_path):
    with pytest.raises(SourceReadError):
        read_table(str(tmp_path), fmt="csv")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sep", [",", ";", "\t", "|"])
def test_csv_delimiter_sniffing(tmp_path, sep):
    frame = synth_frame(5)
    # other delimiters appear inside (quoted) values, the active one is quoted by to_csv
    frame.loc[1, "direccion"] = "Carrera 100 # 15-30, Apto 301; torre 2 | piso 3"
    p = write_csv(tmp_path / "d.csv", frame, sep=sep)
    reader = read_table(p)
    out = collect(reader)
    assert reader.delimiter == sep
    assert list(out.columns) == list(frame.columns)
    assert_same_frames(out, frame)


def test_csv_explicit_delimiter_override(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("a;b\n1,5;x\n2,5;y\n", encoding="utf-8")
    out = collect(read_table(str(p), delimiter=";"))
    assert list(out.columns) == ["a", "b"]
    assert out["a"].tolist() == ["1,5", "2,5"]


def test_csv_quoted_embedded_newlines_and_delimiters(tmp_path):
    p = tmp_path / "q.csv"
    p.write_bytes(
        'id,direccion,nota\n1,"Carrera 100 # 15-30\nApto 301, torre 2","dijo ""hola"""\n2,CL 5 # 38 - 20,ok\n'.encode()
    )
    out = collect(read_table(str(p), chunk_size=1))
    assert len(out) == 2
    assert out.loc[0, "direccion"] == "Carrera 100 # 15-30\nApto 301, torre 2"
    assert out.loc[0, "nota"] == 'dijo "hola"'


def test_csv_utf8_bom_stripped_from_first_header(tmp_path):
    p = tmp_path / "bom.csv"
    p.write_bytes(b"\xef\xbb\xbfid,direccion\n1,CL 5 # 38 - 20\n")
    reader = read_table(str(p))
    out = collect(reader)
    assert list(out.columns) == ["id", "direccion"]


@pytest.mark.parametrize("encoding", ["latin-1", "cp1252"])
def test_csv_legacy_encodings_are_detected(tmp_path, encoding):
    p = tmp_path / "old.csv"
    p.write_bytes("id,dirección,barrio\n1,CL 5 # 38 - 20,Peñón\n".encode(encoding))
    reader = read_table(str(p))
    out = collect(reader)
    assert list(out.columns) == ["id", "dirección", "barrio"]
    assert out.loc[0, "barrio"] == "Peñón"
    assert any("encoding" in w.lower() for w in reader.warnings)


def test_csv_explicit_encoding_override(tmp_path):
    p = tmp_path / "old.csv"
    p.write_bytes("id,barrio\n1,Ñandú\n".encode("cp1252"))
    out = collect(read_table(str(p), encoding="cp1252"))
    assert out.loc[0, "barrio"] == "Ñandú"


def test_csv_wrong_explicit_encoding_is_read_error(tmp_path):
    p = tmp_path / "old.csv"
    p.write_bytes("id,barrio\n1,Peñón\n".encode("utf-8"))
    with pytest.raises(SourceReadError):
        collect(read_table(str(p), encoding="ascii"))


def test_csv_utf16_bom(tmp_path):
    p = tmp_path / "u16.csv"
    p.write_bytes("id,direccion\n1,CL 5 # 38 - 20\n".encode("utf-16"))
    out = collect(read_table(str(p)))
    assert out.loc[0, "direccion"] == ADDR_OK


def test_csv_empty_file_is_actionable_error(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_bytes(b"")
    with pytest.raises(SourceReadError, match="empty"):
        read_table(str(p))


def test_csv_whitespace_only_file_is_error(tmp_path):
    p = tmp_path / "ws.csv"
    p.write_text("   \n\n  \n", encoding="utf-8")
    with pytest.raises(SourceReadError):
        collect(read_table(str(p)))


def test_csv_header_only_yields_columns_and_no_rows(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("id,direccion,barrio\n", encoding="utf-8")
    reader = read_table(str(p))
    assert reader.columns == ["id", "direccion", "barrio"]
    assert list(reader) == []


def test_csv_single_row(tmp_path):
    p = write_csv(tmp_path / "s.csv", synth_frame(1))
    out = collect(read_table(p))
    assert len(out) == 1 and out.loc[0, "direccion"] == ADDR_OK


def test_csv_values_are_kept_as_text_not_coerced(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("id,direccion,cod\n007,CL 5 # 38 - 20,1e3\n", encoding="utf-8")
    out = collect(read_table(str(p)))
    assert out.loc[0, "id"] == "007" and out.loc[0, "cod"] == "1e3"


def test_csv_na_like_strings_are_not_turned_into_missing(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("id,direccion\n1,NA\n2,null\n", encoding="utf-8")
    out = collect(read_table(str(p)))
    assert out["direccion"].tolist() == ["NA", "null"]


def test_csv_very_long_field_survives(tmp_path):
    long_addr = "CL 5 # 38 - 20 " + "x" * 10_000
    p = tmp_path / "long.csv"
    pd.DataFrame({"id": ["1"], "direccion": [long_addr]}).to_csv(p, index=False)
    out = collect(read_table(str(p)))
    assert out.loc[0, "direccion"] == long_addr


def test_csv_duplicate_header_names_are_made_unique(tmp_path):
    p = tmp_path / "dup.csv"
    p.write_text("direccion,direccion,id\nA,B,1\n", encoding="utf-8")
    reader = read_table(str(p))
    assert len(set(reader.columns)) == 3


def test_csv_ragged_rows_fail_with_actionable_message(tmp_path):
    p = tmp_path / "rag.csv"
    p.write_text("a,b\n1,2\n3,4,5,6\n", encoding="utf-8")
    with pytest.raises(SourceReadError, match="delimiter|quot"):
        collect(read_table(str(p)))


def test_csv_metadata_rows_above_header_explicit(tmp_path):
    p = tmp_path / "meta.csv"
    p.write_text("Reporte generado\n\nid,direccion\n1,CL 5 # 38 - 20\n2,x\n", encoding="utf-8")
    reader = read_table(str(p), header_row=2)
    out = collect(reader)
    assert list(out.columns) == ["id", "direccion"] and len(out) == 2
    assert reader.header_row == 2


def test_csv_header_row_autodetected_when_omitted(tmp_path):
    p = tmp_path / "meta.csv"
    p.write_text("Reporte,\nGenerado hoy,\nid,direccion\n1,CL 5 # 38 - 20\n2,CL 6 # 1 - 1\n", encoding="utf-8")
    reader = read_table(str(p))
    assert reader.header_row == 2
    assert any("header" in w.lower() for w in reader.warnings)


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_csv_chunk_boundaries_preserve_rows_and_order(tmp_path, chunk_size):
    frame = synth_frame(7)
    p = write_csv(tmp_path / "d.csv", frame)
    chunks = list(read_table(p, chunk_size=chunk_size))
    assert all(len(c) <= chunk_size for c in chunks)
    assert sum(len(c) for c in chunks) == 7
    assert_same_frames(pd.concat(chunks, ignore_index=True), frame)


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_chunk_size_is_usage_error(tmp_path, bad):
    p = write_csv(tmp_path / "d.csv", synth_frame(2))
    with pytest.raises(UsageError):
        read_table(p, chunk_size=bad)


def test_tsv_forces_tab_delimiter(tmp_path):
    p = write_csv(tmp_path / "d.tsv", synth_frame(3), sep="\t")
    out = collect(read_table(p))
    assert "direccion" in out.columns


def test_txt_with_delimiter_is_a_table(tmp_path):
    p = write_csv(tmp_path / "d.txt", synth_frame(3), sep="|")
    out = collect(read_table(p))
    assert list(out.columns) == ["id", "direccion", "barrio", "extra_unknown"]


def test_txt_without_delimiter_is_a_headerless_address_list(tmp_path):
    p = tmp_path / "list.txt"
    p.write_text("CL 5 # 38 - 20\nCarrera 100 # 15-30\n\nCL 6 # 1 - 1\n", encoding="utf-8")
    reader = read_table(str(p))
    out = collect(reader)
    assert reader.columns == ["direccion"]
    assert out["direccion"].tolist() == ["CL 5 # 38 - 20", "Carrera 100 # 15-30", "", "CL 6 # 1 - 1"]


def test_txt_lines_format_streams_in_chunks(tmp_path):
    p = tmp_path / "list.txt"
    p.write_text("\n".join(f"CL {i} # 1 - 1" for i in range(5)), encoding="utf-8")
    chunks = list(read_table(str(p), fmt="lines", chunk_size=2))
    assert [len(c) for c in chunks] == [2, 2, 1]


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------
def _write_xlsx(path, sheets: dict[str, list[list]]):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for row in rows:
            ws.append(row)
    wb.save(path)
    return str(path)


def test_xlsx_basic_and_types(tmp_path):
    p = _write_xlsx(tmp_path / "a.xlsx", {"S": [
        ["id", "direccion", "n"], ["001", ADDR_OK, 38], ["002", "x", 2.5],
    ]})
    out = collect(read_table(p))
    assert list(out.columns) == ["id", "direccion", "n"]
    assert out["id"].tolist() == ["001", "002"] and out.loc[0, "n"] == 38


def test_xlsx_header_not_on_first_row_explicit(tmp_path):
    p = _write_xlsx(tmp_path / "a.xlsx", {"S": [
        ["Reporte de inspecciones"], [None], ["id", "direccion"], ["1", ADDR_OK], ["2", "x"],
    ]})
    reader = read_table(p, header_row=2)
    out = collect(reader)
    assert list(out.columns) == ["id", "direccion"] and len(out) == 2


def test_xlsx_header_row_autodetected(tmp_path):
    p = _write_xlsx(tmp_path / "a.xlsx", {"S": [
        ["Reporte"], ["generado hoy"], ["id", "direccion"], ["1", ADDR_OK], ["2", "CL 6 # 1 - 1"],
    ]})
    reader = read_table(p)
    assert reader.header_row == 2 and list(collect(reader).columns) == ["id", "direccion"]


def test_xlsx_multiple_sheets_default_first_with_warning_and_explicit_choice(tmp_path):
    p = _write_xlsx(tmp_path / "a.xlsx", {
        "uno": [["id", "direccion"], ["1", ADDR_OK]],
        "dos": [["id", "calle"], ["9", "CL 6 # 1 - 1"], ["8", "x"]],
    })
    first = read_table(p)
    assert list(collect(first).columns) == ["id", "direccion"]
    assert first.sheet_names == ["uno", "dos"]
    assert any("sheet" in w.lower() for w in first.warnings)
    second = read_table(p, sheet="dos")
    assert len(collect(second)) == 2 and not any("sheets" in w.lower() for w in second.warnings)


def test_xlsx_missing_sheet_lists_available(tmp_path):
    p = _write_xlsx(tmp_path / "a.xlsx", {"uno": [["id"], ["1"]]})
    with pytest.raises(SourceReadError, match="uno"):
        read_table(p, sheet="nope")


def test_xlsx_blank_rows_and_ragged_widths(tmp_path):
    p = _write_xlsx(tmp_path / "a.xlsx", {"S": [
        ["id", "direccion", None], ["1", ADDR_OK], [None, None, None], ["2", "x", None],
    ]})
    out = collect(read_table(p))
    assert len(out) == 2 and list(out.columns)[:2] == ["id", "direccion"]


def test_xlsx_header_only_and_empty_sheet(tmp_path):
    p = _write_xlsx(tmp_path / "h.xlsx", {"S": [["id", "direccion"]]})
    reader = read_table(p)
    assert reader.columns == ["id", "direccion"] and list(reader) == []
    q = _write_xlsx(tmp_path / "e.xlsx", {"S": []})
    with pytest.raises(SourceReadError):
        read_table(q)


def test_xlsx_corrupt_file_is_read_error(tmp_path):
    p = tmp_path / "bad.xlsx"
    p.write_bytes(b"this is not a zip")
    with pytest.raises(SourceReadError):
        read_table(str(p))


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_xlsx_chunk_boundaries(tmp_path, chunk_size):
    frame = synth_frame(7)
    p = _write_xlsx(tmp_path / "a.xlsx", {"S": [list(frame.columns)] + frame.values.tolist()})
    chunks = list(read_table(p, chunk_size=chunk_size))
    assert sum(len(c) for c in chunks) == 7
    # blank-address rows are written as None by openpyxl; compare on the non-blank columns
    assert collect(chunks)["id"].tolist() == frame["id"].tolist()


def test_xls_without_xlrd_gives_install_hint(tmp_path, monkeypatch):
    p = tmp_path / "old.xls"
    p.write_bytes(b"\xd0\xcf\x11\xe0not really")
    monkeypatch.setitem(sys.modules, "xlrd", None)
    with pytest.raises((MissingDependencyError, SourceReadError)) as exc:
        read_table(str(p))
    assert "xlrd" in str(exc.value) or "xls" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Parquet / JSON / JSONL / GeoJSON / Shapefile / GPKG
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_parquet_chunks(tmp_path, chunk_size):
    frame = synth_frame(7)
    p = tmp_path / "d.parquet"
    frame.to_parquet(p, index=False)
    chunks = list(read_table(str(p), chunk_size=chunk_size))
    assert sum(len(c) for c in chunks) == 7 and all(len(c) <= chunk_size for c in chunks)
    assert_same_frames(pd.concat(chunks, ignore_index=True), frame)


def test_parquet_empty_has_columns(tmp_path):
    p = tmp_path / "d.parquet"
    synth_frame(3).iloc[0:0].to_parquet(p, index=False)
    reader = read_table(str(p))
    assert reader.columns == ["id", "direccion", "barrio", "extra_unknown"] and list(reader) == []


def test_parquet_corrupt(tmp_path):
    p = tmp_path / "d.parquet"
    p.write_bytes(b"PAR1garbage")
    with pytest.raises(SourceReadError):
        read_table(str(p))


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_jsonl_chunks(tmp_path, chunk_size):
    frame = synth_frame(7)
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in frame.to_dict("records")), encoding="utf-8")
    chunks = list(read_table(str(p), chunk_size=chunk_size))
    assert sum(len(c) for c in chunks) == 7
    assert_same_frames(pd.concat(chunks, ignore_index=True), frame)


def test_jsonl_malformed_line_is_read_error(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"a": 1}\n{oops\n', encoding="utf-8")
    with pytest.raises(SourceReadError, match="line 2"):
        collect(read_table(str(p)))


def test_jsonl_blank_lines_ignored_and_heterogeneous_keys(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"a": 1}\n\n{"a": 2, "b": "x"}\n', encoding="utf-8")
    out = collect(read_table(str(p), chunk_size=1))
    assert len(out) == 2 and "b" in out.columns


def test_json_array_of_objects(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([{"id": "1", "direccion": ADDR_OK}, {"id": "2", "direccion": None}]), encoding="utf-8")
    out = collect(read_table(str(p), chunk_size=1))
    assert len(out) == 2 and out.loc[0, "direccion"] == ADDR_OK


def test_json_wrapped_records(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"records": [{"id": "1", "direccion": ADDR_OK}]}), encoding="utf-8")
    assert len(collect(read_table(str(p)))) == 1


def test_json_with_jsonl_content_falls_back(tmp_path):
    p = tmp_path / "d.json"
    p.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    assert len(collect(read_table(str(p)))) == 2


@pytest.mark.parametrize("content", ["", "   ", "{not json", '"just a string"', "42", '{"a": 1}'])
def test_json_garbage_is_read_error(tmp_path, content):
    p = tmp_path / "d.json"
    p.write_text(content, encoding="utf-8")
    if content == '{"a": 1}':
        # a lone object is not a table
        with pytest.raises(SourceReadError):
            collect(read_table(str(p)))
        return
    with pytest.raises(SourceReadError):
        collect(read_table(str(p)))


def test_json_feature_collection_is_routed_to_geojson(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"direccion": ADDR_OK}, "geometry": {"type": "Point", "coordinates": [-76.5, 3.4]}},
    ]}), encoding="utf-8")
    out = collect(read_table(str(p)))
    assert out.loc[0, "_lon"] == pytest.approx(-76.5)


def test_geojson_points_and_missing_geometry(tmp_path):
    p = tmp_path / "d.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"direccion": ADDR_OK}, "geometry": {"type": "Point", "coordinates": [-76.5, 3.4]}},
        {"type": "Feature", "properties": {"direccion": "x"}, "geometry": None},
    ]}), encoding="utf-8")
    reader = read_table(str(p), chunk_size=1)
    chunks = list(reader)
    assert [len(c) for c in chunks] == [1, 1]
    out = pd.concat(chunks, ignore_index=True)
    assert out.loc[0, "_lat"] == pytest.approx(3.4) and pd.isna(out.loc[1, "_lat"])


def test_geojson_invalid_is_read_error(tmp_path):
    p = tmp_path / "d.geojson"
    p.write_text('{"type": "Feature"}', encoding="utf-8")
    with pytest.raises(SourceReadError):
        read_table(str(p))


def _write_shp(tmp_path, n=3):
    import shapefile

    base = str(tmp_path / "pts")
    with shapefile.Writer(base, shapeType=shapefile.POINT) as w:
        w.field("direccion", "C", size=60)
        w.field("id", "N", size=6)
        for i in range(n):
            w.point(-76.5 + i * 0.001, 3.4)
            w.record(f"CL {i} # 1 - 1", i)
    return base


def test_shapefile_path_streams_in_chunks(tmp_path):
    base = _write_shp(tmp_path, 5)
    chunks = list(read_table(base + ".shp", chunk_size=2))
    assert [len(c) for c in chunks] == [2, 2, 1]
    out = pd.concat(chunks, ignore_index=True)
    assert out.loc[0, "direccion"] == "CL 0 # 1 - 1" and out.loc[2, "_lon"] == pytest.approx(-76.498)


def test_shapefile_zip(tmp_path):
    base = _write_shp(tmp_path, 3)
    zpath = tmp_path / "pts.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for ext in (".shp", ".shx", ".dbf"):
            z.write(base + ext, arcname="pts" + ext)
    out = collect(read_table(str(zpath), chunk_size=2))
    assert len(out) == 3 and "direccion" in out.columns


def test_incomplete_shapefile_zip_is_read_error(tmp_path):
    base = _write_shp(tmp_path, 1)
    zpath = tmp_path / "pts.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.write(base + ".shp", arcname="pts.shp")
    with pytest.raises(SourceReadError, match="shx|dbf"):
        read_table(str(zpath))


def test_gpkg_without_pyogrio_gives_install_hint(tmp_path, monkeypatch):
    p = tmp_path / "d.gpkg"
    p.write_bytes(b"SQLite format 3\x00")
    monkeypatch.setitem(sys.modules, "pyogrio", None)
    with pytest.raises(MissingDependencyError, match="pyogrio"):
        read_table(str(p))


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------
def test_http_url_is_downloaded_and_read(tmp_path):
    write_csv(tmp_path / "remote.csv", synth_frame(4))
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    handler.log_message = lambda *a, **k: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/remote.csv"
        out = collect(read_table(url, chunk_size=3))
        assert len(out) == 4
    finally:
        server.shutdown()
        server.server_close()


def test_http_404_is_source_read_error(tmp_path):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    handler.log_message = lambda *a, **k: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(SourceReadError):
            read_table(f"http://127.0.0.1:{server.server_address[1]}/missing.csv")
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# row-width safety (pandas' chunked parser silently drops the extra fields of a long row)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("chunk_size", [1, 2, 100])
def test_csv_overlong_row_is_an_error_never_silent_truncation(tmp_path, chunk_size):
    p = tmp_path / "long_row.csv"
    p.write_bytes(b"id,direccion\n1,A\n2,B\n3,C,extra,fields\n")
    with pytest.raises(SourceReadError, match="line 4"):
        collect(read_table(str(p), chunk_size=chunk_size))


def test_csv_trailing_separators_and_short_rows_are_tolerated(tmp_path):
    p = tmp_path / "loose.csv"
    p.write_bytes(b"id,direccion\n1,A,\n2\n3,C,,\n")
    out = collect(read_table(str(p)))
    assert out["id"].tolist() == ["1", "2", "3"]
    assert out["direccion"].tolist()[0] == "A" and out["direccion"].tolist()[2] == "C"
    assert out["direccion"].iloc[1] is None


def test_csv_cell_larger_than_the_stdlib_default_limit(tmp_path):
    big = "x" * 300_000
    p = tmp_path / "big.csv"
    p.write_bytes(f"id,direccion\n1,{big}\n".encode())
    assert len(collect(read_table(str(p))).loc[0, "direccion"]) == 300_000


def test_csv_single_column_keeps_blank_lines_as_empty_addresses(tmp_path):
    p = tmp_path / "one.csv"
    p.write_bytes(b"direccion\nCL 5 # 38 - 20\n\nCL 6 # 1 - 1\n")
    assert collect(read_table(str(p)))["direccion"].tolist() == ["CL 5 # 38 - 20", "", "CL 6 # 1 - 1"]


def test_csv_explicit_utf8_with_bom_still_strips_the_bom(tmp_path):
    p = tmp_path / "bom.csv"
    p.write_bytes(b"\xef\xbb\xbfid,direccion\n1,x\n")
    assert read_table(str(p), encoding="utf-8").columns == ["id", "direccion"]


def test_csv_header_row_beyond_end_is_error(tmp_path):
    p = write_csv(tmp_path / "d.csv", synth_frame(2))
    with pytest.raises(SourceReadError, match="beyond"):
        read_table(p, header_row=50)
