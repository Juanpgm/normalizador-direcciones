"""Unit tests for the table-reading and column/header detection layer.

None of these load the model: they cover the parts of ``cali_address.service``
that decide *what* to normalize, which is where a wrong guess silently
normalizes the wrong field.
"""

from __future__ import annotations

import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cali_address.parser import parse_address  # noqa: E402
from cali_address.service import (  # noqa: E402
    CANDIDATE_SCORE_FLOOR,
    OUTPUT_COLUMNS,
    AddressColumnError,
    TableFormatError,
    _rule_violations,
    detect_address_column,
    guess_header_row,
    normalize_strict,
    read_table,
    resolve_address_column,
    results_to_geojson,
    summarize,
)


# ---------------------------------------------------------------------------
# detect_address_column
# ---------------------------------------------------------------------------
def _scores(columns) -> dict:
    _, ranked = detect_address_column(columns)
    return {item["column"]: item["score"] for item in ranked}


def test_detect_exact_accented_name_wins_over_derived_twin():
    """The Fasecolda export has DIRECCIÓN and DIRECCIÓN_NORMALIZADA; the input wins."""
    columns = ["CONSECUTIVO", "MUNICIPIO", "DIRECCIÓN", "DIRECCIÓN_NORMALIZADA"]
    best, ranked = detect_address_column(columns)
    assert best == "DIRECCIÓN"
    assert ranked[0]["score"] == 100.0
    assert ranked[1]["column"] == "DIRECCIÓN_NORMALIZADA"


def test_detect_accepts_unaccented_spelling_too():
    assert detect_address_column(["id", "direccion"])[0] == "direccion"
    assert detect_address_column(["id", "ubicación"])[0] == "ubicación"
    assert detect_address_column(["id", "localizacion"])[0] == "localizacion"


@pytest.mark.parametrize(
    "column",
    ["address", "addr", "Street", "location", "domicile", "ADDR_LINE_1", "Street Address"],
)
def test_detect_english_synonyms(column):
    best, ranked = detect_address_column(["id", "name", column])
    assert best == column
    assert ranked[0]["score"] >= 90.0


def test_detect_returns_none_when_nothing_looks_like_an_address():
    best, ranked = detect_address_column(["id", "nombre", "valor", "fecha_registro"])
    assert best is None
    assert len(ranked) == 4
    assert all(item["score"] < CANDIDATE_SCORE_FLOOR for item in ranked)


def test_detect_ranks_every_column_descending_and_keeps_input_order_on_ties():
    columns = ["zzz", "direccion_uno", "direccion_dos", "aaa"]
    best, ranked = detect_address_column(columns)
    assert [item["column"] for item in ranked][:2] == ["direccion_uno", "direccion_dos"]
    assert ranked[0]["score"] == ranked[1]["score"]
    assert best == "direccion_uno"
    scores = [item["score"] for item in ranked]
    assert scores == sorted(scores, reverse=True)


def test_detect_does_not_fire_on_lookalike_words():
    """`directorio` and `descripcion` must stay below the candidate floor."""
    scores = _scores(["descripcion", "director", "directorio", "calificacion", "predio"])
    for name, score in scores.items():
        assert score < CANDIDATE_SCORE_FLOOR, f"{name} scored {score}"


def test_detect_handles_empty_and_non_string_column_names():
    best, ranked = detect_address_column(["", None, 42, "direccion"])
    assert best == "direccion"
    assert len(ranked) == 4
    assert _scores(["", None, 42])[""] == 0.0


def test_detect_on_zero_columns():
    best, ranked = detect_address_column([])
    assert best is None
    assert ranked == []


# ---------------------------------------------------------------------------
# guess_header_row
# ---------------------------------------------------------------------------
def test_guess_header_skips_metadata_rows_like_the_stickers_export():
    """4 metadata rows, 1 blank row, then 31 named columns."""
    raw = pd.DataFrame(
        [
            ["Evaluaciones ATC-20", None, None, None],
            ["Filtro aplicado:", "Todos los registros", None, None],
            ["Fecha de generacion", "24 de septiembre", None, None],
            ["Registros:", 3697, None, None],
            [None, None, None, None],
            ["id", "direccion", "barrio", "comuna"],
            [1, "KR 1 # 9-80", "San Pedro", 3],
        ]
    )
    assert guess_header_row(raw) == 5


def test_guess_header_finds_the_row_with_most_named_columns():
    """The inspecciones case: the header is denser than any data row."""
    raw = pd.DataFrame(
        [
            ["Descargado el:", "23 de septiembre", None, None, None],
            [None, None, None, None, None],
            ["id_edan", "ObjectID", "direccion", "barrio", "comuna"],
            ["QRYEU", 1, "KR 38 BIS # 5B2 09", None, None],
            ["2QSUY", 2, "KR 43 # 10-50", None, None],
        ]
    )
    assert guess_header_row(raw) == 2


def test_guess_header_prefers_labels_when_density_ties():
    """Every row is full, so the label-likeness tiebreak has to pick row 0."""
    raw = pd.DataFrame(
        [
            ["consecutivo", "direccion", "longitud"],
            [1, "KR 13 60 34", -76.53874],
            [2, "KR 9 A 89 40", -76.546719],
        ]
    )
    assert guess_header_row(raw) == 0


def test_guess_header_on_single_row_and_empty_frames():
    assert guess_header_row(pd.DataFrame([["direccion", "barrio"]])) == 0
    assert guess_header_row(pd.DataFrame()) == 0
    assert guess_header_row(pd.DataFrame([[None, None], [None, None]])) == 0


def test_guess_header_respects_max_scan_rows():
    """A denser header beyond the scan window must not be reached."""
    rows = [["a", None, None]] * 20
    rows.append(["id", "direccion", "barrio"])
    raw = pd.DataFrame(rows)
    assert guess_header_row(raw, max_scan_rows=5) == 0
    assert guess_header_row(raw, max_scan_rows=25) == 20


def test_guess_header_treats_numeric_strings_as_data_not_labels():
    raw = pd.DataFrame([["1", "2", "3"], ["id", "direccion", "barrio"]])
    assert guess_header_row(raw) == 1


# ---------------------------------------------------------------------------
# read_table: xlsx
# ---------------------------------------------------------------------------
def _xlsx_bytes(rows, sheet_name="Hoja1", extra_sheets=None) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name, index=False, header=False)
        for name, other in (extra_sheets or {}).items():
            pd.DataFrame(other).to_excel(writer, sheet_name=name, index=False, header=False)
    return buffer.getvalue()


def test_read_table_xlsx_guesses_the_header_row():
    data = _xlsx_bytes([
        ["Descargado el:", "hoy", None],
        [None, None, None],
        ["id", "direccion", "barrio"],
        [1, "KR 1 # 9-80", "San Pedro"],
    ])
    df = read_table(data, "acciones.xlsx")
    assert list(df.columns) == ["id", "direccion", "barrio"]
    assert len(df) == 1
    assert df.attrs["source_format"] == "xlsx"
    assert df.attrs["header_row"] == 2
    assert any("encabezado" in w for w in df.attrs["warnings"])


def test_read_table_xlsx_explicit_header_overrides_the_guess():
    data = _xlsx_bytes([
        ["Descargado el:", "hoy", None],
        ["id", "direccion", "barrio"],
        [1, "KR 1 # 9-80", "San Pedro"],
    ])
    df = read_table(data, "x.xlsx", header=1)
    assert list(df.columns) == ["id", "direccion", "barrio"]
    assert df.attrs["header_row"] == 1
    assert df.attrs["warnings"] == []


def test_read_table_xlsx_selects_the_requested_sheet():
    data = _xlsx_bytes(
        [["id", "direccion"], [1, "KR 1 # 9-80"]],
        sheet_name="primera",
        extra_sheets={"segunda": [["otro", "campo"], [9, "CL 5 # 38-25"]]},
    )
    df = read_table(data, "x.xlsx", sheet="segunda")
    assert list(df.columns) == ["otro", "campo"]


def test_read_table_xlsx_unknown_sheet_is_a_clear_error():
    data = _xlsx_bytes([["id", "direccion"], [1, "KR 1 # 9-80"]])
    with pytest.raises(TableFormatError):
        read_table(data, "x.xlsx", sheet="no_existe")


def test_read_table_xlsx_header_only_file_has_zero_rows():
    df = read_table(_xlsx_bytes([["id", "direccion"]]), "x.xlsx")
    assert len(df) == 0
    assert list(df.columns) == ["id", "direccion"]


# ---------------------------------------------------------------------------
# read_table: csv
# ---------------------------------------------------------------------------
def test_read_table_csv_utf8_comma():
    data = "id,direccion\n1,KR 1 # 9-80\n2,CL 5 # 38-25\n".encode("utf-8")
    df = read_table(data, "datos.csv")
    assert list(df.columns) == ["id", "direccion"]
    assert len(df) == 2
    assert df.attrs["source_format"] == "csv"


def test_read_table_csv_sniffs_the_semicolon_delimiter():
    data = "id;direccion;barrio\n1;KR 1 # 9-80;San Pedro\n".encode("utf-8")
    df = read_table(data, "datos.csv")
    assert list(df.columns) == ["id", "direccion", "barrio"]
    assert df.iloc[0]["direccion"] == "KR 1 # 9-80"


def test_read_table_csv_falls_back_to_latin1():
    data = "id,direccion\n1,Calle 5 Siloé\n".encode("latin-1")
    df = read_table(data, "datos.csv")
    assert "Siloé" in str(df.iloc[0]["direccion"])
    assert any("latin-1" in w for w in df.attrs["warnings"])


def test_read_table_csv_keeps_accents_in_column_names():
    data = "id,dirección\n1,KR 1 # 9-80\n".encode("utf-8")
    df = read_table(data, "datos.csv")
    assert "dirección" in df.columns


def test_read_table_csv_guesses_header_past_metadata():
    data = (
        "Descargado el:,hoy\n"
        ",\n"
        "id,direccion,barrio\n"
        "1,KR 1 # 9-80,San Pedro\n"
    ).encode("utf-8")
    df = read_table(data, "datos.csv")
    assert list(df.columns) == ["id", "direccion", "barrio"]
    assert df.attrs["header_row"] == 2


def test_read_table_csv_finds_the_header_under_a_padded_metadata_row():
    """A metadata row padded with empty cells keeps the file rectangular from row 0.

    The width-stability guard must not mistake that for "row 0 is the header": the
    real header is row 1 and every row from there down is rectangular.
    """
    data = (
        "Exportado por ArcGIS Pro 3.2,,\n"
        "id,direccion,barrio\n"
        "1,KR 1 # 9-80,San Pedro\n"
    ).encode("utf-8")
    df = read_table(data, "export.csv")
    assert list(df.columns) == ["id", "direccion", "barrio"]
    assert df.attrs["header_row"] == 1
    assert len(df) == 1


def test_read_table_csv_warns_when_it_cannot_find_a_consistent_header():
    """An irregular file pandas CAN still parse must say why row 0 was used.

    Otherwise the caller sees only a confusing 422 about a missing address column,
    with nothing pointing at the real cause.
    """
    data = (
        "meta,,,\n"     # widest row, but only one filled cell
        "a,b,c\n"       # densest row - yet the file never becomes rectangular
        "d,e\n"
    ).encode("utf-8")
    df = read_table(data, "irregular.csv")
    assert df.attrs["header_row"] == 0
    assert df.columns[0] == "meta"
    assert any("encabezado consistente" in w for w in df.attrs["warnings"])


def test_read_table_csv_ragged_from_an_unquoted_delimiter_is_an_error():
    """Growing width means pandas cannot parse it; a clear error beats bad columns."""
    data = (
        "id,direccion\n"
        "1,KR 1 # 9-80\n"
        "2,Calle 5 # 38-25, San Fernando\n"
    ).encode("utf-8")
    with pytest.raises(TableFormatError, match="CSV"):
        read_table(data, "roto.csv")


def test_read_table_rejects_empty_bytes():
    with pytest.raises(TableFormatError, match="vacio"):
        read_table(b"", "datos.csv")


# ---------------------------------------------------------------------------
# read_table: geojson
# ---------------------------------------------------------------------------
def _feature_collection(features) -> bytes:
    return json.dumps({"type": "FeatureCollection", "features": features}).encode("utf-8")


def test_read_table_geojson_properties_become_columns_with_centroid():
    data = _feature_collection([
        {
            "type": "Feature",
            "properties": {"direccion": "KR 1 # 9-80", "barrio": "San Pedro"},
            "geometry": {"type": "Point", "coordinates": [-76.5343, 3.4527]},
        },
        {
            "type": "Feature",
            "properties": {"direccion": "CL 5 # 38-25", "barrio": "San Fernando"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-76.6, 3.4], [-76.4, 3.4], [-76.4, 3.5], [-76.6, 3.5], [-76.6, 3.4]]],
            },
        },
    ])
    df = read_table(data, "predios.geojson")
    assert list(df.columns) == ["direccion", "barrio", "_lon", "_lat"]
    assert df.attrs["geometry_columns"] == ["_lon", "_lat"]
    assert df.iloc[0]["_lon"] == pytest.approx(-76.5343)
    assert df.iloc[1]["_lon"] == pytest.approx(-76.5)
    assert df.iloc[1]["_lat"] == pytest.approx(3.45)


def test_read_table_geojson_null_geometry_yields_nan_coordinates():
    data = _feature_collection([
        {"type": "Feature", "properties": {"direccion": "KR 1 # 9-80"}, "geometry": None},
    ])
    df = read_table(data, "predios.geojson")
    assert pd.isna(df.iloc[0]["_lon"])
    assert any("geometria" in w for w in df.attrs["warnings"])


def test_read_table_geojson_rejects_a_non_feature_collection():
    data = json.dumps({"type": "Feature", "properties": {}, "geometry": None}).encode()
    with pytest.raises(TableFormatError, match="FeatureCollection"):
        read_table(data, "predios.geojson")


def test_read_table_geojson_rejects_malformed_json():
    with pytest.raises(TableFormatError, match="JSON invalido"):
        read_table(b'{"type": "FeatureCollection", "features": [', "predios.geojson")


def test_read_table_geojson_rejects_a_plain_json_array():
    with pytest.raises(TableFormatError, match="FeatureCollection"):
        read_table(b'[{"direccion": "KR 1 # 9-80"}]', "datos.json")


def test_read_table_geojson_empty_feature_list():
    df = read_table(_feature_collection([]), "predios.geojson")
    assert len(df) == 0


# ---------------------------------------------------------------------------
# read_table: zipped shapefile
# ---------------------------------------------------------------------------
def _shapefile_zip(tmp_path, points, prj_wkt=None, drop=()) -> bytes:
    """Write a real shapefile with pyshp and zip the members."""
    import shapefile

    base = os.path.join(str(tmp_path), "predios")
    writer = shapefile.Writer(base)
    writer.field("direccion", "C", 100)
    writer.field("barrio", "C", 60)
    for x, y, direccion, barrio in points:
        writer.point(x, y)
        writer.record(direccion, barrio)
    writer.close()
    if prj_wkt is not None:
        with open(base + ".prj", "w", encoding="utf-8") as fh:
            fh.write(prj_wkt)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for ext in (".shp", ".shx", ".dbf", ".prj"):
            if ext in drop:
                continue
            path = base + ext
            if os.path.exists(path):
                archive.write(path, "predios" + ext)
    return buffer.getvalue()


def test_read_table_shapefile_zip_without_prj_assumes_wgs84(tmp_path):
    data = _shapefile_zip(tmp_path, [(-76.5343, 3.4527, "KR 1 # 9-80", "San Pedro")])
    df = read_table(data, "predios.zip")
    assert df.attrs["source_format"] == "shp"
    assert list(df.columns) == ["direccion", "barrio", "_lon", "_lat"]
    assert df.iloc[0]["_lon"] == pytest.approx(-76.5343, abs=1e-4)
    assert any("prj" in w.lower() for w in df.attrs["warnings"])


def test_read_table_shapefile_zip_with_wgs84_prj_does_not_warn(tmp_path):
    import pyproj

    wkt = pyproj.CRS.from_epsg(4326).to_wkt()
    data = _shapefile_zip(tmp_path, [(-76.5343, 3.4527, "KR 1 # 9-80", "San Pedro")], prj_wkt=wkt)
    df = read_table(data, "predios.zip")
    assert df.iloc[0]["_lat"] == pytest.approx(3.4527, abs=1e-4)
    assert not any("prj" in w.lower() for w in df.attrs["warnings"])


def test_read_table_shapefile_zip_reprojects_from_magna_sirgas(tmp_path):
    """A projected shapefile must come back as lon/lat over Cali, not as metres."""
    import pyproj

    source = pyproj.CRS.from_epsg(3116)
    to_metres = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), source, always_xy=True)
    x, y = to_metres.transform(-76.5343, 3.4527)
    data = _shapefile_zip(tmp_path, [(x, y, "KR 1 # 9-80", "San Pedro")],
                          prj_wkt=source.to_wkt())
    df = read_table(data, "predios.zip")
    assert df.iloc[0]["_lon"] == pytest.approx(-76.5343, abs=1e-5)
    assert df.iloc[0]["_lat"] == pytest.approx(3.4527, abs=1e-5)
    assert any("reproyectad" in w for w in df.attrs["warnings"])


def test_read_table_shapefile_zip_missing_dbf_names_the_member(tmp_path):
    data = _shapefile_zip(tmp_path, [(-76.5, 3.45, "KR 1 # 9-80", "San Pedro")], drop=(".dbf",))
    with pytest.raises(TableFormatError, match=r"\.dbf"):
        read_table(data, "predios.zip")


def test_read_table_shapefile_zip_missing_shx_names_the_member(tmp_path):
    data = _shapefile_zip(tmp_path, [(-76.5, 3.45, "KR 1 # 9-80", "San Pedro")], drop=(".shx",))
    with pytest.raises(TableFormatError, match=r"\.shx"):
        read_table(data, "predios.zip")


def test_read_table_rejects_a_corrupt_zip():
    with pytest.raises(TableFormatError, match="zip"):
        read_table(b"esto no es un zip", "predios.zip")


# ---------------------------------------------------------------------------
# read_table: rejections
# ---------------------------------------------------------------------------
def test_read_table_rejects_a_bare_shp():
    with pytest.raises(TableFormatError, match="comprima|Comprima"):
        read_table(b"\x00\x00\x27\x0a", "predios.shp")


def test_read_table_rejects_a_bare_dbf():
    with pytest.raises(TableFormatError, match="comprima|Comprima"):
        read_table(b"\x03", "predios.dbf")


def test_read_table_rejects_an_unsupported_extension():
    with pytest.raises(TableFormatError, match="no soportada"):
        read_table(b"contenido", "informe.docx")


def test_read_table_rejects_a_file_with_no_extension():
    with pytest.raises(TableFormatError, match="sin extension"):
        read_table(b"contenido", "direcciones")


# ---------------------------------------------------------------------------
# resolve_address_column
# ---------------------------------------------------------------------------
def _frame(columns) -> pd.DataFrame:
    return pd.DataFrame({name: ["KR 1 # 9-80"] for name in columns})


def test_resolve_explicit_column_is_used_verbatim():
    df = _frame(["id", "direccion", "otra"])
    assert resolve_address_column(df, "otra") == "otra"


def test_resolve_explicit_column_matches_case_insensitively():
    df = _frame(["ID", "DIRECCIÓN"])
    assert resolve_address_column(df, "dirección") == "DIRECCIÓN"


def test_resolve_explicit_missing_column_lists_what_is_available():
    df = _frame(["id", "direccion"])
    with pytest.raises(AddressColumnError) as info:
        resolve_address_column(df, "no_existe")
    error = info.value
    assert error.requested == "no_existe"
    assert error.available == ["id", "direccion"]
    assert error.as_dict()["error"] == "address_column_required"


def test_resolve_blank_requested_column_falls_back_to_detection():
    df = _frame(["id", "direccion"])
    assert resolve_address_column(df, "") == "direccion"
    assert resolve_address_column(df, None) == "direccion"


def test_resolve_auto_picks_an_exact_synonym_even_with_a_close_runner_up():
    df = _frame(["DIRECCIÓN", "DIRECCIÓN_NORMALIZADA"])
    assert resolve_address_column(df, None) == "DIRECCIÓN"


def test_resolve_auto_picks_a_clear_leader():
    df = _frame(["id", "direccion_bien", "municipio"])
    assert resolve_address_column(df, None) == "direccion_bien"


@pytest.mark.parametrize(
    "columns",
    [
        ["id", "direccion", "domicilio"],
        ["id", "direccion", "dir"],
        ["id", "direccion", "ubicacion"],
        ["id", "address", "direccion"],
        ["id", "direccion", "domicilio", "ubicacion"],
    ],
)
def test_resolve_auto_raises_when_two_exact_synonyms_tie(columns):
    """Two exact synonym matches both score 100: there is no basis to prefer one.

    Regression: the exactness shortcut used to short-circuit the margin check, so
    the first column by file order won silently with no warning at all.
    """
    with pytest.raises(AddressColumnError) as info:
        resolve_address_column(_frame(columns), None)
    error = info.value
    names = [c["column"] for c in error.candidates]
    for column in columns[1:]:
        assert column in names, f"{column} must be offered as a candidate"
    assert "varias columnas" in error.message
    assert error.available == columns


def test_resolve_auto_still_accepts_an_exact_match_over_a_close_non_tie():
    """The fix must not break the DIRECCIÓN / DIRECCIÓN_NORMALIZADA case (100 vs 95)."""
    assert resolve_address_column(_frame(["DIRECCIÓN", "DIRECCIÓN_NORMALIZADA"]), None) == "DIRECCIÓN"


def test_resolve_auto_accepts_a_lone_exact_synonym():
    assert resolve_address_column(_frame(["id", "domicilio", "barrio"]), None) == "domicilio"
    assert resolve_address_column(_frame(["direccion"]), None) == "direccion"


def test_resolve_auto_raises_on_a_tie_of_compound_names():
    """Two tier-B matches at 95 each: also a tie, also ambiguous."""
    with pytest.raises(AddressColumnError) as info:
        resolve_address_column(_frame(["id", "direccion_actual", "direccion_anterior"]), None)
    names = [c["column"] for c in info.value.candidates]
    assert names[:2] == ["direccion_actual", "direccion_anterior"]


def test_resolve_auto_raises_when_two_columns_are_equally_plausible():
    df = _frame(["id", "direccion_bien", "domicilio_actual"])
    with pytest.raises(AddressColumnError) as info:
        resolve_address_column(df, None)
    error = info.value
    names = [c["column"] for c in error.candidates]
    assert "direccion_bien" in names and "domicilio_actual" in names
    assert error.requested is None
    assert "indique cual" in error.message


def test_resolve_auto_raises_when_nothing_looks_like_an_address():
    df = _frame(["id", "nombre", "valor"])
    with pytest.raises(AddressColumnError) as info:
        resolve_address_column(df, None)
    assert info.value.candidates == []
    assert info.value.available == ["id", "nombre", "valor"]


def test_resolve_error_payload_is_json_serializable():
    df = _frame(["id", "direccion_bien", "domicilio_actual"])
    with pytest.raises(AddressColumnError) as info:
        resolve_address_column(df, None)
    json.dumps(info.value.as_dict())


# ---------------------------------------------------------------------------
# Output serialization helpers
# ---------------------------------------------------------------------------
def test_summarize_always_reports_all_three_statuses():
    df = pd.DataFrame({"estado": ["OK", "OK", "SIN_MATCH"]})
    assert summarize(df) == {"OK": 2, "SIN_MATCH": 1, "NO_PARSEABLE": 0, "total": 3}


def test_summarize_on_an_empty_frame():
    df = pd.DataFrame({"estado": []})
    assert summarize(df) == {"OK": 0, "SIN_MATCH": 0, "NO_PARSEABLE": 0, "total": 0}


# ---------------------------------------------------------------------------
# _rule_violations
# ---------------------------------------------------------------------------
def _violations(query_text, cadastral_text, min_struct=0.0, struct_score=1.0, plate_tolerance=0):
    query = parse_address(query_text)
    cadastral = parse_address(cadastral_text)
    assert query.parse_ok
    assert cadastral.parse_ok
    return _rule_violations(query, cadastral, min_struct, struct_score, plate_tolerance)


def test_rule_violations_via_letters_query_only_is_a_violation():
    violations = _violations("KR 41 E # 5-20", "KR 41 # 5-20")
    assert len(violations) == 1
    assert "letra de via" in violations[0]


def test_rule_violations_via_letters_cadastral_only_is_a_violation():
    violations = _violations("KR 41 # 5-20", "KR 41 E # 5-20")
    assert len(violations) == 1
    assert "letra de via" in violations[0]


def test_rule_violations_identical_via_letters_is_clean():
    assert _violations("KR 41 E # 5-20", "KR 41 E # 5-20") == []


def test_rule_violations_cross_letters_mismatch():
    violations = _violations("KR 5 # 38 A-10", "KR 5 # 38-10")
    assert len(violations) == 1
    assert "letra de cruce" in violations[0]


def test_rule_violations_via_bis_mismatch():
    violations = _violations("CL 13 BIS # 4-10", "CL 13 # 4-10")
    assert len(violations) == 1
    assert "bis" in violations[0]


def test_rule_violations_via_quadrant_query_only_is_now_allowed():
    """Symmetric-lenient quadrant rule: a one-sided quadrant (either side) never
    violates on its own; data showed 17 correct vs 3 wrong for one-sided cases."""
    assert _violations("CL 5 NORTE # 38-10", "CL 5 # 38-10") == []


def test_rule_violations_via_quadrant_cadastral_more_specific_is_allowed():
    assert _violations("CL 5 # 38-10", "CL 5 NORTE # 38-10") == []


def test_rule_violations_via_quadrant_both_sides_differ_is_still_a_violation():
    violations = _violations("CL 5 NORTE # 38-10", "CL 5 SUR # 38-10")
    assert any("cuadrante" in v for v in violations)


def test_rule_violations_quadrant_in_complement_not_in_cadastral():
    violations = _violations("CL 5 # 38-10 NORTE", "CL 5 # 38-10")
    assert any("cuadrante" in v for v in violations)


def test_rule_violations_self_comparison_with_complement_quadrant_is_clean():
    """Regression: real cadastral strings (``AV 4 2 2 OESTE # 13 B - 19``) push a
    quadrant word into ``complement`` on both sides when compared to themselves;
    the complement-quadrant rule must not fire against an address's own parse."""
    for raw in (
        "AV 4 2 2 OESTE # 13 B - 19",
        "KR 72 CL 13 ZONA VERDE OCCIDENTAL SUR",
        "CL 6 BISIS OESTE # KR 49 B - BO",
    ):
        assert _violations(raw, raw) == []


def test_rule_violations_quadrant_in_complement_matches_cadastral_cross_quadrant():
    assert _violations("CL 5 # 38-10 NORTE", "CL 5 # 38 NORTE - 10") == []


def test_rule_violations_single_letter_in_complement_is_not_a_quadrant():
    assert _violations("CL 5 # 38-10 BLQ E", "CL 5 # 38-10") == []


def test_rule_violations_cross_type_mismatch():
    violations = _violations("CL 5 # KR 38-10", "CL 5 # TV 38-10")
    assert any("tipo de cruce" in v for v in violations)


def test_rule_violations_cross_type_one_side_empty_is_allowed():
    assert _violations("CL 5 # 38-10", "CL 5 # KR 38-10") == []


def test_rule_violations_suffix_letters_mismatch():
    violations = _violations("KR 1 A 5 B # 10-20", "KR 1 A 5 # 10-20")
    assert any("sufijo de via" in v for v in violations)


def test_rule_violations_exact_complex_duplicate_is_clean():
    assert _violations("AV 3 E # 59 NORTE - 130 CA 26", "AV 3 E # 59 NORTE - 130 CA 26") == []


def test_rule_violations_via_type_mismatch_still_reported():
    violations = _violations("CL 5 # 38-10", "KR 5 # 38-10")
    assert any("tipo de via" in v for v in violations)


# ---------------------------------------------------------------------------
# _rule_violations: abbreviated quadrant letter (N/O/W) <-> spelled-out quadrant
# ---------------------------------------------------------------------------
def test_rule_violations_abbreviated_north_letter_on_both_sides_is_clean():
    assert _violations("CALLE 6N 2N-120", "CL 6 NORTE # 2 NORTE - 120") == []


def test_rule_violations_abbreviated_west_letter_o_on_both_sides_is_clean():
    assert _violations("AV 12 O 6 O 153", "AV 12 OESTE # 6 OESTE - 153") == []


def test_rule_violations_abbreviated_north_letter_on_cross_only_is_clean():
    assert _violations("Calle 41 5n 23", "CL 41 # 5 NORTE - 23") == []


def test_rule_violations_abbreviated_west_letter_w_is_clean():
    assert _violations("KR 46 W # 19 A - 17", "KR 46 OESTE # 19 A - 17") == []


def test_rule_violations_abbreviated_east_letter_is_not_equivalent_to_este():
    """E/S are excluded on purpose: ESTE never occurs and a bare E is a real
    street letter (``KR 41 E``), so ``CL 5 E`` must not be forgiven against
    ``CL 5 ESTE``."""
    violations = _violations("CL 5 E # 38-10", "CL 5 ESTE # 38-10")
    assert any("letra de via" in v for v in violations)


def test_rule_violations_glued_letter_plus_north_already_parsed_is_clean():
    """``BN`` is parsed by the grammar itself into letter B + quadrant NORTE,
    so this must already be clean with no rule-level equivalence involved."""
    assert _violations("CL 6 BN # 2-10", "CL 6 B NORTE # 2-10") == []


def test_rule_violations_genuine_letter_n_with_no_quadrant_on_either_side_is_clean():
    """The cadastre genuinely has a street letter N with no quadrant at all: the
    equivalence must not fire when the cadastral side has no NORTE to absorb it."""
    assert _violations("CL 6 N # 2-10", "CL 6 N # 2-10") == []


def test_rule_violations_genuine_letter_n_without_a_cadastral_quadrant_still_violates():
    violations = _violations("CL 6 N # 2-10", "CL 6 # 2-10")
    assert any("letra de via" in v for v in violations)


def test_rule_violations_plate_mismatch_reported_without_tolerance():
    violations = _violations("CL 5 # 38-10", "CL 5 # 38-12", plate_tolerance=0)
    assert any("placa" in v for v in violations)


def test_rule_violations_plate_mismatch_within_tolerance_is_allowed():
    violations = _violations("CL 5 # 38-10", "CL 5 # 38-12", plate_tolerance=2)
    assert not any("placa" in v for v in violations)


def test_rule_violations_struct_below_minimum_reported():
    violations = _violations("CL 5 # 38-10", "CL 5 # 38-10", min_struct=0.9, struct_score=0.5)
    assert any("coincidencia estructural" in v for v in violations)


# ---------------------------------------------------------------------------
# nivel_precision (via normalize_strict, using the shared stub normalizer)
# ---------------------------------------------------------------------------
def test_missing_plate_is_ok_with_nivel_esquina(stub_normalizer):
    # "Calle 5 # 38" parses as cross=38, plate=None: the row still binds to a
    # real predio at that corner, but only the corner was confirmed.
    stub = stub_normalizer([{"direccion": "CL 5 # 38 - 10", "predial": "PXYZ"}])
    result = normalize_strict(stub, ["Calle 5 # 38"], gazetteer=None)
    row = result.iloc[0]
    assert row["estado"] == "OK"
    assert row["nivel_precision"] == "esquina"
    assert row["numero_predial_nacional"] == "PXYZ"


def test_missing_cross_is_ok_with_nivel_via(stub_normalizer):
    # "CL 5 # - 38" (the `rest.startswith("- ")` branch) parses as cross=None,
    # plate=38: only the via was confirmed, the cross is inferred from the match.
    stub = stub_normalizer([{"direccion": "CL 5 # 12 - 38"}])
    result = normalize_strict(stub, ["CL 5 # - 38"], gazetteer=None)
    row = result.iloc[0]
    assert row["estado"] == "OK"
    assert row["nivel_precision"] == "via"


def test_complete_address_is_nivel_predio(stub_normalizer):
    stub = stub_normalizer([{"direccion": "CL 5 # 38 - 10"}])
    result = normalize_strict(stub, ["CL 5 # 38 - 10"], gazetteer=None)
    row = result.iloc[0]
    assert row["estado"] == "OK"
    assert row["nivel_precision"] == "predio"


def test_non_ok_rows_have_null_nivel(stub_normalizer):
    stub = stub_normalizer([{"direccion": "CL 9 # 20 - 30"}], scores=[[0.1]], threshold=0.7)
    result = normalize_strict(stub, ["", "asdkjasd1234", "CL 9 # 20 - 30"], gazetteer=None)
    assert list(result["estado"]) == ["SIN_MATCH", "NO_PARSEABLE", "SIN_MATCH"]
    assert result["nivel_precision"].isna().all()


def test_output_columns_include_nivel_precision_in_order():
    assert "nivel_precision" in OUTPUT_COLUMNS
    # the two reliability columns and the margin sit between confianza and nivel_precision
    assert OUTPUT_COLUMNS.index("nivel_precision") == OUTPUT_COLUMNS.index("confianza") + 4


def test_nivel_precision_edge_case_with_plate_but_no_matching_cross(stub_normalizer):
    # The query has a plate; the cadastral candidate has none (a lot recorded
    # only by lot number, `LT 6`). Whichever way the rules land, this must not
    # raise, and nivel must agree with estado.
    stub = stub_normalizer([{"direccion": "KR 72 # - LT 6"}])
    result = normalize_strict(stub, ["KR 72 # 5 - 10"], gazetteer=None)
    row = result.iloc[0]
    if row["estado"] == "OK":
        assert row["nivel_precision"] == "predio"
    else:
        assert pd.isna(row["nivel_precision"])


# ---------------------------------------------------------------------------
# shared addresses (several predios behind one cadastral address)
# ---------------------------------------------------------------------------
def _shared(stub_normalizer, query, n_predios, nunique, direccion="CL 5 # 38 - 10"):
    stub = stub_normalizer([{
        "direccion": direccion, "predial": "PXYZ", "manzana": "MZ7",
        "n_predios": n_predios, "manzana_nunique": nunique,
    }])
    return normalize_strict(stub, [query], gazetteer=None).iloc[0]


def test_shared_address_keeps_predial(stub_normalizer):
    row = _shared(stub_normalizer, "CL 5 # 38 - 10", 5, 1)
    assert row["estado"] == "OK"
    assert row["numero_predial_nacional"] == "PXYZ"
    assert row["manzana"] == "MZ7"
    assert row["nivel_precision"] == "manzana"
    assert row["motivo"] == "direccion compartida por 5 predios"


def test_shared_address_across_manzanas_keeps_predial_and_manzana(stub_normalizer):
    row = _shared(stub_normalizer, "CL 5 # 38 - 10", 5, 2)
    assert row["estado"] == "OK"
    assert row["numero_predial_nacional"] == "PXYZ"
    assert row["manzana"] == "MZ7"
    assert row["nivel_precision"] == "direccion"
    assert row["motivo"] == "direccion compartida por 5 predios"
    assert row["lat"] == 3.4 and row["lon"] == -76.5


def test_unique_address_unchanged(stub_normalizer):
    row = _shared(stub_normalizer, "CL 5 # 38 - 10", 1, 1)
    assert row["numero_predial_nacional"] == "PXYZ"
    assert row["manzana"] == "MZ7"
    assert row["nivel_precision"] == "predio"
    assert row["motivo"] == ""


def test_shared_address_with_missing_plate_keeps_least_precise_level(stub_normalizer):
    row = _shared(stub_normalizer, "Calle 5 # 38", 3, 1)
    assert row["estado"] == "OK"
    assert row["nivel_precision"] == "esquina"  # esquina < manzana
    assert row["numero_predial_nacional"] == "PXYZ"
    assert row["manzana"] == "MZ7"
    assert row["motivo"] == "direccion compartida por 3 predios"


def test_shared_address_across_manzanas_with_missing_plate_is_direccion(stub_normalizer):
    row = _shared(stub_normalizer, "Calle 5 # 38", 3, 2)
    assert row["nivel_precision"] == "direccion"
    assert row["numero_predial_nacional"] == "PXYZ"
    assert row["manzana"] == "MZ7"


def test_shared_address_with_missing_cross_is_via_level(stub_normalizer):
    row = _shared(stub_normalizer, "CL 5 # - 38", 2, 1, direccion="CL 5 # 12 - 38")
    assert row["nivel_precision"] == "via"
    assert row["numero_predial_nacional"] == "PXYZ"


def test_shared_address_accepts_numpy_integer_counts(stub_normalizer):
    stub = stub_normalizer([{"direccion": "CL 5 # 38 - 10"}])
    stub._n_predios = np.array([np.int64(4)])
    stub._manzana_nunique = np.array([np.int32(1)])
    row = normalize_strict(stub, ["CL 5 # 38 - 10"], gazetteer=None).iloc[0]
    assert row["motivo"] == "direccion compartida por 4 predios"
    assert row["nivel_precision"] == "manzana"


def test_shared_address_non_ok_rows_are_untouched(stub_normalizer):
    stub = stub_normalizer(
        [{"direccion": "CL 9 # 20 - 30", "n_predios": 9, "manzana_nunique": 3}],
        scores=[[0.1]], threshold=0.7,
    )
    row = normalize_strict(stub, ["CL 9 # 20 - 30"], gazetteer=None).iloc[0]
    assert row["estado"] == "SIN_MATCH"
    assert pd.isna(row["nivel_precision"])
    assert "compartida" not in row["motivo"]


@pytest.mark.parametrize("level, expected", [
    (("predio", "manzana"), "manzana"),
    (("via", "manzana"), "via"),
    (("esquina", "manzana"), "esquina"),
    (("esquina", "direccion"), "direccion"),
    (("predio", "direccion"), "direccion"),
    (("predio", "predio"), "predio"),
])
def test_least_precise_level_wins(level, expected):
    from cali_address.service import _least_precise
    assert _least_precise(*level) == expected


# ---------------------------------------------------------------------------
# relaxed soft rules (one-sided via letter, plate off by 1-2)
# ---------------------------------------------------------------------------
def _relax(stub_normalizer, query, direccion, plate_tolerance=0, doc_extra=None, **stub_kwargs):
    doc = {"direccion": direccion, "predial": "PXYZ", "manzana": "MZ7"}
    doc.update(doc_extra or {})
    stub = stub_normalizer([doc], **stub_kwargs)
    return normalize_strict(stub, [query], gazetteer=None, plate_tolerance=plate_tolerance).iloc[0]


def _assert_relaxed(row, text):
    assert row["estado"] == "OK"
    assert row["numero_predial_nacional"] == "PXYZ"
    assert row["manzana"] == "MZ7"
    assert row["lat"] == 3.4 and row["lon"] == -76.5
    assert row["nivel_precision"] == "manzana"
    assert row["motivo"].startswith("aproximado:")
    assert text in row["motivo"]


def test_relaxed_letter_in_cadastre_only(stub_normalizer):
    row = _relax(stub_normalizer, "KR 41 # 5-20", "KR 41 E # 5 - 20")
    _assert_relaxed(row, "letra de via - != E")


def test_relaxed_letter_in_query_only(stub_normalizer):
    row = _relax(stub_normalizer, "KR 41 E # 5-20", "KR 41 # 5 - 20")
    _assert_relaxed(row, "letra de via E != -")


def test_letter_both_differ_is_not_relaxed(stub_normalizer):
    row = _relax(stub_normalizer, "KR 41 F # 5-20", "KR 41 E # 5 - 20")
    assert row["estado"] == "SIN_MATCH"
    assert row["numero_predial_nacional"] is None
    assert row["manzana"] is None


@pytest.mark.parametrize("query, direccion", [
    ("CL 5 # 38 - 10", "CL 5 # 38 - 10"),
])
def test_exact_match_is_not_relaxed(stub_normalizer, query, direccion):
    row = _relax(stub_normalizer, query, direccion)
    assert row["nivel_precision"] == "predio"
    assert row["motivo"] == ""


@pytest.mark.parametrize("query_plate", [11, 13, 12, 14])
def test_relaxed_plate_within_two(stub_normalizer, query_plate):
    row = _relax(stub_normalizer, f"CL 5 # 38 - {query_plate}", "CL 5 # 38 - 12")
    if query_plate == 12:
        assert row["nivel_precision"] == "predio" and row["motivo"] == ""
    else:
        _assert_relaxed(row, f"placa {query_plate} != 12")


def test_plate_delta_three_is_sin_match(stub_normalizer):
    row = _relax(stub_normalizer, "CL 5 # 38 - 15", "CL 5 # 38 - 12")
    assert row["estado"] == "SIN_MATCH"
    assert row["numero_predial_nacional"] is None


def test_plate_tolerance_two_makes_delta_two_a_plain_predio(stub_normalizer):
    row = _relax(stub_normalizer, "CL 5 # 38 - 14", "CL 5 # 38 - 12", plate_tolerance=2)
    assert row["estado"] == "OK"
    assert row["nivel_precision"] == "predio"
    assert "aproximado" not in row["motivo"]


def test_plate_delta_two_with_tolerance_one_is_relaxed(stub_normalizer):
    row = _relax(stub_normalizer, "CL 5 # 38 - 14", "CL 5 # 38 - 12", plate_tolerance=1)
    _assert_relaxed(row, "placa 14 != 12")


def test_plate_delta_within_hard_tolerance_is_not_relaxed(stub_normalizer):
    row = _relax(stub_normalizer, "CL 5 # 38 - 13", "CL 5 # 38 - 12", plate_tolerance=1)
    assert row["nivel_precision"] == "predio"
    assert row["motivo"] == ""


def test_two_soft_violations_are_sin_match(stub_normalizer):
    row = _relax(stub_normalizer, "KR 41 # 5-19", "KR 41 E # 5 - 20")
    assert row["estado"] == "SIN_MATCH"
    assert row["numero_predial_nacional"] is None


def test_soft_plus_hard_violation_is_sin_match(stub_normalizer):
    row = _relax(stub_normalizer, "KR 42 # 5-19", "KR 41 # 5 - 20")
    assert row["estado"] == "SIN_MATCH"
    assert row["numero_predial_nacional"] is None
    assert "numero de via" in row["motivo"]


def test_soft_violation_below_confidence_threshold_is_sin_match(stub_normalizer):
    row = _relax(stub_normalizer, "KR 41 # 5-20", "KR 41 E # 5 - 20",
                 scores=[[0.1]], threshold=0.7)
    assert row["estado"] == "SIN_MATCH"
    assert row["numero_predial_nacional"] is None
    assert row["nivel_precision"] is None or pd.isna(row["nivel_precision"])


def test_soft_plus_shared_address_combines_notes(stub_normalizer):
    row = _relax(stub_normalizer, "KR 41 # 5-20", "KR 41 E # 5 - 20",
                 doc_extra={"n_predios": 3})
    assert row["estado"] == "OK"
    assert row["nivel_precision"] == "manzana"
    assert row["numero_predial_nacional"] == "PXYZ"
    assert row["motivo"] == "aproximado: letra de via - != E; direccion compartida por 3 predios"


def test_soft_plus_shared_across_manzanas_is_direccion_level(stub_normalizer):
    row = _relax(stub_normalizer, "KR 41 # 5-20", "KR 41 E # 5 - 20",
                 doc_extra={"n_predios": 3, "manzana_nunique": 2})
    assert row["nivel_precision"] == "direccion"
    assert row["numero_predial_nacional"] == "PXYZ"
    assert row["manzana"] == "MZ7"


def test_soft_with_missing_plate_keeps_least_precise_level(stub_normalizer):
    row = _relax(stub_normalizer, "KR 41 # 5", "KR 41 E # 5 - 20")
    if row["estado"] == "OK":
        assert row["nivel_precision"] in ("esquina", "via", "direccion")


@pytest.mark.parametrize("query, direccion", [
    ("CL 5 # 38 - 10A", "CL 5 # 38 - 12"),
    ("CL 5 # 38 - 12", "CL 5 # 38 - 10A"),
])
def test_alphanumeric_plate_does_not_raise(stub_normalizer, query, direccion):
    row = _relax(stub_normalizer, query, direccion)
    assert row["estado"] in ("OK", "SIN_MATCH")


def test_classify_violations_splits_hard_and_soft():
    from cali_address.service import _classify_violations
    q, c = parse_address("KR 41 # 5-19"), parse_address("KR 41 E # 5-20")
    hard, soft = _classify_violations(q, c, 0.0, 1.0, 0)
    assert hard == []
    assert len(soft) == 2


def test_classify_violations_tunable_moves_plate_out_of_soft():
    from cali_address.service import _classify_violations
    q, c = parse_address("CL 5 # 38-14"), parse_address("CL 5 # 38-12")
    assert _classify_violations(q, c, 0.0, 1.0, 0) == ([], ["placa 14 != 12"])
    assert _classify_violations(q, c, 0.0, 1.0, 2) == ([], [])


def test_classify_violations_non_numeric_plate_is_hard():
    from cali_address.service import _classify_violations
    import dataclasses
    q, c = parse_address("CL 5 # 38-10"), parse_address("CL 5 # 38-12")
    q = dataclasses.replace(q, plate="10A")  # the parser never emits this; guard the int() path
    hard, soft = _classify_violations(q, c, 0.0, 1.0, 0)
    assert soft == [] and any(m.startswith("placa") for m in hard)


def test_classify_violations_unparseable_candidate_is_hard():
    from cali_address.service import _classify_violations
    q = parse_address("CL 5 # 38-10")
    c = parse_address("asdkjasd")
    hard, soft = _classify_violations(q, c, 0.0, 1.0, 0)
    assert hard and not soft


def test_classify_violations_struct_floor_is_hard():
    from cali_address.service import _classify_violations
    q, c = parse_address("CL 5 # 38-10"), parse_address("CL 5 # 38-10")
    hard, soft = _classify_violations(q, c, 0.9, 0.5, 0)
    assert len(hard) == 1 and soft == []


def test_geojson_keeps_rows_without_a_coordinate_as_null_geometry():
    df = pd.DataFrame({
        "direccion_entrada": ["KR 1 # 9-80", "basura"],
        "estado": ["OK", "NO_PARSEABLE"],
        "lat": [3.4527, None],
        "lon": [-76.5343, None],
    })
    collection = results_to_geojson(df)
    assert collection["type"] == "FeatureCollection"
    assert len(collection["features"]) == 2
    assert collection["features"][0]["geometry"] == {
        "type": "Point", "coordinates": [-76.5343, 3.4527],
    }
    assert collection["features"][1]["geometry"] is None
    assert "lat" not in collection["features"][0]["properties"]
    json.dumps(collection)
