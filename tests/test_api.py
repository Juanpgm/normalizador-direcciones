"""End-to-end tests for the HTTP service.

The model, the 330k embeddings and the gazetteer are loaded ONCE for the whole
module (a session-scoped ``TestClient``), which is also the point: every test
here shares the same singletons, so a test that corrupted shared state would
break its successors.
"""

from __future__ import annotations

import io
import json
import os
import sys
import zipfile

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from cali_address.api.main import app  # noqa: E402

ARTIFACTS = os.path.join(ROOT, "artifacts")
BASEMAPS = os.path.join(ROOT, "basemaps")
ESRI = "/arcgis/rest/services/CaliNormalizador/GeocodeServer"

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ARTIFACTS, "model.pt")),
    reason="serving artifacts are not present",
)

#: Addresses with known outcomes, used across several tests.
MATCHING = "Carrera1#9-80"
UNMATCHED = "Calle 5 # 38-25, San Fernando"
RURAL = "Corregimiento los andes, vereda la reforma, casa 117"
JUNK = "no es una direccion"


@pytest.fixture(scope="module")
def client():
    os.environ.setdefault("NORMALIZER_DEVICE", "cpu")
    with TestClient(app) as test_client:
        # Force the background load to finish before the first assertion.
        response = test_client.post("/api/v1/normalize-address", json={"addresses": [MATCHING]})
        assert response.status_code == 200, response.text
        yield test_client


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------
def _xlsx(rows, sheet_name="Hoja1") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name, index=False, header=False)
    return buffer.getvalue()


#: An ArcGIS-style export: two metadata rows, then the real header.
METADATA_XLSX_ROWS = [
    ["Descargado el:", "24 de septiembre de 2026", None],
    [None, None, None],
    ["id", "direccion", "barrio"],
    [1, MATCHING, "San Pedro"],
    [2, UNMATCHED, "San Fernando"],
    [3, RURAL, None],
]

# UNMATCHED contains a comma, so it MUST be quoted: an unquoted comma makes the
# file ragged and no parser can recover the intended columns.
SIMPLE_CSV = (
    "id,direccion\n"
    f'1,"{MATCHING}"\n'
    f'2,"{UNMATCHED}"\n'
    f'3,"{JUNK}"\n'
).encode("utf-8")

#: The same content with the comma left unquoted: genuinely malformed.
RAGGED_CSV = (
    "id,direccion\n"
    f"1,{MATCHING}\n"
    f"2,{UNMATCHED}\n"
).encode("utf-8")


def _upload(name, data, content_type="application/octet-stream"):
    return {"file": (name, data, content_type)}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
def test_health_reports_a_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["catastro_size"] > 300000
    assert body["threshold"] == pytest.approx(0.73)
    assert body["device"] in ("cpu", "cuda")


def test_root_serves_the_upload_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Normalizador de direcciones" in response.text
    assert "/api/v1/inspect" in response.text


def test_integration_docs_page_renders_markdown(client):
    response = client.get("/docs-integracion")
    assert response.status_code == 200
    assert "<h1>" in response.text
    assert "SIN_MATCH" in response.text


def test_openapi_documents_every_endpoint(client):
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    for path in (
        "/health", "/api/v1/inspect", "/api/v1/normalize",
        "/api/v1/normalize-json", "/api/v1/normalize-address",
        f"{ESRI}/findAddressCandidates", f"{ESRI}/geocodeAddresses",
    ):
        assert path in paths, path
    for path, operations in paths.items():
        for method, operation in operations.items():
            assert operation.get("summary"), f"{method.upper()} {path} has no summary"
            assert operation.get("description"), f"{method.upper()} {path} has no description"


# ---------------------------------------------------------------------------
# /api/v1/inspect
# ---------------------------------------------------------------------------
def test_inspect_xlsx_finds_the_header_below_metadata_rows(client):
    response = client.post("/api/v1/inspect", files=_upload("acciones.xlsx", _xlsx(METADATA_XLSX_ROWS)))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["format"] == "xlsx"
    assert body["columns"] == ["id", "direccion", "barrio"]
    assert body["header_row"] == 2
    assert body["row_count"] == 3
    assert body["suggested_column"] == "direccion"
    assert body["candidates"][0] == {"column": "direccion", "score": 100.0}
    assert len(body["candidates"]) == 3
    assert len(body["preview"]) == 3
    assert any("encabezado" in w for w in body["warnings"])
    assert body["sheets"] == ["Hoja1"]


def test_inspect_csv(client):
    response = client.post("/api/v1/inspect", files=_upload("datos.csv", SIMPLE_CSV))
    assert response.status_code == 200
    body = response.json()
    assert body["format"] == "csv"
    assert body["columns"] == ["id", "direccion"]
    assert body["suggested_column"] == "direccion"
    assert body["sheets"] is None
    assert len(body["preview"]) == 3


def test_inspect_never_fails_on_an_ambiguous_column(client):
    """The whole point of /inspect: it must answer even when /normalize would 422."""
    data = b"direccion_bien,domicilio_actual\nKR 1 # 9-80,CL 5 # 38-25\n"
    response = client.post("/api/v1/inspect", files=_upload("ambiguo.csv", data))
    assert response.status_code == 200
    body = response.json()
    assert body["suggested_column"] in ("direccion_bien", "domicilio_actual")
    assert len(body["candidates"]) == 2


def test_inspect_reports_no_suggestion_when_no_column_looks_like_an_address(client):
    data = b"id,nombre,valor\n1,Ana,5\n"
    response = client.post("/api/v1/inspect", files=_upload("otro.csv", data))
    assert response.status_code == 200
    assert response.json()["suggested_column"] is None


def test_inspect_geojson_exposes_the_centroid_columns(client):
    payload = json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"direccion": MATCHING},
            "geometry": {"type": "Point", "coordinates": [-76.5343, 3.4527]},
        }],
    }).encode()
    response = client.post("/api/v1/inspect", files=_upload("predios.geojson", payload))
    assert response.status_code == 200
    body = response.json()
    assert body["format"] == "geojson"
    assert "_lon" in body["columns"] and "_lat" in body["columns"]


def test_inspect_shapefile_zip(client, tmp_path):
    import shapefile

    base = os.path.join(str(tmp_path), "predios")
    writer = shapefile.Writer(base)
    writer.field("direccion", "C", 100)
    writer.point(-76.5343, 3.4527)
    writer.record(MATCHING)
    writer.close()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for ext in (".shp", ".shx", ".dbf"):
            archive.write(base + ext, "predios" + ext)
    response = client.post("/api/v1/inspect", files=_upload("predios.zip", buffer.getvalue()))
    assert response.status_code == 200
    body = response.json()
    assert body["format"] == "shp"
    assert body["suggested_column"] == "direccion"
    assert any("prj" in w.lower() for w in body["warnings"])


def test_inspect_rejects_an_empty_upload(client):
    response = client.post("/api/v1/inspect", files=_upload("vacio.csv", b""))
    assert response.status_code == 400
    assert "vacio" in json.dumps(response.json())


def test_inspect_rejects_an_unsupported_extension(client):
    response = client.post("/api/v1/inspect", files=_upload("informe.docx", b"contenido"))
    assert response.status_code == 400
    assert response.json()["error"] == "bad_file"


def test_inspect_rejects_a_malformed_geojson(client):
    response = client.post("/api/v1/inspect", files=_upload("roto.geojson", b'{"type": "Feature"'))
    assert response.status_code == 400


def test_inspect_rejects_an_incomplete_shapefile_zip(client):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("predios.shp", "x")
    response = client.post("/api/v1/inspect", files=_upload("predios.zip", buffer.getvalue()))
    assert response.status_code == 400
    message = response.json()["message"]
    assert ".dbf" in message and ".shx" in message


def test_inspect_enforces_the_upload_size_cap(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.0001")
    response = client.post("/api/v1/inspect", files=_upload("datos.csv", SIMPLE_CSV))
    assert response.status_code == 413
    assert response.json()["error"] == "payload_too_large"


# ---------------------------------------------------------------------------
# /api/v1/normalize — JSON output
# ---------------------------------------------------------------------------
def test_normalize_xlsx_end_to_end_json(client):
    response = client.post(
        "/api/v1/normalize",
        files=_upload("acciones.xlsx", _xlsx(METADATA_XLSX_ROWS)),
        data={"address_column": "direccion", "output_format": "json"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["address_column"] == "direccion"
    assert body["summary"]["total"] == 3
    assert sum(body["summary"][k] for k in ("OK", "SIN_MATCH", "NO_PARSEABLE")) == 3
    estados = [row["estado"] for row in body["results"]]
    assert estados == ["OK", "SIN_MATCH", "NO_PARSEABLE"]
    matched = body["results"][0]
    assert matched["fuente_normalizacion"] == "catastro"
    assert matched["numero_predial_nacional"] and len(matched["numero_predial_nacional"]) == 30
    assert -77.5 < matched["lon"] < -76.0 and 3.0 < matched["lat"] < 4.0
    assert matched["comuna_corregimiento"].startswith("Comuna ")
    # A rejected row must never carry cadastral fields.
    rejected = body["results"][1]
    assert rejected["numero_predial_nacional"] is None
    assert rejected["lat"] is None and rejected["lon"] is None
    assert rejected["confianza"] is None
    assert rejected["motivo"]


def test_normalize_json_rows_include_calibrated_reliability(client):
    response = client.post(
        "/api/v1/normalize",
        files=_upload("acciones.xlsx", _xlsx(METADATA_XLSX_ROWS)),
        data={"address_column": "direccion", "output_format": "json"},
    )
    matched, rejected = response.json()["results"][:2]
    assert "confiabilidad" in matched and "confiabilidad_manzana" in matched
    assert rejected["confiabilidad"] is None and rejected["confiabilidad_manzana"] is None
    if os.path.exists(os.path.join(ARTIFACTS, "reliability.json")):
        assert 0.0 <= matched["confiabilidad"] <= matched["confiabilidad_manzana"] <= 1.0
    else:
        assert matched["confiabilidad"] is None


def test_normalize_auto_detects_the_column_when_unambiguous(client):
    response = client.post(
        "/api/v1/normalize",
        files=_upload("datos.csv", SIMPLE_CSV),
        data={"output_format": "json"},
    )
    assert response.status_code == 200
    assert response.json()["address_column"] == "direccion"


def test_normalize_422_when_the_column_is_ambiguous(client):
    """Two equally plausible address columns: the caller must choose."""
    data = (
        "id,direccion_bien,domicilio_actual\n"
        f'1,"{MATCHING}","{UNMATCHED}"\n'
    ).encode()
    response = client.post(
        "/api/v1/normalize", files=_upload("ambiguo.csv", data), data={"output_format": "json"}
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "address_column_required"
    names = [c["column"] for c in body["candidates"]]
    assert "direccion_bien" in names and "domicilio_actual" in names
    assert body["available_columns"] == ["id", "direccion_bien", "domicilio_actual"]
    assert body["requested_column"] is None
    # ... and naming one of them resolves it.
    ok = client.post(
        "/api/v1/normalize",
        files=_upload("ambiguo.csv", data),
        data={"address_column": "direccion_bien", "output_format": "json"},
    )
    assert ok.status_code == 200
    assert ok.json()["address_column"] == "direccion_bien"


def test_normalize_422_when_no_column_looks_like_an_address(client):
    data = b"id,nombre,valor\n1,Ana,5\n"
    response = client.post("/api/v1/normalize", files=_upload("otro.csv", data))
    assert response.status_code == 422
    body = response.json()
    assert body["candidates"] == []
    assert body["available_columns"] == ["id", "nombre", "valor"]


def test_normalize_422_when_the_named_column_does_not_exist(client):
    response = client.post(
        "/api/v1/normalize",
        files=_upload("datos.csv", SIMPLE_CSV),
        data={"address_column": "no_existe"},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["requested_column"] == "no_existe"
    assert body["available_columns"] == ["id", "direccion"]


def test_normalize_rejects_an_unknown_output_format(client):
    response = client.post(
        "/api/v1/normalize",
        files=_upload("datos.csv", SIMPLE_CSV),
        data={"address_column": "direccion", "output_format": "pdf"},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# /api/v1/normalize — file outputs
# ---------------------------------------------------------------------------
def test_normalize_xlsx_output_has_three_sheets(client):
    response = client.post(
        "/api/v1/normalize",
        files=_upload("datos.csv", SIMPLE_CSV),
        data={"address_column": "direccion", "output_format": "xlsx"},
    )
    assert response.status_code == 200
    assert "spreadsheetml" in response.headers["content-type"]
    assert "attachment" in response.headers["content-disposition"]
    assert ".xlsx" in response.headers["content-disposition"]
    book = pd.ExcelFile(io.BytesIO(response.content))
    assert book.sheet_names == ["normalizado", "revisar", "resumen"]
    todo = book.parse("normalizado")
    revisar = book.parse("revisar")
    assert len(todo) == 3
    assert (revisar["estado"] != "OK").all()
    assert len(revisar) == len(todo) - int((todo["estado"] == "OK").sum())
    assert set(book.parse("resumen").columns) == {"estado", "filas"}


def test_normalize_csv_output(client):
    response = client.post(
        "/api/v1/normalize",
        files=_upload("datos.csv", SIMPLE_CSV),
        data={"address_column": "direccion", "output_format": "csv"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert response.content.startswith(b"\xef\xbb\xbf"), "Excel needs the UTF-8 BOM"
    table = pd.read_csv(io.BytesIO(response.content), encoding="utf-8-sig")
    assert len(table) == 3
    assert "direccion_normalizada" in table.columns
    assert "comuna_corregimiento" in table.columns


def test_normalize_geojson_output_keeps_unlocated_rows(client):
    response = client.post(
        "/api/v1/normalize",
        files=_upload("datos.csv", SIMPLE_CSV),
        data={"address_column": "direccion", "output_format": "geojson"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/geo+json")
    collection = response.json()
    assert collection["type"] == "FeatureCollection"
    assert len(collection["features"]) == 3, "one feature per row, located or not"
    geometries = [feature["geometry"] for feature in collection["features"]]
    assert geometries[0]["type"] == "Point"
    assert len(geometries[0]["coordinates"]) == 2
    assert geometries[1] is None and geometries[2] is None
    properties = collection["features"][0]["properties"]
    assert "estado" in properties and "lat" not in properties and "lon" not in properties


# ---------------------------------------------------------------------------
# /api/v1/normalize — caps and edge cases
# ---------------------------------------------------------------------------
def test_normalize_enforces_the_row_cap(client, monkeypatch):
    monkeypatch.setenv("MAX_ROWS", "2")
    data = ("direccion\n" + f"{MATCHING}\n" * 5).encode()
    response = client.post(
        "/api/v1/normalize", files=_upload("muchas.csv", data), data={"address_column": "direccion"}
    )
    assert response.status_code == 422
    assert "limite por peticion es 2" in json.dumps(response.json())


def test_normalize_row_cap_allows_a_file_at_the_limit(client, monkeypatch):
    monkeypatch.setenv("MAX_ROWS", "3")
    data = ("direccion\n" + f"{MATCHING}\n" * 3).encode()
    response = client.post(
        "/api/v1/normalize", files=_upload("justas.csv", data), data={"address_column": "direccion"}
    )
    assert response.status_code == 200
    assert response.json()["summary"]["total"] == 3


def test_normalize_handles_blank_and_null_rows_inside_a_valid_file(client):
    rows = [
        ["direccion"], [MATCHING], [""], ["   "], ["N/A"], ["SIN ESPECIFICAR"], [None], ["-"],
    ]
    response = client.post(
        "/api/v1/normalize",
        files=_upload("huecos.xlsx", _xlsx(rows)),
        data={"address_column": "direccion", "output_format": "json"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["total"] == 7
    assert body["summary"]["OK"] == 1
    placeholders = body["results"][1:]
    assert all(row["estado"] == "SIN_MATCH" for row in placeholders)
    assert all("vacio o marcador" in row["motivo"] for row in placeholders)
    assert all(row["numero_predial_nacional"] is None for row in placeholders)


def test_normalize_a_file_with_zero_data_rows(client):
    response = client.post(
        "/api/v1/normalize",
        files=_upload("solo_encabezado.csv", b"id,direccion\n"),
        data={"address_column": "direccion", "output_format": "json"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == {"OK": 0, "SIN_MATCH": 0, "NO_PARSEABLE": 0, "total": 0}
    assert body["results"] == []


def test_normalize_zero_row_file_also_works_for_file_outputs(client):
    for fmt in ("csv", "xlsx", "geojson"):
        response = client.post(
            "/api/v1/normalize",
            files=_upload("solo_encabezado.csv", b"id,direccion\n"),
            data={"address_column": "direccion", "output_format": fmt},
        )
        assert response.status_code == 200, f"{fmt}: {response.text}"


def test_normalize_rejects_a_ragged_csv_instead_of_renaming_columns(client):
    """An unquoted comma must produce a clear 400, never silently shifted columns."""
    response = client.post(
        "/api/v1/inspect", files=_upload("roto.csv", RAGGED_CSV)
    )
    assert response.status_code == 400
    assert response.json()["error"] == "bad_file"


def test_normalize_preserves_accented_and_unicode_content(client):
    data = (
        "id,dirección\n"
        "1,Calle 1 # 2-3 Barrio Siloé\n"
        '2,"Corregimiento La Buitrera, vereda Alto Los Mangos"\n'
    ).encode()
    response = client.post(
        "/api/v1/normalize", files=_upload("acentos.csv", data), data={"output_format": "json"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["address_column"] == "dirección"
    assert "Siloé" in body["results"][0]["direccion_entrada"]
    # The gazetteer must still have normalized the zone from the accented text.
    assert body["results"][0]["barrio_vereda"] or body["results"][0]["comuna_corregimiento"]


def test_normalize_respects_the_gazetteer_switch(client):
    """Disabling the gazetteer must drop the zone columns for a rural address."""
    payload = {"addresses": [RURAL], "gazetteer": False}
    off = client.post("/api/v1/normalize-address", json=payload).json()["results"][0]
    on = client.post("/api/v1/normalize-address", json={"addresses": [RURAL]}).json()["results"][0]
    assert off["barrio_vereda"] is None and off["comuna_corregimiento"] is None
    assert on["comuna_corregimiento"] or on["barrio_vereda"]


def test_normalize_tunables_are_validated(client):
    response = client.post(
        "/api/v1/normalize",
        files=_upload("datos.csv", SIMPLE_CSV),
        data={"address_column": "direccion", "min_struct": "5"},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("endpoint, send", [
    (
        "/api/v1/normalize",
        lambda value: dict(
            files=_upload("datos.csv", SIMPLE_CSV),
            data={"address_column": "direccion", "plate_tolerance": str(value)},
        ),
    ),
    (
        "/api/v1/normalize-address",
        lambda value: dict(json={"addresses": [MATCHING], "plate_tolerance": value}),
    ),
])
def test_api_rejects_plate_tolerance_above_2(client, endpoint, send):
    response = client.post(endpoint, **send(3))
    assert response.status_code == 422


@pytest.mark.parametrize("endpoint, send", [
    (
        "/api/v1/normalize",
        lambda value: dict(
            files=_upload("datos.csv", SIMPLE_CSV),
            data={"address_column": "direccion", "ambiguity_delta": str(value)},
        ),
    ),
    (
        "/api/v1/normalize-address",
        lambda value: dict(json={"addresses": [MATCHING], "ambiguity_delta": value}),
    ),
    (
        "/api/v1/normalize-json",
        lambda value: dict(json={"addresses": [MATCHING], "ambiguity_delta": value}),
    ),
])
def test_api_ambiguity_delta_is_bounded(client, endpoint, send):
    assert client.post(endpoint, **send(0.5)).status_code == 422
    assert client.post(endpoint, **send(-0.01)).status_code == 422
    assert client.post(endpoint, **send(0.02)).status_code == 200
    assert client.post(endpoint, **send(0.2)).status_code == 200


def test_api_tunables_default_ambiguity_delta_is_0_02():
    from cali_address.api.schemas import Tunables

    assert Tunables().ambiguity_delta == 0.02
    assert Tunables(ambiguity_delta=0.0).ambiguity_delta == 0.0


def test_api_multipart_default_ambiguity_delta_is_0_02():
    import inspect

    from cali_address.api import main

    src = inspect.getsource(main)
    assert "ambiguity_delta: float = Form(0.02, ge=0.0, le=0.2)" in src


def test_api_results_expose_margen(client):
    row = client.post("/api/v1/normalize-address", json={"addresses": [MATCHING]}).json()["results"][0]
    assert "margen" in row


def test_api_accepts_plate_tolerance_2(client):
    response = client.post(
        "/api/v1/normalize",
        files=_upload("datos.csv", SIMPLE_CSV),
        data={"address_column": "direccion", "plate_tolerance": "2"},
    )
    assert response.status_code != 422
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# /api/v1/normalize-json
# ---------------------------------------------------------------------------
def test_normalize_json_with_records_and_address_field(client):
    records = [
        {"id": 1, "dir": MATCHING, "otro": "x"},
        {"id": 2, "dir": UNMATCHED, "otro": "y"},
    ]
    response = client.post(
        "/api/v1/normalize-json", json={"records": records, "address_field": "dir"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["address_column"] == "dir"
    assert body["summary"]["total"] == 2
    assert [row["direccion_entrada"] for row in body["results"]] == [MATCHING, UNMATCHED]


def test_normalize_json_with_addresses_only(client):
    response = client.post(
        "/api/v1/normalize-json", json={"addresses": [MATCHING, JUNK]}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["address_column"] is None
    assert [row["estado"] for row in body["results"]] == ["OK", "NO_PARSEABLE"]


def test_normalize_json_detects_the_address_field_when_omitted_is_rejected(client):
    """`records` without `address_field` is a client error, not a guess."""
    response = client.post("/api/v1/normalize-json", json={"records": [{"direccion": MATCHING}]})
    assert response.status_code == 422


@pytest.mark.parametrize("payload", [
    {},
    {"records": [{"dir": "x"}], "address_field": "dir", "addresses": ["x"]},
])
def test_normalize_json_requires_exactly_one_source(client, payload):
    assert client.post("/api/v1/normalize-json", json=payload).status_code == 422


def test_normalize_json_422_when_the_address_field_does_not_exist(client):
    response = client.post(
        "/api/v1/normalize-json",
        json={"records": [{"id": 1, "direccion": MATCHING}], "address_field": "no_existe"},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "address_column_required"
    assert body["requested_column"] == "no_existe"


def test_normalize_json_with_empty_lists(client):
    for payload in ({"addresses": []}, {"records": [], "address_field": "dir"}):
        response = client.post("/api/v1/normalize-json", json=payload)
        assert response.status_code == 200, response.text
        assert response.json()["summary"]["total"] == 0


def test_normalize_json_accepts_null_addresses(client):
    response = client.post("/api/v1/normalize-json", json={"addresses": [None, "", MATCHING]})
    assert response.status_code == 200
    estados = [row["estado"] for row in response.json()["results"]]
    assert estados == ["SIN_MATCH", "SIN_MATCH", "OK"]


def test_normalize_json_enforces_the_row_cap(client, monkeypatch):
    monkeypatch.setenv("MAX_ROWS", "2")
    response = client.post("/api/v1/normalize-json", json={"addresses": [MATCHING] * 3})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# /api/v1/normalize-address
# ---------------------------------------------------------------------------
def test_normalize_address_alias(client):
    response = client.post(
        "/api/v1/normalize-address",
        json={"addresses": [UNMATCHED, MATCHING, RURAL, ""]},
    )
    assert response.status_code == 200
    body = response.json()
    assert [row["estado"] for row in body["results"]] == [
        "SIN_MATCH", "OK", "NO_PARSEABLE", "SIN_MATCH",
    ]
    assert body["summary"] == {"OK": 1, "SIN_MATCH": 2, "NO_PARSEABLE": 1, "total": 4}


def test_normalize_address_requires_the_addresses_field(client):
    assert client.post("/api/v1/normalize-address", json={}).status_code == 422


def test_two_rapid_sequential_requests_share_the_singleton_safely(client):
    """The single worker holds one model; back-to-back calls must both succeed."""
    first = client.post("/api/v1/normalize-address", json={"addresses": [MATCHING, UNMATCHED]})
    second = client.post("/api/v1/normalize-address", json={"addresses": [MATCHING, UNMATCHED]})
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json(), "the engine must be deterministic across calls"
    third = client.get("/health")
    assert third.status_code == 200


# ---------------------------------------------------------------------------
# Esri-shaped endpoints
# ---------------------------------------------------------------------------
def test_find_address_candidates_get(client):
    response = client.get(
        f"{ESRI}/findAddressCandidates", params={"SingleLine": MATCHING, "f": "json"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["spatialReference"]["wkid"] == 4326
    assert len(body["candidates"]) == 1
    candidate = body["candidates"][0]
    assert candidate["attributes"]["Status"] == "OK"
    assert candidate["score"] > 70.0
    assert -77.5 < candidate["location"]["x"] < -76.0
    assert 3.0 < candidate["location"]["y"] < 4.0
    assert len(candidate["attributes"]["Numero_Predial"]) == 30
    extent = candidate["extent"]
    assert extent["xmin"] < candidate["location"]["x"] < extent["xmax"]
    assert extent["ymin"] < candidate["location"]["y"] < extent["ymax"]


@pytest.mark.parametrize("parameter", ["SingleLine", "Address", "text"])
def test_find_address_candidates_accepts_every_alias(client, parameter):
    response = client.get(f"{ESRI}/findAddressCandidates", params={parameter: MATCHING})
    assert response.status_code == 200
    assert response.json()["candidates"][0]["attributes"]["Status"] == "OK"


def test_find_address_candidates_post_form_encoded(client):
    response = client.post(
        f"{ESRI}/findAddressCandidates", data={"SingleLine": MATCHING, "f": "json"}
    )
    assert response.status_code == 200
    assert response.json()["candidates"][0]["attributes"]["Status"] == "OK"


def test_find_address_candidates_still_answers_for_a_non_ok_address(client):
    """A rejected address returns one candidate, but it must be unmistakably bad."""
    response = client.get(f"{ESRI}/findAddressCandidates", params={"SingleLine": JUNK})
    assert response.status_code == 200
    candidates = response.json()["candidates"]
    assert len(candidates) == 1, "clients that assume >=1 candidate must not break"
    candidate = candidates[0]
    assert candidate["attributes"]["Status"] != "OK"
    assert candidate["attributes"]["Status"] in ("SIN_MATCH", "NO_PARSEABLE")
    assert candidate["score"] == 0.0
    assert candidate["location"] == {"x": None, "y": None}
    assert candidate["extent"] is None
    assert candidate["attributes"]["Motivo"]


def test_find_address_candidates_for_an_unmatched_but_parseable_address(client):
    response = client.get(f"{ESRI}/findAddressCandidates", params={"SingleLine": UNMATCHED})
    candidate = response.json()["candidates"][0]
    assert candidate["attributes"]["Status"] == "SIN_MATCH"
    assert candidate["score"] == 0.0
    # The zone detected in the text is still reported: it is real information.
    assert candidate["attributes"]["Comuna_Corregimiento"]


def test_find_address_candidates_with_an_empty_query(client):
    response = client.get(f"{ESRI}/findAddressCandidates", params={"SingleLine": ""})
    assert response.status_code == 200
    assert response.json()["candidates"] == []
    assert client.get(f"{ESRI}/findAddressCandidates").json()["candidates"] == []


def test_geocode_addresses_batch_with_a_junk_record(client):
    payload = {"records": [
        {"attributes": {"OBJECTID": 1, "SingleLine": MATCHING}},
        {"attributes": {"OBJECTID": 2, "SingleLine": UNMATCHED}},
        {"attributes": {"OBJECTID": 99, "SingleLine": JUNK}},
    ]}
    response = client.post(f"{ESRI}/geocodeAddresses", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["spatialReference"]["wkid"] == 4326
    locations = body["locations"]
    assert len(locations) == 3
    assert [loc["attributes"]["ResultID"] for loc in locations] == [1, 2, 99]
    assert locations[0]["attributes"]["Status"] == "OK"
    assert locations[0]["score"] > 70.0
    assert locations[1]["attributes"]["Status"] == "SIN_MATCH"
    assert locations[2]["attributes"]["Status"] == "NO_PARSEABLE"
    assert all(loc["score"] == 0.0 for loc in locations[1:])
    assert all("extent" not in loc for loc in locations)


def test_geocode_addresses_with_an_empty_record_set(client):
    response = client.post(f"{ESRI}/geocodeAddresses", json={"records": []})
    assert response.status_code == 200
    assert response.json()["locations"] == []


def test_geocode_addresses_tolerates_malformed_records(client):
    payload = {"records": [{"attributes": {}}, {"no_attributes": True}]}
    response = client.post(f"{ESRI}/geocodeAddresses", json=payload)
    assert response.status_code == 200
    locations = response.json()["locations"]
    assert len(locations) == 2
    assert all(loc["attributes"]["Status"] != "OK" for loc in locations)


def test_geocode_addresses_without_a_records_key(client):
    response = client.post(f"{ESRI}/geocodeAddresses", json={})
    assert response.status_code == 200
    assert response.json()["locations"] == []
