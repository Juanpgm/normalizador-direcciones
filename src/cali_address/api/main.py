"""FastAPI service around the strict Cali address normalizer.

Startup
-------
The model (13 MB), the 330k cadastral embeddings (169 MB) and the IDESC
gazetteer are loaded ONCE, in a background thread, so the port binds
immediately and a platform health check does not time out while torch warms up.
``GET /health`` answers 503 until the load finishes; the normalization
endpoints wait for it (up to ``WARMUP_TIMEOUT_S``) and then answer 503 too.

Configuration (environment variables)
-------------------------------------
``NORMALIZER_DEVICE``   cpu (default) or cuda
``ARTIFACTS_DIR``       directory holding model.pt, catastro_emb.pt,
                        catastro_docs.parquet, tuning.json, gazetteer.pkl
``BASEMAPS_DIR``        directory holding the two IDESC GeoJSON basemaps
``DOCS_DIR``            directory holding API.md (served at /docs-integracion)
``MAX_UPLOAD_MB``       request body cap, default 25
``MAX_ROWS``            row cap per normalization request, default 20000
``WARMUP_TIMEOUT_S``    how long a request waits for the model, default 300

Privacy
-------
No default path in this service points at ``context/`` or ``outputs/``. Those
directories hold real citizen data (insured-property registry, disaster-victim
registry with names and ID numbers, field inspection reports) and must never be
committed to git nor baked into a container image; ``.gitignore`` and
``.dockerignore`` exclude both. Uploaded bytes are read into memory, used for
the response and dropped: nothing is written to disk.
"""

from __future__ import annotations

import html
import os
import re
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..gazetteer import ZONE_BUFFER_M
from ..inference import AddressNormalizer
from ..service import (
    CLI_BARRIO_BUFFER_M,
    DEFAULT_MIN_STRUCT,
    AddressColumnError,
    TableFormatError,
    detect_address_column,
    load_gazetteer,
    normalize_strict,
    read_table,
    resolve_address_column,
    results_to_geojson,
    results_to_records,
    summarize,
    write_csv_bytes,
    write_xlsx_bytes,
)
from .esri import ESRI_CAVEAT, build_router
from .schemas import (
    AddressColumnErrorResponse,
    AddressesRequest,
    ErrorResponse,
    HealthResponse,
    InspectResponse,
    NormalizeJsonRequest,
    NormalizeResponse,
)

__all__ = ["app", "create_app"]

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
STATIC_DIR = os.path.join(_HERE, "static")

#: Rows shown in an /inspect preview.
PREVIEW_ROWS = 5


# ---------------------------------------------------------------------------
# Settings (read per call so tests can override them with environment vars)
# ---------------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        return default


def artifacts_dir() -> str:
    return os.environ.get("ARTIFACTS_DIR") or os.path.join(PROJECT_ROOT, "artifacts")


def basemaps_dir() -> str:
    return os.environ.get("BASEMAPS_DIR") or os.path.join(PROJECT_ROOT, "basemaps")


def docs_dir() -> str:
    return os.environ.get("DOCS_DIR") or os.path.join(PROJECT_ROOT, "docs")


def device() -> str:
    return os.environ.get("NORMALIZER_DEVICE") or "cpu"


def max_upload_bytes() -> int:
    return int(_env_float("MAX_UPLOAD_MB", 25.0) * 1024 * 1024)


def _size(num_bytes: float) -> str:
    """Human size that stays informative below 1 MB, for a tightened cap."""
    if num_bytes >= 1048576:
        return f"{num_bytes / 1048576:.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{int(num_bytes)} B"


def max_rows() -> int:
    return _env_int("MAX_ROWS", 20000)


def warmup_timeout() -> float:
    return _env_float("WARMUP_TIMEOUT_S", 300.0)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
class _Engine:
    """The singletons every request shares."""

    def __init__(self) -> None:
        self.ready = threading.Event()
        self.normalizer: AddressNormalizer | None = None
        self.gazetteer = None
        self.device = "cpu"
        self.error: str | None = None
        self.warnings: list[str] = []
        self.load_seconds: float | None = None

    def load(self) -> None:
        started = time.monotonic()
        try:
            self.device = device()
            self.gazetteer = load_gazetteer(basemaps_dir(), warn=self.warnings.append)
            self.normalizer = AddressNormalizer(artifacts_dir(), device=self.device)
        except Exception as exc:  # keep the process alive so /health can explain
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.load_seconds = time.monotonic() - started
            self.ready.set()

    def wait(self) -> "_Engine":
        """Block until loaded; raise 503 on timeout or load failure."""
        if not self.ready.wait(warmup_timeout()):
            raise HTTPException(
                status_code=503,
                detail="el modelo aun se esta cargando; reintente en unos segundos",
            )
        if self.error is not None or self.normalizer is None:
            raise HTTPException(
                status_code=503, detail=f"el modelo no se pudo cargar ({self.error})"
            )
        return self


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = _Engine()
    app.state.engine = engine
    thread = threading.Thread(target=engine.load, name="normalizer-load", daemon=True)
    thread.start()
    try:
        yield
    finally:
        app.state.engine = None


def _engine(request: Request) -> _Engine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="el servicio no esta inicializado")
    return engine.wait()


# ---------------------------------------------------------------------------
# JSON sanitation
# ---------------------------------------------------------------------------
def _py(value):
    """Turn a pandas/numpy cell into something json.dumps can handle."""
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _sanitize(records: list[dict]) -> list[dict]:
    return [{str(k): _py(v) for k, v in record.items()} for record in records]


# ---------------------------------------------------------------------------
# Engine invocation
# ---------------------------------------------------------------------------
def _check_row_cap(count: int) -> None:
    cap = max_rows()
    if count > cap:
        raise HTTPException(
            status_code=422,
            detail=(
                f"el archivo trae {count} filas y el limite por peticion es {cap}. "
                "Divida el archivo o ajuste MAX_ROWS en el servidor."
            ),
        )


def _run(
    engine: _Engine,
    raws: list,
    *,
    min_struct: float = DEFAULT_MIN_STRUCT,
    plate_tolerance: int = 0,
    barrio_buffer: float = CLI_BARRIO_BUFFER_M,
    zone_buffer: float = ZONE_BUFFER_M,
    gate_escalate: bool = True,
    gazetteer: bool = True,
    ambiguity_delta: float = 0.02,
) -> pd.DataFrame:
    _check_row_cap(len(raws))
    return normalize_strict(
        engine.normalizer,
        raws,
        min_struct=min_struct,
        plate_tolerance=plate_tolerance,
        gazetteer=engine.gazetteer if gazetteer else None,
        barrio_buffer_m=barrio_buffer,
        zone_buffer_m=zone_buffer,
        gate_escalate=gate_escalate,
        ambiguity_delta=ambiguity_delta,
    )


def _payload(result: pd.DataFrame, warnings: list[str] | None = None,
             address_column: str | None = None) -> dict:
    return {
        "summary": summarize(result),
        "results": _sanitize(results_to_records(result)),
        "warnings": list(warnings or []),
        "address_column": address_column,
    }


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------
class PayloadTooLarge(Exception):
    """Body above ``MAX_UPLOAD_MB``. Handled into the documented 413 shape."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


async def _read_upload(file: UploadFile) -> bytes:
    """Read an upload fully, enforcing the size cap on the real byte count.

    The Content-Length middleware is the cheap first line of defence; this is the
    one that holds for a chunked body, which declares no length at all.
    """
    data = await file.read()
    cap = max_upload_bytes()
    if len(data) > cap:
        raise PayloadTooLarge(
            f"el archivo pesa {_size(len(data))} y el limite es {_size(cap)}"
        )
    if not data:
        # TableFormatError so every bad-file answer has one shape.
        raise TableFormatError("el archivo esta vacio (0 bytes)")
    return data


def _read_table_or_400(data: bytes, filename: str, sheet: str | None,
                       header: int | None) -> pd.DataFrame:
    """Read the upload; ``TableFormatError`` becomes the documented 400 body."""
    return read_table(data, filename or "", sheet=sheet, header=header)


def _sheet_names(data: bytes, source_format: str) -> list[str] | None:
    if source_format != "xlsx":
        return None
    try:
        import io

        with pd.ExcelFile(io.BytesIO(data)) as book:
            return [str(name) for name in book.sheet_names]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Minimal Markdown renderer for /docs-integracion
# ---------------------------------------------------------------------------
def _inline(text: str) -> str:
    out = html.escape(text)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', out)
    return out


def _render_markdown(text: str) -> str:
    """Enough Markdown for docs/API.md: headings, fences, lists, tables, paragraphs."""
    lines = text.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("```"):
            i += 1
            block: list[str] = []
            while i < n and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(block)) + "</code></pre>")
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            i += 1
            continue
        if re.match(r"^\s*([-*_])\s*\1\s*\1[\s\-*_]*$", line):
            out.append("<hr>")
            i += 1
            continue
        if line.lstrip().startswith("|") and line.rstrip().endswith("|"):
            rows: list[list[str]] = []
            while i < n and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c or "") for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                head = "".join(f"<th>{_inline(c)}</th>" for c in rows[0])
                body = "".join(
                    "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>"
                    for row in rows[1:]
                )
                out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
            continue
        if re.match(r"^\s*[-*+]\s+", line):
            items = []
            while i < n and re.match(r"^\s*[-*+]\s+", lines[i]):
                items.append(_inline(re.sub(r"^\s*[-*+]\s+", "", lines[i])))
                i += 1
            out.append("<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>")
            continue
        if re.match(r"^\s*\d+[.)]\s+", line):
            items = []
            while i < n and re.match(r"^\s*\d+[.)]\s+", lines[i]):
                items.append(_inline(re.sub(r"^\s*\d+[.)]\s+", "", lines[i])))
                i += 1
            out.append("<ol>" + "".join(f"<li>{item}</li>" for item in items) + "</ol>")
            continue
        if not line.strip():
            i += 1
            continue
        paragraph = []
        while i < n and lines[i].strip() and not re.match(
            r"^(#{1,6}\s|```|\s*[-*+]\s|\s*\d+[.)]\s|\s*\|)", lines[i]
        ):
            paragraph.append(lines[i].strip())
            i += 1
        out.append("<p>" + _inline(" ".join(paragraph)) + "</p>")
    return "\n".join(out)


_DOCS_PAGE = """<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Documentacion de integracion</title>
<link rel="stylesheet" href="/static/style.css">
</head><body class="doc">
<header><h1>Normalizador de direcciones de Cali</h1>
<nav><a href="/">Cargar archivo</a> <a href="/docs">Swagger</a></nav></header>
<main class="prose">{body}</main>
</body></html>
"""


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
DESCRIPTION = f"""
Normaliza direcciones de Santiago de Cali contra el catastro (330 000 direcciones)
usando un codificador neuronal de caracteres, un reordenamiento fusionado y una reja
geografica sobre los basemaps IDESC.

**Contrato de estado.** Cada fila sale con `estado` en `OK`, `SIN_MATCH` o
`NO_PARSEABLE`. Solo `OK` trae numero predial, manzana y coordenada: una fila que no
pasa el umbral de confianza, la reja geografica o las reglas estructurales no recibe
candidato alguno, para que una conjetura de baja calidad no pueda confundirse con una
coincidencia real. El motivo exacto del rechazo queda en `motivo`.

**Flujo recomendado.** `POST /api/v1/inspect` para descubrir columnas y elegir la de
direccion, luego `POST /api/v1/normalize`. Si ya tiene los datos en memoria use
`POST /api/v1/normalize-json`.

**Compatibilidad Esri.** {ESRI_CAVEAT}
"""


def create_app() -> FastAPI:
    app = FastAPI(
        title="Normalizador de direcciones - Cali",
        version="1.0.0",
        description=DESCRIPTION,
        lifespan=lifespan,
    )

    if os.path.isdir(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # -- size cap -----------------------------------------------------------
    @app.middleware("http")
    async def _cap_request_size(request: Request, call_next):
        """Reject an oversized body before it reaches the single worker."""
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                size = int(declared)
            except ValueError:
                size = 0
            cap = max_upload_bytes()
            if size > cap:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "payload_too_large",
                        "message": (
                            f"el cuerpo de la peticion pesa {_size(size)} y el "
                            f"limite es {_size(cap)}"
                        ),
                    },
                )
        return await call_next(request)

    @app.exception_handler(AddressColumnError)
    async def _address_column_handler(request: Request, exc: AddressColumnError):
        return JSONResponse(status_code=422, content=exc.as_dict())

    @app.exception_handler(TableFormatError)
    async def _table_format_handler(request: Request, exc: TableFormatError):
        return JSONResponse(
            status_code=400, content={"error": "bad_file", "message": str(exc)}
        )

    @app.exception_handler(PayloadTooLarge)
    async def _too_large_handler(request: Request, exc: PayloadTooLarge):
        return JSONResponse(
            status_code=413, content={"error": "payload_too_large", "message": exc.message}
        )

    # -- pages --------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def index():
        path = os.path.join(STATIC_DIR, "index.html")
        if not os.path.exists(path):
            return HTMLResponse("<h1>Normalizador de direcciones</h1>", status_code=200)
        return FileResponse(path, media_type="text/html")

    @app.get("/docs-integracion", include_in_schema=False)
    async def integration_docs():
        path = os.path.join(docs_dir(), "API.md")
        if not os.path.exists(path):
            return HTMLResponse(
                _DOCS_PAGE.format(body="<p>docs/API.md no esta disponible en este despliegue.</p>"),
                status_code=200,
            )
        with open(path, "r", encoding="utf-8") as fh:
            return HTMLResponse(_DOCS_PAGE.format(body=_render_markdown(fh.read())))

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return Response(status_code=204)

    # -- health -------------------------------------------------------------
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Servicio"],
        summary="Estado del servicio",
        description=(
            "Devuelve 200 cuando el modelo, el indice catastral y el gazetteer estan en "
            "memoria, y 503 mientras la carga esta en curso o si fallo. Es el endpoint "
            "que debe apuntar el health check de la plataforma."
        ),
        responses={503: {"model": HealthResponse, "description": "Cargando o con error de carga"}},
    )
    async def health(request: Request):
        engine = getattr(request.app.state, "engine", None)
        if engine is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error", "model_loaded": False, "device": device(),
                    "catastro_size": 0, "gazetteer_loaded": False, "threshold": None,
                    "detail": "el servicio no esta inicializado",
                },
            )
        if not engine.ready.is_set():
            return JSONResponse(
                status_code=503,
                content={
                    "status": "loading", "model_loaded": False, "device": device(),
                    "catastro_size": 0, "gazetteer_loaded": False, "threshold": None,
                    "detail": "cargando modelo e indice catastral",
                },
            )
        if engine.error is not None or engine.normalizer is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error", "model_loaded": False, "device": engine.device,
                    "catastro_size": 0, "gazetteer_loaded": engine.gazetteer is not None,
                    "threshold": None, "detail": engine.error,
                },
            )
        return {
            "status": "ok",
            "model_loaded": True,
            "device": engine.normalizer.device,
            "catastro_size": int(len(engine.normalizer.docs)),
            "gazetteer_loaded": engine.gazetteer is not None,
            "threshold": float(engine.normalizer.threshold),
            "detail": None,
        }

    # -- inspect ------------------------------------------------------------
    @app.post(
        "/api/v1/inspect",
        response_model=InspectResponse,
        tags=["Normalizacion"],
        summary="Inspeccionar un archivo antes de normalizar",
        description=(
            "Lee el archivo SIN normalizar y responde con el formato detectado, las "
            "columnas, la columna de direccion sugerida, todas las columnas puntuadas y "
            "una muestra de hasta 5 filas.\n\n"
            "Este endpoint nunca falla por ambiguedad de columnas: es precisamente el "
            "paso con el que un cliente descubre que enviar despues. Acepta .xlsx, .xls, "
            ".csv, .geojson/.json (FeatureCollection) y un .zip con shapefile "
            "(.shp + .shx + .dbf, con .prj opcional)."
        ),
        responses={
            400: {"model": ErrorResponse, "description": "Archivo vacio o formato no soportado"},
            413: {"model": ErrorResponse, "description": "Archivo demasiado grande"},
        },
    )
    async def inspect(
        file: UploadFile = File(description="Archivo tabular o geoespacial a inspeccionar."),
        sheet: str | None = Form(None, description="Hoja a leer cuando el archivo es Excel."),
        header: int | None = Form(None, description="Fila de encabezado 0-based; se infiere si se omite."),
    ):
        data = await _read_upload(file)
        df = _read_table_or_400(data, file.filename or "", sheet, header)
        columns = [str(c) for c in df.columns]
        suggested, ranked = detect_address_column(columns)
        preview = _sanitize(
            df.head(PREVIEW_ROWS).astype(object).where(pd.notna(df.head(PREVIEW_ROWS)), None)
            .to_dict(orient="records")
        )
        return {
            "format": df.attrs.get("source_format", "csv"),
            "row_count": int(len(df)),
            "columns": columns,
            "suggested_column": suggested,
            "candidates": ranked,
            "preview": preview,
            "warnings": list(df.attrs.get("warnings", [])),
            "header_row": df.attrs.get("header_row"),
            "sheets": _sheet_names(data, df.attrs.get("source_format", "")),
        }

    # -- normalize (file) ---------------------------------------------------
    @app.post(
        "/api/v1/normalize",
        tags=["Normalizacion"],
        summary="Normalizar un archivo completo",
        description=(
            "Normaliza la columna de direccion de un archivo y devuelve JSON "
            "(`output_format=json`) o el archivo resultante (`xlsx`, `csv`, `geojson`).\n\n"
            "El xlsx trae tres hojas: `normalizado` (todo), `revisar` (las filas que no "
            "quedaron en OK) y `resumen` (conteo por estado). El geojson lleva geometria "
            "Point donde hubo coordenada y `null` donde no, de modo que el numero de "
            "features siempre coincide con el de filas.\n\n"
            "Si no se envia `address_column` y la deteccion es ambigua, responde 422 con "
            "la lista de candidatas: ese es el caso en el que el usuario debe definir el "
            "campo. Use `/api/v1/inspect` para elegirla."
        ),
        responses={
            200: {"description": "JSON con resumen y filas, o el archivo solicitado"},
            400: {"model": ErrorResponse, "description": "Archivo vacio o formato no soportado"},
            413: {"model": ErrorResponse, "description": "Archivo demasiado grande"},
            422: {
                "model": AddressColumnErrorResponse,
                "description": "Hay que indicar la columna de direccion, o se excedio MAX_ROWS",
            },
        },
    )
    async def normalize_file(
        request: Request,
        file: UploadFile = File(description="Archivo con las direcciones."),
        sheet: str | None = Form(None, description="Hoja a leer cuando el archivo es Excel."),
        header: int | None = Form(None, description="Fila de encabezado 0-based; se infiere si se omite."),
        address_column: str | None = Form(None, description="Columna de direccion; se detecta si se omite."),
        output_format: str = Form("json", description="json (por omision), xlsx, csv o geojson."),
        min_struct: float = Form(DEFAULT_MIN_STRUCT, ge=0.0, le=1.0),
        plate_tolerance: int = Form(0, ge=0, le=2),
        barrio_buffer: float = Form(CLI_BARRIO_BUFFER_M, ge=0.0, le=20000.0),
        zone_buffer: float = Form(ZONE_BUFFER_M, ge=0.0, le=20000.0),
        gate_escalate: bool = Form(True),
        gazetteer: bool = Form(True),
        ambiguity_delta: float = Form(0.02, ge=0.0, le=0.2),
    ):
        fmt = (output_format or "json").strip().lower()
        if fmt not in ("json", "xlsx", "csv", "geojson"):
            raise HTTPException(
                status_code=422,
                detail="output_format debe ser json, xlsx, csv o geojson",
            )
        data = await _read_upload(file)
        df = _read_table_or_400(data, file.filename or "", sheet, header)
        column = resolve_address_column(df, address_column)
        engine = _engine(request)
        result = _run(
            engine, df[column].tolist(),
            min_struct=min_struct, plate_tolerance=plate_tolerance,
            barrio_buffer=barrio_buffer, zone_buffer=zone_buffer,
            gate_escalate=gate_escalate, gazetteer=gazetteer, ambiguity_delta=ambiguity_delta,
        )
        warnings = list(df.attrs.get("warnings", []))
        stem = os.path.splitext(os.path.basename(file.filename or "direcciones"))[0] or "direcciones"

        if fmt == "json":
            return JSONResponse(_payload(result, warnings, column))
        if fmt == "xlsx":
            return Response(
                content=write_xlsx_bytes(result),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f'attachment; filename="{stem}_normalizado.xlsx"'},
            )
        if fmt == "csv":
            return Response(
                content=write_csv_bytes(result),
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{stem}_normalizado.csv"'},
            )
        return JSONResponse(
            content=results_to_geojson(result),
            media_type="application/geo+json",
            headers={"Content-Disposition": f'attachment; filename="{stem}_normalizado.geojson"'},
        )

    # -- normalize (json records) ------------------------------------------
    @app.post(
        "/api/v1/normalize-json",
        response_model=NormalizeResponse,
        tags=["Normalizacion"],
        summary="Normalizar datos tabulares ya en memoria",
        description=(
            "Punto de integracion para otras fuentes que ya tienen los datos y no quieren "
            "subir un archivo. Envie `records` (filas completas) junto con `address_field`, "
            "o bien `addresses` (lista simple). Debe indicarse exactamente uno de los dos.\n\n"
            "La respuesta tiene la misma forma que `/api/v1/normalize` con "
            "`output_format=json`: `summary` con el conteo por estado y `results` con una "
            "fila por entrada, en el mismo orden."
        ),
        responses={
            422: {
                "model": AddressColumnErrorResponse,
                "description": "address_field inexistente, cuerpo invalido o se excedio MAX_ROWS",
            }
        },
    )
    async def normalize_json(request: Request, payload: NormalizeJsonRequest):
        if payload.records is not None:
            frame = pd.DataFrame(payload.records)
            if frame.empty:
                return _payload(_run(_engine(request), []), [], payload.address_field)
            column = resolve_address_column(frame, payload.address_field)
            raws = frame[column].tolist()
        else:
            column, raws = None, list(payload.addresses or [])
        result = _run(
            _engine(request), raws,
            min_struct=payload.min_struct, plate_tolerance=payload.plate_tolerance,
            barrio_buffer=payload.barrio_buffer, zone_buffer=payload.zone_buffer,
            gate_escalate=payload.gate_escalate, gazetteer=payload.gazetteer,
            ambiguity_delta=payload.ambiguity_delta,
        )
        return _payload(result, [], column)

    # -- normalize (bare address list) -------------------------------------
    @app.post(
        "/api/v1/normalize-address",
        response_model=NormalizeResponse,
        tags=["Normalizacion"],
        summary="Normalizar una lista de direcciones",
        description=(
            "Alias minimo de `/api/v1/normalize-json`: solo acepta `addresses`. Pensado "
            "para pruebas de humo e integraciones triviales. Misma forma de respuesta."
        ),
    )
    async def normalize_address(request: Request, payload: AddressesRequest):
        result = _run(
            _engine(request), list(payload.addresses),
            min_struct=payload.min_struct, plate_tolerance=payload.plate_tolerance,
            barrio_buffer=payload.barrio_buffer, zone_buffer=payload.zone_buffer,
            gate_escalate=payload.gate_escalate, gazetteer=payload.gazetteer,
            ambiguity_delta=payload.ambiguity_delta,
        )
        return _payload(result)

    # -- Esri-shaped endpoints ---------------------------------------------
    def run_addresses(request: Request, raws: list) -> list[dict]:
        result = _run(_engine(request), list(raws))
        return _sanitize(results_to_records(result))

    app.include_router(build_router(run_addresses))
    return app


app = create_app()
