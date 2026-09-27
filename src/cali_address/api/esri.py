"""Esri-compatible *response shapes* for two ArcGIS Geocode Service operations.

WHAT THIS IS
------------
Two endpoints that answer with the JSON shape of ArcGIS REST
``findAddressCandidates`` and ``geocodeAddresses``, so tooling that already knows
how to read an Esri geocoder response can consume this normalizer with no
translation layer.

WHAT THIS IS NOT
----------------
This is **not** a registered or certified ArcGIS custom Locator. A real Locator
is a provider implemented against Esri's Locator Provider SDK and registered
inside ArcGIS Server / ArcGIS Pro; a bare REST endpoint cannot satisfy that
contract no matter how faithfully it mimics the JSON. Concretely, this service
does NOT implement the rest of the Geocode Service surface an ArcGIS client may
probe or require: the service metadata document (``addressFields``,
``candidateFields``, ``locatorProperties``, ``capabilities``, ``spatialReference``
negotiation), ``reverseGeocode``, ``suggest``, ``outFields`` / ``outSR``
projection, ``searchExtent`` filtering, ``magicKey`` suggestion tokens, batch
``geocodeAddresses`` token authentication, or the ``Addr_type`` /
``Match_addr`` / ``Loc_name`` field taxonomy ArcGIS uses to rank results.
Registering this as an ArcGIS Enterprise "Locator" item requires Esri-side
configuration and software that lives outside this service.

Use it as a convenience adapter, and read ``attributes.Status`` before trusting
a location: a ``SIN_MATCH`` or ``NO_PARSEABLE`` row still comes back as one
candidate (so clients that assume at least one candidate do not break) but with
score 0 and no coordinates.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Form, Query, Request

__all__ = ["build_router", "ESRI_CAVEAT", "SERVICE_PATH"]

#: The honest one-liner, reused by the docs page so code and prose cannot drift.
ESRI_CAVEAT = (
    "Estos endpoints reproducen unicamente la FORMA JSON de findAddressCandidates y "
    "geocodeAddresses de ArcGIS REST. No constituyen un Locator de ArcGIS registrado "
    "ni certificado: eso exige implementar el Locator Provider SDK de Esri y registrarlo "
    "dentro del software ArcGIS, algo que un endpoint REST por si solo no puede cumplir."
)

SERVICE_PATH = "/arcgis/rest/services/CaliNormalizador/GeocodeServer"

#: Half-side of the synthetic candidate extent, in degrees (~55 m at this latitude).
#: ArcGIS clients use `extent` to zoom to a candidate; a cadastral centroid has no
#: real extent, so a small fixed box is the honest approximation.
_EXTENT_HALF_DEG = 0.0005

_SPATIAL_REFERENCE = {"wkid": 4326, "latestWkid": 4326}


def _text(value) -> str:
    """Esri attributes are strings; null becomes an empty string, not "None"."""
    return "" if value is None else str(value)


def _number(value) -> float | None:
    """A finite float, else None (NaN from a pandas round trip becomes null)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _attributes(row: dict) -> dict:
    return {
        "Status": _text(row.get("estado")),
        "Numero_Predial": _text(row.get("numero_predial_nacional")),
        "Manzana": _text(row.get("manzana")),
        "Barrio_Vereda": _text(row.get("barrio_vereda")),
        "Comuna_Corregimiento": _text(row.get("comuna_corregimiento")),
        "Motivo": _text(row.get("motivo")),
        "Confiabilidad": _number(row.get("confiabilidad")),
        "Margen": _number(row.get("margen")),
    }


def _score(row: dict) -> float:
    """Esri scores run 0-100. A non-OK row scores 0 so score filters work."""
    if row.get("estado") != "OK":
        return 0.0
    confidence = row.get("confianza")
    try:
        return round(float(confidence) * 100.0, 2)
    except (TypeError, ValueError):
        return 0.0


def _location(row: dict) -> dict:
    lon, lat = row.get("lon"), row.get("lat")
    try:
        return {"x": float(lon), "y": float(lat)}
    except (TypeError, ValueError):
        return {"x": None, "y": None}


def _extent(location: dict) -> dict | None:
    x, y = location.get("x"), location.get("y")
    if x is None or y is None:
        return None
    return {
        "xmin": x - _EXTENT_HALF_DEG, "ymin": y - _EXTENT_HALF_DEG,
        "xmax": x + _EXTENT_HALF_DEG, "ymax": y + _EXTENT_HALF_DEG,
    }


def _address_text(row: dict) -> str:
    """Prefer the normalized address; fall back to the input so the field is never empty."""
    return _text(row.get("direccion_normalizada") or row.get("direccion_entrada"))


def _candidate(row: dict) -> dict:
    location = _location(row)
    return {
        "address": _address_text(row),
        "location": location,
        "score": _score(row),
        "attributes": _attributes(row),
        "extent": _extent(location),
    }


def build_router(run_addresses: Callable[[Request, list], list[dict]]) -> APIRouter:
    """Wire the Esri-shaped endpoints to the engine.

    ``run_addresses(request, raws)`` must return one result row dict per input, in
    order, using the service defaults. Injected instead of imported so this module
    never depends on ``main``.
    """
    router = APIRouter(tags=["Compatibilidad Esri (best-effort)"])

    @router.get(
        f"{SERVICE_PATH}/findAddressCandidates",
        summary="Candidatos para una direccion (forma ArcGIS findAddressCandidates)",
        description=(
            "Normaliza UNA direccion y responde con la forma JSON de findAddressCandidates.\n\n"
            + ESRI_CAVEAT
            + "\n\nSiempre devuelve exactamente un candidato, incluso cuando no hubo "
            "coincidencia: en ese caso `score` es 0 y `attributes.Status` vale SIN_MATCH o "
            "NO_PARSEABLE. Filtre por `attributes.Status` o por `score` antes de usar la "
            "coordenada. Acepta `SingleLine`, `Address` o `text` como parametro de entrada."
        ),
    )
    async def find_address_candidates_get(
        request: Request,
        SingleLine: str | None = Query(None, description="Direccion en una sola linea."),
        Address: str | None = Query(None, description="Alias de SingleLine."),
        text: str | None = Query(None, description="Alias de SingleLine."),
        f: str = Query("json", description="Formato de salida; solo se admite json."),
    ):
        return _find_candidates(request, SingleLine or Address or text)

    @router.post(
        f"{SERVICE_PATH}/findAddressCandidates",
        summary="Candidatos para una direccion (POST form-encoded, como llaman los clientes ArcGIS)",
        description=(
            "Identico al GET. Existe porque los clientes ArcGIS envian estas consultas "
            "como POST con cuerpo `application/x-www-form-urlencoded`.\n\n" + ESRI_CAVEAT
        ),
    )
    async def find_address_candidates_post(
        request: Request,
        SingleLine: str | None = Form(None),
        Address: str | None = Form(None),
        text: str | None = Form(None),
        f: str = Form("json"),
    ):
        return _find_candidates(request, SingleLine or Address or text)

    def _find_candidates(request: Request, raw: str | None) -> dict:
        if raw is None or str(raw).strip() == "":
            # Esri clients treat an empty candidate list as "nothing found", which
            # is the truthful answer to an empty query.
            return {"spatialReference": _SPATIAL_REFERENCE, "candidates": []}
        rows = run_addresses(request, [raw])
        return {
            "spatialReference": _SPATIAL_REFERENCE,
            "candidates": [_candidate(rows[0])] if rows else [],
        }

    @router.post(
        f"{SERVICE_PATH}/geocodeAddresses",
        summary="Lote de direcciones (forma ArcGIS geocodeAddresses)",
        description=(
            "Recibe un record-set de Esri "
            '(`{"records": [{"attributes": {"OBJECTID": 1, "SingleLine": "..."}}]}`) '
            "y responde con `locations`, donde `attributes.ResultID` repite el OBJECTID "
            "de entrada.\n\n"
            + ESRI_CAVEAT
            + "\n\nCada entrada produce siempre una salida, en el mismo orden. "
            "Revise `attributes.Status` para distinguir una geocodificacion real de un rechazo."
        ),
    )
    async def geocode_addresses(request: Request, payload: dict[str, Any]):
        records = payload.get("records")
        if not isinstance(records, list):
            records = []
        raws: list[Any] = []
        result_ids: list[Any] = []
        for position, record in enumerate(records):
            attributes = record.get("attributes") if isinstance(record, dict) else None
            attributes = attributes if isinstance(attributes, dict) else {}
            raws.append(
                attributes.get("SingleLine")
                or attributes.get("Address")
                or attributes.get("text")
            )
            result_ids.append(attributes.get("OBJECTID", attributes.get("ResultID", position + 1)))
        rows = run_addresses(request, raws) if raws else []
        locations = []
        for result_id, row in zip(result_ids, rows):
            candidate = _candidate(row)
            candidate["attributes"] = {"ResultID": result_id, **candidate["attributes"]}
            candidate.pop("extent", None)
            locations.append(candidate)
        return {"spatialReference": _SPATIAL_REFERENCE, "locations": locations}

    return router
