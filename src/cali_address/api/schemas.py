"""Request and response models for the normalization service.

Field descriptions are in Spanish on purpose: FastAPI renders them in
``/docs``, which is the documentation an integrator reads first.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..service import CLI_BARRIO_BUFFER_M, DEFAULT_MIN_STRUCT
from ..gazetteer import ZONE_BUFFER_M

__all__ = [
    "AddressColumnErrorResponse",
    "AddressesRequest",
    "ColumnCandidate",
    "ErrorResponse",
    "HealthResponse",
    "InspectResponse",
    "NormalizeJsonRequest",
    "NormalizeResponse",
    "ResultRow",
    "Summary",
    "Tunables",
]


class Tunables(BaseModel):
    """Los parametros del motor. Los valores por omision son los calibrados."""

    min_struct: float = Field(
        DEFAULT_MIN_STRUCT, ge=0.0, le=1.0,
        description="Coincidencia estructural minima (0-1) exigida al candidato catastral.",
    )
    plate_tolerance: int = Field(
        0, ge=0, le=2,
        description="Diferencia maxima aceptada en la placa. 0 exige placa exacta.",
    )
    barrio_buffer: float = Field(
        CLI_BARRIO_BUFFER_M, ge=0.0, le=20000.0,
        description="Holgura en metros alrededor del barrio detectado en el texto.",
    )
    zone_buffer: float = Field(
        ZONE_BUFFER_M, ge=0.0, le=20000.0,
        description="Holgura en metros alrededor de la comuna o corregimiento detectado.",
    )
    gate_escalate: bool = Field(
        True,
        description="Si ningun candidato queda dentro del barrio, ampliar a la comuna "
                    "o corregimiento en lugar de descartar la fila.",
    )
    ambiguity_delta: float = Field(
        0.02, ge=0.0, le=0.2,
        description="Abstencion por ambiguedad (0 = desactivada). Si es mayor que 0, una "
                    "fila cuyo margen frente al mejor candidato de otra manzana sea menor que este "
                    "valor pasa a SIN_MATCH con motivo 'ambiguo: ...'.",
    )
    gazetteer: bool = Field(
        True,
        description="Usar el gazetteer IDESC para limpiar el texto y aplicar la reja geografica.",
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"] = Field(description="Estado del servicio.")
    model_loaded: bool = Field(description="True cuando el modelo ya esta en memoria.")
    device: str = Field(description="Dispositivo de inferencia en uso (cpu o cuda).")
    catastro_size: int = Field(description="Cantidad de direcciones catastrales indexadas.")
    gazetteer_loaded: bool = Field(description="True si los basemaps IDESC estan disponibles.")
    threshold: float | None = Field(
        None, description="Umbral de confianza calibrado con el que se acepta una coincidencia."
    )
    detail: str | None = Field(None, description="Motivo cuando el estado no es 'ok'.")


class ColumnCandidate(BaseModel):
    column: str = Field(description="Nombre de la columna tal como viene en el archivo.")
    score: float = Field(description="Puntaje 0-100 de parecido con una columna de direccion.")


class InspectResponse(BaseModel):
    """Lo que se necesita saber antes de normalizar: columnas y muestra."""

    format: Literal["xlsx", "csv", "geojson", "shp"] = Field(
        description="Formato detectado a partir de la extension."
    )
    row_count: int = Field(description="Filas de datos leidas.")
    columns: list[str] = Field(description="Todas las columnas del archivo, en orden.")
    suggested_column: str | None = Field(
        description="Mejor candidata a columna de direccion, o null si ninguna es plausible."
    )
    candidates: list[ColumnCandidate] = Field(
        description="Todas las columnas ordenadas por puntaje descendente."
    )
    preview: list[dict[str, Any]] = Field(description="Primeras filas del archivo (maximo 5).")
    warnings: list[str] = Field(description="Avisos de lectura: encabezado inferido, reproyeccion, etc.")
    header_row: int | None = Field(
        None, description="Fila de encabezado utilizada (solo xlsx/csv)."
    )
    sheets: list[str] | None = Field(
        None, description="Hojas disponibles cuando el archivo es un libro de Excel."
    )


class ResultRow(BaseModel):
    """Una fila normalizada. Es el mismo contrato que produce la CLI."""

    direccion_entrada: Any | None = Field(description="Texto original recibido.")
    estado: Literal["OK", "SIN_MATCH", "NO_PARSEABLE"] = Field(
        description="OK solo si paso umbral, reja geografica y reglas estructurales."
    )
    motivo: str | None = Field(
        description="Razon del rechazo; en filas OK vacio o una nota: 'aproximado: <regla "
                    "blanda>' (letra de via de un solo lado o placa a 1-2 unidades) y/o "
                    "'direccion compartida por N predios'."
    )
    direccion_normalizada: str | None = Field(
        description="Direccion catastral cuando hay coincidencia, o la forma canonica por reglas."
    )
    fuente_normalizacion: str | None = Field(
        description="'catastro' (coincidencia real), 'reglas' (solo gramatica) o null."
    )
    numero_predial_nacional: str | None = Field(description="Numero predial nacional del predio.")
    manzana: str | None = Field(description="Codigo de manzana catastral.")
    lat: float | None = Field(description="Latitud del centroide del predio (EPSG:4326).")
    lon: float | None = Field(description="Longitud del centroide del predio (EPSG:4326).")
    confianza: float | None = Field(description="Puntaje fusionado 0-1 de la coincidencia (sin calibrar).")
    confiabilidad: float | None = Field(
        description="Probabilidad calibrada (0-1) de que el predio entregado sea el correcto. "
                    "Estimada sobre un conjunto de validacion pequeno con etiquetas ruidosas "
                    "(tiende a subestimar); solo informativa, no cambia el estado. Null si el "
                    "estado no es OK o si el modelo de calibracion no esta disponible."
    )
    confiabilidad_manzana: float | None = Field(
        description="Probabilidad calibrada (0-1) de que la manzana entregada sea la correcta; "
                    "siempre >= confiabilidad. Misma salvedad y mismos nulos que confiabilidad."
    )
    margen: float | None = Field(
        None,
        description="Diferencia de puntaje fusionado entre el predio entregado y el mejor candidato "
                    "alternativo que tambien pasa las reglas y esta en OTRA manzana. Cuanto menor, "
                    "mas ambigua la eleccion de manzana. Null si no hay competidor o si el estado "
                    "no es OK. Solo informativo."
    )
    nivel_precision: str | None = Field(
        description="Que tan confirmado quedo el predio: 'esquina' (sin placa), "
                    "'via' (sin numero de cruce), 'manzana' (coincidencia aproximada o "
                    "direccion compartida en una manzana), 'direccion' (direccion "
                    "compartida entre manzanas) o 'predio' (via, cruce y placa "
                    "confirmados por el texto de entrada); null si el estado no es OK."
    )
    barrio_vereda: str | None = Field(description="Nombre oficial IDESC del barrio o vereda.")
    comuna_corregimiento: str | None = Field(
        description="Nombre oficial de la comuna ('Comuna 19') o del corregimiento."
    )

    model_config = {"extra": "allow"}


class Summary(BaseModel):
    OK: int = Field(description="Filas con coincidencia catastral aceptada.")
    SIN_MATCH: int = Field(description="Filas sin coincidencia asignable.")
    NO_PARSEABLE: int = Field(description="Filas cuya gramatica de direccion no se reconocio.")
    total: int = Field(description="Filas procesadas.")


class NormalizeResponse(BaseModel):
    summary: Summary
    results: list[ResultRow]
    warnings: list[str] = Field(default_factory=list, description="Avisos de lectura del archivo.")
    address_column: str | None = Field(
        None, description="Columna de direccion efectivamente utilizada."
    )


class AddressColumnErrorResponse(BaseModel):
    """Cuerpo del 422 cuando hay que elegir la columna de direccion."""

    error: Literal["address_column_required"]
    message: str
    requested_column: str | None = None
    available_columns: list[str]
    candidates: list[ColumnCandidate]


class ErrorResponse(BaseModel):
    error: str
    message: str


class NormalizeJsonRequest(Tunables):
    """Datos tabulares ya en memoria: `records` + `address_field`, o `addresses`."""

    records: list[dict[str, Any]] | None = Field(
        None, description="Filas completas. Requiere address_field."
    )
    address_field: str | None = Field(
        None, description="Clave de cada record que contiene la direccion."
    )
    addresses: list[str | None] | None = Field(
        None, description="Lista simple de direcciones, sin campos adicionales."
    )

    @model_validator(mode="after")
    def _exactly_one_source(self):
        has_records = self.records is not None
        has_addresses = self.addresses is not None
        if has_records == has_addresses:
            raise ValueError("indique exactamente uno de 'records' o 'addresses'")
        if has_records and not (self.address_field or "").strip():
            raise ValueError("'records' requiere 'address_field'")
        return self


class AddressesRequest(Tunables):
    """Alias minimo: solo una lista de direcciones."""

    addresses: list[str | None] = Field(description="Direcciones a normalizar.")
