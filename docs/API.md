# API del normalizador de direcciones de Cali

Servicio HTTP que normaliza direcciones de Santiago de Cali contra el catastro
(330 387 direcciones indexadas), devuelve el numero predial nacional, la manzana,
la coordenada del centroide del predio y el barrio/vereda y comuna/corregimiento
oficiales del IDESC.

El servicio expone tambien `/docs` (Swagger UI generado por FastAPI, con todos
los esquemas de peticion y respuesta) y `/docs-integracion`, que es esta misma
pagina renderizada en HTML.

---

## Contrato de estado

Cada fila de salida trae un `estado` con exactamente uno de tres valores:

| `estado` | Significado |
| --- | --- |
| `OK` | La direccion se parseo, coincidio con un predio por encima del umbral de confianza calibrado, el predio cae dentro del lugar detectado en el texto y la direccion catastral concuerda estructuralmente con la entrada. |
| `SIN_MATCH` | No se puede asignar ningun predio. El campo `motivo` dice por que: dato vacio o marcador sin informacion, confianza por debajo del umbral, una regla estructural violada por el mejor candidato, o todos los candidatos fuera del barrio/comuna/corregimiento detectado. |
| `NO_PARSEABLE` | La gramatica de direcciones no reconocio el texto. Las columnas de zona se llenan igual si se detecto un lugar. |

**Solo las filas `OK` traen `numero_predial_nacional`, `manzana`, `lat`, `lon`,
`confianza`, `confiabilidad`, `confiabilidad_manzana` y `margen`.** Una fila que no pasa el umbral, la reja geografica o las reglas
estructurales no recibe candidato alguno. Esto es deliberado: evita que una
conjetura de baja calidad se confunda con una coincidencia real. El motivo exacto
del rechazo siempre queda en `motivo`.

**Confiabilidad.** `confiabilidad` y `confiabilidad_manzana` son probabilidades
calibradas (regresion logistica pequena, `artifacts/reliability.json`, ajustada con
`scripts/fit_reliability.py`) de que el predio, respectivamente la manzana, entregados
sean los correctos. Son **solo informativas**: el `estado` no depende de ellas. Se
estimaron sobre un conjunto de validacion pequeno (~530 filas con verdad terreno) y las
etiquetas de predio son ruidosas (verdad terreno por GPS; predios vecinos a ~6 m), por lo
que `confiabilidad` tiende a **subestimar** la probabilidad real y discrimina poco entre
filas `OK`. Si `reliability.json` no esta disponible, ambas salen `null` (nunca se
sustituyen por `confianza`). `reliability.json` admite dos formatos (campo `kind`): `logistic` (el actual; sin `kind` se trata como `logistic`) y `cells` (tabla de tasas por tipo de fila y banda de `margen`, suavizadas hacia la tasa global; ver `scripts/fit_reliability.py --kind`).

`fuente_normalizacion` indica de donde sale `direccion_normalizada`:

- `catastro` — el registro catastral que coincidio (solo en filas `OK`);
- `reglas` — la forma canonica producida por la gramatica, en filas parseables
  pero sin coincidencia;
- `null` — no hubo nada que normalizar.

### Columnas de salida

| Columna | Tipo | Descripcion |
| --- | --- | --- |
| `direccion_entrada` | texto | El texto original recibido. |
| `estado` | texto | `OK`, `SIN_MATCH` o `NO_PARSEABLE`. |
| `motivo` | texto | Razon del rechazo. En filas `OK` es vacio o una nota: `aproximado: <regla>` (letra de via presente en un solo lado, o placa a 1-2 unidades; solo garantiza manzana) y/o `direccion compartida por N predios` (se reporta el primer predio por OBJECTID). |
| `nivel_precision` | texto o null | `predio`, `via`, `esquina`, `manzana` (aproximado o direccion compartida) o `direccion` (compartida entre manzanas); null si no es `OK`. |
| `direccion_normalizada` | texto o null | Direccion catastral o forma canonica por reglas. |
| `fuente_normalizacion` | texto o null | `catastro`, `reglas` o null. |
| `numero_predial_nacional` | texto o null | 30 digitos del predio. |
| `manzana` | texto o null | Codigo de manzana catastral. |
| `lat` | numero o null | Latitud del centroide del predio (EPSG:4326). |
| `lon` | numero o null | Longitud del centroide del predio (EPSG:4326). |
| `confianza` | numero o null | Puntaje fusionado 0-1, **sin calibrar** (no es una probabilidad: en el rango 0.9-1.0 solo ~64 % acierta el predio exacto). |
| `confiabilidad` | numero o null | Probabilidad **calibrada** (0-1) de que `numero_predial_nacional` sea el predio correcto. |
| `confiabilidad_manzana` | numero o null | Probabilidad calibrada (0-1) de que `manzana` sea la correcta; siempre `>= confiabilidad`. |
| `margen` | numero o null | Diferencia de puntaje fusionado entre el predio entregado y el mejor candidato alternativo que tambien pasa las reglas y esta en **otra manzana**. Cuanto menor, mas ambigua la manzana; `null` si no hay competidor o la fila no es `OK`. Con el `ambiguity_delta` por defecto (0.02) las filas con `margen < 0.02` pasan a `SIN_MATCH`; con `ambiguity_delta=0` es solo informativo. En la validacion, las filas `OK` con `margen < 0.05` (~5 % de las `OK`) acertaron la manzana solo ~29 % de las veces frente a ~84 % en general. |
| `barrio_vereda` | texto o null | Nombre oficial IDESC del barrio o de la vereda. |
| `comuna_corregimiento` | texto o null | `Comuna 19` o el nombre del corregimiento. |

---

## Parametros del motor

Todos los endpoints de normalizacion aceptan los mismos parametros. Los valores
por omision son los calibrados sobre la particion de validacion catastral
retenida; cambielos solo con una razon medida.

| Parametro | Omision | Que hace |
| --- | --- | --- |
| `min_struct` | `0.6` | Coincidencia estructural minima (0-1) exigida al candidato. Por debajo, la fila pasa a `SIN_MATCH` con el motivo `coincidencia estructural X < Y`. |
| `plate_tolerance` | `0` | Diferencia maxima aceptada entre la placa de entrada y la del predio, entre `0` y `2`. `0` exige placa exacta; valores mas altos no se aceptan porque amplian falsos OK. Dentro de la tolerancia no hay violacion (nivel `predio`); una diferencia de 1-2 unidades por encima de ella se acepta como `aproximado` (nivel `manzana`). |
| `barrio_buffer` | `1000` | Holgura en metros alrededor del barrio detectado en el texto. |
| `zone_buffer` | `300` | Holgura en metros alrededor de la comuna o corregimiento detectado. |
| `gate_escalate` | `true` | Si ningun candidato queda dentro del barrio, ampliar a la comuna o corregimiento en vez de descartar la fila. |
| `ambiguity_delta` | `0.02` | Abstencion por ambiguedad, entre `0` y `0.2`. `0` la desactiva (los estados no cambian). Si es mayor que 0, una fila `OK` cuyo `margen` sea **menor** que este valor pasa a `SIN_MATCH` con el motivo `ambiguo: N candidatos cercanos en otras manzanas`. Reduce las `OK` a cambio de quitar filas con mayor probabilidad de error; una fila sin competidor (`margen` `null`) nunca se abstiene. En la CLI es `--ambiguity-delta` (por defecto 0.02). |
| `gazetteer` | `true` | Usar el gazetteer IDESC para limpiar el texto (`Barrio Siloé calle 1 # 2-3` no parsea; `calle 1 # 2-3` si) y aplicar la reja geografica. |

El umbral de confianza es **`0.73`**, calibrado y leido de `artifacts/tuning.json`;
no es un parametro de la peticion, por lo que una integracion no puede relajarlo
por accidente. Se puede consultar en `/health`.

Por que `barrio_buffer` vale 1000 y no 200: medido sobre `stickers` (3 697 filas
con coordenada GPS del inspector), la reja de 200 m retenia 138 filas de las
cuales el 86 % estaba a menos de 100 m del punto GPS registrado. Los nombres de
barrio de ese archivo vienen de un geocodificador inverso cuyas extensiones
coloquiales no coinciden con los poligonos oficiales del IDESC: el barrio del
texto concuerda con el poligono del predio solo el 77 % de las veces, mientras la
comuna concuerda el 92 %. Con 1000 m mas escalamiento se conservan 2 436 filas
`OK` con la misma proporcion (3,1 %) de errores mayores a 500 m.

---

## `GET /health`

Estado del servicio. Devuelve **503** mientras el modelo se esta cargando o si la
carga fallo, y **200** cuando el servicio puede atender. Es el endpoint que debe
apuntar el health check de la plataforma.

```bash
curl -s https://<app>.up.railway.app/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu",
  "catastro_size": 330387,
  "gazetteer_loaded": true,
  "threshold": 0.73,
  "detail": null
}
```

Mientras carga (503):

```json
{
  "status": "loading",
  "model_loaded": false,
  "device": "cpu",
  "catastro_size": 0,
  "gazetteer_loaded": false,
  "threshold": null,
  "detail": "cargando modelo e indice catastral"
}
```

---

## `POST /api/v1/inspect`

Lee un archivo **sin normalizarlo** y responde con lo que hace falta saber para
la siguiente llamada: formato detectado, columnas, columna de direccion sugerida,
todas las columnas puntuadas y una muestra de hasta 5 filas.

Este endpoint nunca falla por ambiguedad de columnas. Es exactamente el paso con
el que un cliente (la interfaz web o un sistema externo) descubre que enviar
despues.

**Cuerpo:** `multipart/form-data`

| Campo | Requerido | Descripcion |
| --- | --- | --- |
| `file` | si | El archivo. |
| `sheet` | no | Hoja a leer, cuando es un libro de Excel. |
| `header` | no | Fila de encabezado 0-based. Si se omite, se infiere. |

### Formatos aceptados

- `.xlsx`, `.xls` — se lee con `pandas.read_excel`. Si no se envia `header`, se
  detecta la fila de encabezado: se toma la fila con mas celdas no vacias y se
  desempata a favor de la que parezca una fila de rotulos cortos en vez de datos.
  Esto resuelve los exportes de ArcGIS de este proyecto, que traen 2 a 5 filas de
  metadatos antes del encabezado real.
- `.csv` — se intenta UTF-8 y luego latin-1; el delimitador (`,` o `;`) se
  detecta contando ocurrencias en la primera linea no vacia.
- `.geojson`, `.json` — debe ser un `FeatureCollection`. Las `properties` de cada
  feature se vuelven columnas y el centroide de la geometria se agrega como
  `_lon` y `_lat`.
- `.zip` con un shapefile — debe contener `.shp`, `.shx` y `.dbf`; el `.prj` es
  opcional y, si esta, se usa para reproyectar los centroides a EPSG:4326. Sin
  `.prj` se asume EPSG:4326 y se avisa en `warnings`. Un `.shp` suelto se rechaza
  con un mensaje que pide comprimir los tres (o cuatro) archivos juntos.

```bash
curl -s -X POST https://<app>.up.railway.app/api/v1/inspect \
  -F "file=@inspecciones.xlsx"
```

```json
{
  "format": "xlsx",
  "row_count": 2218,
  "columns": ["id_edan", "ObjectID", "direccion", "direccion_norm", "barrio"],
  "suggested_column": "direccion",
  "candidates": [
    {"column": "direccion", "score": 100.0},
    {"column": "direccion_norm", "score": 95.0},
    {"column": "barrio", "score": 40.0},
    {"column": "ObjectID", "score": 33.3},
    {"column": "id_edan", "score": 30.0}
  ],
  "preview": [{"id_edan": "QRYEU", "direccion": "KR 38 BIS # 5B2 09", "barrio": "San Fernando Nuevo"}],
  "warnings": ["fila de encabezado detectada automaticamente: 2"],
  "header_row": 2,
  "sheets": ["inspecciones"]
}
```

`candidates` esta ordenado de mayor a menor puntaje e incluye **todas** las
columnas, de modo que sirve directamente para poblar un desplegable.
`suggested_column` es `null` cuando ninguna columna alcanza un puntaje plausible:
en ese caso la interfaz debe pedirle al usuario que elija.

---

## `POST /api/v1/normalize`

Normaliza un archivo completo.

**Cuerpo:** `multipart/form-data`

| Campo | Omision | Descripcion |
| --- | --- | --- |
| `file` | — | El archivo. |
| `sheet` | primera hoja | Hoja de Excel. |
| `header` | inferido | Fila de encabezado 0-based. |
| `address_column` | detectada | Columna de direccion. |
| `output_format` | `json` | `json`, `xlsx`, `csv` o `geojson`. |
| `min_struct`, `plate_tolerance`, `barrio_buffer`, `zone_buffer`, `gate_escalate`, `ambiguity_delta`, `gazetteer` | ver arriba | Parametros del motor. |

### Salida `json`

```bash
curl -s -X POST https://<app>.up.railway.app/api/v1/normalize \
  -F "file=@inspecciones.xlsx" \
  -F "address_column=direccion" \
  -F "output_format=json"
```

```json
{
  "summary": {"OK": 1487, "SIN_MATCH": 612, "NO_PARSEABLE": 119, "total": 2218},
  "address_column": "direccion",
  "warnings": ["fila de encabezado detectada automaticamente: 2"],
  "results": [
    {
      "direccion_entrada": "Carrera1#9-80",
      "estado": "OK",
      "motivo": "",
      "direccion_normalizada": "KR 1 # 9 - 80",
      "fuente_normalizacion": "catastro",
      "numero_predial_nacional": "760010100031100010003000000000",
      "manzana": "76001010003110001",
      "lat": 3.452734,
      "lon": -76.534302,
      "confianza": 0.9354,
      "confiabilidad": 0.66,
      "confiabilidad_manzana": 0.86,
      "barrio_vereda": "San Pedro",
      "comuna_corregimiento": "Comuna 3"
    },
    {
      "direccion_entrada": "Calle 5 # 38-25, San Fernando",
      "estado": "SIN_MATCH",
      "motivo": "placa 25 != 21",
      "direccion_normalizada": "CL 5 # 38 - 25",
      "fuente_normalizacion": "reglas",
      "numero_predial_nacional": null,
      "manzana": null,
      "lat": null,
      "lon": null,
      "confianza": null,
      "confiabilidad": null,
      "confiabilidad_manzana": null,
      "barrio_vereda": "San Fernando Nuevo",
      "comuna_corregimiento": "Comuna 19"
    }
  ]
}
```

### Salidas de archivo

Con `output_format=xlsx`, `csv` o `geojson` la respuesta es el archivo, con
`Content-Disposition: attachment`.

- **xlsx** — tres hojas: `normalizado` (todas las filas), `revisar` (solo las que
  no quedaron en `OK`) y `resumen` (conteo por estado). Es el mismo libro que
  produce `scripts/normalizar.py --output salida.xlsx`.
- **csv** — la tabla plana, codificada en UTF-8 con BOM para que Excel la abra
  bien en Windows.
- **geojson** — un `FeatureCollection` con geometria `Point` donde hubo
  coordenada y `"geometry": null` donde no. Las filas nunca se descartan: el
  numero de features siempre coincide con el de filas, de modo que se puede ver
  que no se pudo ubicar. Las demas columnas van en `properties`.

```bash
curl -X POST https://<app>.up.railway.app/api/v1/normalize \
  -F "file=@inspecciones.xlsx" -F "address_column=direccion" \
  -F "output_format=xlsx" -o inspecciones_normalizado.xlsx
```

---

## `POST /api/v1/normalize-json`

Punto de integracion para otras fuentes que **ya tienen los datos en memoria** y
no quieren subir un archivo. Se envia uno de dos cuerpos, nunca los dos:

```json
{"records": [{"id": 1, "dir": "Calle 5 # 38-25"}], "address_field": "dir"}
```

```json
{"addresses": ["Calle 5 # 38-25", "Carrera 1 # 9-80"]}
```

Los parametros del motor van en el mismo objeto JSON (`"min_struct": 0.7`, etc.).
La respuesta tiene la misma forma que `/api/v1/normalize` con
`output_format=json`, y `results` conserva el orden de entrada.

```bash
curl -s -X POST https://<app>.up.railway.app/api/v1/normalize-json \
  -H "Content-Type: application/json" \
  -d '{"records": [{"id": 1, "dir": "Carrera1#9-80"}], "address_field": "dir"}'
```

### Ejemplo en Python con `requests`

```python
import requests

BASE = "https://<app>.up.railway.app"

# Filas que ya vienen de una base de datos, un CSV leido localmente, etc.
filas = [
    {"id": 1, "direccion": "Carrera1#9-80"},
    {"id": 2, "direccion": "Calle 5 # 38-25, San Fernando"},
    {"id": 3, "direccion": "Corregimiento los andes, vereda la reforma, casa 117"},
]

respuesta = requests.post(
    f"{BASE}/api/v1/normalize-json",
    json={
        "records": filas,
        "address_field": "direccion",
        "barrio_buffer": 1000,
        "zone_buffer": 300,
        "gate_escalate": True,
    },
    timeout=600,
)
respuesta.raise_for_status()
datos = respuesta.json()

print(datos["summary"])
# Los resultados vienen en el mismo orden que las filas enviadas.
for fila, salida in zip(filas, datos["results"]):
    if salida["estado"] == "OK":
        print(fila["id"], salida["numero_predial_nacional"],
              salida["lat"], salida["lon"], salida["barrio_vereda"])
    else:
        print(fila["id"], salida["estado"], salida["motivo"])
```

Para volumenes grandes, envie lotes de unos pocos miles de filas: el limite por
peticion es `MAX_ROWS` (20 000 por omision) y el servicio corre con un solo
worker.

---

## `POST /api/v1/normalize-address`

Alias minimo del anterior: solo acepta `addresses`. Existe para pruebas de humo e
integraciones triviales, y responde con la misma forma.

```bash
curl -s -X POST https://<app>.up.railway.app/api/v1/normalize-address \
  -H "Content-Type: application/json" \
  -d '{"addresses": ["Carrera1#9-80", "no es una direccion"]}'
```

```json
{
  "summary": {"OK": 1, "SIN_MATCH": 0, "NO_PARSEABLE": 1, "total": 2},
  "address_column": null,
  "warnings": [],
  "results": [
    {"direccion_entrada": "Carrera1#9-80", "estado": "OK", "confianza": 0.9354, "...": "..."},
    {"direccion_entrada": "no es una direccion", "estado": "NO_PARSEABLE", "...": "..."}
  ]
}
```

---

## Endpoints con forma Esri

`GET|POST /arcgis/rest/services/CaliNormalizador/GeocodeServer/findAddressCandidates`
y
`POST /arcgis/rest/services/CaliNormalizador/GeocodeServer/geocodeAddresses`
responden con la forma JSON de las operaciones homonimas de ArcGIS REST.

**Reproducen unicamente la forma de la respuesta. No constituyen un Locator de
ArcGIS registrado ni certificado.** Los detalles y las limitaciones estan en
[ESRI_INTEGRATION.md](ESRI_INTEGRATION.md).

---

## Errores

Todos los errores traen un cuerpo JSON. Los tres que una integracion debe manejar:

### 422 — hay que definir la columna de direccion

Se devuelve cuando no se envio `address_column` (o `address_field`) y la
deteccion automatica no es concluyente: ninguna columna es plausible, o hay
varias igualmente plausibles.

```json
{
  "error": "address_column_required",
  "message": "varias columnas podrian ser la direccion; indique cual usar",
  "requested_column": null,
  "available_columns": ["id", "direccion_bien", "domicilio", "barrio"],
  "candidates": [
    {"column": "direccion_bien", "score": 95.0},
    {"column": "domicilio", "score": 95.0}
  ]
}
```

La reaccion correcta es mostrar `candidates` al usuario y repetir la peticion con
`address_column`. Si se envio un nombre que no existe, `requested_column` lo
repite y `available_columns` lista lo que si hay.

El mismo codigo 422 se usa cuando el archivo excede `MAX_ROWS`, con un cuerpo
`{"detail": "el archivo trae N filas y el limite por peticion es M..."}`.

### 413 — archivo demasiado grande

```json
{
  "error": "payload_too_large",
  "message": "el cuerpo de la peticion pesa 48.2 MB y el limite es 25 MB"
}
```

El limite se configura con `MAX_UPLOAD_MB` (25 por omision).

### 400 — archivo invalido

```json
{
  "error": "bad_file",
  "message": "el .zip no contiene un shapefile completo; falta .dbf. Comprima juntos .shp, .shx y .dbf (y .prj si lo tiene)."
}
```

Tambien cubre un archivo de 0 bytes, un JSON malformado, un GeoJSON que no es
`FeatureCollection` y una extension no soportada.

### 503 — el servicio aun no esta listo

El modelo se carga en un hilo aparte para que el puerto quede disponible de
inmediato. Mientras eso ocurre, `/health` responde 503 con
`"status": "loading"` y los endpoints de normalizacion esperan hasta
`WARMUP_TIMEOUT_S` segundos antes de responder 503. Un cliente debe reintentar.

---

## Variables de entorno

| Variable | Omision | Para que sirve |
| --- | --- | --- |
| `NORMALIZER_DEVICE` | `cpu` | `cpu` o `cuda`. |
| `ARTIFACTS_DIR` | `<raiz>/artifacts` | Directorio con `model.pt`, `catastro_emb.pt`, `catastro_docs.parquet`, `tuning.json`, `gazetteer.pkl`, `reliability.json`. |
| `BASEMAPS_DIR` | `<raiz>/basemaps` | Directorio con los dos GeoJSON del IDESC. |
| `DOCS_DIR` | `<raiz>/docs` | Directorio desde donde se sirve `/docs-integracion`. |
| `MAX_UPLOAD_MB` | `25` | Tamano maximo del cuerpo de la peticion. |
| `MAX_ROWS` | `20000` | Filas maximas por peticion de normalizacion. |
| `WARMUP_TIMEOUT_S` | `300` | Segundos que una peticion espera a que el modelo termine de cargar. |

---

## Nota de privacidad

Ningun valor por omision de este servicio apunta a `context/` ni a `outputs/`.
Esos directorios contienen datos reales de ciudadanos (registro de inmuebles
asegurados, registro unico de damnificados con nombres y numeros de documento,
reportes de inspeccion de campo) y estan excluidos en `.gitignore`,
`.dockerignore` y `.railwayignore`: nunca deben subirse a un repositorio, ni
incluirse en una imagen de contenedor, ni desplegarse. Los bytes que llegan por
`multipart/form-data` se procesan en memoria y se descartan al terminar la
peticion; el servicio no escribe archivos subidos en disco.
