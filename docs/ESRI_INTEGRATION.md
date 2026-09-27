# Integracion con ArcGIS / Esri

Este servicio ofrece dos endpoints que responden con la **forma JSON** de dos
operaciones de un ArcGIS REST Geocode Service, para que una herramienta que ya
sabe leer una respuesta de geocodificador Esri pueda consumir el normalizador sin
capa de traduccion.

---

## Lo que esto es y lo que no es

**Lo que es:** dos endpoints HTTP que devuelven `candidates` y `locations` con la
estructura `{"address", "location": {"x","y"}, "score", "attributes"}` y un
`spatialReference` con `wkid: 4326`.

**Lo que NO es: un Locator de ArcGIS.** Un Locator real es un proveedor
implementado contra el *Locator Provider SDK* de Esri y **registrado dentro del
software ArcGIS** (ArcGIS Server, ArcGIS Pro). Un endpoint REST por si solo no
puede cumplir ese contrato, por fiel que sea su JSON. Concretamente, este
servicio **no** implementa el resto de la superficie que un cliente ArcGIS puede
consultar o exigir:

- el documento de metadatos del servicio (`addressFields`, `candidateFields`,
  `locatorProperties`, `capabilities`, negociacion de `spatialReference`);
- `reverseGeocode`;
- `suggest` ni los tokens `magicKey`;
- proyeccion mediante `outSR`, seleccion de campos mediante `outFields`;
- filtrado por `searchExtent`;
- autenticacion por token para `geocodeAddresses` en lote;
- la taxonomia de campos `Addr_type` / `Match_addr` / `Loc_name` con la que
  ArcGIS clasifica y ordena resultados.

En consecuencia: **publicar esto como un item "Locator" de ArcGIS Enterprise
requiere configuracion y software del lado de Esri, fuera del alcance de este
servicio.** Si el objetivo final es un Locator institucional, el camino realista
es usar `/api/v1/normalize-json` para poblar una tabla o feature class y
construir el Locator con las herramientas de Esri sobre esos datos (`Create
Locator` de ArcGIS Pro), no apuntar ArcGIS a estos endpoints.

**Honestidad sobre los resultados.** Para no romper clientes que asumen al menos
un candidato, una direccion sin coincidencia **tambien** devuelve un candidato,
pero con `score` igual a `0`, `location` con `x` e `y` en `null` y
`attributes.Status` en `SIN_MATCH` o `NO_PARSEABLE`. Un cliente que filtre por
`score` o por `Status` puede distinguirlo sin ambiguedad. El servicio nunca
presenta un rechazo como una geocodificacion buena.

---

## `findAddressCandidates`

Una direccion por llamada. Acepta `SingleLine`, `Address` o `text` como nombre
del parametro, en `GET` (query string) o en `POST`
(`application/x-www-form-urlencoded`), porque asi es como llaman los clientes
ArcGIS.

```bash
curl -s -G "https://<app>.up.railway.app/arcgis/rest/services/CaliNormalizador/GeocodeServer/findAddressCandidates" \
  --data-urlencode "SingleLine=Carrera1#9-80" \
  --data-urlencode "f=json"
```

```json
{
  "spatialReference": {"wkid": 4326, "latestWkid": 4326},
  "candidates": [
    {
      "address": "KR 1 # 9 - 80",
      "location": {"x": -76.534302, "y": 3.452734},
      "score": 93.54,
      "attributes": {
        "Status": "OK",
        "Numero_Predial": "760010100031100010003000000000",
        "Manzana": "76001010003110001",
        "Barrio_Vereda": "San Pedro",
        "Comuna_Corregimiento": "Comuna 3",
        "Motivo": ""
      },
      "extent": {
        "xmin": -76.534802, "ymin": 3.452234,
        "xmax": -76.533802, "ymax": 3.453234
      }
    }
  ]
}
```

`extent` es una caja sintetica de unos 55 m de lado alrededor del centroide: un
centroide catastral no tiene extension propia, y `extent` solo se usa para que el
cliente encuadre el mapa.

Una direccion no resuelta:

```json
{
  "spatialReference": {"wkid": 4326, "latestWkid": 4326},
  "candidates": [
    {
      "address": "no es una direccion",
      "location": {"x": null, "y": null},
      "score": 0.0,
      "attributes": {
        "Status": "NO_PARSEABLE",
        "Numero_Predial": "", "Manzana": "",
        "Barrio_Vereda": "", "Comuna_Corregimiento": "",
        "Motivo": "gramatica de direccion no reconocida"
      },
      "extent": null
    }
  ]
}
```

Una consulta vacia devuelve `"candidates": []`, que es la respuesta veraz.

---

## `geocodeAddresses`

Lote, con el record-set de Esri. `attributes.ResultID` de la salida repite el
`OBJECTID` de la entrada, y siempre hay una salida por entrada, en el mismo
orden.

```bash
curl -s -X POST "https://<app>.up.railway.app/arcgis/rest/services/CaliNormalizador/GeocodeServer/geocodeAddresses" \
  -H "Content-Type: application/json" \
  -d '{"records": [
        {"attributes": {"OBJECTID": 1, "SingleLine": "Carrera1#9-80"}},
        {"attributes": {"OBJECTID": 2, "SingleLine": "Calle 5 # 38-25, San Fernando"}},
        {"attributes": {"OBJECTID": 3, "SingleLine": "asdfqwer"}}
      ]}'
```

```json
{
  "spatialReference": {"wkid": 4326, "latestWkid": 4326},
  "locations": [
    {
      "address": "KR 1 # 9 - 80",
      "location": {"x": -76.534302, "y": 3.452734},
      "score": 93.54,
      "attributes": {"ResultID": 1, "Status": "OK", "Numero_Predial": "760010100031100010003000000000",
                     "Manzana": "76001010003110001", "Barrio_Vereda": "San Pedro",
                     "Comuna_Corregimiento": "Comuna 3", "Motivo": ""}
    },
    {
      "address": "CL 5 # 38 - 25",
      "location": {"x": null, "y": null},
      "score": 0.0,
      "attributes": {"ResultID": 2, "Status": "SIN_MATCH", "Numero_Predial": "", "Manzana": "",
                     "Barrio_Vereda": "San Fernando Nuevo", "Comuna_Corregimiento": "Comuna 19",
                     "Motivo": "placa 25 != 21"}
    },
    {
      "address": "asdfqwer",
      "location": {"x": null, "y": null},
      "score": 0.0,
      "attributes": {"ResultID": 3, "Status": "NO_PARSEABLE", "Numero_Predial": "", "Manzana": "",
                     "Barrio_Vereda": "", "Comuna_Corregimiento": "",
                     "Motivo": "gramatica de direccion no reconocida"}
    }
  ]
}
```

Nota: en `locations` no se incluye `extent` (ArcGIS no lo espera en la respuesta
de lote).

---

## Uso recomendado desde ArcGIS Pro: `/api/v1/normalize-json` + `InsertCursor`

Esta es la via que recomendamos para trabajo real en ArcGIS, porque entrega todos
los campos del contrato (predial, manzana, barrio, comuna, motivo del rechazo) y
no solo lo que cabe en la forma de un geocodificador.

Pegue esto en la ventana de Python de ArcGIS Pro o en un notebook de ArcGIS.
Requiere `requests`, que viene en el entorno `arcgispro-py3`.

```python
"""Normaliza direcciones de una tabla y escribe los resultados en una feature class."""
import arcpy
import requests

BASE = "https://<app>.up.railway.app"
TABLA_ENTRADA = r"C:\datos\proyecto.gdb\inspecciones"   # tabla o feature class origen
CAMPO_DIRECCION = "direccion"
CAMPO_ID = "OBJECTID"
SALIDA_GDB = r"C:\datos\proyecto.gdb"
SALIDA_NOMBRE = "direcciones_normalizadas"
LOTE = 2000            # el servicio limita a MAX_ROWS (20 000) por peticion

# ---------------------------------------------------------------- 1. leer origen
filas = []
with arcpy.da.SearchCursor(TABLA_ENTRADA, [CAMPO_ID, CAMPO_DIRECCION]) as cursor:
    for id_origen, direccion in cursor:
        filas.append({"id_origen": id_origen, "direccion": direccion})
arcpy.AddMessage(f"{len(filas)} filas leidas de {TABLA_ENTRADA}")

# ------------------------------------------------------- 2. crear la salida
arcpy.management.CreateFeatureclass(
    SALIDA_GDB, SALIDA_NOMBRE, "POINT",
    spatial_reference=arcpy.SpatialReference(4326),
)
salida = f"{SALIDA_GDB}\\{SALIDA_NOMBRE}"
campos = [
    ("id_origen", "LONG", None),
    ("direccion_entrada", "TEXT", 255),
    ("estado", "TEXT", 20),
    ("motivo", "TEXT", 255),
    ("direccion_normalizada", "TEXT", 255),
    ("fuente_normalizacion", "TEXT", 20),
    ("numero_predial_nacional", "TEXT", 40),
    ("manzana", "TEXT", 30),
    ("confianza", "DOUBLE", None),
    ("barrio_vereda", "TEXT", 120),
    ("comuna_corregimiento", "TEXT", 120),
]
for nombre, tipo, longitud in campos:
    arcpy.management.AddField(salida, nombre, tipo, field_length=longitud)

campos_cursor = [nombre for nombre, _, _ in campos] + ["SHAPE@XY"]

# ------------------------------------------------- 3. normalizar por lotes
with arcpy.da.InsertCursor(salida, campos_cursor) as cursor:
    for inicio in range(0, len(filas), LOTE):
        lote = filas[inicio:inicio + LOTE]
        respuesta = requests.post(
            f"{BASE}/api/v1/normalize-json",
            json={"records": lote, "address_field": "direccion"},
            timeout=900,
        )
        respuesta.raise_for_status()
        datos = respuesta.json()
        arcpy.AddMessage(f"lote {inicio // LOTE + 1}: {datos['summary']}")

        # results conserva el orden de entrada, asi que se puede emparejar directo.
        for origen, resultado in zip(lote, datos["results"]):
            lon, lat = resultado.get("lon"), resultado.get("lat")
            # Una fila sin coincidencia se inserta igual, con geometria nula:
            # perder las filas rechazadas esconde el trabajo que falta revisar.
            geometria = (lon, lat) if lon is not None and lat is not None else None
            cursor.insertRow((
                origen["id_origen"],
                resultado.get("direccion_entrada"),
                resultado.get("estado"),
                resultado.get("motivo"),
                resultado.get("direccion_normalizada"),
                resultado.get("fuente_normalizacion"),
                resultado.get("numero_predial_nacional"),
                resultado.get("manzana"),
                resultado.get("confianza"),
                resultado.get("barrio_vereda"),
                resultado.get("comuna_corregimiento"),
                geometria,
            ))

arcpy.AddMessage(f"listo: {salida}")
```

Despues de esto, `estado = 'OK'` es la seleccion geocodificada y confiable, y
`estado <> 'OK'` con el campo `motivo` es la lista de trabajo pendiente de
revision manual.

---

## Uso de los endpoints con forma Esri desde un script propio

Para una herramienta de bajo codigo que ya habla el JSON del geocodificador Esri,
basta apuntarla a la URL del servicio. Desde un script propio:

```python
import requests

GEOCODER = ("https://<app>.up.railway.app"
            "/arcgis/rest/services/CaliNormalizador/GeocodeServer")

def geocodificar_una(direccion):
    respuesta = requests.get(
        f"{GEOCODER}/findAddressCandidates",
        params={"SingleLine": direccion, "f": "json"},
        timeout=120,
    )
    respuesta.raise_for_status()
    candidatos = respuesta.json()["candidates"]
    if not candidatos:
        return None
    mejor = candidatos[0]
    # Verificacion obligatoria: un rechazo tambien vuelve como candidato.
    if mejor["attributes"]["Status"] != "OK":
        return None
    return mejor["location"]["x"], mejor["location"]["y"], mejor["score"]


def geocodificar_lote(direcciones):
    """Devuelve {OBJECTID: (x, y, score)} solo para las que quedaron en OK."""
    cuerpo = {"records": [
        {"attributes": {"OBJECTID": i + 1, "SingleLine": d}}
        for i, d in enumerate(direcciones)
    ]}
    respuesta = requests.post(f"{GEOCODER}/geocodeAddresses", json=cuerpo, timeout=900)
    respuesta.raise_for_status()
    salida = {}
    for ubicacion in respuesta.json()["locations"]:
        atributos = ubicacion["attributes"]
        if atributos["Status"] == "OK":
            salida[atributos["ResultID"]] = (
                ubicacion["location"]["x"], ubicacion["location"]["y"], ubicacion["score"],
            )
    return salida


print(geocodificar_una("Carrera1#9-80"))
print(geocodificar_lote(["Carrera1#9-80", "Calle 5 # 38-25", "asdfqwer"]))
```

El patron importante en ambas funciones es el mismo: **revisar
`attributes.Status` antes de usar la coordenada.** Un `score` de 0 y unas
coordenadas nulas significan que el normalizador no asigno predio, no que la
direccion este en el origen de coordenadas.

---

## Sistema de referencia

Todas las coordenadas que devuelve el servicio estan en **EPSG:4326**
(longitud/latitud en grados, `x` = longitud, `y` = latitud). Si su geodatabase
trabaja en MAGNA-SIRGAS / Cali (EPSG:3115 o 3116), proyecte despues de insertar
con `arcpy.management.Project`, o cree la feature class directamente en el sistema
de destino y deje que ArcGIS haga la transformacion al insertar geometrias con un
`arcpy.SpatialReference(4326)` explicito.
