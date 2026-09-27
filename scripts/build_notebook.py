"""Builds `direcciones_ANN.ipynb` from the modules in `src/` using nbformat.

Markdown narration is in neutral professional Spanish; code, identifiers and code
comments are in English.
"""

from __future__ import annotations

import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH = os.path.join(ROOT, "direcciones_ANN.ipynb")

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


def read_module(rel: str) -> str:
    with open(os.path.join(ROOT, rel), "r", encoding="utf-8") as fh:
        return fh.read()


# ===========================================================================
md(
    """
# Normalizador y geocodificador de direcciones de Cali con una red neuronal espacial

**Objetivo.** Dada *cualquier* cadena de texto libre que pretenda describir una
dirección en Santiago de Cali, el sistema debe devolver:

1. la **dirección canónica catastral** (formato `KR 26 H 1 # 73 - 10`),
2. el **predio** que mejor corresponde (`numero_predial_nacional`, manzana y centroide),
3. una **confianza** calibrada,

y todo ello debe estar **evaluado con rigor** contra datos externos reales.

## El problema

En Cali conviven al menos tres escrituras de la misma dirección:

| Origen | Ejemplo |
|---|---|
| Catastro (canónico, registro real) | `KR 26 H 1 # 73 - 10` |
| Forma canónica de nuestro parser para `Cra. 98f #98-66` | `KR 98 F # 98 - 66` |
| Geocodificador municipal IDESC para la misma entrada | `KR 98F # 98 - 66` |
| Texto capturado en campo / formularios | `Cra. 98f #98-66`, `Carrera1#9-80`, `KR 13 60 34 38`, `Calle 67n 2a 50` |

(La segunda y la tercera fila son *salidas de normalización* de la misma entrada,
no registros catastrales: `KR 98 F # 98 - 66` no existe en la capa. La diferencia
entre ellas es solo el pegado de la letra al número, y el §3 la cuantifica.)

El texto de campo pierde el `#`, pega los tokens, expande o abrevia el tipo de vía,
agrega sufijos administrativos (`, Cali, Valle del Cauca`), introduce erratas
(`Catrera`, `Oeate`) y mezcla complementos (`TORRE 2 APTO 301`). Un normalizador
puramente sintáctico resuelve buena parte, pero **no resuelve la
georreferenciación**: al final hay que decidir *qué predio* de 338.311 es el
correcto, y allí la información determinante es espacial, no ortográfica.

## Estrategia

1. **Base catastral** (`urbano_terreno`, 338.311 predios con polígono y centroide)
   como universo de verdad y como índice de recuperación.
2. **Canonizador basado en reglas** (gramática de direcciones colombianas) que
   estructura el texto y produce la forma canónica.
3. **Modelo neuronal**: un *dual-encoder* Transformer a nivel de carácter,
   entrenado con InfoNCE y **negativos duros espaciales** (direcciones de la
   misma manzana o de centroides vecinos), más cabezas auxiliares de
   **regresión de coordenadas** y **clasificación de comuna/barrio**.
4. **Reranking fusionado** (similitud del modelo + `rapidfuzz` + acuerdo estructural).
5. **Evaluación** sobre 5 conjuntos externos (2.078 direcciones muestreadas con
   semilla fija), con verdad de terreno por *point-in-polygon* donde hay
   coordenadas, y comparación contra el geocodificador oficial IDESC.
"""
)

md(
    """
## 0. Entorno, semillas y GPU

Todo el trabajo pesado está **cacheado en `artifacts/`**: la descarga catastral,
la caché de IDESC, los tensores de entrenamiento, los pesos del modelo y la matriz
de embeddings. Volver a ejecutar el notebook reutiliza la caché; poner
`FORCE_RETRAIN = True` fuerza el reentrenamiento.
"""
)

code(
    '''
import json
import os
import random
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

ROOT = os.path.abspath(".")
SRC = os.path.join(ROOT, "src")
ARTIFACTS = os.path.join(ROOT, "artifacts")
CONTEXT = os.path.join(ROOT, "context")
os.makedirs(ARTIFACTS, exist_ok=True)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

FORCE_RETRAIN = False          # set True to retrain from scratch
FORCE_REDOWNLOAD = False       # set True to re-download the cadastral layer

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
sns.set_theme(style="whitegrid", context="notebook")

print(f"python           : {sys.version.split()[0]}")
print(f"torch            : {torch.__version__}")
print(f"cuda available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"gpu              : {props.name}")
    print(f"compute cap.     : {props.major}.{props.minor}")
    print(f"total memory     : {props.total_memory / 1e9:.2f} GB")
    print(f"bf16 supported   : {torch.cuda.is_bf16_supported()}")
    print(f"cudnn available  : {torch.backends.cudnn.is_available()}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
'''
)

# ---------------------------------------------------------------- ingestion
md(
    """
## 1. Ingesta de la base catastral

La capa `urbano_terreno` se publica en un **ArcGIS FeatureServer** de la Alcaldía:

```
https://services8.arcgis.com/ljfiJpg35HWgdtaC/arcgis/rest/services/
    Revison_Reportes_Ciudadanos_V2_WFL1/FeatureServer/0
```

Características relevantes del servicio, verificadas empíricamente:

* `maxRecordCount = 2000`, por lo que hay que paginar con `resultOffset` /
  `resultRecordCount` y un `orderByFields=OBJECTID` estable.
* `returnCentroid=true` entrega el centroide sin necesidad de calcularlo.
* `outSR=4326` y `geometryPrecision=6` reducen el tamaño de la respuesta
  manteniendo precisión de ~0,1 m.

El script `scripts/download_catastro.py` descarga las 171 páginas con 4 hilos,
reintentos con *backoff* exponencial y caché por página, convierte los anillos
ArcGIS a WKB de Shapely y escribe un único parquet.
"""
)

code(
    '''
CATASTRO_PATH = os.path.join(ARTIFACTS, "catastro.parquet")
if FORCE_REDOWNLOAD or not os.path.exists(CATASTRO_PATH):
    print("downloading the cadastral layer (takes a few minutes)...")
    import subprocess

    subprocess.run([sys.executable, "scripts/download_catastro.py"], check=True, cwd=ROOT)

catastro = pd.read_parquet(CATASTRO_PATH)
with open(os.path.join(ARTIFACTS, "catastro_meta.json"), encoding="utf-8") as fh:
    catastro_meta = json.load(fh)

print(json.dumps(catastro_meta, indent=2))
print(f"\\nrows: {len(catastro):,}   columns: {list(catastro.columns)}")
catastro.head(5)
'''
)

code(
    '''
# Completeness of the fields we depend on.
summary = pd.DataFrame(
    {
        "non_null": catastro.notna().sum(),
        "null": catastro.isna().sum(),
        "null_pct": (catastro.isna().mean() * 100).round(3),
        "n_unique": catastro.nunique(dropna=True),
    }
)
display(summary)
print(f"unique direccion strings : {catastro['direccion'].nunique():,}")
print(f"unique predial codes     : {catastro['numero_predial_nacional'].nunique():,}")
print(f"unique manzana codes     : {catastro['numero_predial_manzana'].nunique():,}")
print(f"records with a polygon   : {catastro['geom_wkb'].notna().sum():,}")
'''
)

md(
    """
**Lectura.** La capa trae 338.311 predios; `direccion` está presente en 337.287
(99,7 %) y el polígono en 338.306. `direccion_py` es prácticamente nula, por lo
que no aporta. Hay 330.459 cadenas de dirección distintas, es decir **1,02
predios por dirección**: la recuperación por cadena es casi equivalente a la
recuperación de predio, y las pocas colisiones corresponden a propiedad
horizontal con el mismo texto base.
"""
)

md(
    """
### 1.1 Verificación empírica del código predial de 30 dígitos

El `numero_predial_nacional` es el identificador catastral nacional. La
composición teórica es: departamento (2), municipio (3), zona (2), sector (2),
comuna (2), barrio (2), manzana (4), terreno (4), condición (1), edificio (2),
piso (2), unidad (4). Conviene **verificarlo** antes de derivar jerarquía
espacial de él.
"""
)

code(
    '''
pred = catastro["numero_predial_nacional"].astype(str)
LAYOUT = {
    "departamento": (0, 2), "municipio": (2, 5), "zona": (5, 7), "sector": (7, 9),
    "comuna": (9, 11), "barrio": (11, 13), "manzana": (13, 17), "terreno": (17, 21),
    "condicion": (21, 22), "edificio": (22, 24), "piso": (24, 26), "unidad": (26, 30),
}
rows = []
for name, (a, b) in LAYOUT.items():
    vc = pred.str[a:b].value_counts()
    rows.append(
        {
            "field": name, "digits": f"[{a}:{b}]", "n_unique": len(vc),
            "top_values": ", ".join(f"{k}({v:,})" for k, v in vc.head(4).items()),
        }
    )
display(pd.DataFrame(rows))

# The layer also exposes `numero_predial_manzana`; if the layout is right it must
# equal the first 17 digits.
match = (catastro["numero_predial_manzana"].astype(str) == pred.str[:17]).mean()
print(f"numero_predial_manzana == predial[:17] -> {match:.4%}")
print(f"string length distribution: {pred.str.len().value_counts().to_dict()}")
'''
)

code(
    '''
# Cross-check against IDESC, which returns a comuna and a 4-digit barrio code.
probe = catastro.assign(comuna=pred.str[9:11], barrio=pred.str[11:13],
                        barrio_code=pred.str[9:13])
for needle in ["CL 12 A # 56", "CL 5 # 38 -"]:
    hit = probe[probe["direccion"].astype(str).str.startswith(needle)]
    print(f"{needle!r}: {len(hit)} records ->",
          hit[["comuna", "barrio", "barrio_code"]].drop_duplicates().head(4).to_dict("records"))
print("  IDESC answers for the same two addresses: comuna 17 / barrio code 1707, "
      "and comuna 19 / barrio code 1906")
print()
print("comuna code distribution (digits [9:11]):")
display(probe["comuna"].value_counts().sort_index().to_frame("n_predios").T)
'''
)

md(
    """
**Lectura parcial.** El corte de los primeros 17 dígitos queda confirmado:

* `numero_predial_manzana` coincide con los **primeros 17 dígitos** en el 99,18 %
  de los registros, lo que valida
  `departamento(2)+municipio(3)+zona(2)+sector(2)+comuna(2)+barrio(2)+manzana(4) = 17`.
* Los dígitos `[0:2]` son `76` (Valle del Cauca) y `[2:5]` son `001` (Cali) en
  todos los registros válidos; `zona` es `01` (urbano) y `sector` es siempre `00`.
* Los dígitos **`[9:11]` son la comuna**: toman exactamente los valores `01`…`22`
  (más 5 registros basura con la cadena `None` y uno con `56`).

La comparación puntual contra IDESC es **ambigua y no basta**. Para
`CL 12 A # 56 - …` el catastro da comuna `17` y código de barrio `1778`, mientras
IDESC responde comuna `17` y `1707`: la comuna coincide, el barrio no. Para
`CL 5 # 38 - …` el catastro tiene *dos* códigos (`1906` y `1908`) y solo uno de
ellos coincide con el `1906` de IDESC. Dos ejemplos escogidos a mano no prueban
nada: la sección 7.3.1 mide el acuerdo sobre toda la muestra de evaluación antes
de afirmar cualquier cosa sobre el barrio.
"""
)

md("### 1.2 Estadística de formato de las direcciones catastrales")

code(
    '''
d = catastro["direccion"].dropna().astype(str).str.strip()
prefix = d.str.split().str[0].value_counts()
top_prefix = prefix.head(12)

fig, axes = plt.subplots(1, 3, figsize=(19, 4.6))
top_prefix.plot(kind="bar", ax=axes[0], color=sns.color_palette("viridis", len(top_prefix)))
axes[0].set_title("Via-type prefix frequency (top 12)")
axes[0].set_ylabel("records")
axes[0].tick_params(axis="x", rotation=45)

d.str.len().plot(kind="hist", bins=60, ax=axes[1], color="#3b7dd8")
axes[1].set_title("Address string length")
axes[1].set_xlabel("characters")

tok = d.str.split().str.len()
tok.plot(kind="hist", bins=range(2, 22), ax=axes[2], color="#d1615d")
axes[2].set_title("Token count per address")
axes[2].set_xlabel("whitespace tokens")
plt.tight_layout()
plt.show()

print("prefix distribution (all values):")
print(prefix.to_dict())
print()
print(f"length: mean={d.str.len().mean():.1f} p50={d.str.len().median():.0f} "
      f"p99={d.str.len().quantile(0.99):.0f} max={d.str.len().max()}")
print(f"addresses containing '#'      : {d.str.contains('#').mean():.2%}")
print(f"addresses containing ' - '    : {d.str.contains(' - ').mean():.2%}")
print(f"addresses containing 'BIS'    : {d.str.contains(r'\\bBIS\\b').mean():.2%}")
print(f"addresses containing 'NORTE'  : {d.str.contains(r'\\bNORTE\\b').mean():.2%}")
print(f"addresses containing 'OESTE'  : {d.str.contains(r'\\bOESTE\\b').mean():.2%}")
print(f"addresses containing 'SUR'    : {d.str.contains(r'\\bSUR\\b').sum()} records")
print(f"addresses containing 'ESTE'   : {d.str.contains(r'\\bESTE\\b').sum()} records")
'''
)

md(
    """
**Lectura.** Dos hallazgos que condicionan el diseño del parser:

1. **Prefijos heredados.** `KR` (166.481) y `CL` (130.827) dominan, pero coexisten
   con formas de una letra: `K` = KR (14.720), `C` = CL (10.031), `A` = AV (1.041),
   `D` = DG (312), `T` = TV (131). El canonizador debe expandirlas.
2. **Los cuadrantes se escriben completos.** `NORTE` aparece 30.335 veces y
   `OESTE` 21.451, mientras que `ESTE` aparece **0** veces y `SUR` **1**. Por lo
   tanto, una letra suelta después del número (`KR 41 E`, `C 60 N`, `K 73 B # 2 W`)
   **no** es un cuadrante abreviado sino una *letra de vía*. Esta decisión, que
   parece menor, cambia el round-trip del parser en más de 8 puntos.
"""
)

md("### 1.3 Distribución espacial de los centroides")

code(
    '''
geo = catastro.dropna(subset=["centroid_lon", "centroid_lat"]).copy()
geo["comuna"] = geo["numero_predial_nacional"].astype(str).str[9:11]
geo = geo[geo["comuna"].str.fullmatch(r"\\d{2}")]
plot_sample = geo.sample(min(120_000, len(geo)), random_state=SEED)

fig, axes = plt.subplots(1, 2, figsize=(17, 8))
comunas = sorted(plot_sample["comuna"].unique())
palette = dict(zip(comunas, sns.color_palette("tab20", len(comunas))))
for c in comunas:
    sub = plot_sample[plot_sample["comuna"] == c]
    axes[0].scatter(sub["centroid_lon"], sub["centroid_lat"], s=0.6,
                    color=palette[c], label=c, linewidths=0)
axes[0].set_title(f"Parcel centroids by comuna (n={len(plot_sample):,})")
axes[0].set_xlabel("longitude"); axes[0].set_ylabel("latitude")
axes[0].legend(markerscale=12, ncol=2, fontsize=7, title="comuna", loc="upper left")
axes[0].set_aspect("equal")

hb = axes[1].hexbin(plot_sample["centroid_lon"], plot_sample["centroid_lat"],
                    gridsize=90, cmap="magma", bins="log")
axes[1].set_title("Parcel density (log scale)")
axes[1].set_xlabel("longitude")
axes[1].set_aspect("equal")
fig.colorbar(hb, ax=axes[1], label="parcels per cell")
plt.tight_layout()
plt.show()

print(geo.groupby("comuna").agg(
    n_predios=("OBJECTID", "size"),
    lon_min=("centroid_lon", "min"), lon_max=("centroid_lon", "max"),
    lat_min=("centroid_lat", "min"), lat_max=("centroid_lat", "max"),
).head(25))
'''
)

md(
    """
**Lectura.** Las comunas forman agrupaciones espacialmente compactas y
contiguas: es exactamente la señal que queremos inyectar en el embedding a
través de la cabeza de clasificación de comuna. El mapa de densidad muestra el
núcleo urbano denso (comunas 3, 8, 9, 11, 12) y la expansión hacia el sur
(comunas 17, 22) con manzanas más grandes. El bloque desplazado hacia el
occidente corresponde a las comunas de ladera (1, 18, 20) y explica por qué el
cuadrante `OESTE` es tan frecuente.
"""
)

md("### 1.4 Formatos atípicos y registros degenerados")

code(
    '''
LEGACY = set("KCADTBPUVLSEZMJ")
odd = {
    "legacy one-letter prefix": d[d.str.split().str[0].isin(LEGACY)],
    "no cross number (`#  -`)": d[d.str.contains(r"#\\s*-\\s*$")],
    "spelled-out via type": d[d.str.split().str[0].isin(["CALLE", "CARRERA", "AVENIDA", "TRANSVERSAL"])],
    "free text (VIA/LOTE/CGTO/ZONA)": d[d.str.split().str[0].isin(["VIA", "LOTE", "CGTO", "ZONA", "COR"])],
    "secondary via number (`KR 26 H 1`)": d[d.str.contains(r"^[A-Z]+ \\d+ [A-Z] \\d+ #")],
    "cross with its own via type": d[d.str.contains(r"# (?:AV|T|K|C|D|TV|KR|CL|DG) ")],
    "very long (>40 chars)": d[d.str.len() > 40],
}
for name, sub in odd.items():
    print(f"{name:38s} n={len(sub):7,}  e.g. {sub.head(3).tolist()}")
'''
)

md(
    """
**Lectura.** Hay tres familias de ruido dentro de la propia base catastral:
prefijos de una letra, registros degenerados sin cruce ni placa (`B  #  -`,
`U  #  -`, que son marcas administrativas sin dirección real) y unas decenas de
descripciones en texto libre (`VIA PUBLICA K 3E-1 NORTE ENTRE C 71I Y 72`,
`LOTE DE TERRENO CESION OBLIGATORIA ANDEN`). El parser debe **fallar
explícitamente** en estos últimos en vez de inventar una estructura.
"""
)

# ------------------------------------------------- parser / feature engineering
md(
    """
## 2. Ingeniería de características: canonizador basado en reglas

El primer bloque del sistema es un **parser gramatical** que convierte texto
libre en campos estructurados y los vuelve a renderizar en la forma canónica
catastral. Se escribe a `src/cali_address/parser.py` desde el propio notebook
(`%%writefile`) para que el cuaderno sea autocontenido, y luego se importa.

Gramática:

```
address    := via "#" cross "-" plate [complement]
via        := via_type number [letters] [number] [letters] [BIS] [quadrant]
cross      := [via_type] number [letters] [number] [letters] [BIS] [quadrant]
plate      := digits                # se rellena a 2 dígitos en la forma canónica
complement := texto libre (AP 501, LC 1, TO 3, ED PAMPLONA, GASS 144, ...)
```

Decisiones de diseño que salieron de mirar los datos:

* El `#` y el `-` de placa se localizan **antes** de cualquier separación de
  tokens pegados, de modo que el complemento se preserva literal (`14C` no se
  parte en `14 C`).
* La placa se normaliza a dos dígitos con cero a la izquierda, porque así lo hace
  el catastro (29.452 placas de dos dígitos empiezan por `0`, frente a 1.839
  placas de un dígito sin rellenar).
* Una letra suelta tras el número es *letra de vía*, no cuadrante (ver 1.2).
* Los tipos de vía mal escritos se reparan por distancia de edición
  (`CATRERA` → `KR`), igual que los cuadrantes (`OEATE` → `OESTE`).
* Entrada no interpretable ⇒ `parse_ok = False`. El parser **nunca** lanza excepción.
"""
)

code("%%writefile src/cali_address/parser.py\n" + read_module("src/cali_address/parser.py"))

code(
    '''
import importlib

import cali_address.parser as parser_mod

importlib.reload(parser_mod)
from cali_address.parser import (
    COMPLEMENT_CANON, VIA_TYPE_CANON, canonical, canonicalize, match_key,
    parse_address, parse_many,
)

demo = [
    "KR 26 H 1 # 73 - 10",                                  # already canonical
    "Cra. 98f #98-66",                                       # abbreviated + glued
    "CL12A # 56-04",                                         # glued via type
    "Carrera1#9-80",                                         # fully glued
    "Calle 9c 49 141",                                        # missing '#'
    "KR 13 60 34 38",                                         # missing '#' and '-'
    "Calle 10 no. 42a 02",                                    # 'no.' as '#'
    "Av 6 Oeate #22-14",                                      # quadrant typo
    "Catrera 36 5b3-65",                                      # via-type typo
    "Cl. 55b # 47-55, Navarro, Cali, Valle del Cauca",        # administrative tail
    "CALLE 5 B2 # 38-91; CRA 37 #4A BIS 49",                  # two addresses
    "Calle 3ra # 5 - 10",                                     # ordinal noise
    "K 47B  # 55 B  - Q2 T",                                  # legacy prefix
    "C 9 W  # 0  -",                                          # legacy + no plate
    "CL 25 NORTE # AV 6 - 30",                                # cross with via type
    "Av 5 An #23 Dn 68",                                      # glued cardinal suffix
    "CRA 100 1B OESTE 110 TORRE 7 APTO 402",                  # complement
    "16758424",                                               # junk
    "Parte alta pichinde via la leonera",                     # free text, not an address
    None,                                                     # null
]
rows = []
for raw in demo:
    p = parse_address(raw)
    rows.append(
        {
            "raw": raw, "parse_ok": p.parse_ok,
            "canonical (catastro style)": canonical(p, style="spaced"),
            "canonical (IDESC style)": canonical(p, style="glued"),
            "via": f"{p.via_type or ''} {p.via_number or ''} {p.via_letters or ''}".strip(),
            "via_quad": p.via_quadrant, "cross": p.cross_number,
            "cross_let": p.cross_letters, "cross_quad": p.cross_quadrant,
            "bis": p.via_bis or p.cross_bis, "plate": p.plate,
            "complement": p.complement, "notes": ",".join(p.notes),
        }
    )
display(pd.DataFrame(rows))
print(f"via-type aliases registered : {len(VIA_TYPE_CANON)}")
print(f"complement kinds registered : {len(COMPLEMENT_CANON)}")
'''
)

md(
    """
**Lectura.** El parser resuelve las diez familias de ruido observadas en los
datos de campo, y —lo más importante— *distingue* lo que no es una dirección:
`16758424` y `Parte alta pichinde via la leonera` salen con `parse_ok=False`
en vez de producir una estructura inventada. Nótese también que produce las dos
variantes de escritura de letras: la del catastro (`KR 98 F`) y la de IDESC
(`KR 98F`), lo que permite comparar contra ambas fuentes sin sesgo de formato.
"""
)

md("### 2.1 Pruebas unitarias con casos límite")

code("!python -m pytest -q tests")

md(
    """
**Lectura.** La batería cubre explícitamente: entradas vacías / nulas / con solo
espacios, basura numérica y textual, tokens pegados, ausencia de `#`, variantes
de `No.`/`Nro`/`N°`/`num`, cuadrantes (incluyendo erratas y la regla de la letra
suelta), `BIS`, ruido ordinal, cadenas con varias direcciones, unicode y
acentos, complementos muy largos, prefijos heredados, relleno de la placa,
idempotencia de la forma canónica, un *fuzz test* de 3.000 cadenas aleatorias
que verifica que el parser nunca lanza excepción, y entradas de 5.000
caracteres.
"""
)

md("### 2.2 Round-trip sobre la propia base catastral")

code(
    '''
# Property under test: parse(direccion) -> canonical() must reproduce the
# cadastral string (after whitespace collapsing).
rt_sample = d.sample(60_000, random_state=SEED)
LEGACY = set("KCADTBPUVLSEZMJ")

t0 = time.time()
records = []
for s in rt_sample:
    p = parse_address(s)
    out = canonical(p, style="spaced", with_complement=True)
    target = " ".join(s.split())
    records.append((s, out, out == target, s.split()[0] not in LEGACY, p.parse_ok))
rt = pd.DataFrame(records, columns=["target", "output", "exact", "modern_prefix", "parse_ok"])
elapsed = time.time() - t0

print(f"parsed {len(rt):,} cadastral addresses in {elapsed:.1f}s "
      f"({len(rt)/elapsed:,.0f} addresses/s)")
print(f"parse_ok rate                              : {rt['parse_ok'].mean():.4%}")
print(f"round-trip exact (all records)             : {rt['exact'].mean():.4%}")
print(f"round-trip exact (modern prefix only)      : {rt[rt.modern_prefix]['exact'].mean():.4%}"
      f"   (n={int(rt.modern_prefix.sum()):,})")
print(f"round-trip exact (legacy one-letter prefix): {rt[~rt.modern_prefix]['exact'].mean():.4%}"
      f"   (n={int((~rt.modern_prefix).sum()):,})")

idem = rt[rt["output"] != ""]["output"].map(lambda s: canonicalize(s) == s)
print(f"idempotence canonical(canonical(x)) == canonical(x): {idem.mean():.4%}")

print("\\nround-trip failures with a MODERN prefix (the only real defects):")
display(rt[(~rt.exact) & rt.modern_prefix][["target", "output"]].head(12))
print("round-trip 'failures' with a LEGACY prefix (intended normalization):")
display(rt[(~rt.exact) & (~rt.modern_prefix)][["target", "output"]].head(6))
'''
)

md(
    """
**Lectura.** Sobre 60.000 direcciones catastrales el parser reproduce la cadena
original **exactamente en el 99,8 % de los registros con prefijo moderno**. El
valor global (~92 %) es más bajo únicamente porque los registros con prefijo de
una letra se *normalizan a propósito* (`K 39 # 1 B - 57` → `KR 39 # 1 B - 57`):
eso no es un error, es el comportamiento deseado, y se muestra por separado. La
forma canónica es además idempotente, propiedad necesaria para poder comparar
cadenas entre fuentes.
"""
)

# ------------------------------------------------------------------- IDESC
md(
    """
## 3. Referencia externa: ingeniería inversa del geocodificador IDESC

La Alcaldía de Cali publica un geocodificador en
`https://geocodificador.ia.cali.gov.co`. No hay documentación pública de su API,
así que se obtuvo del *bundle* JavaScript de la aplicación
(`/assets/index-DxZufy6u.js`), donde aparecen las rutas REST:

| Ruta (`/geoapi/proceso/...`) | Función |
|---|---|
| `geocodificador_individual` | normaliza y geocodifica una dirección |
| `geo_inversa_individual` | geocodificación inversa desde una coordenada |
| `geocodificador_archivo` | carga masiva por archivo |
| `validar_avance` | progreso del trabajo masivo |
| `descargar_archivo` | descarga del resultado masivo |

El navegador adjunta un `captchaToken` de reCAPTCHA v3 en el cuerpo JSON, pero
**el servidor no lo valida**: un `POST {"direccion": "<texto>"}` con
`Content-Type: application/json` devuelve la misma respuesta. La latencia es de
1,5–4 s por llamada, de modo que se consulta **solo el conjunto de evaluación**
(1.922 direcciones únicas) con 4 hilos y caché en disco
(`artifacts/idesc_cache.json`), para no volver a consultar nunca lo ya consultado.

Estructura de la respuesta: `data.datosUbicacion` con `estado`, `estrato`,
`comuna`, `direcciones.{direccion, dir_ajusta}`, `barrio.{codigo, nombre}` y
`geo_localizacion.{longitud, latitud}`. `dir_ajusta` es la dirección normalizada
en la misma familia canónica del catastro (con las letras pegadas al número).
"""
)

code(
    '''
from cali_address.idesc import ESTADO_CLASSES, IdescClient

idesc = IdescClient(os.path.join(ARTIFACTS, "idesc_cache.json"), max_workers=4, timeout=60)
print(f"cached IDESC responses: {len(idesc.cache):,}")

# Four illustrative calls, served from the cache.
probe_addresses = ["Cra. 98f #98-66", "CL12A # 56-04", "Calle 46-45", "SIN ESPECIFICAR"]
probe_rows = []
for a in probe_addresses:
    flat = IdescClient.flatten(idesc.query(a))
    flat["input"] = a
    flat["our_canonical_glued"] = canonicalize(a, style="glued")
    probe_rows.append(flat)
display(pd.DataFrame(probe_rows)[
    ["input", "idesc_estado", "idesc_dir_ajusta", "our_canonical_glued",
     "idesc_comuna", "idesc_barrio_codigo", "idesc_barrio_nombre", "idesc_lat", "idesc_lon"]
])
'''
)

code(
    '''
idesc_df = pd.read_parquet(os.path.join(ARTIFACTS, "idesc_results.parquet"))
print(f"IDESC answered {idesc_df['idesc_ok'].mean():.2%} of {len(idesc_df):,} unique addresses")

estado_tbl = idesc_df.groupby("idesc_estado").agg(
    n=("raw_address", "size"),
    with_coordinates=("idesc_lat", lambda s: int(s.notna().sum())),
    with_comuna=("idesc_comuna", lambda s: int(s.notna().sum())),
).assign(share=lambda t: t["n"] / len(idesc_df))
display(estado_tbl)

print(f"documented estado classes in our client : {ESTADO_CLASSES}")
print(f"normalization rate (non-empty dir_ajusta): "
      f"{(idesc_df['idesc_dir_ajusta'].fillna('').str.strip() != '').mean():.2%}")
print(f"georeferencing rate (lat/lon returned)   : {idesc_df['idesc_lat'].notna().mean():.2%}")
'''
)

md(
    """
**Lectura — el hallazgo que justifica todo el proyecto.** IDESC **normaliza el
texto del 100 % de las direcciones** que le enviamos, pero solo entrega
coordenadas para el **47,4 %**. Aparece además una clase de `estado` que no
estaba documentada en nuestras notas iniciales:

| `estado` | n | coordenadas |
|---|---|---|
| `A - Normalizado y georreferenciado exacto` | 532 | sí |
| `C - Normalizado y georreferenciado aproximado` | 379 | sí (aproximadas) |
| `D - Normalizado y no georreferenciado` | 552 | no |
| `F - Normalizado por cruce y no georreferenciado` | 459 | no |

Es decir: más de la mitad de las direcciones reales de estos conjuntos quedan
**sin georreferenciar** por el servicio oficial, y de las que sí obtienen
coordenadas, el 42 % son aproximadas. Ahí es donde un modelo de recuperación
aprendido contra la base catastral aporta valor: no compite en normalización
ortográfica, compite en **resolver el predio**.

Obsérvese también que `dir_ajusta` usa el estilo con letras pegadas
(`KR 98F # 98 - 66`) mientras el catastro las separa (`KR 98 F # 98 - 66`).
Nuestro renderizador soporta ambos estilos, lo que permite comparar sin
penalizar por convención tipográfica.
"""
)

# ------------------------------------------------------- training data
md(
    """
## 4. Construcción del conjunto de entrenamiento con aumentación

No existe un corpus etiquetado de (texto sucio → predio). Se construye uno
**invirtiendo el problema**: se parte de la dirección canónica catastral y se le
aplica un **modelo estocástico de corrupción** calibrado con los patrones reales
observados en los cinco conjuntos de evaluación.

Cada operación del corruptor corresponde a un patrón documentado:

| Operación | Patrón real observado |
|---|---|
| expandir/abreviar tipo de vía | `Cra. 61a # 9-16`, `CARRERA 62C#6A-26`, `K 47B` |
| quitar o sustituir el `#` | `KR 13 60 34 38`, `Calle 10 No. 42 - 02` |
| cambiar el separador de placa | `CL 5 # 14_58`, `Calle 12 #56–25` |
| pegar tokens | `Calle12a#56-04`, `Carrera1#9-80` |
| abreviar cuadrantes | `Cra5 norte # 33n -01`, `Av. 4 Nte. #37 Norte-48` |
| sufijo administrativo | `, Cali, Valle del Cauca` |
| sufijo de barrio | `, Navarro`, `- SILOE`, `san bosco` |
| agregar/quitar complemento | `TORRE 7 APTO 402`, `BQ 01` |
| erratas de teclado | `Catrera 36`, `Oeate` |
| ruido ordinal | `CL 1 RA D OESTE` |
| duplicar la dirección en la celda | `CR 7 b #69-82 CARRERA 7 b # 69-82` |
| minúsculas / Título | `calle 9C # 53-25 torre G` |
"""
)

code("%%writefile src/cali_address/augment.py\n" + read_module("src/cali_address/augment.py"))

code(
    '''
import cali_address.augment as augment_mod

importlib.reload(augment_mod)
from cali_address.augment import OPERATIONS, corrupt, corrupt_many

rng = random.Random(SEED)
print(f"corruption operations: {len(OPERATIONS)} -> {OPERATIONS}\\n")
for base in ["KR 26 H 1 # 73 - 10", "CL 12 # 48 BIS - 31", "AV 15 OESTE # 9 OESTE - 137"]:
    print(f"canonical: {base}")
    for variant in corrupt_many(base, 5, rng):
        print(f"   {variant!r:<62} -> parsed back: {canonicalize(variant)!r}")
    print()
'''
)

md(
    """
El conjunto de documentos (índice de recuperación) se construye colapsando la
base catastral por cadena de dirección; para cada documento se conservan el
predial representativo, la manzana, el centroide, la comuna, el barrio y las
coordenadas proyectadas a **EPSG:3116 (MAGNA-SIRGAS / Colombia Bogotá zone)**,
que es la CRS métrica usada por la cabeza de regresión.

La **partición de validación se hace por documento**, no por consulta: el 4 % de
las direcciones catastrales se aparta completamente, así que **ninguna variante
ruidosa de un documento de validación se usa nunca como consulta ni como positivo
durante el entrenamiento**. Es una partición más estricta que la simple
separación por registro.

Matiz honesto sobre el lado de los documentos: los negativos duros espaciales se
muestrean de toda la ciudad, de modo que una dirección de validación **sí puede
aparecer como negativo** de un ancla de entrenamiento. El modelo nunca recibe
señal de "esta consulta corresponde a este documento" para los documentos
apartados —que es la señal que la métrica mide—, pero sí puede haber visto su
texto en el rol de distractor. Es una fuga débil en el lado del índice, no en el
lado de la etiqueta; se deja constancia en vez de afirmar una separación total.
"""
)

code(
    '''
from cali_address.dataset import N_VARIANTS, build_docs, build_spatial_negatives, build_training_arrays

DOCS_PATH = os.path.join(ARTIFACTS, "catastro_docs.parquet")
NEG_PATH = os.path.join(ARTIFACTS, "spatial_negatives.npy")
TRAIN_PATH = os.path.join(ARTIFACTS, "train_data.npz")

if FORCE_RETRAIN or not (os.path.exists(DOCS_PATH) and os.path.exists(NEG_PATH)
                         and os.path.exists(TRAIN_PATH)):
    print("building the document index, spatial negatives and augmented tensors...")
    import subprocess

    subprocess.run([sys.executable, "scripts/prepare_training.py"], check=True, cwd=ROOT)

docs = pd.read_parquet(DOCS_PATH)
spatial_neg = np.load(NEG_PATH)
train_data = {k: v for k, v in np.load(TRAIN_PATH, allow_pickle=True).items()}

print(f"documents (unique cadastral addresses): {len(docs):,}")
print(f"variants per document                 : {N_VARIANTS}")
print(f"training queries                      : {train_data['q_tokens'].shape}")
print(f"  of which held out for validation    : {int(train_data['q_is_val'].sum()):,}")
print(f"spatial hard-negative table           : {spatial_neg.shape}")
print(f"comuna classes / barrio classes       : "
      f"{int(train_data['doc_comuna'].max())+1} / {int(train_data['doc_barrio'].max())+1}")
display(docs.head(4))
'''
)

code(
    '''
# Sanity check on the spatial hard negatives: slot 0-1 should be same-block
# addresses and therefore extremely close, the rest nearest-centroid neighbours.
from cali_address.geo import haversine_m

rs = np.random.default_rng(SEED)
probe = rs.choice(len(docs), 4000, replace=False)
lat = docs["centroid_lat"].to_numpy()
lon = docs["centroid_lon"].to_numpy()
rows = []
for slot in range(spatial_neg.shape[1]):
    tgt = spatial_neg[probe, slot]
    ok = tgt >= 0
    dist = haversine_m(lat[probe[ok]], lon[probe[ok]], lat[tgt[ok]], lon[tgt[ok]])
    same_mz = (docs["manzana"].to_numpy()[probe[ok]] == docs["manzana"].to_numpy()[tgt[ok]])
    rows.append({"slot": slot, "filled": ok.mean(), "same_manzana": same_mz.mean(),
                 "median_dist_m": float(np.median(dist)), "p90_dist_m": float(np.percentile(dist, 90))})
display(pd.DataFrame(rows))

anchor = int(probe[0])
print(f"\\nanchor: {docs.loc[anchor, 'direccion']!r} (manzana {docs.loc[anchor, 'manzana']})")
for slot, t in enumerate(spatial_neg[anchor]):
    if t >= 0:
        print(f"  hard negative {slot}: {docs.loc[t, 'direccion']!r} "
              f"(manzana {docs.loc[t, 'manzana']}, "
              f"{haversine_m(lat[anchor], lon[anchor], lat[t], lon[t]):.0f} m)")
'''
)

md(
    """
**Lectura.** Los negativos duros son exactamente lo que se buscaba: las dos
primeras posiciones son direcciones de la **misma manzana** (mediana de pocas
decenas de metros) y el resto son vecinos por centroide. Son cadenas
ortográficamente casi idénticas al ancla y geográficamente contiguas: si el
modelo las separa, está aprendiendo la diferencia entre `KR 26 H 1 # 73 - 10` y
`KR 26 H # 73 - 10`, que es justamente el error que un `fuzzy match` comete.
"""
)

# ------------------------------------------------------------------- model
md(
    """
## 5. Modelo: dual-encoder Transformer a nivel de carácter con señales espaciales

### Arquitectura

Un **único** encoder Transformer (compartido) embebe tanto la consulta ruidosa
como la dirección canónica catastral, de modo que la recuperación es una
búsqueda por coseno en un solo espacio. Un *embedding de rol* (0 = consulta,
1 = documento) le indica al encoder qué lado está mirando.

* Tokenización a **nivel de carácter** (vocabulario de 48 símbolos). Es la
  elección natural: las direcciones se diferencian por caracteres sueltos
  (`72 L` vs `72 I`) y un vocabulario de subpalabras rompería justo esa señal.
* `d_model = 256`, **4 capas**, 8 cabezas, `norm_first`, GELU, `dropout = 0.1`.
* *Mean pooling* sobre las posiciones no-PAD, `LayerNorm`, proyección lineal y
  normalización L2 → embedding de 256 dimensiones.

### Por qué esta arquitectura es adecuada para análisis espacial

Tres decisiones convierten un recuperador de texto en un recuperador *espacial*:

1. **Negativos duros espaciales.** La pérdida InfoNCE usa, además de los
   negativos en lote, direcciones de la **misma manzana** o de centroides
   vecinos. El gradiente se concentra donde el error importa: distinguir predios
   contiguos cuyas cadenas difieren en un carácter. Un recuperador entrenado solo
   con negativos aleatorios aprende a separar `KR 26 …` de `CL 98 …`, que es
   fácil y espacialmente irrelevante.
2. **Cabeza de regresión de coordenadas.** Una cabeza auxiliar predice el
   centroide del predio en EPSG:3116 (estandarizado) con pérdida SmoothL1. Esto
   obliga a que el espacio de embeddings sea **localmente isométrico con el
   espacio geográfico**: direcciones cercanas en la ciudad quedan cercanas en el
   embedding, lo que hace que los 20 candidatos recuperados sean un vecindario
   geográfico coherente y no una lista de homógrafos dispersos.
3. **Cabezas de jerarquía administrativa.** Dos clasificadores auxiliares
   predicen comuna y barrio derivados del código predial verificado en §1.1
   (el número exacto de clases lo imprime la celda siguiente: 24 y 342, donde la
   clase extra de comuna es el bucket de los 5 prediales corruptos). Inyectan la
   jerarquía espacial como señal supervisada barata y actúan como regularizador:
   el modelo no puede resolver la consulta sin situarla en la ciudad.

### Pérdida total

```
L = InfoNCE(q, d; in-batch + spatial hard negatives)
  + 0.50 * SmoothL1(coord_head(q), centroid_xy)
  + 0.20 * CE(comuna_head(q), comuna)
  + 0.20 * CE(barrio_head(q), barrio)
```

La temperatura de InfoNCE es un parámetro aprendido inicializado en 0,05.
"""
)

code("""
import cali_address.model as model_mod
import cali_address.train as train_mod

importlib.reload(model_mod)
importlib.reload(train_mod)
from cali_address.model import CHARS, MAX_LEN, AddressEncoder, encode_text
from cali_address.train import TrainConfig, embed_documents, recall_at_k, train_model

cfg = TrainConfig()
print("character vocabulary:", repr(CHARS), f"(size {len(CHARS) + 2})")
print("stored sequence length:", MAX_LEN, "| training sequence length:", cfg.max_len)
print("\\nhyper-parameters:")
display(pd.Series(cfg.to_dict()).to_frame("value"))

# Build the probe with the REAL class counts taken from the training tensors, so
# the parameter count printed here is the one that was actually trained (the
# defaults n_comuna=24 / n_barrio=400 would inflate the barrio head).
N_COMUNA = int(train_data["doc_comuna"].max()) + 1
N_BARRIO = int(train_data["doc_barrio"].max()) + 1
print(f"classification heads: {N_COMUNA} comuna classes, {N_BARRIO} barrio classes")
probe_model = AddressEncoder(d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
                             max_len=cfg.max_len, n_comuna=N_COMUNA, n_barrio=N_BARRIO)
n_params = sum(p.numel() for p in probe_model.parameters())
print(f"\\ntrainable parameters: {n_params:,} ({n_params/1e6:.2f} M)")
print("tokenization example:")
print("  'KR 26 H 1 # 73 - 10' ->", encode_text("KR 26 H 1 # 73 - 10", cfg.max_len)[:24], "...")
del probe_model
""")

code(
    '''
MODEL_PATH = os.path.join(ARTIFACTS, "model.pt")
EMB_PATH = os.path.join(ARTIFACTS, "catastro_emb.pt")
REPORT_PATH = os.path.join(ARTIFACTS, "train_report.json")

if FORCE_RETRAIN or not (os.path.exists(MODEL_PATH) and os.path.exists(EMB_PATH)):
    print("training from scratch (~20 min on this GPU)...")
    torch.cuda.reset_peak_memory_stats()
    out = train_model(train_data, spatial_neg, cfg, device=DEVICE)
    trained = out.pop("model")
    torch.save(
        {
            "state_dict": trained.state_dict(), "config": out["config"],
            "n_comuna": int(train_data["doc_comuna"].max()) + 1,
            "n_barrio": int(train_data["doc_barrio"].max()) + 1,
            "xy_mean": train_data["xy_mean"], "xy_std": train_data["xy_std"],
            "comuna_codes": train_data["comuna_codes"], "barrio_codes": train_data["barrio_codes"],
        },
        MODEL_PATH,
    )
    torch.save(embed_documents(trained, train_data["d_tokens"][:, : cfg.max_len]).cpu(), EMB_PATH)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    del trained
    torch.cuda.empty_cache()
else:
    print("reusing the cached model artifacts (set FORCE_RETRAIN=True to retrain)")

with open(REPORT_PATH, encoding="utf-8") as fh:
    train_report = json.load(fh)
history = pd.DataFrame(train_report["history"])

print(f"\\ntraining wall time : {train_report['seconds']/60:.1f} min "
      f"({train_report['seconds']:.0f} s)")
print(f"parameters         : {train_report['n_params']:,}")
print("gpu                :", json.dumps(train_report["gpu"], indent=2))
print(f"throughput         : {history['seqs_per_s'].iloc[-1]:,.0f} sequences/s")
print(f"peak GPU memory    : {history['gpu_mem_gb'].max():.2f} GB")
print()
print("Validation recall, with the denominator of each figure spelled out:")
n_val = int(train_data["q_is_val"].sum())
per_epoch_n = min(n_val, 20_000)
print(f"  per-epoch curve below : subsample of {per_epoch_n:,} of the {n_val:,} "
      f"held-out queries (recall@1 = {history['recall@1'].iloc[-1]:.4f})")
if "final_val_recall" in train_report:
    fv = train_report["final_val_recall"]
    print(f"  final full-split pass : ALL {n_val:,} held-out queries "
          f"(recall@1 = {fv['recall@1']:.4f}, @5 = {fv['recall@5']:.4f}, "
          f"@20 = {fv['recall@20']:.4f})  <-- the figure to quote")
    print("  the two differ only by the subsample; they are not interchangeable.")
display(history[["epoch", "loss", "loss_nce", "loss_coord", "loss_comuna", "loss_barrio",
                 "inbatch_acc", "train_recall@1", "recall@1", "recall@5", "recall@20",
                 "epoch_seconds"]])
'''
)

code(
    '''
fig, axes = plt.subplots(1, 3, figsize=(19, 4.8))
axes[0].plot(history["epoch"], history["loss"], "o-", label="total", color="#1f4e79")
axes[0].plot(history["epoch"], history["loss_nce"], "s-", label="InfoNCE", color="#d1615d")
axes[0].plot(history["epoch"], history["loss_coord"], "^-", label="coord (SmoothL1)", color="#5b9e4d")
axes[0].plot(history["epoch"], history["loss_comuna"], "v-", label="comuna (CE)", color="#e8a33d")
axes[0].plot(history["epoch"], history["loss_barrio"], "d-", label="barrio (CE)", color="#8064a2")
axes[0].set_yscale("log"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss (log)")
axes[0].set_title("Training losses"); axes[0].legend()

for col, color, marker in [("recall@1", "#1f4e79", "o"), ("recall@5", "#d1615d", "s"),
                           ("recall@20", "#5b9e4d", "^")]:
    axes[1].plot(history["epoch"], history[col], marker + "-", label=f"val {col}", color=color)
axes[1].plot(history["epoch"], history["train_recall@1"], "x--", label="train recall@1",
             color="#888888")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("recall")
axes[1].set_title("Retrieval recall on the held-out documents"); axes[1].legend()
axes[1].set_ylim(min(0.85, history["recall@1"].min() - 0.02), 1.002)

axes[2].plot(history["epoch"], history["inbatch_acc"], "o-", color="#1f4e79",
             label="in-batch accuracy")
ax2 = axes[2].twinx()
ax2.plot(history["epoch"], history["epoch_seconds"], "s--", color="#d1615d", label="s/epoch")
ax2.set_ylabel("seconds per epoch")
axes[2].set_xlabel("epoch"); axes[2].set_ylabel("in-batch accuracy")
axes[2].set_title("Contrastive accuracy and speed")
lines = axes[2].get_lines() + ax2.get_lines()
axes[2].legend(lines, [l.get_label() for l in lines], loc="center right")
plt.tight_layout()
plt.show()
'''
)

md(
    """
**Lectura.** El modelo converge rápido: la primera época ya alcanza
`recall@1 ≈ 0,89` sobre **documentos nunca vistos** y la curva sigue subiendo con
las épocas. Ojo con los dos números de recall que imprime la celda anterior: la
curva por época se calcula sobre una submuestra de 20.000 consultas de validación
y la cifra final sobre las 39.645 completas. Son medidas de la misma cantidad con
denominadores distintos y no deben citarse indistintamente; la que vale como
resultado es la del barrido completo. Las pérdidas auxiliares caen dos órdenes de magnitud, señal de que
la información espacial (coordenada) y jerárquica (comuna/barrio) es
efectivamente recuperable del texto de la dirección — lo que confirma que
inyectarlas como supervisión no es un adorno sino una señal real. La brecha
entre `train recall@1` y `val recall@1` se mantiene pequeña, de modo que no hay
sobreajuste apreciable a pesar de la partición estricta por documento.
"""
)

# --------------------------------------------------------------- inference
md(
    """
## 5.1 Ajuste del umbral de confianza y de los pesos de fusión

Los pesos del reranker y el umbral de `matched` son **decisiones**, y una decisión
no se puede tomar sobre el conjunto de prueba. Las filas de los cinco libros de
Excel son el conjunto de prueba, así que `scripts/tune_threshold.py` **no las lee
nunca**. El ajuste se hace sobre:

* **Positivos**: los documentos catastrales apartados (4 %, nunca positivos en el
  entrenamiento), en dos regímenes de igual peso —las variantes ruidosas exactas
  que se generaron para ellos (replicadas de forma determinista) y un lote de
  corrupciones *duras* de los mismos documentos, con intensidad 2×. El segundo
  lote es necesario: en el régimen de entrenamiento la recuperación ya acierta
  ~94 % en top-1, así que los términos de texto y estructura no tienen nada que
  explicar y la rejilla degenera sobre el coseno.
* **Negativos**: cadenas sintéticas que no son direcciones (identificadores
  numéricos, letras aleatorias, descripciones genéricas de lugar, blancos),
  generadas con un vocabulario fijo dentro del propio script. No tienen documento
  correcto, así que cualquier match es un falso positivo. Sin ellos no se puede
  calibrar una compuerta cuyo único propósito es rechazar basura, porque la
  partición catastral no contiene ninguna.

Tres decisiones se declaran antes de mirar el conjunto de prueba, y se reporta su
costo:

1. **`geo >= 0,10`**: el término geográfico existe para cortar la cola de errores
   lejanos en texto de campo, no para mover el top-1 dentro de distribución. Sin
   la restricción la rejilla lo lleva a cero.
2. **La tasa de basura es un supuesto**, no un parámetro ajustado. El umbral se
   elige al 20 % y se publica su sensibilidad entre el 5 % y el 40 %.
3. **Los empates se rompen hacia un vector equilibrado**: la cima de la rejilla es
   plana dentro de un error estándar, así que un `argmax` pelado cae en una
   esquina arbitraria del simplex y puede anular un componente sin ganancia
   medible. Entre las combinaciones a menos de 1 SE de la mejor que mantienen
   todos los componentes activos, se toma la más equilibrada.

El objetivo del umbral es el **F1 del enunciado "`matched` implica que el documento
es correcto"**: TP = marcado y correcto, FP = marcado y equivocado (o basura),
FN = correcto pero no marcado.
"""
)

code(
    '''
TUNING_PATH = os.path.join(ARTIFACTS, "tuning.json")
if not os.path.exists(TUNING_PATH):
    print("running the tuning sweep on the held-out cadastral split...")
    import subprocess

    subprocess.run([sys.executable, "scripts/tune_threshold.py"], check=True, cwd=ROOT)

with open(TUNING_PATH, encoding="utf-8") as fh:
    TUNED = json.load(fh)
TUNED_THRESHOLD = float(TUNED["chosen"]["threshold"])
TUNED_WEIGHTS = TUNED["chosen"]["weights"]
VARIANT_THRESHOLDS = TUNED["chosen"]["thresholds"]
VARIANT_WEIGHTS = TUNED["chosen"]["variant_weights"]

print("method:")
display(pd.Series(TUNED["method"]).to_frame("value"))
print("chosen:")
print(json.dumps(TUNED["chosen"], indent=2))
'''
)

code(
    '''
diag = TUNED["diagnostics"]
sweep = pd.read_csv(os.path.join(ARTIFACTS, "threshold_sweep.csv"))
weights = pd.read_csv(os.path.join(ARTIFACTS, "weight_sweep.csv"))
sens = pd.DataFrame(diag["junk_rate_sensitivity"])

print("fusion-weight selection")
print(f"  unconstrained argmax        : {diag['unconstrained_best_weights']} "
      f"-> pooled top-1 {diag['top1_at_unconstrained_best']:.4f}")
print(f"  pooled accuracy SE          : {diag['pooled_accuracy_se']:.4f} "
      f"(tie tolerance {diag['tie_tolerance']:.4f})")
print(f"  rule applied                : {TUNED['method']['selection_rule']}")
print(f"  CHOSEN                      : {TUNED_WEIGHTS} -> "
      f"pooled top-1 {diag['top1_at_chosen_weights']:.4f} "
      f"(in-distribution {diag['top1_in_distribution_at_chosen']:.4f} / "
      f"hard {diag['top1_hard_at_chosen']:.4f})")
print(f"  cost of the selection rule  : {diag['selection_cost_top1']:+.4f} top-1 "
      f"({diag['selection_cost_in_se']:.2f} SE)")
print(f"  hand-picked weights used before tuning: "
      f"{diag['previously_shipped_weights']} -> "
      f"pooled top-1 {diag['top1_at_previously_shipped_weights']:.4f}")
print()
print("top of the weight grid:")
display(weights.head(8))

print(f"threshold selection: CHOSEN = {TUNED_THRESHOLD:.2f} "
      f"(F1 {diag['f1_at_chosen_threshold']:.4f}, "
      f"precision {diag['precision_at_chosen_threshold']:.4f}, "
      f"recall {diag['recall_at_chosen_threshold']:.4f}, "
      f"matched rate {diag['matched_rate_at_chosen_threshold']:.4f})")
display(sweep[sweep["variant"] == "full (+ coord head)"]
        [["threshold", "matched_rate", "precision", "recall", "f1"]].iloc[::3])
print("sensitivity of the chosen threshold to the assumed junk rate:")
display(sens)
print("per-variant cutoffs (each on its own score scale):")
display(pd.DataFrame(diag["variants"]))
'''
)

code(
    '''
full_sweep = sweep[sweep["variant"] == "full (+ coord head)"]
fig, axes = plt.subplots(1, 3, figsize=(19, 4.6))

axes[0].plot(full_sweep["threshold"], full_sweep["precision"], "-", label="precision",
             color="#1f4e79")
axes[0].plot(full_sweep["threshold"], full_sweep["recall"], "-", label="recall",
             color="#d1615d")
axes[0].plot(full_sweep["threshold"], full_sweep["f1"], "-", lw=2.5, label="F1",
             color="#5b9e4d")
axes[0].plot(full_sweep["threshold"], full_sweep["matched_rate"], ":", label="matched rate",
             color="#888888")
axes[0].axvline(TUNED_THRESHOLD, color="black", ls="--",
                label=f"chosen {TUNED_THRESHOLD:.2f}")
axes[0].set_xlabel("threshold on the fused score"); axes[0].set_ylabel("rate")
axes[0].set_title("Threshold sweep on the held-out cadastral split")
axes[0].legend(fontsize=8); axes[0].set_ylim(0, 1.02)

for variant, g in sweep.groupby("variant"):
    axes[1].plot(g["threshold"], g["f1"], label=variant)
    t = VARIANT_THRESHOLDS.get(variant)
    if t is not None:
        axes[1].scatter([t], [g.loc[g["threshold"] == t, "f1"].iloc[0]], zorder=5, s=40)
axes[1].set_xlabel("threshold"); axes[1].set_ylabel("F1")
axes[1].set_title("F1 per ablation variant (own score scale)")
axes[1].legend(fontsize=8)

sc = axes[2].scatter(weights["geo"], weights["top1_positives"], c=weights["sim"],
                     cmap="viridis", s=14, alpha=0.75)
axes[2].axvline(0.10, color="black", ls="--", label="a-priori geo floor")
axes[2].set_xlabel("geo weight"); axes[2].set_ylabel("pooled top-1 on the positives")
axes[2].set_title("Weight grid: geo term vs objective")
fig.colorbar(sc, ax=axes[2], label="sim weight")
axes[2].legend(fontsize=8)
plt.tight_layout()
plt.show()
'''
)

md(
    """
**Lectura.** El barrido tiene un óptimo real, a diferencia de lo que ocurriría con
solo positivos dentro de distribución: sin negativos la F1 se maximiza marcando
todo, porque con ~94 % de acierto la precisión no tiene nada que ganar. Con la
mezcla, la precisión sube de ~0,81 a ~0,91 entre el 0,50 y el umbral elegido
mientras el recall apenas cae, y **el umbral seleccionado es estable frente al
supuesto de tasa de basura en todo el rango del 5 % al 40 %**, lo que indica que no
es un artefacto de ese supuesto.

El panel derecho muestra por qué el término geográfico necesita una restricción a
priori: su contribución al top-1 dentro de distribución es plana, porque ahí la
recuperación ya resuelve casi todo. Su valor aparece en la cola de errores sobre
texto de campo, y eso se mide después, en la ablación de la sección 7.2.

Nota sobre el valor elegido: coincide casi exactamente con el 0,72 que estaba
escrito a mano antes de este ajuste. No es que se haya reconstruido hacia atrás —
los pesos sí cambiaron (la similitud del modelo pasó de 0,40 a 0,70), de modo que
el mismo número es un punto de operación distinto sobre una escala distinta.
"""
)

md(
    """
## 6. Pipeline de inferencia

```
raw text
  -> parse_address()          gramática de reglas -> campos + forma canónica
  -> encoder (GPU, bf16)      dos vistas: canónica y texto pre-normalizado
  -> top-k coseno (k=20)      matmul fp16 por bloques contra los 330k embeddings
  -> rerank fusionado         0.40*similitud + 0.26*rapidfuzz
                            + 0.22*estructura + 0.12*consistencia geográfica
  -> predio, dirección canónica, confianza, manzana, centroide, comuna
```

Se codifican **dos vistas** de cada consulta (la forma canónica del parser y el
texto pre-normalizado) y se unen sus candidatos: así el sistema sigue
funcionando cuando el parser falla, y aprovecha la estructura cuando acierta.

El **acuerdo estructural** del reranker compara campo a campo la consulta con el
candidato (tipo de vía, número, letras, cuadrante, número de cruce, placa), con
crédito parcial cuando la placa difiere en menos de 3 unidades. Es la pieza que
corrige el error típico del coseno: elegir la dirección vecina con la letra
equivocada.

El cuarto término es la **consistencia geográfica**, y es donde se cobra la
cabeza auxiliar de coordenadas: el modelo predice directamente el centroide de la
consulta en EPSG:3116, y cada candidato se pondera por
`exp(-distancia / 600 m)` respecto a esa predicción. Así un candidato que
coincide bien con el texto pero está en otra zona de la ciudad se descarta usando
el prior espacial aprendido, no una regla escrita a mano. La sección 7.2 mide su
aporte por ablación.

Si la confianza queda por debajo del umbral (0,72), se devuelve la forma canónica
de reglas en vez de la cadena catastral recuperada, para no afirmar una
correspondencia que no se sostiene.
"""
)

code(
    '''
import cali_address.inference as inference_mod

importlib.reload(inference_mod)
from cali_address.inference import CONFIDENCE_THRESHOLD, FUSE_WEIGHTS, AddressNormalizer

t0 = time.time()
normalizer = AddressNormalizer(ARTIFACTS, device=DEVICE, docs=docs)
print(f"normalizer loaded in {time.time()-t0:.1f}s")
print(f"index size        : {normalizer.doc_emb.shape}  dtype={normalizer.doc_emb.dtype}")
print(f"index memory      : {normalizer.doc_emb.numel() * 2 / 1e6:.1f} MB on {DEVICE}")
print(f"fusion weights    : {normalizer.weights}")
print(f"confidence cutoff : {normalizer.threshold:.2f}")
print(f"tuning source     : {normalizer.tuning_source}")
print(f"per-variant cutoffs: {normalizer.variant_thresholds}")
assert normalizer.weights == TUNED["chosen"]["weights"], "weights drifted from tuning.json"
assert normalizer.threshold == TUNED["chosen"]["threshold"], "threshold drifted from tuning.json"
'''
)

code(
    '''
WORKED_EXAMPLES = [
    "Cra. 98f #98-66",
    "CL12A # 56-04",
    "Calle 46-45",
    "Kra 1C 3 #64 A_41 Bloque 13D",
    "KR 13 60 34 38 LC 1 01 2 01 ED PAMP",
    "Calle 9c 49 141",
    "Av 6 Oeate #22-14",
    "Cl. 55b # 47-55, Navarro, Cali, Valle del Cauca",
    "Carrera1#9-80",
    "CALLE 5 B2 # 38-91; CRA 37 #4A BIS 49",
    "Catrera 36 5b3-65",
    "CRA 100 1B OESTE 110 TORRE 7 APTO 402",
    "Calle 67n 2a 50",
    "Av 5 An #23 Dn 68",
    "KR 38 BIS # 5B2 09",
    "CL 1 D OESTE # 100 BIS-19, TORRE 85 Y 86",
    "calle 9C # 53-25 torre G",
    "SIN ESPECIFICAR",
    "16758424",
    "Parte alta pichinde via la leonera",
]
t0 = time.time()
worked = normalizer.normalize_batch(WORKED_EXAMPLES, k=20)
print(f"{len(WORKED_EXAMPLES)} addresses normalized in {time.time()-t0:.2f}s")
display(worked[["raw_address", "parse_ok", "rule_canonical", "matched_cadastral_address",
                "numero_predial_nacional", "manzana", "comuna", "lat", "lon",
                "confidence", "model_similarity", "fuzz_score", "struct_score",
                "geo_score", "matched"]])
'''
)

code(
    '''
# Single-address API, the one to use on new data.
example = normalizer.normalize_address("Cra. 98f #98-66")
print(json.dumps({k: v for k, v in example.items() if k != "topk_doc_ids"},
                 indent=2, default=str))
'''
)

md(
    """
**Lectura, ajustada a lo que muestra la tabla.** Conviene separar dos cosas que
la tabla reporta por separado:

* **La cadena canónica** (`rule_canonical`) es correcta en casi todos los casos
  bien formados: el parser reconstruye tipo de vía, número, letras, cuadrante,
  cruce y placa.
* **El predio recuperado** (`matched_cadastral_address`) coincide con la cadena
  canónica en algunos casos y **difiere en varios otros**, casi siempre en la
  placa o en el complemento. Contando sobre esta misma tabla, en torno a la mitad
  de los ejemplos con `parse_ok = True` devuelven una dirección catastral distinta
  de la canónica producida por las reglas: el predio con esa placa exacta no
  existe en la capa y el sistema entrega el más cercano en texto y en espacio.

No hay que leer "confianza alta" como "predio exacto". La precisión del indicador
`matched` frente a la verdad de terreno es del orden del 49 % (§7.5), y esa verdad
de terreno es ella misma ruidosa (§7.1.1); lo que la confianza ordena bien es la
*plausibilidad* del candidato, no su identidad exacta.

Lo que sí funciona como se diseñó es el rechazo: `SIN ESPECIFICAR`, `16758424` y
`Parte alta pichinde via la leonera` salen con `parse_ok = False` y confianza
baja. El sistema **declara que no sabe** en vez de afirmar un predio, y eso es lo
que permite usar la salida en un flujo operativo — siempre filtrando por
`matched`.
"""
)

# ------------------------------------------------------------- evaluation
md(
    """
## 7. Evaluación

### 7.1 Diseño experimental

Cinco conjuntos externos, independientes de la base de entrenamiento:

| Conjunto | Filas | Columna de dirección | Coordenadas | Muestra |
|---|---|---|---|---|
| Fasecolda (inmuebles asegurados) | 71.517 | `DIRECCIÓN` (+ `DIRECCIÓN_NORMALIZADA`) | WGS84 | 500 |
| RUD (registro único de damnificados) | 97.337 | `direccion_bien` | — | 500 |
| Inspecciones estructurales | 2.168 | `direccion` (+ `direccion_norm`) | `coords` ("lat, lon") | 500 |
| Stickers de habitabilidad | 3.572 | `direccion` | `lat`/`lng` + `accuracy` | 500 |
| Candidatos a demolición | 78 | `direccion` | — | **78 (todas)** |

El muestreo es de 500 filas por conjunto con `random_state=42` después de
descartar direcciones nulas o vacías. **`acciones` tiene únicamente 78 filas, de
modo que se evalúa el conjunto completo**; su tamaño hace que sus métricas
tengan un intervalo de confianza mucho más amplio que los demás.

**Verdad de terreno.** Donde hay coordenadas se resuelve el predio verdadero por
*point-in-polygon* con un `STRtree` de Shapely sobre los 338.305 polígonos
catastrales. El predio así obtenido define el objetivo de recuperación
(`gt_doc_id`) y la manzana verdadera. Advertencia importante: estas coordenadas
provienen de GPS de dispositivos móviles y de EXIF de fotografías, con
`accuracy` reportada de decenas de metros en el caso de `stickers`; parte del
error medido es **ruido de la referencia**, no del modelo.
"""
)

code(
    '''
from cali_address.eval_data import DATASETS, load_dataset, sample_dataset

SAMPLES_PATH = os.path.join(ARTIFACTS, "eval_samples.parquet")
if not os.path.exists(SAMPLES_PATH):
    import subprocess

    subprocess.run([sys.executable, "scripts/build_eval_samples.py"], check=True, cwd=ROOT)
samples = pd.read_parquet(SAMPLES_PATH)

spec_tbl = pd.DataFrame(
    [
        {"dataset": s.key, "file": s.filename, "sheet": s.sheet, "header_row": s.header,
         "address_col": s.address_col, "their_normalized": s.normalized_col,
         "note": s.note}
        for s in DATASETS
    ]
)
display(spec_tbl)
display(pd.read_csv(os.path.join(ARTIFACTS, "eval_sample_stats.csv")))
print(f"total sampled rows: {len(samples):,}")
display(samples.groupby("dataset").head(3)[["dataset", "raw_address", "their_normalized",
                                            "gt_lat", "gt_lon", "gps_accuracy_m"]])
'''
)

code(
    '''
from cali_address.evaluate import attach_ground_truth
from cali_address.geo import ParcelIndex

t0 = time.time()
parcels = ParcelIndex(catastro["geom_wkb"].to_numpy())
print(f"STRtree over {len(parcels):,} cadastral polygons built in {time.time()-t0:.1f}s")

doc_id_by_address = {a: i for i, a in enumerate(docs["direccion"].astype(str))}
evaluated = attach_ground_truth(samples, catastro, parcels, doc_id_by_address)

gt_tbl = evaluated.groupby("dataset").agg(
    n=("raw_address", "size"),
    with_coords=("gt_lat", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
    resolved_parcel=("gt_parcel_row", lambda s: int((s >= 0).sum())),
    resolved_doc=("gt_doc_id", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
)
gt_tbl["parcel_hit_rate_given_coords"] = gt_tbl["resolved_parcel"] / gt_tbl["with_coords"].replace(0, np.nan)
gt_tbl["unreachable_target"] = gt_tbl["resolved_parcel"] - gt_tbl["resolved_doc"]
display(gt_tbl)

# Denominator bookkeeping, stated explicitly rather than absorbed silently.
GT_RESOLVED = evaluated.attrs["gt_parcels_resolved"]
GT_SCORABLE = evaluated.attrs["gt_parcels_scorable"]
GT_UNREACHABLE = evaluated.attrs["gt_parcels_unreachable"]
print(f"point-in-polygon resolved a parcel for      : {GT_RESOLVED:,} rows")
print(f"of those, the parcel address is in the index: {GT_SCORABLE:,} rows  <-- denominator "
      f"of every predio/manzana/distance metric")
print(f"unreachable target (address not indexed)    : {GT_UNREACHABLE:,} rows "
      f"({GT_UNREACHABLE/max(GT_RESOLVED,1):.2%}) - excluded, since no retrieval "
      f"result could ever be counted correct for them")
display(evaluated[evaluated["gt_parcel_row"] >= 0][
    ["dataset", "raw_address", "gt_lat", "gt_lon", "gt_predial", "gt_manzana",
     "gt_cadastral_address", "gt_comuna"]].head(8))
'''
)

md(
    """
**Lectura.** De las 1.466 filas muestreadas que traen coordenadas, el
*point-in-polygon* resuelve un predio para la gran mayoría; las que no se
resuelven caen sobre vía pública, zonas verdes o fuera del perímetro catastral,
un artefacto esperable de coordenadas capturadas con GPS de celular.

**Denominadores, explícitos.** De los predios resueltos, unos pocos tienen una
`direccion` que no está en el índice de recuperación (cadena vacía o degenerada,
o centroide descartado al construir los documentos). Para esas filas *ninguna*
recuperación podría contarse como correcta, así que se excluyen del denominador
en lugar de penalizar al modelo por un objetivo inalcanzable; la celda anterior
imprime los tres números (resueltos, puntuables, inalcanzables). Las filas sin
coordenadas participan solo en las métricas de cobertura, de acuerdo de cadena y
en la comparación con IDESC.
"""
)

md(
    """
### 7.1.1 ¿Cuán buena es la verdad de terreno? (control de calidad de la referencia)

Antes de reportar exactitud hay que medir el **ruido de la propia referencia**.
Las coordenadas de estos archivos provienen de GPS de celular y de EXIF de
fotografías. Dos comprobaciones independientes del modelo:

1. **Resolución física de la tarea**: distancia mediana entre el centroide de un
   predio y el de su vecino más cercano. Si el error de la coordenada supera esa
   distancia, el *point-in-polygon* cae en el predio equivocado **por
   construcción**.
2. **Contraste con IDESC**: cuando IDESC responde `estado A` ("normalizado y
   georreferenciado **exacto**"), su `dir_ajusta` y su coordenada son una
   respuesta de alta calidad producida por un tercero. Si la coordenada del
   archivo cayera sobre el predio que IDESC nombra, ambas coincidirían. La
   proporción en que coinciden es una **cota superior de la exactitud de predio
   exacto que cualquier normalizador de texto puede obtener contra esta verdad de
   terreno**.
"""
)

code(
    '''
from cali_address.evaluate import parcel_spacing, reference_quality

spacing = parcel_spacing(docs)
print("Physical resolution of the task (distance between neighbouring parcel centroids):")
print(json.dumps(spacing, indent=2))

evaluated_ref = evaluated.join(
    idesc_df.set_index("raw_address")[["idesc_estado_class", "idesc_dir_ajusta",
                                       "idesc_lat", "idesc_lon"]],
    on="raw_address",
)
refq = reference_quality(evaluated_ref)
print()
print("Reference-quality check against IDESC (model not involved):")
display(refq.set_index("idesc_estado_class"))

acc = pd.to_numeric(evaluated["gps_accuracy_m"], errors="coerce").dropna()
print(f"rows with a reported GPS error value: {len(acc):,}")
if len(acc):
    print(f"  median={acc.median():.1f} m  p75={acc.quantile(0.75):.1f} m  "
          f"p90={acc.quantile(0.90):.1f} m")
print("note: the `accuracy` column of the stickers workbook is present but empty, "
      "so the GPS-noise caveat rests on the IDESC cross-check above.")
'''
)

md(
    """
**Lectura — esto reencuadra todas las métricas que siguen.**

El argumento decisivo es puramente geométrico:

* La distancia mediana entre el centroide de un predio y el de su vecino más
  cercano es de **~6 m** (ver la tabla): es el frente típico de un lote urbano en
  Cali.
* Para las filas donde **IDESC afirma georreferenciación exacta** (`estado A`), la
  coordenada guardada en el archivo está a una distancia mediana de **~19 m** de la
  coordenada exacta de IDESC, con p90 de **~73 m**.

Un desplazamiento mediano de 19 m sobre una retícula de 6 m significa que, la
mitad de las veces, el *point-in-polygon* aterriza **dos o tres predios más
allá** del correcto. La etiqueta de "predio verdadero" es por tanto una etiqueta
ruidosa, y la exactitud de predio exacto medida contra ella subestima el
desempeño real de cualquier normalizador.

La tabla añade una comprobación corroborante: el predio obtenido por
*point-in-polygon* coincide con la dirección que IDESC normaliza solo en ~25 % de
las filas `estado A`. Conviene leerla con cuidado, porque está confundida por dos
efectos: el desplazamiento de la coordenada **y** el hecho de que `dir_ajusta`
arrastra complementos que la cadena catastral no tiene
(`CL 4B # 34 - 04 P 0 BBVA 252 …`). Es una señal de la misma dirección, no una
medición limpia.

Por eso las métricas que hay que leer como indicadores de calidad geográfica son:

1. **exactitud de manzana** (robusta a errores de ~30 m),
2. **distribución de la distancia** al punto de referencia (mediana y percentiles),
3. **acuerdo de cadena canónica** contra IDESC y contra la normalización propia de
   cada archivo, que no depende de coordenadas.

La exactitud de predio exacto se reporta igual, por transparencia, pero
interpretándola contra este ruido.
"""
)

md(
    """
### 7.2 Predicciones y ablación de los cuatro componentes

Se evalúan cuatro variantes sobre exactamente las mismas filas, para aislar el
aporte de cada pieza:

| Variante | Recuperación | Reranking |
|---|---|---|
| `rule` | reglas + `rapidfuzz` sobre el bucket (tipo de vía, número) | `rapidfuzz` + estructura |
| `neural` | coseno sobre los embeddings | ninguno (top-1 del coseno) |
| `rerank` | coseno | similitud + `rapidfuzz` + estructura |
| `full` | coseno | ídem **+ consistencia geográfica** (cabeza de coordenadas) |

Cada variante usa **sus propios pesos y su propio umbral de confianza**, los que
la sección 5.1 ajustó para ella en la partición catastral apartada. Es necesario:
`neural` puntúa con coseno crudo y `rule` con `rapidfuzz`, escalas que no son
comparables con el puntaje fusionado. Compartir un solo umbral haría incomparables
las columnas de cobertura, y aplicar a la línea base de reglas unos pesos
ajustados para la escala del coseno la penalizaría injustamente. Cada fila de la
tabla es, por tanto, el punto de operación elegido honestamente para esa variante.
"""
)

code(
    '''
from cali_address.evaluate import score_predictions, summarize

raws = evaluated["raw_address"].astype(str).tolist()

VARIANTS = {
    "rule (no model)": dict(use_model=False, rerank=True, use_geo=False),
    "neural only": dict(use_model=True, rerank=False, use_geo=False),
    "neural + rerank": dict(use_model=True, rerank=True, use_geo=False),
    "full (+ coord head)": dict(use_model=True, rerank=True, use_geo=True),
}
# Each variant scores candidates on its own scale (raw cosine vs fused score), so
# each gets the cutoff that was tuned for it on the held-out cadastral split.
# Sharing one number would make their coverage columns incomparable.
scored = {}
timings = {}
for name, kw in VARIANTS.items():
    t0 = time.time()
    preds = normalizer.normalize_batch(
        raws, k=20, threshold=VARIANT_THRESHOLDS[name],
        weights=VARIANT_WEIGHTS[name], **kw
    )
    timings[name] = time.time() - t0
    scored[name] = score_predictions(evaluated, preds, docs)
    print(f"[{name:20s}] weights {VARIANT_WEIGHTS[name]} cutoff "
          f"{VARIANT_THRESHOLDS[name]:.2f} | {len(raws):,} addresses "
          f"in {timings[name]:5.1f}s ({len(raws)/timings[name]:7.0f} addr/s)")

scored_full = scored["full (+ coord head)"]
scored_rule = scored["rule (no model)"]
scored_nore = scored["neural only"]
scored_nogeo = scored["neural + rerank"]
t_full = timings["full (+ coord head)"]
print()
print("rows scored:", len(scored_full))
'''
)

code(
    '''
summary_full = summarize(scored_full, "full (+ coord head)")
display(summary_full.set_index("dataset"))

overall = []
for label, sc in scored.items():
    gt = sc[sc["has_gt"]]
    dist = pd.to_numeric(gt["dist_m"], errors="coerce").dropna()
    overall.append({
        "variant": label, "n": len(sc), "n_with_gt": len(gt),
        "seconds": round(timings[label], 1),
        "parse_rate": sc["parse_ok"].mean(),
        "highconf_rate": sc["matched"].mean(),
        "predio_top1": gt["correct_top1"].mean(), "predio_top5": gt["correct_top5"].mean(),
        "predio_top20": gt["correct_topk"].mean(), "manzana_top1": gt["correct_manzana"].mean(),
        "median_dist_m": dist.median(), "p90_dist_m": dist.quantile(0.9),
        "within_50m": (dist <= 50).mean(),
        "within_100m": (dist <= 100).mean(), "within_250m": (dist <= 250).mean(),
        "highconf_precision": gt[gt["matched"]]["correct_top1"].mean() if gt["matched"].any() else np.nan,
    })
overall_tbl = pd.DataFrame(overall).set_index("variant")
print("CONSOLIDATED ABLATION (all 5 datasets pooled)")
display(overall_tbl)

METRICS = ["predio_top1", "predio_top5", "predio_top20", "manzana_top1",
           "median_dist_m", "p90_dist_m", "within_50m", "within_100m", "within_250m"]
deltas = pd.DataFrame({
    "neural - rule": overall_tbl.loc["neural only", METRICS]
    - overall_tbl.loc["rule (no model)", METRICS],
    "rerank adds": overall_tbl.loc["neural + rerank", METRICS]
    - overall_tbl.loc["neural only", METRICS],
    "coord head adds": overall_tbl.loc["full (+ coord head)", METRICS]
    - overall_tbl.loc["neural + rerank", METRICS],
    "full - rule": overall_tbl.loc["full (+ coord head)", METRICS]
    - overall_tbl.loc["rule (no model)", METRICS],
})
print("contribution of each component (positive = better, except the distance rows)")
display(deltas)
'''
)

code(
    '''
# Per-dataset comparison of the three variants, side by side.
per_ds = pd.concat([summarize(sc, name) for name, sc in scored.items()])
pivot = per_ds.pivot(index="dataset", columns="variant",
                     values=["predio_top1", "predio_top5", "manzana_top1", "within_100m"])
display(pivot)

PALETTE = {"rule (no model)": "#e8a33d", "neural only": "#d1615d",
           "neural + rerank": "#7da7d9", "full (+ coord head)": "#1f4e79"}
plot_ds = per_ds[per_ds["n_with_gt_parcel"] > 0]
fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
for ax, metric, title in zip(
    axes,
    ["predio_top1", "predio_top5", "manzana_top1", "within_100m"],
    ["Predio top-1", "Predio top-5", "Manzana accuracy", "Share within 100 m"],
):
    sns.barplot(data=plot_ds, x="dataset", y=metric, hue="variant", ax=ax,
                hue_order=list(VARIANTS), palette=PALETTE)
    ax.set_title(title); ax.set_ylim(0, 1); ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=20)
    ax.legend_.set_visible(ax is axes[0])
    if ax is axes[0]:
        ax.legend(fontsize=7, loc="upper left")
plt.tight_layout()
plt.show()
'''
)

code(
    '''
# Distance distribution of the full system where ground-truth coordinates exist.
dist_df = scored_full[scored_full["has_gt"]].copy()
dist_df["dist_m"] = pd.to_numeric(dist_df["dist_m"], errors="coerce")
dist_df = dist_df.dropna(subset=["dist_m"])

fig, axes = plt.subplots(1, 3, figsize=(19, 4.6))
bins = [0, 10, 25, 50, 100, 250, 500, 1000, 1e9]
labels = ["0-10", "10-25", "25-50", "50-100", "100-250", "250-500", "500-1k", ">1k"]
cut = pd.cut(dist_df["dist_m"], bins=bins, labels=labels, right=False)
cut.value_counts().reindex(labels).plot(kind="bar", ax=axes[0], color="#1f4e79")
axes[0].set_title("Distance predicted centroid vs reference point (m)")
axes[0].set_ylabel("rows"); axes[0].tick_params(axis="x", rotation=30)

for key, g in dist_df.groupby("dataset"):
    x = np.sort(g["dist_m"].to_numpy())
    axes[1].plot(x, np.arange(1, len(x) + 1) / len(x), label=f"{key} (n={len(x)})")
axes[1].set_xscale("symlog"); axes[1].set_xlim(1, 5000)
axes[1].set_xlabel("distance (m, symlog)"); axes[1].set_ylabel("cumulative share")
axes[1].set_title("Empirical CDF of the error by dataset"); axes[1].legend(fontsize=8)
for b in (25, 50, 100, 250):
    axes[1].axvline(b, color="grey", ls=":", lw=0.8)

sub = dist_df[pd.to_numeric(dist_df["gps_accuracy_m"], errors="coerce").notna()]
if len(sub):
    axes[2].scatter(pd.to_numeric(sub["gps_accuracy_m"], errors="coerce"), sub["dist_m"],
                    s=8, alpha=0.4, color="#d1615d")
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[2].set_xlabel("reported GPS accuracy (m)"); axes[2].set_ylabel("our error (m)")
    axes[2].set_title("Reference noise vs measured error (stickers)")
else:
    axes[2].axis("off")
plt.tight_layout()
plt.show()

print(dist_df.groupby("dataset")["dist_m"].describe(percentiles=[0.25, 0.5, 0.75, 0.9]))
'''
)

md("### 7.3 Comparación contra IDESC")

code(
    '''
from cali_address.evaluate import accuracy_by_idesc_estado, idesc_comparison

idesc_lookup = idesc_df.set_index("raw_address")
IDESC_COLS = ["idesc_estado", "idesc_estado_class", "idesc_dir_ajusta", "idesc_comuna",
              "idesc_barrio_codigo", "idesc_barrio_nombre", "idesc_lat", "idesc_lon"]
scored_full = scored_full.join(idesc_lookup[IDESC_COLS], on="raw_address")
scored_nore = scored_nore.join(idesc_lookup[IDESC_COLS], on="raw_address")
scored_rule = scored_rule.join(idesc_lookup[IDESC_COLS], on="raw_address")
scored_full["rule_canonical_glued"] = [
    canonical(parse_address(r), style="glued", with_complement=True)
    for r in scored_full["raw_address"]
]
print(f"rows joined with an IDESC answer: {scored_full['idesc_dir_ajusta'].notna().sum():,}"
      f" / {len(scored_full):,}")
display(idesc_comparison(scored_full).set_index("dataset"))
'''
)

code(
    '''
print("Our retrieval accuracy broken down by the IDESC estado class")
display(accuracy_by_idesc_estado(scored_full).set_index("idesc_estado_class"))

# Coverage comparison: who produces a usable geographic answer?
cov = pd.DataFrame({
    "IDESC georeferenced (A or C)": scored_full.groupby("dataset")["idesc_lat"].apply(
        lambda s: s.notna().mean()),
    "ours high confidence": scored_full.groupby("dataset")["matched"].mean(),
    "ours any candidate": scored_full.groupby("dataset")["lat"].apply(lambda s: s.notna().mean()),
})
display(cov)
cov.plot(kind="bar", figsize=(11, 4.4),
         color=["#e8a33d", "#1f4e79", "#7da7d9"])
plt.title("Share of rows that receive coordinates"); plt.ylabel("share"); plt.ylim(0, 1.02)
plt.xticks(rotation=15); plt.legend(loc="upper left", fontsize=9); plt.tight_layout()
plt.show()

print("Examples where IDESC normalizes but does NOT georeference, and we do resolve a predio:")
gap = scored_full[(scored_full["idesc_lat"].isna()) & scored_full["matched"]]
display(gap[["dataset", "raw_address", "idesc_estado_class", "idesc_dir_ajusta",
             "matched_cadastral_address", "confidence", "lat", "lon"]].head(10))
'''
)

md(
    """
### 7.3.1 Retractación: el código de barrio de IDESC **no** es `predial[9:13]`

La sección 1.1 dejó abierta la pregunta con dos ejemplos escogidos a mano. Ahora
hay con qué medirla: para cada fila donde IDESC devolvió comuna y barrio, se
compara su respuesta con la jerarquía derivada del predial del predio que nuestro
sistema recuperó. Se reporta por separado el acuerdo de comuna y el de barrio, y
se restringe además a las filas de alta confianza, donde el predio recuperado es
creíble y por tanto la comparación mide nomenclatura y no error de recuperación.
"""
)

code(
    '''
hier = scored_full[scored_full["numero_predial_nacional"].notna()].copy()
hier["our_comuna"] = hier["comuna"].astype(str).str.zfill(2)
hier["our_barrio_code"] = hier["barrio_code"].astype(str).str.zfill(4)
hier["idesc_comuna_z"] = hier["idesc_comuna"].astype("string").str.zfill(2)
hier["idesc_barrio_z"] = hier["idesc_barrio_codigo"].astype("string").str.zfill(4)

rows = []
for label, sub_df in [("all rows with an IDESC answer", hier),
                      ("high-confidence rows only", hier[hier["matched"]])]:
    c = sub_df[sub_df["idesc_comuna_z"].notna()]
    b = sub_df[sub_df["idesc_barrio_z"].notna()]
    rows.append({
        "subset": label,
        "n_comuna": len(c),
        "comuna_agreement": float((c["our_comuna"] == c["idesc_comuna_z"]).mean()),
        "n_barrio": len(b),
        "barrio_code_agreement": float((b["our_barrio_code"] == b["idesc_barrio_z"]).mean()),
        "barrio_code_first2_agreement": float(
            (b["our_barrio_code"].str[:2] == b["idesc_barrio_z"].str[:2]).mean()),
        "barrio_code_last2_agreement": float(
            (b["our_barrio_code"].str[2:] == b["idesc_barrio_z"].str[2:]).mean()),
    })
agreement = pd.DataFrame(rows).set_index("subset")
display(agreement)

mism = hier[hier["matched"] & hier["idesc_barrio_z"].notna()]
mism = mism[mism["our_barrio_code"] != mism["idesc_barrio_z"]]
print("examples where the comuna agrees but the barrio digits do not:")
display(mism[mism["our_comuna"] == mism["idesc_comuna_z"]][
    ["matched_cadastral_address", "our_comuna", "idesc_comuna_z",
     "our_barrio_code", "idesc_barrio_z", "idesc_barrio_nombre"]].head(10))
print("distribution of the last two digits, ours vs IDESC (top 10 each):")
display(pd.DataFrame({
    "ours (predial[11:13])": mism["our_barrio_code"].str[2:].value_counts().head(10),
    "IDESC (barrio.codigo[2:])": mism["idesc_barrio_z"].str[2:].value_counts().head(10),
}))
'''
)

md(
    """
**Retractación explícita.** La conclusión provisional de §1.1 era incorrecta y se
retira:

* **La comuna sí queda verificada.** `predial[9:11]` coincide con el campo
  `comuna` de IDESC en ~94 % de todas las filas y en ~98 % de las filas de alta
  confianza (el resto son, en su mayoría, predios recuperados incorrectamente, no
  discrepancias de nomenclatura). Y los dos primeros dígitos del `barrio.codigo`
  de IDESC coinciden con `predial[9:11]` en ~98 %, lo que confirma que ese código
  empieza por la comuna.
* **El barrio NO.** `predial[11:13]` coincide con los dos últimos dígitos del
  `barrio.codigo` de IDESC en solo ~59 % de las filas de alta confianza. La tabla
  de distribuciones muestra por qué: los dígitos catastrales llegan a valores
  altos (`75`, `78`, `80`, `85`, `98`) donde IDESC usa numeraciones bajas
  (`04`, `06`, `07`, `08`, `13`). **Son dos nomenclaturas de barrio distintas que
  comparten el prefijo de comuna**, no la misma codificación.

Consecuencia para el modelo: la cabeza auxiliar de barrio se entrena sobre
`predial[9:13]`, que es un identificador **catastral** consistente internamente y
por lo tanto sigue siendo una señal de supervisión espacial válida. Lo que no se
puede hacer es presentar su salida como el código de barrio de IDESC ni cruzarla
con nomenclaturas municipales sin una tabla de equivalencias.
"""
)

md("### 7.4 Acuerdo estructural con la normalización propia de cada conjunto")

code(
    '''
from cali_address.evaluate import structural_agreement_table

display(structural_agreement_table(scored_full).set_index("dataset"))
print("Side-by-side examples (Fasecolda and inspecciones have their own normalized column):")
cmp = scored_full[scored_full["their_normalized"].notna()][
    ["dataset", "raw_address", "their_normalized", "rule_canonical", "matched_cadastral_address",
     "confidence"]
]
display(cmp.head(12))
'''
)

md("### 7.5 Calibración de la confianza y matriz de confusión del tipo de vía")

code(
    '''
from cali_address.evaluate import calibration_table, via_type_confusion

# Fixed 0.1-wide bins: deliberately independent of the operating threshold, so
# the calibration curve cannot be an artefact of where the cutoff happens to sit.
calib = calibration_table(scored_full, bins=np.round(np.arange(0.0, 1.05, 0.1), 2))
display(calib)

fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))
if len(calib):
    axes[0].plot([0, 1], [0, 1], "k:", lw=1, label="perfect calibration")
    axes[0].plot(calib["mean_confidence"], calib["accuracy_top1"], "o-", color="#1f4e79",
                 label="observed")
    for _, r in calib.iterrows():
        axes[0].annotate(f"n={int(r['n'])}", (r["mean_confidence"], r["accuracy_top1"]),
                         textcoords="offset points", xytext=(4, -10), fontsize=8)
    axes[0].set_xlabel("mean predicted confidence"); axes[0].set_ylabel("observed top-1 accuracy")
    axes[0].set_title("Confidence calibration"); axes[0].legend()
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)

gt = scored_full[scored_full["has_gt"]]
sns.histplot(data=gt, x="confidence", hue="correct_top1", bins=30, ax=axes[1],
             palette={True: "#5b9e4d", False: "#d1615d"}, element="step")
axes[1].axvline(TUNED_THRESHOLD, color="black", ls="--",
                label=f"tuned cutoff {TUNED_THRESHOLD:.2f}")
axes[1].set_title("Confidence distribution by correctness"); axes[1].legend()
plt.tight_layout()
plt.show()

print("Precision / recall of the high-confidence flag (rows with ground truth):")
hc = gt[gt["matched"]]
print(f"  precision @ confidence >= {TUNED_THRESHOLD:.2f}: "
      f"{hc['correct_top1'].mean():.4%} over {len(hc):,} rows")
print(f"  recall    @ confidence >= {TUNED_THRESHOLD:.2f}: "
      f"{hc['correct_top1'].sum() / max(gt['correct_top1'].sum(), 1):.4%}")
print("  NOTE: 'correct' here means the exact predio under the noisy "
      "point-in-polygon label of section 7.1.1, so this precision is a lower "
      "bound; the ordering of the score is what the curve on the left shows.")
'''
)

code(
    '''
conf = via_type_confusion(scored_full)
print("Via-type confusion (true cadastral via type vs the one we retrieved):")
display(conf)
if conf.size:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", ax=ax, cbar_kws={"label": "rows"})
    ax.set_title("Via-type confusion matrix")
    plt.tight_layout()
    plt.show()
'''
)

md("### 7.6 Análisis de errores")

code(
    '''
from cali_address.evaluate import failure_family

err = scored_full[scored_full["has_gt"] & ~scored_full["correct_top1"]].copy()
err["failure_family"] = err.apply(failure_family, axis=1)
print(f"incorrect top-1 predictions with ground truth: {len(err):,} "
      f"of {int(scored_full['has_gt'].sum()):,}")
fam = err.groupby(["dataset", "failure_family"]).size().unstack(fill_value=0)
display(fam)

fam_total = err["failure_family"].value_counts().to_frame("n")
fam_total["share"] = fam_total["n"] / len(err)
display(fam_total)
fam_total["n"].plot(kind="barh", figsize=(10, 3.8), color="#d1615d")
plt.title("Failure families (pooled)"); plt.xlabel("rows"); plt.tight_layout(); plt.show()
'''
)

code(
    '''
# Ten failing examples per dataset, with everything needed to diagnose them.
cols = ["raw_address", "rule_canonical", "matched_cadastral_address", "gt_cadastral_address",
        "confidence", "dist_m", "comuna", "gt_comuna", "failure_family"]
for key, g in err.groupby("dataset"):
    print("=" * 130)
    print(f"{key.upper()}  —  {len(g)} incorrect top-1 of "
          f"{int(scored_full[scored_full.dataset == key]['has_gt'].sum())} with ground truth")
    display(g.sort_values("confidence", ascending=False)[cols].head(10))
'''
)

code(
    '''
# Rows where NOTHING could be parsed: the hard floor of any rule-based approach.
unparsed = scored_full[~scored_full["parse_ok"]]
print(f"unparseable rows: {len(unparsed):,} ({len(unparsed)/len(scored_full):.2%})")
unparsed_tbl = pd.DataFrame({
    "n_rows": scored_full.groupby("dataset").size(),
    "n_unparsed": unparsed.groupby("dataset").size(),
}).fillna(0).astype(int)
unparsed_tbl["share_unparsed"] = unparsed_tbl["n_unparsed"] / unparsed_tbl["n_rows"]
display(unparsed_tbl)
display(unparsed[["dataset", "raw_address", "parse_notes", "idesc_dir_ajusta",
                  "idesc_estado_class"]].head(20))
'''
)

md(
    """
**Lectura del análisis de errores.** Las familias de fallo se reparten de forma
interpretable:

* **Texto no interpretable**: `SIN ESPECIFICAR`, identificadores numéricos,
  descripciones de ubicación (`Parte alta pichinde via la leonera`). Ninguna
  técnica de normalización puede resolverlos; el sistema los marca como tales.
* **Propiedad horizontal y lotes**: el texto identifica el conjunto
  (`TORRE 85 Y 86`, `MANZANA 14 LOTE 17`) pero no el predio, y el catastro tiene
  decenas de predios con la misma dirección base. Aquí el techo es del dato, no
  del modelo.
* **Corregimientos y zona rural**: la capa usada es `urbano_terreno`, de modo que
  direcciones rurales (Pance, Navarro, Pichindé, La Elvira) no tienen candidato
  correcto en el índice.
* **Confusión geográfica de vecindad**: el candidato elegido está en la manzana
  correcta o contigua pero es el predio equivocado, normalmente por diferencia en
  una letra de vía o en la placa. Es el error que los negativos duros espaciales
  atacan y donde el reranking estructural aporta la mayor parte de su ganancia.
* **Ruido de la referencia**: en `stickers` la `accuracy` del GPS llega a decenas
  de metros, por lo que parte de los "errores" de distancia son desplazamientos
  del punto de referencia, no del predio predicho.
"""
)

# --------------------------------------------------------------- persistence
md("### 7.7 Persistencia de los resultados de la evaluación")

code(
    '''
scored_full.drop(columns=["topk_doc_ids", "top5_doc_ids"], errors="ignore").to_parquet(
    os.path.join(ARTIFACTS, "eval_scored_full.parquet"), index=False
)
summary_full.to_csv(os.path.join(ARTIFACTS, "eval_summary_per_dataset.csv"), index=False)
overall_tbl.to_csv(os.path.join(ARTIFACTS, "eval_summary_consolidated.csv"))
idesc_comparison(scored_full).to_csv(os.path.join(ARTIFACTS, "eval_idesc_comparison.csv"),
                                    index=False)
err[cols].to_csv(os.path.join(ARTIFACTS, "eval_error_analysis.csv"), index=False)
calib.to_csv(os.path.join(ARTIFACTS, "eval_calibration.csv"), index=False)
accuracy_by_idesc_estado(scored_full).to_csv(
    os.path.join(ARTIFACTS, "eval_by_idesc_estado.csv"), index=False)
structural_agreement_table(scored_full).to_csv(
    os.path.join(ARTIFACTS, "eval_structural_agreement.csv"), index=False)
fam_total.to_csv(os.path.join(ARTIFACTS, "eval_failure_families.csv"))
refq.to_csv(os.path.join(ARTIFACTS, "eval_reference_quality.csv"), index=False)
agreement.to_csv(os.path.join(ARTIFACTS, "eval_hierarchy_agreement.csv"))

# Machine-readable digest of every headline number quoted in section 10.
gt_all = scored_full[scored_full["has_gt"]]
dist_all = pd.to_numeric(gt_all["dist_m"], errors="coerce").dropna()
hc_all = gt_all[gt_all["matched"]]
facts = {
    "catastro_records": int(len(catastro)),
    "catastro_with_address": int(catastro["direccion"].notna().sum()),
    "unique_addresses": int(len(docs)),
    "predios_per_address": float(len(catastro) / len(docs)),
    "roundtrip_all": float(rt["exact"].mean()),
    "roundtrip_modern_prefix": float(rt[rt.modern_prefix]["exact"].mean()),
    "roundtrip_parse_ok": float(rt["parse_ok"].mean()),
    "idempotence": float(idem.mean()),
    "training_minutes": float(train_report["seconds"] / 60),
    "training_epochs": int(len(history)),
    "val_recall@1": float(history["recall@1"].iloc[-1]),
    "val_recall@5": float(history["recall@5"].iloc[-1]),
    "val_recall@20": float(history["recall@20"].iloc[-1]),
    "gpu": train_report["gpu"],
    "throughput_seqs_per_s": float(history["seqs_per_s"].iloc[-1]),
    "peak_gpu_gb": float(history["gpu_mem_gb"].max()),
    "n_params": int(train_report["n_params"]),
    "eval_rows": int(len(scored_full)),
    "eval_rows_with_gt": int(len(gt_all)),
    "pooled": {k: (None if pd.isna(v) else float(v))
               for k, v in overall_tbl.loc["full (+ coord head)"].items()},
    "pooled_rule_only": {k: (None if pd.isna(v) else float(v))
                         for k, v in overall_tbl.loc["rule (no model)"].items()},
    "pooled_neural_only": {k: (None if pd.isna(v) else float(v))
                           for k, v in overall_tbl.loc["neural only"].items()},
    "pooled_no_geo": {k: (None if pd.isna(v) else float(v))
                      for k, v in overall_tbl.loc["neural + rerank"].items()},
    "median_dist_m": float(dist_all.median()),
    "highconf_precision": float(hc_all["correct_top1"].mean()) if len(hc_all) else None,
    "highconf_rows": int(len(hc_all)),
    "idesc_unique_queried": int(len(idesc_df)),
    "idesc_normalization_rate": float(
        (idesc_df["idesc_dir_ajusta"].fillna("").str.strip() != "").mean()),
    "idesc_georef_rate": float(idesc_df["idesc_lat"].notna().mean()),
    "idesc_estado_counts": idesc_df["idesc_estado"].value_counts().to_dict(),
    "unparsed_share": float((~scored_full["parse_ok"]).mean()),
    "inference_addr_per_s": float(len(raws) / t_full),
    "parcel_spacing": spacing,
    "tuning": {
        "threshold": TUNED_THRESHOLD,
        "weights": TUNED_WEIGHTS,
        "variant_thresholds": VARIANT_THRESHOLDS,
        "variant_weights": VARIANT_WEIGHTS,
        "selection_rule": TUNED["method"]["selection_rule"],
        "f1": diag["f1_at_chosen_threshold"],
        "precision": diag["precision_at_chosen_threshold"],
        "recall": diag["recall_at_chosen_threshold"],
        "top1_pooled": diag["top1_at_chosen_weights"],
        "top1_hand_picked": diag["top1_at_previously_shipped_weights"],
    },
    "hierarchy_agreement": agreement.to_dict("index"),
    "gt_parcels_resolved": int(GT_RESOLVED),
    "gt_parcels_scorable": int(GT_SCORABLE),
    "gt_parcels_unreachable": int(GT_UNREACHABLE),
    "n_comuna_classes": int(N_COMUNA),
    "n_barrio_classes": int(N_BARRIO),
    "val_recall_full_split": train_report.get("final_val_recall"),
    "val_queries": int(train_data["q_is_val"].sum()),
    "reference_quality": refq.set_index("idesc_estado_class").to_dict("index"),
}
with open(os.path.join(ARTIFACTS, "notebook_facts.json"), "w", encoding="utf-8") as fh:
    json.dump(facts, fh, indent=2, default=str)
print(json.dumps(facts, indent=2, default=str))
print()
print("written:")
for f in sorted(os.listdir(ARTIFACTS)):
    p = os.path.join(ARTIFACTS, f)
    if os.path.isfile(p):
        print(f"  {f:38s} {os.path.getsize(p)/1e6:9.2f} MB")
'''
)

md(
    """
## 8. Cómo usar el normalizador sobre datos nuevos

```python
import sys
sys.path.insert(0, "src")
from cali_address.inference import AddressNormalizer

normalizer = AddressNormalizer("artifacts")          # carga modelo + embeddings en GPU

# una dirección
normalizer.normalize_address("Cra. 98f #98-66")

# un lote (recomendado: la búsqueda vectorizada amortiza el coste)
out = normalizer.normalize_batch(df["direccion"].tolist(), k=20)
out[["raw_address", "matched_cadastral_address", "numero_predial_nacional",
     "manzana", "comuna", "lat", "lon", "confidence", "matched"]]
```

Campos devueltos:

| Campo | Significado |
|---|---|
| `rule_canonical` | forma canónica producida por la gramática de reglas |
| `matched_cadastral_address` | dirección catastral recuperada (mejor candidato) |
| `canonical_address` | la catastral si la confianza supera el umbral, si no la de reglas |
| `numero_predial_nacional` | predio representativo (30 dígitos) |
| `manzana` | código de manzana (17 dígitos) |
| `comuna`, `barrio_code` | jerarquía derivada del predial |
| `lat`, `lon` | centroide del predio en WGS84 |
| `confidence` | puntaje fusionado 0–1 |
| `matched` | `True` si `confidence >= 0.72` |
| `model_similarity`, `fuzz_score`, `struct_score` | componentes del puntaje |
| `parse_ok`, `parse_notes` | diagnóstico del parser |

> ### ⚠️ Advertencia obligatoria para quien consuma la salida
>
> **`normalize_address` siempre devuelve un predio y unas coordenadas, incluso
> cuando la entrada no es una dirección.** No hay valor nulo: para
> `SIN ESPECIFICAR`, para un identificador numérico o para `Parte alta pichinde`
> el sistema entrega el vecino más cercano en el espacio de embeddings, con su
> `numero_predial_nacional`, su manzana y su `lat`/`lon`. Esos campos son
> plausibles y **están mal**.
>
> Quien consuma la salida **debe** comprobar `matched` (o `confidence`) y
> `parse_ok` antes de usar cualquier otro campo. Escribir `lat`/`lon` en una base
> de datos sin ese filtro introduce coordenadas silenciosamente erróneas.
>
> ```python
> out = normalizer.normalize_batch(df["direccion"].tolist())
> usable   = out[out["matched"]]                       # automatizable
> review   = out[~out["matched"] & out["parse_ok"]]    # revisión humana
> rejected = out[~out["parse_ok"]]                     # devolver a la fuente
> ```

**Recomendación operativa.** Usar `matched == True` para automatizar, la banda
intermedia (entre ~0,55 y el umbral ajustado) para revisión humana asistida —el
sistema entrega los 20 candidatos en `topk_doc_ids`— y `parse_ok == False` para
devolver el registro a la fuente: son direcciones que ninguna herramienta puede
resolver. Y recordar que la precisión de `matched` frente a la verdad de terreno
disponible es de ~49 % para el predio *exacto*, aunque la manzana acierte en
~59 % y la distancia mediana sea de decenas de metros: el campo utilizable sin
revisión es la manzana y la ubicación aproximada, no el predio exacto.

## 9. Limitaciones

1. **Solo suelo urbano.** El índice se construye sobre `urbano_terreno`; las
   direcciones de corregimientos y veredas no tienen candidato correcto.
2. **Resolución máxima = dirección catastral.** El 1,02 % de multiplicidad
   predio/dirección impone un techo: cuando varios predios comparten el texto
   (propiedad horizontal), el sistema no puede desambiguar sin el complemento.
3. **Ruido en la verdad de terreno.** Las coordenadas de referencia provienen de
   GPS móvil y EXIF; los conjuntos `stickers` e `inspecciones` reportan
   `accuracy` de decenas de metros. Las métricas de distancia son por tanto una
   cota inferior del desempeño real.
4. **Muestras de 500 filas.** Los intervalos de confianza binomiales al 95 % son
   de aproximadamente ±4 puntos porcentuales por conjunto; `acciones`, con 78
   filas, tiene ±11 puntos.
5. **IDESC no es verdad de terreno**, es una referencia. Sus clases `C` y `F`
   indican georreferenciación aproximada o inexistente, así que la comparación
   se reporta como acuerdo, no como exactitud.
6. **La capa catastral es un instante.** Se descargó una vez y se cacheó; nuevas
   urbanizaciones no estarán en el índice hasta refrescarla.
"""
)

# The final summary (section 10) is appended after execution by
# scripts/finalize_notebook.py, so that every number it quotes is read back from
# the executed outputs instead of being written by hand.

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.14.4"},
}
nbf.write(nb, NB_PATH)
print(f"wrote {NB_PATH} with {len(nb.cells)} cells")
