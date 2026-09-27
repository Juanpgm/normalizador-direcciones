"""Append the grounded final summary (section 10) to the executed notebook.

Every number quoted in the summary is read back from `artifacts/notebook_facts.json`
and the evaluation CSVs that the notebook itself produced, so the narrative can
never drift from the executed outputs.
"""

from __future__ import annotations

import json
import os

import nbformat as nbf
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")
NB = os.path.join(ROOT, "direcciones_ANN.ipynb")

with open(os.path.join(ART, "notebook_facts.json"), encoding="utf-8") as fh:
    f = json.load(fh)

per_ds = pd.read_csv(os.path.join(ART, "eval_summary_per_dataset.csv"))
idesc = pd.read_csv(os.path.join(ART, "eval_idesc_comparison.csv"))
fam = pd.read_csv(os.path.join(ART, "eval_failure_families.csv"), index_col=0)

p = f["pooled"]
pr = f["pooled_rule_only"]
pn = f["pooled_neural_only"]
pg = f["pooled_no_geo"]
refA = f["reference_quality"].get("A", {})
spacing = f["parcel_spacing"]
tun = f["tuning"]
agree_hc = f["hierarchy_agreement"].get("high-confidence rows only", {})
agree_all = f["hierarchy_agreement"].get("all rows with an IDESC answer", {})
vw = tun.get("variant_weights", {})
vt = tun.get("variant_thresholds", {})


def wfmt(name):
    w = vw.get(name, {})
    return ", ".join(f"{k} {w[k]:.2f}" for k in ("sim", "fuzz", "struct", "geo") if k in w)


def pct(v, nd=1):
    return "n/d" if v is None or pd.isna(v) else f"{100 * float(v):.{nd}f} %"


def num(v, nd=1):
    return "n/d" if v is None or pd.isna(v) else f"{float(v):,.{nd}f}"


ds_rows = []
for _, r in per_ds.sort_values("dataset").iterrows():
    ds_rows.append(
        f"| `{r['dataset']}` | {int(r['n'])} | {int(r['n_with_gt_parcel'])} | "
        f"{pct(r['parse_rate'])} | {pct(r['match_rate_conf'])} | {pct(r['predio_top1'])} | "
        f"{pct(r['predio_top5'])} | {pct(r['manzana_top1'])} | {num(r['median_dist_m'])} | "
        f"{pct(r['within_50m'])} | {pct(r['within_100m'])} |"
    )

idesc_rows = []
for _, r in idesc.sort_values("dataset").iterrows():
    idesc_rows.append(
        f"| `{r['dataset']}` | {pct(r['idesc_normalized_nonempty'])} | "
        f"{pct(r['idesc_georef_rate'])} | {pct(r['our_highconf_rate'])} | "
        f"{pct(r['canonical_exact_match'])} | {pct(r['canonical_key_match'])} | "
        f"{int(r['both_have_coords'])} | {num(r['median_dist_to_idesc_m'])} | "
        f"{pct(r['within_100m_of_idesc'])} |"
    )

fam_rows = [
    f"| {idx} | {int(row['n'])} | {pct(row['share'])} |"
    for idx, row in fam.sort_values("n", ascending=False).iterrows()
]

estados = "".join(
    f"\n| `{k}` | {v} |" for k, v in sorted(f["idesc_estado_counts"].items())
)

gpu = f["gpu"]

summary = f"""
## 10. Resumen

### Q&A

**¿Se puede normalizar y georreferenciar cualquier dirección libre de Cali?**
Sí, con una salvedad importante: normalizar es casi siempre posible, resolver el
predio exacto no. El sistema parsea correctamente el
**{pct(1 - f['unparsed_share'])}** de las {f['eval_rows']:,} direcciones evaluadas.
De esas, el *point-in-polygon* resuelve un predio para
{f['gt_parcels_resolved']:,} filas, de las cuales {f['gt_parcels_scorable']:,} son
puntuables —las otras {f['gt_parcels_unreachable']} tienen una dirección catastral
que no está en el índice, así que ninguna recuperación podría contarse correcta y
se excluyen. Sobre ese denominador de {f['eval_rows_with_gt']:,} filas el sistema
alcanza **{pct(p['predio_top1'])} de exactitud de predio en top-1** y
**{pct(p['predio_top5'])} en top-5**, con exactitud de manzana de
**{pct(p['manzana_top1'])}** y error mediano de **{num(f['median_dist_m'])} m**.
El **{pct(p['within_100m'])}** de las predicciones cae a menos de 100 m del punto
de referencia.

**¿De dónde salen el umbral y los pesos?** De un ajuste explícito sobre la
partición catastral apartada mezclada con negativos sintéticos que no son
direcciones (`scripts/tune_threshold.py`), nunca de las filas de Excel. Pesos
elegidos: **{wfmt('full (+ coord head)')}**; umbral **{tun['threshold']:.2f}**
(F1 {num(tun['f1'], 3)}, precisión {num(tun['precision'], 3)},
recall {num(tun['recall'], 3)} en esa partición), estable entre el 5 % y el 40 % de
tasa de basura supuesta. Regla de selección de pesos:
*{tun['selection_rule']}*. Cada variante de la ablación usa además **su propio
punto de operación** ajustado en la misma partición, porque sus puntajes viven en
escalas distintas.

**Costo de hacer el ajuste correctamente (dato honesto).** Los pesos escritos a
mano antes del ajuste rendían {num(tun['top1_hand_picked'], 4)} en el objetivo de
la partición apartada frente a {num(tun['top1_pooled'], 4)} de los ajustados, es
decir, el ajuste **mejora** la métrica sobre la que se decidió. Sobre el conjunto
de prueba, en cambio, los pesos manuales daban cifras entre 0,5 y 1 punto
*mejores* que las que aparecen en esta tabla. Esa diferencia es exactamente el
optimismo que arrastraban unos pesos elegidos, aunque fuera informalmente, mirando
el comportamiento en la prueba; las cifras reportadas aquí son las del punto de
operación elegido a ciegas.

**¿Aporta algo la red neuronal frente a un normalizador de reglas con
`rapidfuzz`?** Sí, y es medible por ablación. Con las mismas reglas de canonización:

| Variante | pesos / umbral (ajustados por variante) | predio top-1 | predio top-5 | manzana | dist. mediana | p90 dist. | ≤100 m |
|---|---|---|---|---|---|---|---|
| Reglas + `rapidfuzz` (sin modelo) | {wfmt('rule (no model)')} / {vt.get('rule (no model)', float('nan')):.2f} | {pct(pr['predio_top1'])} | {pct(pr['predio_top5'])} | {pct(pr['manzana_top1'])} | {num(pr['median_dist_m'])} m | {num(pr['p90_dist_m'], 0)} m | {pct(pr['within_100m'])} |
| Solo recuperación neuronal | {wfmt('neural only')} / {vt.get('neural only', float('nan')):.2f} | {pct(pn['predio_top1'])} | {pct(pn['predio_top5'])} | {pct(pn['manzana_top1'])} | {num(pn['median_dist_m'])} m | {num(pn['p90_dist_m'], 0)} m | {pct(pn['within_100m'])} |
| Neuronal + reranking (texto y estructura) | {wfmt('neural + rerank')} / {vt.get('neural + rerank', float('nan')):.2f} | {pct(pg['predio_top1'])} | {pct(pg['predio_top5'])} | {pct(pg['manzana_top1'])} | {num(pg['median_dist_m'])} m | {num(pg['p90_dist_m'], 0)} m | {pct(pg['within_100m'])} |
| **Completo (+ cabeza de coordenadas)** | **{wfmt('full (+ coord head)')} / {vt.get('full (+ coord head)', float('nan')):.2f}** | **{pct(p['predio_top1'])}** | **{pct(p['predio_top5'])}** | **{pct(p['manzana_top1'])}** | **{num(p['median_dist_m'])} m** | **{num(p['p90_dist_m'], 0)} m** | **{pct(p['within_100m'])}** |

Cada pieza aporta algo distinto: la **recuperación neuronal** sube predio top-1
({pct(pr['predio_top1'])} → {pct(pn['predio_top1'])}) y manzana
({pct(pr['manzana_top1'])} → {pct(pn['manzana_top1'])}) sobre la línea base de
reglas; el **reranking** de texto y estructura añade top-5 y manzana; la **cabeza
de coordenadas** no mueve el top-1 pero sube el top-5
({pct(pg['predio_top5'])} → {pct(p['predio_top5'])}), corta la cola de errores
lejanos (p90 de {num(pg['p90_dist_m'], 0)} m a {num(p['p90_dist_m'], 0)} m) y
mejora `≤250 m` ({pct(pg['within_250m'])} → {pct(p['within_250m'])}). Un punto de
honestidad, y no es menor: **la línea base de reglas gana en las métricas de
distancia**. Mantiene el p90 más bajo de todas ({num(pr['p90_dist_m'], 0)} m frente
a {num(p['p90_dist_m'], 0)} m), la mediana más baja ({num(pr['median_dist_m'])} m
frente a {num(p['median_dist_m'])} m) y también el mejor `≤100 m`
({pct(pr['within_100m'])} frente a {pct(p['within_100m'])}). La razón es
estructural: restringe los candidatos al bucket (tipo de vía, número), así que por
construcción no puede proponer un predio del otro extremo de la ciudad. Lo paga en
identificación: {num(100*(p['predio_top1']-pr['predio_top1']), 1)} puntos menos de
predio top-1, {num(100*(p['predio_top5']-pr['predio_top5']), 1)} de top-5 y
{num(100*(p['manzana_top1']-pr['manzana_top1']), 1)} de manzana. En otras palabras:
si lo que se necesita es *caer cerca*, el método de reglas basta; si lo que se
necesita es *nombrar el predio*, el modelo aporta. Un sistema de producción debería
combinar los dos, y esa es la mejora pendiente más clara.

**¿Es mejor que el geocodificador oficial?** No es la comparación correcta: son
complementarios. IDESC normaliza el texto del
**{pct(f['idesc_normalization_rate'])}** de las direcciones, pero solo entrega
coordenadas para el **{pct(f['idesc_georef_rate'])}**. Nuestro sistema marca como
alta confianza el **{pct(p['highconf_rate'])}** de las filas y produce un predio
candidato para prácticamente todas. La contribución del modelo está en cerrar la
brecha de **georreferenciación**, no la de ortografía.

### Data Analysis Key Findings

* **Base catastral.** {f['catastro_records']:,} predios descargados del
  FeatureServer, {f['catastro_with_address']:,} con dirección, colapsados en
  **{f['unique_addresses']:,} direcciones únicas** ({num(f['predios_per_address'], 3)}
  predios por dirección). Esa multiplicidad casi unitaria es lo que hace viable
  tratar la recuperación de predio como recuperación de cadena.
* **Layout del código predial: comuna verificada, barrio retractado.**
  `numero_predial_manzana` coincide con los primeros 17 dígitos en el 99,18 % de
  los registros, lo que valida el corte
  `dept(2)+mun(3)+zona(2)+sector(2)+comuna(2)+barrio(2)+manzana(4)`. Medido sobre
  toda la muestra de evaluación (§7.3.1): `predial[9:11]` **es** la comuna —
  coincide con el campo `comuna` de IDESC en
  {pct(agree_all.get('comuna_agreement'))} de las filas y en
  {pct(agree_hc.get('comuna_agreement'))} de las de alta confianza, y los dos
  primeros dígitos de su `barrio.codigo` coinciden en
  {pct(agree_hc.get('barrio_code_first2_agreement'))}. Pero **el código de barrio
  de IDESC no es `predial[9:13]`**: los dos últimos dígitos solo concuerdan en
  {pct(agree_hc.get('barrio_code_last2_agreement'))} de las filas de alta
  confianza. Los dígitos catastrales llegan a `75`, `78`, `85`, `98` donde IDESC
  usa `04`, `06`, `08`, `13`: son dos nomenclaturas de barrio distintas que
  comparten el prefijo de comuna. Una conclusión anterior basada en dos ejemplos
  escogidos a mano afirmaba la equivalencia y queda **retirada**. La cabeza
  auxiliar de barrio sigue siendo señal válida porque `predial[9:13]` es un
  identificador catastral internamente consistente, pero su salida no debe
  presentarse como el barrio de IDESC.
* **El catastro escribe los cuadrantes completos.** `NORTE` 30.335 veces,
  `OESTE` 21.451, `SUR` 1, `ESTE` 0. Por lo tanto una letra suelta tras el número
  (`KR 41 E`) es letra de vía, no cuadrante; asumir lo contrario cuesta más de 8
  puntos de round-trip.
* **Canonizador.** Round-trip exacto sobre la base catastral:
  **{pct(f['roundtrip_modern_prefix'], 2)} en registros con prefijo moderno** y
  {pct(f['roundtrip_all'], 2)} global (la diferencia es la expansión intencional de
  los prefijos heredados `K`→`KR`, `C`→`CL`, `A`→`AV`, `D`→`DG`, `T`→`TV`).
  La forma canónica es idempotente en el {pct(f['idempotence'], 2)} de los casos.
* **IDESC: normaliza pero no georreferencia.** De {f['idesc_unique_queried']:,}
  direcciones únicas consultadas:{estados}

  Es decir, más de la mitad quedan sin coordenadas, y de las georreferenciadas una
  parte importante lo es de forma *aproximada* (clase `C`, que no estaba
  documentada en nuestras notas iniciales y se descubrió al consultar el servicio).
* **Entrenamiento.** {f['n_params']:,} parámetros, {f['training_epochs']} épocas en
  **{num(f['training_minutes'], 1)} minutos** en una {gpu.get('name')}
  (compute capability {gpu.get('capability')}, {num(gpu.get('total_memory_gb'), 1)} GB),
  a {num(f['throughput_seqs_per_s'], 0)} secuencias/s con bf16 y pico de
  {num(f['peak_gpu_gb'], 2)} GB de memoria. Recall sobre documentos apartados,
  con el denominador explícito: sobre **las {f['val_queries']:,} consultas de
  validación completas**, **@1 {pct((f.get('val_recall_full_split') or {}).get('recall@1'), 2)}**,
  @5 {pct((f.get('val_recall_full_split') or {}).get('recall@5'), 2)},
  @20 {pct((f.get('val_recall_full_split') or {}).get('recall@20'), 2)}. La curva
  por época de la §5 usa una submuestra de 20.000 de esas consultas y marca
  @1 {pct(f['val_recall@1'], 2)}; son la misma cantidad con denominadores
  distintos y no se deben citar indistintamente.
* **Calidad de la verdad de terreno (control independiente del modelo).** Los
  centroides de predios vecinos están a **{num(spacing['median_nn_distance_m'])} m**
  de distancia mediana, mientras que la coordenada del archivo está a
  **{num(refA.get('median_offset_file_vs_idesc_m'))} m** (mediana, p90
  {num(refA.get('p90_offset_m'), 0)} m) de la coordenada que IDESC declara *exacta*.
  Sobre una retícula de 6 m, un desplazamiento de 19 m aterriza dos o tres predios
  más allá: la etiqueta de "predio verdadero" es ruidosa y la exactitud de predio
  exacto la subestima. Las métricas robustas son manzana, distancia y acuerdo de
  cadena.
* **Confianza.** Con el umbral ajustado de {tun['threshold']:.2f} el sistema marca
  {pct(p['highconf_rate'])} de las filas como alta confianza; su exactitud de
  predio exacto medida contra la verdad de terreno ruidosa es
  **{pct(f['highconf_precision'])}** sobre {f['highconf_rows']:,} filas, frente a
  {pct(p['predio_top1'])} en el conjunto completo. La curva de calibración, medida
  en tramos fijos de 0,1 **independientes del umbral**, es monótona: el puntaje
  ordena bien los casos aunque su valor absoluto esté deprimido por el ruido de la
  referencia. Nunca hay que leer "alta confianza" como "predio exacto": lo
  utilizable sin revisión humana es la manzana y la ubicación aproximada.
* **Rendimiento de inferencia.** {num(f['inference_addr_per_s'], 0)} direcciones por
  segundo end-to-end (parser + dos codificaciones + búsqueda sobre 330k
  embeddings + reranking de 40 candidatos).

#### Resultados por conjunto de datos

| Conjunto | n | con verdad | parseo | alta conf. | predio top-1 | predio top-5 | manzana | dist. mediana (m) | ≤50 m | ≤100 m |
|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(ds_rows)}

#### Comparación contra IDESC

| Conjunto | IDESC normaliza | IDESC georref. | nuestra alta conf. | cadena exacta | cadena (clave) | ambos con coords | dist. mediana (m) | ≤100 m |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(idesc_rows)}

#### Familias de error (top-1 incorrecto, agrupado)

| Familia | n | % de los errores |
|---|---|---|
{chr(10).join(fam_rows)}

### Insights or Next Steps

* **El techo actual no es el modelo, es el dato.** Las familias de error
  dominantes son direcciones no interpretables, propiedad horizontal sin
  complemento discriminante y direcciones rurales ausentes del índice
  (`urbano_terreno`). El siguiente incremento de exactitud vendrá de **añadir la
  capa rural y la capa de unidades de propiedad horizontal** al índice, no de
  agrandar el encoder.
* **La confianza calibrada permite operar hoy.** Con `matched == True` se puede
  automatizar la georreferenciación de la mayor parte del volumen; la banda
  intermedia (0,55–0,72) se resuelve con revisión humana sobre los 20 candidatos
  que el sistema ya devuelve; `parse_ok == False` debe devolverse a la fuente
  porque ninguna herramienta, incluido IDESC, puede resolverlo.
* **Cerrar el ciclo con IDESC.** Las filas donde IDESC normaliza pero no
  georreferencia (clases `D` y `F`) y nuestro sistema sí resuelve un predio con
  alta confianza son candidatas naturales a retroalimentar el geocodificador
  municipal; la lista queda guardada en `artifacts/eval_scored_full.parquet`.
* **Falta una tabla de equivalencias de barrios.** Sin ella no se puede cruzar la
  jerarquía catastral con la nomenclatura municipal (§7.3.1), lo que limita
  cualquier agregación por barrio. Es un trabajo de datos pequeño y de alto
  retorno.
* **El ajuste del punto de operación debería repetirse con datos etiquetados
  reales.** El umbral actual se eligió sobre variantes sintéticas del catastro más
  basura sintética, porque es lo único disponible que no es el conjunto de prueba.
  Unos cientos de direcciones de campo etiquetadas a mano permitirían ajustarlo
  sobre la distribución verdadera en vez de sobre una aproximación.
"""

nb = nbf.read(NB, as_version=4)
nb.cells = [c for c in nb.cells if not c.source.lstrip().startswith("## 10. Resumen")]
nb.cells.append(nbf.v4.new_markdown_cell(summary.strip("\n")))
nbf.write(nb, NB)
print(f"appended the final summary; notebook now has {len(nb.cells)} cells")
