# Normalizing any dataset

`python -m cali_address` reads addresses from almost any table (file, URL or
database), normalizes them against the Cali cadastre with the same engine the API
uses, and writes the result to almost any format. Your original columns are kept;
the normalization columns are appended.

```
python -m cali_address normalize INPUT -o OUTPUT [options]
python -m cali_address inspect INPUT          # what is in this file? (does not normalize)
python -m cali_address formats                # supported input / output formats
```

Install with `pip install -e .` (see the README) and use the `cali-address` command,
or run `python -m cali_address` from a checkout with `PYTHONPATH=src`. The examples
below use `python -m cali_address`; `cali-address` is an exact alias.
Exit codes: `0` ok, `2` usage / column-mapping / config error, `3` I/O error
(missing or corrupt input, unreachable database, unwritable output). User errors
print one `error:` line with the fix, never a traceback.

## Quickstart by source

Always start with `inspect` when you do not know the file: it shows the detected
format, delimiter, encoding, header row, the columns, the guessed address column
and the first 5 rows.

```
python -m cali_address inspect data.csv
```

**CSV with an odd delimiter or encoding.** Delimiter (`, ; TAB |`) and encoding
(BOM, UTF-8, cp1252/latin-1, UTF-16) are sniffed. Override when the sniffing is wrong:

```
python -m cali_address normalize data.csv -o out.csv
python -m cali_address normalize data.txt -o out.csv --delimiter "|" --encoding cp1252
python -m cali_address normalize list.txt -o out.csv     # .txt with no delimiter: one address per line
```

**Excel with the header on row 3.** `--header-row` is 0-based (row 3 is `2`). When omitted
the header row is detected (works when the header is the densest of the first 15
rows). Only the first sheet is read unless you pass `--sheet`.

```
python -m cali_address normalize report.xlsx -o out.xlsx --sheet Inspections --header-row 2
```

**Address split across columns.** Parts are joined in the given order with
`--parts-sep` (default one space); blank parts are skipped, integral numbers lose the
`.0`.

```
python -m cali_address normalize data.csv -o out.csv --address-parts via,numero,complemento
python -m cali_address normalize data.csv -o out.csv --address-parts via,numero --parts-sep " "
```

**Database (SQLAlchemy URL).** Use `--table [schema.]name` or a read-only `--sql-query`
(a single `SELECT`/`WITH`). Rows are fetched in chunks through a server-side cursor.
Table names are validated identifiers and no user value is ever formatted into SQL.
The query text is trusted operator input: it runs as written. Passwords are masked
in every message. Drivers are optional: `pip install sqlalchemy` plus the driver
(`psycopg2-binary` for Postgres); a missing one gives an install hint.

```
python -m cali_address normalize sqlite:///local.db -o out.parquet --table direcciones
python -m cali_address normalize postgresql://user:pw@host/db -o out.csv \
    --sql-query "SELECT id, via, numero FROM predios WHERE municipio = 'Cali'" --address-parts via,numero
```

**GeoJSON, shapefile, GeoPackage.** GeoJSON (`FeatureCollection`) and shapefile
(`.shp` beside its `.shx/.dbf/.prj`, or a `.zip`) use the existing dependencies
(`shapely`, `pyshp`, `pyproj`); each feature's centroid becomes `_lon` / `_lat` in
the passthrough columns (informational). `.gpkg` needs `pip install pyogrio geopandas`
(`--table` selects the layer).

```
python -m cali_address normalize points.geojson -o out.geojson --address-col direccion
```

Other inputs: `.tsv`, `.parquet`, `.json` (list of objects, `{"records": [...]}` or
a FeatureCollection), `.jsonl`, `.xls` (needs `xlrd`), http(s) URLs (downloaded to a
temp file first), and `--format` to override the extension. Outputs: `.csv` (UTF-8
with BOM), `.tsv`, `.xlsx` (sheets `normalizado` / `revisar` / `resumen`), `.parquet`,
`.json`, `.jsonl`, `.geojson` (a Point per located row, `null` geometry otherwise).

## Column mapping

| Option | Meaning |
|---|---|
| `--address-col NAME` | Address column. Auto-detected (Spanish/English synonyms) when omitted. If two columns are equally plausible (`direccion` and `domicilio`) the command stops and lists the candidates: it never guesses. |
| `--address-parts A,B,C` | Join several columns into one address. Mutually exclusive with `--address-col`. |
| `--id-col NAME` | Always carried to the output, even with `--keep-columns none`. |
| `--municipality-col NAME` | Out-of-area guard (see below). Off unless given. |
| `--lat-col` / `--lon-col` | Validated and reported only. Coordinates never influence matching. |
| `--keep-columns` | `all` (default), `none`, or a comma list of input columns to carry. |

A missing or unknown column fails before the model is loaded and lists the detected
columns.

## Output columns

Your columns first (all of them by default), then these, always in this order.
A source column whose name equals an output column (or a coordinate-like name such as
`lat`, `lon`, `latitud`, `longitud`, which are renamed as a pair) is carried as
`<name>_original`. When the address is a single kept column, `direccion_entrada` is
omitted because it would duplicate that column.

| Column | Type | Meaning | Null / empty when |
|---|---|---|---|
| `direccion_entrada` | text | The address text received (joined text for `--address-parts`). | never; omitted with a single kept address column |
| `estado` | text | `OK`, `SIN_MATCH`, `NO_PARSEABLE`, plus `FUERA_DE_AREA` and `ERROR` (below). | never |
| `motivo` | text | Why the row was not OK. On OK rows it is empty or a note: `aproximado: ...` and / or `direccion compartida por N predios`. | empty on a plain OK row |
| `direccion_normalizada` | text | The matched cadastral address (OK) or the deterministic canonical form (`reglas`). | blank/unparseable input, `ERROR`, `FUERA_DE_AREA` |
| `fuente_normalizacion` | text | `catastro` (real match), `reglas` (grammar only) or empty. | when `direccion_normalizada` is null |
| `numero_predial_nacional` | text | Predial number of the matched predio. | non-OK rows |
| `manzana` | text | Cadastral block code. | non-OK rows |
| `lat`, `lon` | float | Centroid of the matched predio (EPSG:4326). | non-OK rows |
| `confianza` | float 0-1 | Fused match score, uncalibrated. | non-OK rows |
| `confiabilidad` | float 0-1 | Calibrated probability the predio is right. Informational; never changes `estado`. | non-OK rows, or if `reliability.json` is absent |
| `confiabilidad_manzana` | float 0-1 | Same for the manzana (always >= `confiabilidad`). | same as above |
| `margen` | float | Score gap to the best rule-passing candidate in ANOTHER manzana; small = ambiguous. | non-OK rows, or no competitor |
| `nivel_precision` | text | `predio` (via, cross and plate confirmed by the text), `via` (no cross number), `esquina` (no plate), `manzana` (approximate or shared address within a block), `direccion` (shared across blocks). | non-OK rows |
| `barrio_vereda` | text | Official IDESC barrio / vereda. For OK rows from the matched predio; otherwise from a place detected in the text. | nothing known |
| `comuna_corregimiento` | text | `Comuna 19` style name or corregimiento. Same sourcing. | nothing known |

`ERROR` and `FUERA_DE_AREA` rows carry `direccion_entrada` and `motivo` only; every
other result column is empty.

## Error semantics

* **One bad row never aborts the run.** If the normalizer raises for a row, that row
  becomes `estado='ERROR'`, `motivo='<ExceptionType>: <short message>'`, and the rest
  of its chunk is still scored normally (the failing batch is bisected to isolate the
  offending rows). The summary counts them and prints up to 5 samples. `--on-error raise`
  aborts instead.
* **Blank, placeholder and non-text values** (`None`, `NaN`, `N/A`, dates, numbers)
  never crash: they become `SIN_MATCH` / `NO_PARSEABLE` like any other unusable address.
  Text formats are read as text, so ids like `007` and addresses like `NA` are preserved.
* **Out-of-area guard.** With `--municipality-col`, a non-empty municipality that is not
  Cali is reported as `FUERA_DE_AREA` without matching. Accepted as Cali (accent and case
  insensitive): `Cali`, `Santiago de Cali`, `Cali - Valle`, `Cali, Valle del Cauca`,
  `76001`. Empty or placeholder values are not guarded.
* **Order and count** of the input are preserved for any `--chunk-size`; output is
  deterministic. The output file is written to a temporary file and moved into place
  only on success, so a failed run never leaves a truncated file or clobbers a previous one.
* Matching results are identical to the legacy `scripts/normalizar.py` and the API for the
  same rows and options (the run is `normalize_strict` on every chunk).

## Run summary

Printed to stderr and, with `--summary-json PATH`, saved: `rows`, `ok`, `sin_match`,
`no_parseable`, `fuera_de_area`, `error`, `by_estado`, `by_nivel_precision`, `chunks`,
`seconds`, `rows_per_sec`, `error_samples`, the gazetteer gate counters, reader warnings
and the resolved column mapping. `--dry-run N` processes only the first N rows, prints
them and writes nothing.

## Config file

`--config run.toml` accepts the same options (`-` or `_`), flat or grouped in tables.
Precedence: defaults < config file < command line. Unknown keys and malformed TOML are
errors (exit 2).

```toml
input = "reporte.xlsx"
output = "salida.csv"
sheet = "Inspecciones"
header_row = 2
address_parts = ["via", "numero", "complemento"]
id_col = "id"
municipality_col = "municipio"
chunk_size = 5000

[tunables]
min_struct = 0.6
ambiguity_delta = 0.02
```

The tunables are the same as the legacy CLI: `--min-struct`, `--plate-tolerance`,
`--ambiguity-delta`, `--barrio-buffer`, `--zone-buffer`, `--gate-escalate`,
`--gazetteer/--no-gazetteer`, `--soft-rules`, `--max-soft`, `--gate-fallback`,
`--threshold`, `--device`, `--artifacts-dir`, `--basemaps`. Defaults equal today's
CLI and API behaviour.

## Performance

* `--chunk-size` (default 20 000) bounds memory: CSV/TSV/TXT, XLSX (read-only mode),
  Parquet, JSONL, `.shp` and SQL stream, so files larger than RAM work. `.xls`, `.json`,
  GeoJSON, zipped shapefiles and GPKG are read whole and then sliced. The `.xlsx` output is
  buffered (Excel caps a sheet at 1 048 575 rows); use csv or parquet for huge results.
* Chunk size never changes the results. Larger chunks amortize model batching (better on
  GPU); smaller ones give steadier progress. If a chunk fails, only the bisection retries
  cost extra.
* Encoding detection scans the file once for UTF-8 validity before reading.

## Limits

* The reference catalog is the **Cali cadastre** (IDESC / catastro parquet) and the model
  is trained on its address grammar. Addresses from another city will be matched against
  Cali predios, which is wrong. Another city needs its own catalog, gazetteer and a
  retrained model (see `docs/model-artifacts.md`); this layer only removes the file-format
  and column-mapping obstacles.
* Coordinates in your data are never used to decide a match.
* Header auto-detection is a heuristic; when `inspect` shows a wrong header, pass
  `--header-row`.
* A CSV row with more non-empty fields than the header is an error (exit 3, with the line number), never silently truncated; short rows are padded and trailing separators ignored.
* A `.txt` with no delimiter is treated as one address per line without a header.
* The HTTP API keeps its own upload path (three statuses only); it does not emit `ERROR` /
  `FUERA_DE_AREA`.
