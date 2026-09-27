# cali-address

Match free-text addresses from **any tabular dataset** (CSV, Excel, Parquet, JSON, GeoJSON, shapefile, SQL, ...)
against the Cali (Colombia) cadastre and get back the predial number, block (manzana), barrio and comuna,
with an explicit status and a precision level per row. A character-level Transformer retrieves candidates,
then structural and geographic rules decide. Your original columns are kept; the result columns are appended.

## Scope

- It normalizes addresses **in Santiago de Cali only**. The reference catalog is the Cali cadastre
  (330,387 addresses) and the model is trained on its grammar. Addresses from another city would be matched
  against Cali predios, which is wrong: another city needs its own reference catalog, gazetteer and a
  retrained model.
- The input dataset can be anything; column mapping and format handling are the flexible part
  ([docs/any-dataset.md](docs/any-dataset.md)).

## Install

Python 3.11+. Install CPU-only torch first, so pip does not pick the ~3 GB CUDA build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .                    # core: the cali-address command
pip install -e ".[excel,sql,geo]"   # optional input/output formats
pip install -e ".[api]"             # HTTP API
pip install -e ".[dev]"             # tests
pip install -e ".[all]"             # everything except dev
```

| Extra | Adds |
|-------|------|
| `excel` | `.xlsx` output and `.xls` input (openpyxl, xlrd) |
| `sql` | SQL sources (SQLAlchemy; you also need the database driver) |
| `geo` | GeoPackage input (geopandas, pyogrio) |
| `api` | FastAPI service (fastapi, uvicorn, python-multipart) |
| `train` | training / evaluation scripts under `scripts/` |
| `dev` | pytest, httpx, scikit-learn, ruff |

### Model artifacts (not in the wheel)

The wheel contains code only. The normalizer needs `model.pt`, `catastro_emb.pt` and `catastro_docs.parquet`
(plus optional `tuning.json`, `reliability.json`, `gazetteer.pkl`). Point the CLI at them with
`--artifacts-dir` or the `CALI_ARTIFACTS_DIR` environment variable. In a git clone, everything except
`catastro_emb.pt` (161 MB, above the GitHub file-size limit) is tracked in `artifacts/`; regenerate the
embeddings by following [docs/model-artifacts.md](docs/model-artifacts.md). Without them, `normalize` exits
with code 3 and lists the missing files.

## 60-second quickstart

```bash
cali-address formats                       # supported input / output formats
cali-address inspect data.csv              # format, header, columns, guessed address column, preview

cali-address normalize data.csv -o out.csv
cali-address normalize data.csv -o out.csv --dry-run 20          # preview 20 rows, write nothing

# Excel with the header on row 3 (0-based --header-row), a specific sheet
cali-address normalize report.xlsx -o out.xlsx --sheet Inspections --header-row 2

# Address split across several columns
cali-address normalize data.csv -o out.csv --address-parts via,numero,complemento

# SQL source (SQLAlchemy URL) with a read-only query
cali-address normalize sqlite:///local.db -o out.parquet --table direcciones
cali-address normalize postgresql://user:pw@host/db -o out.csv \
    --sql-query "SELECT id, via, numero FROM predios" --address-parts via,numero

# Same options from a TOML file (command line flags override it)
cali-address normalize --config run.toml
```

`run.toml`:

```toml
input = "reporte.xlsx"
output = "salida.csv"
sheet = "Inspecciones"
header_row = 2
address_parts = ["via", "numero", "complemento"]
id_col = "id"
chunk_size = 5000

[tunables]
min_struct = 0.6
ambiguity_delta = 0.02
```

`python -m cali_address ...` is equivalent. Exit codes: `0` ok, `2` usage / mapping / config error,
`3` I/O error (including missing artifacts).

## Output columns

Your columns first, then: `direccion_entrada`, `estado`, `motivo`, `direccion_normalizada`,
`fuente_normalizacion`, `numero_predial_nacional`, `manzana`, `lat`, `lon`, `confianza`, `confiabilidad`,
`confiabilidad_manzana`, `margen`, `nivel_precision`, `barrio_vereda`, `comuna_corregimiento`.
The full data dictionary is in [docs/any-dataset.md](docs/any-dataset.md).

- **`estado`**: `OK` (matched a predio), `SIN_MATCH` (parsed, no acceptable match), `NO_PARSEABLE`
  (not an address), plus `FUERA_DE_AREA` (municipality other than Cali) and `ERROR` (that row failed; the run continues).
- **`nivel_precision`** (OK rows): `predio` (street, cross and plate confirmed by the text), `via` (no cross number),
  `esquina` (no plate), `manzana` (approximate, or shared by several predios in one block),
  `direccion` (shared across blocks). Rows degrade to the level that is true instead of claiming a predio.
- **`confiabilidad`**: calibrated probability that the predio is right (`confiabilidad_manzana` for the block).

### How far to trust it

Measured on the frozen test split (2,000 rows, see [docs/model-artifacts.md](docs/model-artifacts.md)
and [docs/splits.md](docs/splits.md)), production configuration:

- Predio precision on **text-consistent "gold" rows: about 0.93** (0.927).
- Manzana precision on **all rows: about 0.69** (0.694). It is low mostly because the ground truth is a GPS
  point-in-polygon lookup and neighbouring parcels are ~6 m apart, so many labels are noisy, not because
  the matches are wrong.
- Coordinates (yours or the predio centroid) are never a correctness criterion; judge a match by manzana / predial
  and the matched address text.
- `confiabilidad` is only **weakly informative**: it tends to underestimate the real probability and
  discriminates little among OK rows. Use `estado` and `nivel_precision` first.

## HTTP API, Docker, Railway

The FastAPI service (`uvicorn cali_address.api.main:app`, extra `api`) is documented in
[docs/API.md](docs/API.md) (Spanish). The `Dockerfile` and `railway.toml` build the serving image; the
deployment guide is [README_DEPLOY.md](README_DEPLOY.md) (Spanish).

## Testing

```bash
pip install -e ".[dev,api,sql,excel,geo]"
python -m pytest -q
```

Tests that need the real serving artifacts are marked `requires_artifacts` and are skipped when
`catastro_emb.pt` is absent (as in CI). Everything else runs against stub models.

## Project layout

| Path | Contents |
|------|----------|
| `src/cali_address/` | the package: parser, model, inference, gazetteer, `service` (decision layer), `tables` and `io/` (readers, writers, pipeline), `cli.py`, `api/` |
| `scripts/` | training, tuning, evaluation, split building, reliability fitting (need a checkout) |
| `tests/` | pytest suite |
| `docs/` | [any-dataset](docs/any-dataset.md), [model-artifacts](docs/model-artifacts.md), [experiments](docs/experiments.md), [splits](docs/splits.md), [contributing](docs/contributing.md), API |
| `artifacts/` | model, config and cadastre index (see `docs/model-artifacts.md` for what is tracked) |
| `basemaps/` | public IDESC boundaries used by the gazetteer |
| `deploy/` | helper to assemble the serving artifacts for the image |
| `context/`, `outputs/` | **private**, gitignored (see below) |

## Privacy

`context/` (real citizen data: insured-property registry, disaster-victim registry with names and ID numbers,
inspection reports), `outputs/` and `artifacts/splits/` (raw evaluation addresses) are private and gitignored.
Never commit or ship them, and never build them into a Docker image (`.dockerignore` and `.railwayignore` exclude them).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Not yet specified.
