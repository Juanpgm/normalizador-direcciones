# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

First packaged release.

### Added

- Three-step local use: `cali-address setup` checks the artifacts, regenerates `catastro_emb.pt` from
  `model.pt` + `catastro_docs.parquet` (atomic write, progress, `--force`, `--device`, `--batch-size`) and
  self-checks with 3 invented addresses; `cali-address web` starts the existing FastAPI upload page
  (`--host`, `--port`, `--no-browser`). `train.embed_documents` gained an optional `progress` callback.
- Any-dataset input and output: `cali-address normalize | inspect | formats` reads CSV, TSV, Excel,
  Parquet, JSON/JSONL, GeoJSON, shapefile, GeoPackage, URLs and SQL databases, maps the address column
  (auto-detected, single column, or several parts), streams in chunks, and writes CSV, Excel, Parquet,
  JSON, JSONL or GeoJSON atomically. TOML `--config` files, `--dry-run`, run summaries and exit codes
  (0 ok, 2 usage, 3 I/O). See `docs/any-dataset.md`.
- Friendly error when the model artifacts are missing: the CLI lists the exact files and points to
  `docs/model-artifacts.md`. `CALI_ARTIFACTS_DIR` sets the default artifacts directory.
- Decision layer with soft-rule parameters (`--soft-rules`, `--max-soft`, `--gate-fallback`) that can also
  be carried by `tuning.json`.
- Experiment infrastructure: isolated experiment directories, leaderboard, seed repeats, real-pair
  training, plate negatives and realistic-noise augmentation (`docs/experiments.md`).
- Gold-quality labels and leakage-free data splits (`docs/splits.md`).
- Packaging: `pyproject.toml` with optional extras (`api`, `sql`, `excel`, `geo`, `train`, `dev`, `all`),
  the `cali-address` console script, `__version__`, and a GitHub Actions workflow.

### Changed

- `normalize INPUT` without `-o` now writes `<input stem>_normalizado<ext>` next to a local input (same format;
  xls -> xlsx, txt -> csv, shp/zip/gpkg -> geojson) instead of failing with exit 2. URL and database inputs still
  require `-o`.
- SQL input is table/view based by design (`--table [schema.]name`, session read-only where the driver supports it): there is no free-form SQL option because a text filter cannot be a security boundary for SQL, so create a database view for joins or filters.
- `--help`, `inspect` and `formats` no longer import torch (table helpers moved to `cali_address.tables`;
  every name is still importable from `cali_address.service`).
- Tests that need the real serving artifacts carry the `requires_artifacts` marker and are skipped when
  the artifacts are absent.

### Notes

- Model weights and embeddings are not shipped in the wheel.
