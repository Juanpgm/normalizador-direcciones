# Model artifacts

## The model

A character-level Transformer dual encoder (4 layers, `d_model` 256, 8 heads, 256-d output, 3,375,601
parameters, max length 40). Address strings and cadastral documents are embedded into the same space and matched by
cosine similarity, then re-ranked by structural and geographic rules (`cali_address.service.normalize_strict`).
It is trained on a **synthetic cadastre**: each cadastral address is corrupted by augmentation operators to simulate
field text. No real citizen data is used by the production recipe.

Held-out synthetic validation recall (`artifacts/train_report.json`, 8 epochs, about 16 min on one GPU):

| recall@1 | recall@5 | recall@20 |
|----------|----------|-----------|
| 0.9352 | 0.9637 | 0.9869 |

## Tracked in git

| File | Size | Why tracked |
|------|------|-------------|
| `artifacts/model.pt` | 14 MB | The trained weights |
| `artifacts/tuning.json` | 5 KB | Fusion weights, threshold, decision options |
| `artifacts/reliability.json` | 4 KB | Calibrated confidence model |
| `artifacts/train_report.json` | 5 KB | Training history and validation recall |
| `artifacts/gazetteer.pkl` | 5 MB | Parsed public basemaps (cache) |
| `artifacts/catastro_docs.parquet` | 21 MB | Public cadastre: address, parcel id, block, barrio, centroid (14 columns, 330,387 rows, no personal data) |
| `artifacts/experiments/leaderboard.csv` | 3 KB | One metrics row per experiment |
| `artifacts/experiments/<tag>/{experiment,tuning,reliability}.json` | small | Per-experiment config and calibration |
| `basemaps/*.geojson` | 12 MB | Public administrative boundaries |

## Not tracked, and why

| File | Reason |
|------|--------|
| `artifacts/catastro_emb.pt` (161 MB) | Above GitHub's 100 MB per-file limit. Regenerable, see below. |
| `artifacts/experiments/*/model.pt`, `catastro_emb.pt`, `*.npy`, `*.npz` | Large, regenerable |
| `artifacts/train_data.npz`, `spatial_negatives.npy` | Large, regenerable |
| `artifacts/splits/`, `strict_eval_*`, `eval_*.parquet` and row-level CSVs | Contain raw addresses from private sources (privacy) |
| `artifacts/catastro.parquet`, `catastro_parts/` | Raw download with geometry (87 MB), regenerable |
| `context/`, `outputs/` | Real citizen data (Fasecolda, RUD with names and ID numbers, inspections) |
| `deploy/artifacts/` | A copy of the six serving files |
| `direcciones_ANN.ipynb`, `.atl/` | Large or local tooling |

## Regenerating the untracked files

Run from the repository root. Steps 1-2 only need network access; the data is public.

```bash
# 1. Download the cadastral layer (resumable) -> artifacts/catastro.parquet
python scripts/download_catastro.py

# 2. Build docs, spatial negatives and training tensors
#    -> catastro_docs.parquet, spatial_negatives.npy, train_data.npz
python scripts/prepare_training.py

# 3. Train (writes model.pt, train_report.json, catastro_emb.pt; needs a GPU for reasonable time)
PYTHONPATH=src python scripts/train.py

# 3b. If model.pt exists but catastro_emb.pt is missing, embed without retraining
PYTHONPATH=src python scripts/finish_training_artifacts.py

# 4. Fusion weights and threshold -> tuning.json
PYTHONPATH=src python scripts/tune_threshold.py

# 5. Calibrated confidence -> reliability.json (needs dev rows, see below)
PYTHONPATH=src python scripts/fit_reliability.py --rows <dev rows parquet>

# 6. Assemble the six serving files into deploy/artifacts/
python deploy/prepare_artifacts.py
```

`gazetteer.pkl` is rebuilt automatically from `basemaps/` the first time `Gazetteer.load` runs.
The six serving files are `model.pt`, `catastro_emb.pt`, `catastro_docs.parquet`, `tuning.json`,
`gazetteer.pkl` and `reliability.json` (see `README_DEPLOY.md`).

Experiments write to `artifacts/experiments/<tag>/` and never touch production; see [experiments.md](experiments.md).

## Splits

`scripts/build_splits.py` builds `artifacts/splits/` (`frozen_test`, `dev`, `train_pairs`, `unlabeled_addresses`) and
needs the private `context/` workbooks, so it cannot run from a fresh clone. Rules and rationale are in
[splits.md](splits.md). Step 5 and the strict evaluation (`scripts/eval_strict.py`) also depend on those splits.

## Experiment status

- Stage 2b/3: retraining variants (real pairs, plate negatives, realistic noise, fine-tuning, seed repeats) did not
  beat the seed-noise floor, so the production weights are unchanged.
- Decision-layer candidate `s3_final` (weights 0.7/0/0.2/0.1, threshold 0.70, soft rules `letra_via`,
  `letra_cruce_cardinal`, `bis_via`, `bis_cruce`, `cuadrante_compl`, `max_soft` 1, `gate_fallback`) on `frozen_test`:
  - all-rows OK: 1036 -> 1090
  - gold OK: 398 -> 405
  - gold predial precision: 0.927 -> 0.9259
  - all-rows manzana precision: 0.694 -> 0.6807 (-0.0133, mostly noisy-label rows)
- **Not promoted.** Production serving files are unchanged. Details and selection rules: [experiments.md](experiments.md).
