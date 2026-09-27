# Training experiments

Experiment infrastructure (Stage 2a). Every experiment lives in its own directory
`artifacts/experiments/<tag>/`; production artifacts are never written.

## Run one

```bash
PYTHONPATH=src python scripts/run_experiment.py --tag mixed_r30 \
    --set real_frac=0.3 --set plate_neg_weight=0.3 --set epochs=6
PYTHONPATH=src python scripts/run_experiment.py --tag ft_lowlr --mode finetune \
    --set lr=2e-4 --set epochs=3 --set freeze_aux=true
PYTHONPATH=src python scripts/run_experiment.py --tag baseline_prod --eval-only --from-dir artifacts
```

Pipeline: `train.py` -> (`finish_training_artifacts.py` if embeddings are missing) ->
`tune_threshold.py` -> `eval_strict.py` on **dev** (all rows and `--gold-only`) -> one row in
`artifacts/experiments/leaderboard.csv`. The experiment dir holds only what differs from production
(`model.pt`, `catastro_emb.pt`, `tuning.json`, `experiment.json`, `train_report.json`, eval files, `run.log`).
`catastro_docs.parquet`, `train_data.npz`, `spatial_negatives.npy` and plate-negative tables fall back to `artifacts/`.
`--smoke` runs a tiny pass (about 1 minute on the GPU) to prove the pipeline; its row has `smoke=True` and must be
ignored when selecting.

The runner refuses to start when `--tag` is unsafe, an override key is unknown, the experiment dir is non-empty
(`--overwrite` to reuse) or `frozen_test` is requested without `--final`. It hashes the six serving files before and
after and exits non-zero (and records `production_untouched` in `experiment.json`) if they changed.

## Add a variant

1. Prefer a `--set key=value` override; no code needed for any hyper-parameter below.
2. A new behaviour (loss term, sampler, augmentation) goes behind a `TrainConfig` field with a default that
   reproduces today's recipe, plus unit tests (tiny CPU fixtures in `tests/tiny.py`).
3. Add the key to `experiment.py::EXTRA_DEFAULTS` if it is not a `TrainConfig` field, so `--set` accepts it.
4. Run a `--smoke` first, then the real run with a descriptive tag.

## Hyper-parameters (defaults reproduce production)

| Group | Keys (default) |
|-------|----------------|
| Optimisation | `lr` 3e-3, `weight_decay` 0.01, `epochs` 8, `batch_size` 2048, `pct_start` 0.15, `div_factor` 25, `final_div_factor` 100, `grad_clip` 1.0, `seed` 42, `max_minutes` 24 (script) |
| Loss | `label_smoothing` 0.02, `init_temperature` 0.05, `w_coord` 0.5, `w_comuna` 0.2, `w_barrio` 0.2 |
| Architecture | `d_model` 256, `n_layers` 4, `n_heads` 8, `d_out` 256, `dropout` 0.1, `max_len` 40 |
| Views / negatives | `p_raw_view` 0.25, `n_hard_neg` 1, `plate_neg_weight` 0.0, `plate_neg_count` 8, `plate_neg_max_delta` 20 |
| Real pairs | `real_frac` 0.0 (share of each batch), `real_oversample` 1, `real_val_frac` 0.10, `real_split_seed` 42 |
| Mode | `mode` mixed / finetune, `freeze_aux` false, `init_from` (finetune start, default `artifacts/model.pt`), `steps_per_epoch` (override) |
| Data | `realistic_noise` false, `n_variants` 3, `real_pairs` (path override) |

Finetune presets (only for keys you did not set): `lr=3e-4`, `real_frac=0.5`. Architecture always comes from the checkpoint.
`plate_neg_weight` is the probability that a hard-negative slot is a plate neighbour (same block face, plate off by
1..`plate_neg_max_delta`, or same plate on the swapped via/cross) instead of a spatial neighbour. It needs `n_hard_neg >= 1`.
`--realistic-noise` measures structural pattern frequencies on real strings (train pairs + unlabeled RUD strings),
rescales existing augmentation operators (clamped 0.25..4x, none removed) and rebuilds `train_data.npz` inside the
experiment dir. The measured table (numbers only, no text) is stored in `experiment.json["realistic_noise"]`.

## Reading the leaderboard

`syn_recall@k`: held-out synthetic queries (cadastral docs never seen as positives). `real_val_recall@k`: recall on
the group-split real validation fold (10% of `train_pairs` by `address_key`, fixed by `real_split_seed`, so every run is
measured on the same fold; the real training pairs of a run never include it). `dev_*`: the production path
(`normalize_strict`) on `dev.parquet`; `dev_gold_*` is the same on gold rows only. `dev_gold_ok_coverage` = OK gold rows /
gold rows. `dev_lost_correct` = rows the model found but a rule/gate/threshold discarded.
Baseline row: `baseline_prod` (dev gold OK-coverage 0.958, gold predial precision 0.982).

## Selection and leakage rules

- Primary: highest `dev_gold_predial_precision` among runs with `dev_gold_ok_coverage` >= the baseline's.
  Tie-break (within one gold row): `dev_manzana_precision` on all dev rows. Recalls are diagnostics only.
- `frozen_test` is scored only with `--final`, once, on the already-selected candidate. Never select, tune or
  compare variants on it. `eval_strict.py` itself also refuses `frozen_test` without `--final`.
- Real pairs come only from `train_pairs.parquet`; dev/frozen rows are never read for training. The validation fold is
  split by `address_key`, so no address appears on both sides. Note the split notes in `docs/splits.md` (units of one
  building can share a predio across keys).
- Dev is small (289 gold rows): one row is 0.35 points of coverage. Treat differences under ~1-2 rows as noise.

## Retrain / promotion order

1. Experiment runs (never touches production) -> pick by the rule above.
2. `--final` on the single selected candidate (frozen_test).
3. Fit reliability for the winner: `python scripts/fit_reliability.py --artifacts-dir artifacts/experiments/<tag> --rows <dev rows parquet>`
   (writes `reliability.json` there).
4. Only after explicit user approval: back up `artifacts/`, copy the winner's `model.pt`, `catastro_emb.pt`, `tuning.json`,
   `reliability.json` over the production ones (`catastro_docs.parquet` and `gazetteer.pkl` are unchanged), then
   `python deploy/prepare_artifacts.py`.
5. The training tensors (`train_data.npz`) only change with `--realistic-noise`; the production one stays as is.

Standalone scripts keep working with their old defaults and gained `--artifacts-dir`
(`train.py --out-dir`, `finish_training_artifacts.py`, `tune_threshold.py`, `fit_reliability.py`, `eval_strict.py --out-dir`).
The env var `CALI_ARTIFACTS_DIR` sets the default for all of them; `AddressNormalizer(artifacts_dir)` accepts an experiment dir.

## Decision-layer options (Stage 3)

The strict decision layer (`service.normalize_strict`) has three options that default to the historical behaviour and are
read from an optional `chosen.decision` block of the artifact's `tuning.json` when the call does not pass them:

| Key | Meaning | Default |
|-----|---------|---------|
| `soft_rules` | rule classes downgraded from hard to soft (row stays OK, `nivel_precision` capped at `manzana`): `letra_via`, `letra_cruce_cardinal`, `letra_cruce_una_cara`, `letra_cruce`, `bis_via`, `bis_cruce`, `cuadrante_compl` | none |
| `max_soft` | soft violations an OK row may carry | 1 |
| `gate_fallback` | when the geographic gate would leave no candidate, keep the text-ranked ones (note + `manzana` cap) | false |

Identity rules (via type/number, cross number, cross type, plate beyond the soft window, structural floor) are never
softened. `eval_strict.py` takes `--soft-rules/--max-soft/--gate-fallback` overrides; without them it scores the
artifact's own block. A decision-only experiment has no model files: its `experiment.json` declares
`"inherits_model": true` and `AddressNormalizer` reads `model.pt`/`catastro_emb.pt` from `artifacts/` (`paths.model_path`).
