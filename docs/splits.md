# Data splits

Built by `scripts/build_splits.py` (CPU only) into `artifacts/splits/`.

| File | Content | Use |
|------|---------|-----|
| `frozen_test.parquet` | 2,000 Fasecolda rows with GT, unique address keys, none in dev | Final measurement only |
| `dev.parquet` | the 2,078 rows of `eval_scored_full.parquet` + `gt_quality` | Tuning rules / reliability |
| `train_pairs.parquet` | real `raw_address -> gt_doc_id` pairs, gold rows only (Fasecolda, inspecciones, stickers) | Training only |
| `unlabeled_addresses.parquet` | `raw_address`, `dataset` only (RUD, acciones) | Unsupervised / augmentation seeds |
| `splits_report.json` | counts, overlap matrix, gold ratios | Audit |

## Rule

- `frozen_test` is NEVER used for tuning (thresholds, weights, rules, reliability, model selection).
- `dev` may be used for tuning.
- `train_pairs` is used only for training.

## Why

1. **Label noise.** GT is a GPS point-in-polygon lookup and neighbouring parcels are ~6 m apart, so the
   hit is often the neighbour of the parcel the address names. `gt_quality` cross-checks the input text
   against the GT parcel's own cadastral address:
   - `gold`: via (type, number, letters, suffix, BIS, quadrant), cross number AND plate all agree.
   - `noisy`: has GT but the text disagrees or a plate/cross cannot be determined.
   - `no_gt`: no ground-truth parcel.

   Only gold rows are trusted as supervision. Noisy rows stay in the test files (flagged) so results can
   be reported on all rows and on gold-only rows (`eval_strict.py --gold-only`).
2. **Leakage.** Splits are disjoint on `address_key` (parser canonical form incl. complement, then
   alphanumerics only), so spelling variants of one address cannot straddle splits. The script exits
   non-zero if any key is shared between `train_pairs`, `dev` and `frozen_test`.

## Notes

- `frozen_test` samples only rows that resolved to a parcel; keys mapping to conflicting GT are dropped.
- Two units of one building have different keys (complement is part of the key), so the same building
  can appear in train and test under different apartments. Use `gt_predial` to check when that matters.
- The 09-24 inspecciones/stickers exports are merged with the 09-23 ones and deduped by key.
- RUD contains personal data: only the address string leaves the workbook.
