"""Fast smoke test of the inference + evaluation path (300 rows) before running
the full notebook, so bugs surface in seconds instead of minutes."""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
ART = os.path.join(ROOT, "artifacts")

from cali_address.evaluate import (  # noqa: E402
    accuracy_by_idesc_estado, attach_ground_truth, calibration_table, failure_family,
    idesc_comparison, score_predictions, structural_agreement_table, summarize,
    via_type_confusion,
)
from cali_address.geo import ParcelIndex  # noqa: E402
from cali_address.inference import CONFIDENCE_THRESHOLD, AddressNormalizer  # noqa: E402
from cali_address.parser import canonical, parse_address  # noqa: E402

pd.set_option("display.width", 220)
catastro = pd.read_parquet(os.path.join(ART, "catastro.parquet"))
docs = pd.read_parquet(os.path.join(ART, "catastro_docs.parquet"))
samples = pd.read_parquet(os.path.join(ART, "eval_samples.parquet"))
idesc_df = pd.read_parquet(os.path.join(ART, "idesc_results.parquet"))

samples = samples.groupby("dataset", group_keys=False).head(60).reset_index(drop=True)
print("smoke rows:", len(samples))

t0 = time.time()
parcels = ParcelIndex(catastro["geom_wkb"].to_numpy())
print(f"parcel index {len(parcels):,} in {time.time()-t0:.1f}s")
doc_id_by_address = {a: i for i, a in enumerate(docs["direccion"].astype(str))}
evaluated = attach_ground_truth(samples, catastro, parcels, doc_id_by_address)
print(evaluated.groupby("dataset")["gt_doc_id"].apply(lambda s: s.notna().sum()).to_dict())

norm = AddressNormalizer(ART, docs=docs)
raws = evaluated["raw_address"].astype(str).tolist()
t0 = time.time()
pred_full = norm.normalize_batch(raws, k=20, use_model=True, rerank=True)
print(f"full {time.time()-t0:.1f}s")
t0 = time.time()
pred_nore = norm.normalize_batch(raws, k=20, use_model=True, rerank=False)
print(f"no-rerank {time.time()-t0:.1f}s")
t0 = time.time()
pred_rule = norm.normalize_batch(raws, k=20, use_model=False, rerank=True)
print(f"rule-only {time.time()-t0:.1f}s")

scored_full = score_predictions(evaluated, pred_full, docs)
scored_nore = score_predictions(evaluated, pred_nore, docs)
scored_rule = score_predictions(evaluated, pred_rule, docs)
print(summarize(scored_full, "full").to_string())
print(summarize(scored_rule, "rule").to_string())

IDESC_COLS = ["idesc_estado", "idesc_estado_class", "idesc_dir_ajusta", "idesc_comuna",
              "idesc_barrio_codigo", "idesc_lat", "idesc_lon"]
lookup = idesc_df.set_index("raw_address")
scored_full = scored_full.join(lookup[IDESC_COLS], on="raw_address")
scored_full["rule_canonical_glued"] = [
    canonical(parse_address(r), style="glued", with_complement=True)
    for r in scored_full["raw_address"]
]
print(idesc_comparison(scored_full).to_string())
print(accuracy_by_idesc_estado(scored_full).to_string())
print(structural_agreement_table(scored_full).to_string())
print(calibration_table(scored_full).to_string())
print(via_type_confusion(scored_full).to_string())

err = scored_full[scored_full["has_gt"] & ~scored_full["correct_top1"]].copy()
err["failure_family"] = err.apply(failure_family, axis=1)
print(err["failure_family"].value_counts().to_string())
cols = ["raw_address", "rule_canonical", "matched_cadastral_address", "gt_cadastral_address",
        "confidence", "dist_m", "comuna", "gt_comuna", "failure_family"]
print(err[cols].head(6).to_string())

unparsed = scored_full[~scored_full["parse_ok"]]
t = pd.DataFrame({"n_rows": scored_full.groupby("dataset").size(),
                  "n_unparsed": unparsed.groupby("dataset").size()}).fillna(0).astype(int)
t["share_unparsed"] = t["n_unparsed"] / t["n_rows"]
print(t.to_string())

overall = []
for label, sc in [("full", scored_full), ("nore", scored_nore), ("rule", scored_rule)]:
    gt = sc[sc["has_gt"]]
    dist = pd.to_numeric(gt["dist_m"], errors="coerce").dropna()
    overall.append({"variant": label, "predio_top1": gt["correct_top1"].mean(),
                    "predio_top5": gt["correct_top5"].mean(),
                    "manzana_top1": gt["correct_manzana"].mean(),
                    "within_100m": (dist <= 100).mean(), "median_dist_m": dist.median(),
                    "highconf_rate": sc["matched"].mean()})
print(pd.DataFrame(overall).set_index("variant").to_string())
print("SMOKE OK")
