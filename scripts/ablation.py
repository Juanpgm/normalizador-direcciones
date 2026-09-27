"""Ablation of the reranking terms on the full evaluation sample.

Variants:
  rule        - rule canonicalization + rapidfuzz lookup, no neural model
  neural      - neural retrieval only, top-1 by cosine, no rerank
  rerank      - neural + fused rerank WITHOUT the coordinate-head term
  full        - neural + fused rerank WITH the coordinate-head geographic term
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
ART = os.path.join(ROOT, "artifacts")

from cali_address.evaluate import attach_ground_truth, score_predictions, summarize  # noqa: E402
from cali_address.geo import ParcelIndex  # noqa: E402
from cali_address.inference import AddressNormalizer  # noqa: E402

pd.set_option("display.width", 240)
catastro = pd.read_parquet(os.path.join(ART, "catastro.parquet"))
docs = pd.read_parquet(os.path.join(ART, "catastro_docs.parquet"))
samples = pd.read_parquet(os.path.join(ART, "eval_samples.parquet"))

parcels = ParcelIndex(catastro["geom_wkb"].to_numpy())
doc_id_by_address = {a: i for i, a in enumerate(docs["direccion"].astype(str))}
evaluated = attach_ground_truth(samples, catastro, parcels, doc_id_by_address)
norm = AddressNormalizer(ART, docs=docs)
raws = evaluated["raw_address"].astype(str).tolist()

VARIANTS = {
    "rule": dict(use_model=False, rerank=True, use_geo=False),
    "neural": dict(use_model=True, rerank=False, use_geo=False),
    "rerank": dict(use_model=True, rerank=True, use_geo=False),
    "full": dict(use_model=True, rerank=True, use_geo=True),
}
rows = []
for name, kw in VARIANTS.items():
    t0 = time.time()
    preds = norm.normalize_batch(raws, k=20, **kw)
    dt = time.time() - t0
    sc = score_predictions(evaluated, preds, docs)
    gt = sc[sc["has_gt"]]
    dist = pd.to_numeric(gt["dist_m"], errors="coerce").dropna()
    rows.append(
        {
            "variant": name, "seconds": round(dt, 1),
            "predio_top1": gt["correct_top1"].mean(),
            "predio_top5": gt["correct_top5"].mean(),
            "predio_top20": gt["correct_topk"].mean(),
            "manzana_top1": gt["correct_manzana"].mean(),
            "median_dist_m": dist.median(),
            "p90_dist_m": dist.quantile(0.9),
            "within_50m": (dist <= 50).mean(),
            "within_100m": (dist <= 100).mean(),
            "within_250m": (dist <= 250).mean(),
            "highconf_rate": sc["matched"].mean(),
            "highconf_precision": gt[gt["matched"]]["correct_top1"].mean(),
        }
    )
    print(rows[-1], flush=True)
print()
print(pd.DataFrame(rows).set_index("variant").to_string())
