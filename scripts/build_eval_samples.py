import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from cali_address.eval_data import DATASETS, load_dataset, sample_dataset  # noqa: E402

CONTEXT = os.path.join(ROOT, "context")
OUT = os.path.join(ROOT, "artifacts", "eval_samples.parquet")

frames = []
stats = []
for spec in DATASETS:
    df = load_dataset(spec, CONTEXT)
    total = len(df)
    s = sample_dataset(df, spec.sample_n)
    frames.append(s)
    stats.append(
        {
            "dataset": spec.key,
            "rows_total": total,
            "rows_with_address": int(
                (df["raw_address"].notna() & (df["raw_address"].astype(str).str.strip() != "")).sum()
            ),
            "sampled": len(s),
            "with_coords": int(s["gt_lat"].notna().sum()),
        }
    )
    print(stats[-1], flush=True)
all_df = pd.concat(frames, ignore_index=True)
all_df.to_parquet(OUT, index=False)
pd.DataFrame(stats).to_csv(os.path.join(ROOT, "artifacts", "eval_sample_stats.csv"), index=False)
print("total sampled", len(all_df), "->", OUT)
print(all_df.groupby("dataset")["raw_address"].apply(lambda s: s.head(3).tolist()))
