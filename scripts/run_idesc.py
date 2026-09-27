import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from cali_address.idesc import IdescClient  # noqa: E402

samples = pd.read_parquet(os.path.join(ROOT, "artifacts", "eval_samples.parquet"))
addresses = samples["raw_address"].astype(str).tolist()
uniq = sorted(set(addresses))
print(f"{len(addresses)} rows, {len(uniq)} unique addresses", flush=True)

client = IdescClient(os.path.join(ROOT, "artifacts", "idesc_cache.json"), max_workers=4, timeout=60)
print(f"cache already holds {len(client.cache)} entries", flush=True)
client.query_many(uniq)
client.save()

flat = pd.DataFrame([client.flatten(client.cache.get(a, {})) for a in uniq])
flat.insert(0, "raw_address", uniq)
flat.to_parquet(os.path.join(ROOT, "artifacts", "idesc_results.parquet"), index=False)
print("ok rate", flat["idesc_ok"].mean())
print(flat["idesc_estado_class"].value_counts(dropna=False).to_dict())
print("with coords", flat["idesc_lat"].notna().sum())
