import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from cali_address.dataset import build_docs, build_spatial_negatives, build_training_arrays  # noqa: E402

ART = os.path.join(ROOT, "artifacts")
t0 = time.time()

catastro = pd.read_parquet(os.path.join(ART, "catastro.parquet"))
docs = build_docs(catastro)
docs.to_parquet(os.path.join(ART, "catastro_docs.parquet"), index=False)
print(f"docs: {len(docs)} unique cadastral addresses ({time.time()-t0:.0f}s)", flush=True)

neg = build_spatial_negatives(docs)
np.save(os.path.join(ART, "spatial_negatives.npy"), neg)
print(f"spatial negatives: {neg.shape} ({time.time()-t0:.0f}s)", flush=True)

arrays = build_training_arrays(docs)
np.savez_compressed(os.path.join(ART, "train_data.npz"), **arrays)
print(
    f"train_data.npz written: queries={arrays['q_tokens'].shape} "
    f"docs={arrays['d_tokens'].shape} val_queries={int(arrays['q_is_val'].sum())} "
    f"({time.time()-t0:.0f}s)",
    flush=True,
)
