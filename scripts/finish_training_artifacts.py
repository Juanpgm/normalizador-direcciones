"""Recover `catastro_emb.pt` and `train_report.json` from an already-saved model.pt.

Used after the embedding step crashed (positional-embedding length mismatch) while
the trained weights had already been persisted; avoids retraining for 16 minutes.
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import torch

import argparse  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from cali_address.paths import resolve_artifacts_dir, shared_path  # noqa: E402
from cali_address.train import embed_documents, load_checkpoint, recall_at_k, resolve_device  # noqa: E402

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--artifacts-dir", default=None,
                 help="experiment directory holding model.pt (env CALI_ARTIFACTS_DIR; default artifacts/)")
ART = resolve_artifacts_dir(_ap.parse_args().artifacts_dir)
DEVICE = resolve_device("cuda")

model, ckpt = load_checkpoint(os.path.join(ART, "model.pt"), DEVICE, dropout=0.0)
cfg = ckpt["config"]
model.eval()

data = {k: v for k, v in np.load(shared_path("train_data.npz", ART), allow_pickle=True).items()}
emb = embed_documents(model, data["d_tokens"], device=DEVICE)
torch.save(emb.cpu(), os.path.join(ART, "catastro_emb.pt"))
print("catastro_emb.pt", tuple(emb.shape), emb.dtype)

val = np.flatnonzero(data["q_is_val"])
metrics = recall_at_k(model, data["q_tokens"][val], data["q_doc"][val], emb, device=DEVICE)
print("recomputed val recall:", metrics)

history = []
log_path = os.path.join(ART, "train.log")
report_path = os.path.join(ART, "train_report.json")
if os.path.exists(log_path):
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('{"epoch"'):
                history.append(json.loads(line))
elif os.path.exists(report_path):
    with open(report_path, encoding="utf-8") as fh:
        history = json.load(fh).get("history", [])
report = {
    "history": history,
    "seconds": history[-1]["seconds"] if history else 0.0,
    "gpu": ({
        "name": torch.cuda.get_device_name(0),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        "max_memory_allocated_gb": round(max((h["gpu_mem_gb"] for h in history), default=0.0), 3),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "cudnn": torch.backends.cudnn.is_available(),
    } if DEVICE == "cuda" else {}),
    "n_params": sum(p.numel() for p in model.parameters()),
    "config": cfg,
    "final_val_recall": metrics,
}
with open(report_path, "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, default=str)
print(f"train_report.json written with {len(history)} epochs, "
      f"{report['seconds']/60:.1f} min, {report['n_params']:,} params")
