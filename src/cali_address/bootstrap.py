"""First-run helpers behind ``cali-address setup``.

``catastro_emb.pt`` (161 MB) is the only serving artifact that is not in git. It is a pure function of
``model.pt`` and ``catastro_docs.parquet``: the encoder embedding of every cadastral address, tokenised
exactly like ``dataset.build_training_arrays`` builds ``d_tokens`` (``direccion.upper()``). This module
regenerates it with the same ``train.embed_documents`` that training and ``scripts/finish_training_artifacts.py``
use, and publishes it atomically.
"""

from __future__ import annotations

import os
import time
from typing import Callable

import pandas as pd
import torch

from .model import encode_batch
from .paths import model_path, shared_path
from .train import embed_documents, load_checkpoint, resolve_device

EMB_FILENAME = "catastro_emb.pt"

#: Invented addresses for the post-setup self-check (never real data).
SELF_CHECK_ADDRESSES = ("CALLE 5 # 10-20", "CARRERA 100 # 15-30", "AVENIDA 3 NORTE # 45-10")


def regenerate_embeddings(
    artifacts_dir: str,
    device: str | None = None,
    batch_size: int = 4096,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Embed every document of ``catastro_docs.parquet`` with ``model.pt`` and write ``catastro_emb.pt``.

    The file is written to a hidden temp file next to the target and moved into place with ``os.replace``,
    so an interrupted run (error, Ctrl+C, full disk) leaves no partial file and keeps any previous one.
    Raises ``ValueError`` when the embedding row count differs from the number of documents.
    """
    device = resolve_device(device)
    model, _ = load_checkpoint(model_path("model.pt", artifacts_dir), device, dropout=0.0)
    docs = pd.read_parquet(shared_path("catastro_docs.parquet", artifacts_dir), columns=["direccion"])
    tokens = encode_batch(docs["direccion"].fillna("").astype(str).str.upper().tolist())
    started = time.time()
    emb = embed_documents(model, tokens, device=device, batch=batch_size, progress=progress).cpu()
    if emb.shape[0] != len(docs):
        raise ValueError(f"embedding rows ({emb.shape[0]}) != catastro_docs.parquet rows ({len(docs)}); nothing written")

    target = os.path.join(artifacts_dir, EMB_FILENAME)
    partial = os.path.join(artifacts_dir, f".{EMB_FILENAME}.{os.getpid()}.partial")
    try:
        torch.save(emb, partial)
        os.replace(partial, target)
    except BaseException:
        try:
            os.remove(partial)
        except OSError:
            pass
        raise
    return {"path": target, "rows": int(emb.shape[0]), "dim": int(emb.shape[1]), "device": device,
            "seconds": time.time() - started}


def self_check(normalizer_factory: Callable, artifacts_dir: str, device: str | None) -> dict:
    """Load the real normalizer and run the invented addresses through it.

    Returns ``{"rows_per_sec", "estados"}``. Raises when the model cannot be loaded or every row errors.
    """
    from .service import normalize_strict

    normalizer = normalizer_factory(artifacts_dir, device)
    warm = normalize_strict(normalizer, list(SELF_CHECK_ADDRESSES), gazetteer=None)
    if len(warm) != len(SELF_CHECK_ADDRESSES) or (warm["estado"] == "ERROR").all():
        raise RuntimeError(f"the normalizer failed on the sample addresses: {warm['estado'].tolist()}")
    batch = list(SELF_CHECK_ADDRESSES) * 50
    started = time.time()
    normalize_strict(normalizer, batch, gazetteer=None)
    elapsed = max(time.time() - started, 1e-9)
    return {"rows_per_sec": len(batch) / elapsed, "estados": warm["estado"].tolist()}
