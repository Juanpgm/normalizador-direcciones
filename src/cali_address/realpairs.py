"""Real ``raw_address -> cadastral document`` pairs for training.

The base model only ever sees synthetic corruptions of the cadastre. This module
loads the *gold* real pairs produced by ``scripts/build_splits.py``
(``artifacts/splits/train_pairs.parquet``), cleans them, tokenizes them with the
same encoder input the synthetic queries use, and splits a validation fold **by
``address_key``** so that no address (or spelling variant of it) is in both the
training and the validation side.

``frozen_test`` and ``dev`` are never read here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .dataset import query_text_for
from .model import encode_batch

__all__ = ["RealPairs", "load_real_pairs", "clean_real_pairs", "split_by_group", "sample_real_batch"]

DEFAULT_PAIRS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "artifacts", "splits", "train_pairs.parquet",
)


@dataclass
class RealPairs:
    """Tokenized real pairs, already split into a train and a validation side."""

    train_tokens: np.ndarray            # canonical-ish view (what synthetic queries use)
    train_tokens_raw: np.ndarray        # pre-normalized raw view
    train_doc: np.ndarray               # int64 doc ids
    train_comuna: np.ndarray            # int64 comuna class, -1 = unknown (fall back to the doc's)
    val_tokens: np.ndarray
    val_doc: np.ndarray
    stats: dict = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return int(len(self.train_doc))

    @property
    def n_val(self) -> int:
        return int(len(self.val_doc))


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or not str(v).strip()


def clean_real_pairs(df: pd.DataFrame, n_docs: int) -> tuple[pd.DataFrame, dict]:
    """Drop unusable rows; return the cleaned frame and a count of what was dropped.

    Dropped: blank/None/NaN ``raw_address``, missing or out-of-index
    ``gt_doc_id`` (a document that is not in the index cannot be a target), and
    exact duplicate ``(raw_address, gt_doc_id)`` pairs.
    """
    stats = {"input": int(len(df)), "dropped_blank": 0, "dropped_missing_doc": 0, "dropped_duplicate": 0}
    if df is None or len(df) == 0:
        stats["kept"] = 0
        return pd.DataFrame(columns=["raw_address", "gt_doc_id", "address_key", "gt_comuna"]), stats
    df = df.copy()
    blank = df["raw_address"].map(_blank).to_numpy()
    stats["dropped_blank"] = int(blank.sum())
    df = df[~blank]
    doc = pd.to_numeric(df["gt_doc_id"], errors="coerce")
    bad = (doc.isna() | (doc < 0) | (doc >= n_docs)).to_numpy()
    stats["dropped_missing_doc"] = int(bad.sum())
    df = df[~bad].copy()
    df["gt_doc_id"] = pd.to_numeric(df["gt_doc_id"]).astype(np.int64)
    df["raw_address"] = df["raw_address"].astype(str)
    before = len(df)
    df = df.drop_duplicates(subset=["raw_address", "gt_doc_id"]).reset_index(drop=True)
    stats["dropped_duplicate"] = int(before - len(df))
    if "address_key" not in df.columns:
        df["address_key"] = df["raw_address"]
    df["address_key"] = df["address_key"].where(~df["address_key"].map(_blank), df["raw_address"]).astype(str)
    if "gt_comuna" not in df.columns:
        df["gt_comuna"] = None
    stats["kept"] = int(len(df))
    return df, stats


def split_by_group(groups: np.ndarray, val_frac: float, seed: int) -> np.ndarray:
    """Boolean validation mask; every group lands wholly on one side.

    ``val_frac`` 0 (or a single group) gives an empty validation fold; the mask
    never covers every group, so the training side is never empty when there is
    more than one group.
    """
    groups = np.asarray(groups)
    is_val = np.zeros(len(groups), dtype=bool)
    if len(groups) == 0 or val_frac <= 0:
        return is_val
    uniq = np.unique(groups)
    if len(uniq) < 2:
        return is_val
    n_val = int(round(min(val_frac, 1.0) * len(uniq)))
    n_val = min(max(n_val, 1), len(uniq) - 1)
    rng = np.random.default_rng(seed)
    val_groups = set(rng.permutation(uniq)[:n_val].tolist())
    return np.fromiter((g in val_groups for g in groups.tolist()), dtype=bool, count=len(groups))


def load_real_pairs(
    n_docs: int,
    comuna_codes=None,
    path: str | None = None,
    val_frac: float = 0.10,
    seed: int = 42,
    max_len: int = 40,
) -> RealPairs:
    """Load, clean, tokenize and split the real pairs. A missing/empty file gives an empty set."""
    path = path or DEFAULT_PAIRS_PATH
    if os.path.exists(path):
        raw = pd.read_parquet(path)
    else:
        raw = pd.DataFrame(columns=["raw_address", "gt_doc_id", "address_key", "gt_comuna"])
    df, stats = clean_real_pairs(raw, n_docs)
    is_val = split_by_group(df["address_key"].to_numpy(), val_frac, seed) if len(df) else np.zeros(0, bool)
    tr, va = df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)

    def tokens(frame):
        pairs = [query_text_for(t) for t in frame["raw_address"].tolist()]
        a = encode_batch([p[0] for p in pairs], max_len)
        b = encode_batch([p[1] for p in pairs], max_len)
        return a, b

    tr_a, tr_b = tokens(tr)
    va_a, _ = tokens(va)
    cmap = {str(c): i for i, c in enumerate(comuna_codes)} if comuna_codes is not None else {}
    comuna = np.array(
        [cmap.get(str(c).strip(), -1) if not _blank(c) else -1 for c in tr["gt_comuna"].tolist()],
        dtype=np.int64,
    )
    stats.update({"n_train": int(len(tr)), "n_val": int(len(va)), "val_frac": float(val_frac)})
    return RealPairs(
        train_tokens=tr_a, train_tokens_raw=tr_b, train_doc=tr["gt_doc_id"].to_numpy(np.int64),
        train_comuna=comuna, val_tokens=va_a, val_doc=va["gt_doc_id"].to_numpy(np.int64), stats=stats,
    )


def sample_real_batch(stream: np.ndarray, cursor: int, n: int, rng: np.random.Generator):
    """Take ``n`` indices from a cyclic, reshuffled ``stream``; returns ``(indices, new_cursor, stream)``."""
    if n <= 0 or len(stream) == 0:
        return np.empty(0, dtype=np.int64), cursor, stream
    out = []
    need = n
    while need > 0:
        if cursor >= len(stream):
            stream = rng.permutation(stream)
            cursor = 0
        take = min(need, len(stream) - cursor)
        out.append(stream[cursor: cursor + take])
        cursor += take
        need -= take
    return np.concatenate(out).astype(np.int64), cursor, stream
