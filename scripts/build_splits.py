"""Build the frozen-test / dev / train-pairs / unlabeled splits from ``context/``.

Why: the GPS-derived ground truth is noisy (neighbouring parcels are ~6 m
apart), and ``artifacts/eval_scored_full.parquet`` was already used to tune
rules and reliability, so it is a DEV set. This script produces

* ``frozen_test.parquet``      Fasecolda-only rows with GT, never used for tuning
* ``dev.parquet``              the existing eval rows + ``gt_quality``
* ``train_pairs.parquet``      real (raw_address -> doc_id) pairs, GOLD rows only
* ``unlabeled_addresses.parquet``  address strings only (RUD, acciones)
* ``splits_report.json``       counts, overlap matrix, gold ratios

and exits non-zero if any normalized address key is shared between
train_pairs, dev and frozen_test. See ``docs/splits.md``.

Usage (CPU only, no model is loaded):
    PYTHONPATH=src python scripts/build_splits.py --seed 42 --frozen-size 2000
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address.labels import address_key, label_quality  # noqa: E402

DEFAULT_OUT_DIR = os.path.join(ROOT, "artifacts", "splits")
CONTEXT_DIR = os.path.join(ROOT, "context")
ARTIFACTS = os.path.join(ROOT, "artifacts")

TRAIN_COLUMNS = [
    "raw_address", "address_key", "dataset", "gt_doc_id", "gt_manzana", "gt_predial",
    "gt_comuna", "source_row_id",
]
UNLABELED_COLUMNS = ["raw_address", "dataset"]
SPLIT_NAMES = ("train_pairs", "dev", "frozen_test")

#: Later exports of the same source; they are near-supersets of the 09-23 files,
#: so duplicates are removed by address key.
EXTRA_FILES = {
    "inspecciones": "inspecciones_2026-09-24_18-51.xlsx",
    "stickers": "stickers_2026-09-24_18-56.xlsx",
}


class LeakageError(RuntimeError):
    """Raised when two splits share a normalized address key."""


# ---------------------------------------------------------------------------
# Pure functions (unit-tested)
# ---------------------------------------------------------------------------
def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``address_key`` and drop rows whose address is blank/NaN."""
    out = df.copy()
    out["address_key"] = [address_key(a) for a in out["raw_address"]]
    return out[out["address_key"] != ""].reset_index(drop=True)


def _drop_conflicting_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Drop keys that map to more than one distinct ``gt_doc_id``."""
    nun = df.groupby("address_key")["gt_doc_id"].transform("nunique")
    return df[nun <= 1]


def select_frozen(pool: pd.DataFrame, dev_keys: set, size: int, seed: int = 42) -> pd.DataFrame:
    """Sample the frozen test set: Fasecolda only, with GT, unique keys, not in dev.

    Keys with conflicting GT are dropped; noisy rows are kept (flagged by
    ``gt_quality``). ``size`` larger than the pool returns the whole pool.
    """
    df = pool[(pool["dataset"] == "fasecolda") & pool["has_gt"].astype(bool)]
    df = df[~df["address_key"].isin(dev_keys)]
    df = _drop_conflicting_keys(df)
    df = df.drop_duplicates("address_key", keep="first")
    n = max(0, min(int(size), len(df)))
    if n == 0:
        return df.iloc[0:0].reset_index(drop=True)
    return df.sample(n, random_state=seed).sort_index().reset_index(drop=True)


def build_train_pairs(labeled: pd.DataFrame, exclude_keys: set) -> pd.DataFrame:
    """Gold-quality (raw_address -> doc_id) pairs, deduped, minus excluded keys."""
    if len(labeled) == 0 or "gt_quality" not in labeled.columns:
        return pd.DataFrame({c: [] for c in TRAIN_COLUMNS})
    df = labeled[(labeled["gt_quality"] == "gold") & ~labeled["address_key"].isin(exclude_keys)]
    df = _drop_conflicting_keys(df).drop_duplicates("address_key", keep="first")
    if len(df) == 0:
        return pd.DataFrame({c: [] for c in TRAIN_COLUMNS})
    if "source_file" in df.columns:
        tag = df["source_file"].astype(str)
    else:
        tag = df["dataset"].astype(str)
    comuna = df["gt_comuna"] if "gt_comuna" in df.columns else df["gt_predial"].astype(str).str[9:11]
    out = pd.DataFrame({
        "raw_address": df["raw_address"].astype(str).to_numpy(),
        "address_key": df["address_key"].to_numpy(),
        "dataset": df["dataset"].to_numpy(),
        "gt_doc_id": df["gt_doc_id"].astype("int64").to_numpy(),
        "gt_manzana": df["gt_manzana"].astype(str).to_numpy(),
        "gt_predial": df["gt_predial"].astype(str).to_numpy(),
        "gt_comuna": np.asarray(comuna, dtype=object),
        "source_row_id": (tag + ":" + df["source_row"].astype(str)).to_numpy(),
    })
    return out[TRAIN_COLUMNS].reset_index(drop=True)


def build_unlabeled(df: pd.DataFrame, exclude_keys: set) -> pd.DataFrame:
    """Address strings ONLY (``raw_address``, ``dataset``); no other column survives."""
    if len(df) == 0:
        return pd.DataFrame({c: [] for c in UNLABELED_COLUMNS})
    keep = df[~df["address_key"].isin(exclude_keys)].drop_duplicates("address_key")
    return keep[UNLABELED_COLUMNS].reset_index(drop=True)


def overlap_matrix(keys: dict[str, set]) -> dict[str, dict[str, int]]:
    """Pairwise count of shared address keys (diagonal = split size)."""
    return {a: {b: len(keys[a] & keys[b]) for b in keys} for a in keys}


def assert_no_leakage(keys: dict[str, set]) -> None:
    m = overlap_matrix(keys)
    bad = {(a, b): m[a][b] for a in m for b in m if a < b and m[a][b] > 0}
    if bad:
        raise LeakageError(f"address keys shared between splits: {bad}")


def _counts(df: pd.DataFrame) -> dict:
    out = {"total": int(len(df))}
    if len(df) and "dataset" in df.columns:
        out["by_dataset"] = {k: int(v) for k, v in df["dataset"].value_counts().items()}
    else:
        out["by_dataset"] = {}
    if "gt_quality" in df.columns:
        out["by_gt_quality"] = {k: int(v) for k, v in df["gt_quality"].value_counts().items()}
    return out


def build_report(frozen, dev, train, unlabeled, universe: dict, seed: int) -> dict:
    keys = {
        "train_pairs": set(train["address_key"]),
        "dev": set(dev["address_key"]) if "address_key" in dev.columns else set(),
        "frozen_test": set(frozen["address_key"]),
    }
    gold_ratio = {}
    for name, df in universe.items():
        q = df["gt_quality"] if "gt_quality" in df.columns else pd.Series([], dtype=object)
        with_gt = int((q != "no_gt").sum())
        gold_ratio[name] = round(float((q == "gold").sum()) / with_gt, 4) if with_gt else None
    return {
        "seed": seed,
        "counts": {
            "frozen_test": _counts(frozen),
            "dev": _counts(dev),
            "train_pairs": _counts(train),
            "unlabeled_addresses": _counts(unlabeled),
        },
        "overlap": overlap_matrix(keys),
        "gold_ratio_by_dataset": gold_ratio,
        "universe_gt_quality_by_dataset": {n: _counts(d)["by_gt_quality"] if "gt_quality" in d.columns else {}
                                           for n, d in universe.items()},
    }


# ---------------------------------------------------------------------------
# I/O orchestration
# ---------------------------------------------------------------------------
def _load_all(catastro, docs, parcels):
    """Load every workbook and attach GT + gt_quality where coordinates exist."""
    from cali_address import eval_data
    from cali_address.evaluate import attach_ground_truth

    doc_id_by_address = {a: i for i, a in enumerate(docs["direccion"].astype(str))}
    labeled: dict[str, pd.DataFrame] = {}
    unlabeled: list[pd.DataFrame] = []
    for spec in eval_data.DATASETS:
        variants = [(spec, os.path.splitext(spec.filename)[0])]
        if spec.key in EXTRA_FILES:
            s2 = copy.copy(spec)
            s2.filename = EXTRA_FILES[spec.key]
            variants.append((s2, os.path.splitext(s2.filename)[0]))
        frames = []
        for sp, tag in variants:
            t0 = time.time()
            df = eval_data.load_dataset(sp, CONTEXT_DIR)
            print(f"loaded {tag}: {len(df):,} rows in {time.time() - t0:.1f}s", file=sys.stderr)
            df["source_file"] = tag
            frames.append(df)
        df = add_keys(pd.concat(frames, ignore_index=True))
        if spec.key in ("rud", "acciones"):
            # Address strings only: nothing else from these workbooks is carried forward.
            unlabeled.append(df[["raw_address", "dataset", "address_key"]])
            continue
        df = attach_ground_truth(df, catastro, parcels, doc_id_by_address)
        df["has_gt"] = pd.to_numeric(df["gt_doc_id"], errors="coerce").notna()
        labeled[spec.key] = label_quality(df, docs)
    return labeled, pd.concat(unlabeled, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frozen-size", type=int, default=2000)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--dev", default=os.path.join(ARTIFACTS, "eval_scored_full.parquet"))
    args = ap.parse_args(argv)

    from cali_address.geo import ParcelIndex

    catastro = pd.read_parquet(os.path.join(ARTIFACTS, "catastro.parquet"))
    docs = pd.read_parquet(os.path.join(ARTIFACTS, "catastro_docs.parquet"))
    parcels = ParcelIndex(catastro["geom_wkb"].to_numpy())

    dev = pd.read_parquet(args.dev)  # read-only: the original file is never modified
    dev = label_quality(dev, docs)
    dev["address_key"] = [address_key(a) for a in dev["raw_address"]]
    dev_keys = set(dev["address_key"]) - {""}

    labeled, unlabeled_all = _load_all(catastro, docs, parcels)

    frozen = select_frozen(labeled["fasecolda"], dev_keys, args.frozen_size, args.seed)
    frozen["idesc_dir_ajusta"] = None
    frozen_keys = set(frozen["address_key"])

    exclude = dev_keys | frozen_keys
    train = build_train_pairs(
        pd.concat([labeled[k] for k in ("fasecolda", "inspecciones", "stickers")], ignore_index=True),
        exclude,
    )
    unlabeled = build_unlabeled(unlabeled_all, exclude)

    keys = {"train_pairs": set(train["address_key"]), "dev": dev_keys, "frozen_test": frozen_keys}
    try:
        assert_no_leakage(keys)
    except LeakageError as exc:
        print(f"LEAKAGE: {exc}", file=sys.stderr)
        return 2
    assert list(unlabeled.columns) == UNLABELED_COLUMNS

    os.makedirs(args.out_dir, exist_ok=True)
    frozen.to_parquet(os.path.join(args.out_dir, "frozen_test.parquet"), index=False)
    dev.to_parquet(os.path.join(args.out_dir, "dev.parquet"), index=False)
    train.to_parquet(os.path.join(args.out_dir, "train_pairs.parquet"), index=False)
    unlabeled.to_parquet(os.path.join(args.out_dir, "unlabeled_addresses.parquet"), index=False)
    report = build_report(frozen, dev, train, unlabeled, labeled, args.seed)
    with open(os.path.join(args.out_dir, "splits_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
