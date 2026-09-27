"""Real-pair loading, cleaning and group-split (``cali_address.realpairs``)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tiny import tiny_docs, write_pairs  # noqa: F401

from cali_address.realpairs import clean_real_pairs, load_real_pairs, sample_real_batch, split_by_group


def _df(rows):
    return pd.DataFrame(rows, columns=["raw_address", "gt_doc_id", "address_key", "gt_comuna"])


def test_clean_drops_blank_missing_and_duplicates():
    df = _df([
        ("CL 5 # 10 - 20", 1, "a", "01"),
        ("CL 5 # 10 - 20", 1, "a", "01"),      # exact duplicate
        ("cl 5 # 10 - 20", 1, "a", "01"),      # different text, same doc: kept
        ("", 2, "b", "01"),                    # blank
        (None, 3, "c", "01"),                  # None
        ("   ", 3, "c", "01"),                 # whitespace
        ("KR 1 # 2 - 3", 999, "d", "01"),      # doc not in the index
        ("KR 1 # 2 - 3", -1, "d", "01"),       # negative doc
        ("KR 1 # 2 - 3", None, "d", "01"),     # null doc
    ])
    out, stats = clean_real_pairs(df, n_docs=10)
    assert len(out) == 2 and stats["kept"] == 2
    assert stats["dropped_duplicate"] == 1
    assert stats["dropped_blank"] == 3
    assert stats["dropped_missing_doc"] == 3
    assert out["gt_doc_id"].dtype == np.int64


def test_clean_empty_frame():
    out, stats = clean_real_pairs(_df([]), n_docs=5)
    assert len(out) == 0 and stats["kept"] == 0


def test_split_is_grouped_and_never_mixes_a_key():
    groups = np.array([f"k{i % 20}" for i in range(200)])
    val = split_by_group(groups, 0.25, seed=1)
    assert val.any() and (~val).any()
    assert not set(groups[val]) & set(groups[~val])
    assert len(set(groups[val])) == 5


def test_split_boundaries():
    groups = np.array(["a", "a", "b", "c"])
    assert not split_by_group(groups, 0.0, 1).any()          # val_frac 0 -> empty fold
    v = split_by_group(groups, 1.0, 1)                       # val_frac 1 still leaves training data
    assert v.any() and (~v).any()
    assert not split_by_group(np.array(["a", "a"]), 0.5, 1).any()   # single group
    assert split_by_group(np.array([]), 0.5, 1).shape == (0,)


def test_split_is_deterministic():
    g = np.array([f"k{i}" for i in range(50)])
    assert np.array_equal(split_by_group(g, 0.2, 5), split_by_group(g, 0.2, 5))


def test_load_real_pairs_maps_comuna_and_splits(tmp_path):
    docs = tiny_docs(48)
    path = write_pairs(str(tmp_path / "p.parquet"), docs, n=20)
    rp = load_real_pairs(48, comuna_codes=["01", "02", "03"], path=path, val_frac=0.25, seed=1)
    assert rp.n_train + rp.n_val == 20 and rp.n_val > 0
    assert set(rp.train_comuna.tolist()) <= {0, 1, 2}
    assert rp.train_tokens.shape == (rp.n_train, 40) and rp.val_tokens.shape[0] == rp.n_val


def test_load_real_pairs_unknown_comuna_is_minus_one(tmp_path):
    docs = tiny_docs(48)
    path = write_pairs(str(tmp_path / "p.parquet"), docs, n=6)
    rp = load_real_pairs(48, comuna_codes=["99"], path=path, val_frac=0.0)
    assert (rp.train_comuna == -1).all() and rp.n_val == 0


def test_missing_and_empty_pair_files_give_an_empty_set(tmp_path):
    rp = load_real_pairs(10, path=str(tmp_path / "nope.parquet"))
    assert rp.n_train == 0 and rp.n_val == 0 and rp.stats["kept"] == 0
    empty = tmp_path / "empty.parquet"
    _df([]).to_parquet(empty)
    rp = load_real_pairs(10, path=str(empty))
    assert rp.n_train == 0 and rp.train_tokens.shape == (0, 40)


def test_sample_real_batch_cycles_and_handles_empty():
    rng = np.random.default_rng(0)
    stream = np.arange(5)
    idx, cur, stream = sample_real_batch(stream, 0, 12, rng)
    assert len(idx) == 12 and set(idx.tolist()) <= set(range(5))
    assert sample_real_batch(np.empty(0, np.int64), 0, 4, rng)[0].size == 0
    assert sample_real_batch(stream, 0, 0, rng)[0].size == 0
