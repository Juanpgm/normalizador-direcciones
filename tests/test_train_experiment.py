"""``train_model`` with real pairs, plate negatives, finetune mode and device fallback (CPU, tiny)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tiny import tiny_config, tiny_data_dir, tiny_docs, write_pairs

from cali_address.dataset import build_plate_negatives
from cali_address.realpairs import RealPairs, load_real_pairs
from cali_address.train import (
    AUX_HEADS, TrainConfig, load_checkpoint, resolve_device, save_checkpoint, train_model,
)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("world")
    docs, data, neg = tiny_data_dir(str(d))
    pairs = write_pairs(str(d / "pairs.parquet"), docs, n=24)
    real = load_real_pairs(len(docs), comuna_codes=data["comuna_codes"], path=pairs, val_frac=0.25, seed=1)
    plate = build_plate_negatives(docs, count=4, max_delta=30)
    return {"docs": docs, "data": data, "neg": neg, "real": real, "plate": plate, "dir": d}


def _empty_real(n_val=0):
    z = np.zeros((0, 40), dtype=np.uint8)
    return RealPairs(z, z, np.empty(0, np.int64), np.empty(0, np.int64), z, np.empty(0, np.int64))


def test_defaults_reproduce_the_original_config():
    cfg = TrainConfig()
    assert (cfg.lr, cfg.batch_size, cfg.epochs, cfg.label_smoothing) == (3e-3, 2048, 8, 0.02)
    assert (cfg.pct_start, cfg.real_frac, cfg.plate_neg_weight, cfg.mode) == (0.15, 0.0, 0.0, "mixed")
    assert cfg.w_coord == 0.5 and cfg.w_comuna == 0.2 and cfg.w_barrio == 0.2


@pytest.mark.parametrize("bad", [
    {"mode": "nope"}, {"real_frac": 1.5}, {"real_frac": -0.1}, {"plate_neg_weight": 2.0},
    {"real_oversample": 0}, {"batch_size": 0}, {"epochs": 0},
])
def test_invalid_config_is_rejected(bad, world):
    with pytest.raises(ValueError):
        train_model(world["data"], world["neg"], tiny_config(**bad), device="cpu")


def test_baseline_run_reports_synthetic_recall_and_no_real_columns(world):
    out = train_model(world["data"], world["neg"], tiny_config(), device="cpu")
    row = out["history"][-1]
    assert 0.0 <= row["recall@1"] <= row["recall@20"] <= 1.0
    assert row["real_val_recall@1"] is None
    assert out["batch_split"] == {"synthetic": 16, "real": 0}


def test_mixed_run_reports_real_val_next_to_synthetic(world):
    cfg = tiny_config(real_frac=0.5, real_oversample=3, epochs=2)
    out = train_model(world["data"], world["neg"], cfg, device="cpu", real=world["real"])
    assert out["batch_split"] == {"synthetic": 8, "real": 8}
    assert len(out["history"]) == 2
    for row in out["history"]:
        assert row["real_val_recall@20"] is not None and "recall@20" in row
    assert np.isfinite(out["history"][-1]["loss"])


def test_real_frac_zero_ignores_real_pairs_but_still_reports_val(world):
    out = train_model(world["data"], world["neg"], tiny_config(real_frac=0.0), device="cpu", real=world["real"])
    assert out["batch_split"]["real"] == 0
    assert out["history"][-1]["real_val_recall@1"] is not None


def test_real_frac_one_trains_on_real_only(world):
    out = train_model(world["data"], world["neg"], tiny_config(real_frac=1.0, real_oversample=2),
                      device="cpu", real=world["real"])
    assert out["batch_split"] == {"synthetic": 0, "real": 16}
    assert out["steps_per_epoch"] >= 1 and np.isfinite(out["history"][-1]["loss"])


def test_empty_real_set_falls_back_to_synthetic_only(world):
    out = train_model(world["data"], world["neg"], tiny_config(real_frac=0.5), device="cpu", real=_empty_real())
    assert out["batch_split"]["real"] == 0
    assert out["history"][-1]["real_val_recall@1"] is None


def test_plate_negatives_are_used_and_zero_neighbour_docs_are_safe(world):
    plate = world["plate"].copy()
    plate[:10] = -1                                   # docs with zero plate neighbours
    out = train_model(world["data"], world["neg"], tiny_config(plate_neg_weight=1.0), device="cpu",
                      plate_negatives=plate)
    assert np.isfinite(out["history"][-1]["loss"])
    allneg = np.full_like(plate, -1)                  # nothing available anywhere
    out = train_model(world["data"], world["neg"], tiny_config(plate_neg_weight=1.0), device="cpu",
                      plate_negatives=allneg)
    assert np.isfinite(out["history"][-1]["loss"])


def test_no_hard_negatives_at_all(world):
    out = train_model(world["data"], world["neg"], tiny_config(n_hard_neg=0, plate_neg_weight=1.0), device="cpu",
                      plate_negatives=world["plate"])
    assert np.isfinite(out["history"][-1]["loss"])


def test_device_falls_back_to_cpu_when_cuda_is_unavailable(monkeypatch, world):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cuda") == "cpu" and resolve_device(None) == "cpu" and resolve_device("cpu") == "cpu"
    out = train_model(world["data"], world["neg"], tiny_config(), device="cuda")
    assert out["device"] == "cpu"


def test_finetune_starts_from_checkpoint_and_can_freeze_aux(world, tmp_path):
    base = train_model(world["data"], world["neg"], tiny_config(), device="cpu")
    ckpt = str(tmp_path / "model.pt")
    save_checkpoint(ckpt, base["model"], base["config"], world["data"])
    before = {k: v.clone() for k, v in base["model"].state_dict().items()}

    cfg = tiny_config(mode="finetune", lr=1e-3, real_frac=0.5, freeze_aux=True, d_model=999)
    out = train_model(world["data"], world["neg"], cfg, device="cpu", real=world["real"], init_checkpoint=ckpt)
    assert out["config"]["d_model"] == 16                        # architecture comes from the checkpoint
    assert out["initial_real_val"] is not None
    after = out["model"].state_dict()
    for name in AUX_HEADS:
        for k in before:
            if k.startswith(name + "."):
                assert torch.equal(before[k], after[k]), k        # frozen heads did not move
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith("encoder."))


def test_finetune_requires_a_checkpoint_and_matching_heads(world, tmp_path):
    with pytest.raises(ValueError):
        train_model(world["data"], world["neg"], tiny_config(mode="finetune"), device="cpu")
    base = train_model(world["data"], world["neg"], tiny_config(), device="cpu")
    ckpt = str(tmp_path / "m.pt")
    save_checkpoint(ckpt, base["model"], base["config"], world["data"])
    bad = dict(world["data"])
    bad["doc_comuna"] = bad["doc_comuna"] + 5
    with pytest.raises(ValueError):
        train_model(bad, world["neg"], tiny_config(mode="finetune"), device="cpu", init_checkpoint=ckpt)


def test_checkpoint_roundtrip_and_temperature(world, tmp_path):
    out = train_model(world["data"], world["neg"], tiny_config(init_temperature=0.1, epochs=1), device="cpu")
    p = str(tmp_path / "m.pt")
    save_checkpoint(p, out["model"], out["config"], world["data"])
    model, ckpt = load_checkpoint(p)
    assert ckpt["config"]["init_temperature"] == 0.1
    assert torch.equal(model.state_dict()["proj.weight"], out["model"].state_dict()["proj.weight"])


def test_time_budget_stops_early(world):
    out = train_model(world["data"], world["neg"], tiny_config(epochs=3, max_minutes=0.0), device="cpu")
    assert len(out["history"]) == 1
