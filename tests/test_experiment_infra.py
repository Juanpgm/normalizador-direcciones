"""Paths, overrides, guards, leaderboard, scripts/train.py and scripts/run_experiment.py (CPU, tiny)."""

from __future__ import annotations

import csv
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from tiny import tiny_config, tiny_data_dir, write_pairs

import cali_address.paths as paths
from cali_address import experiment as ex
from cali_address.inference import AddressNormalizer
from cali_address.train import save_checkpoint, train_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- paths
def test_resolve_artifacts_dir_precedence(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.ENV_ARTIFACTS_DIR, raising=False)
    assert paths.resolve_artifacts_dir() == os.path.abspath(paths.DEFAULT_ARTIFACTS_DIR)
    monkeypatch.setenv(paths.ENV_ARTIFACTS_DIR, str(tmp_path / "env"))
    assert paths.resolve_artifacts_dir() == str(tmp_path / "env")
    assert paths.resolve_artifacts_dir(str(tmp_path / "cli")) == str(tmp_path / "cli")   # CLI wins


def test_shared_path_prefers_experiment_dir_then_falls_back(monkeypatch, tmp_path):
    default, exp = tmp_path / "default", tmp_path / "exp"
    default.mkdir(); exp.mkdir()
    monkeypatch.setattr(paths, "DEFAULT_ARTIFACTS_DIR", str(default))
    (default / "a.bin").write_bytes(b"1")
    assert paths.shared_path("a.bin", str(exp)) == str(default / "a.bin")     # absent in exp -> default
    (exp / "a.bin").write_bytes(b"2")
    assert paths.shared_path("a.bin", str(exp)) == str(exp / "a.bin")         # present -> exp
    assert paths.shared_path("missing.bin", str(exp)) == str(default / "missing.bin")


def test_sha256_and_fingerprint(tmp_path):
    f = tmp_path / "x"
    f.write_bytes(b"abc")
    assert paths.sha256_file(str(f)) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert paths.sha256_file(str(tmp_path / "nope")) is None
    assert set(paths.sha256_serving_files(str(tmp_path))) == set(paths.SERVING_FILES)
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("x = 1\n")
    (src / "b.txt").write_text("ignored")
    fp = paths.code_fingerprint(str(src))
    assert fp == paths.code_fingerprint(str(src))
    (src / "a.py").write_text("x = 2\n")
    assert fp != paths.code_fingerprint(str(src))
    (src / "b.txt").write_text("still ignored")
    assert paths.code_fingerprint(str(src)) != fp


def test_normalizer_loads_from_experiment_dir_with_docs_fallback(monkeypatch, tmp_path):
    default, exp = tmp_path / "default", tmp_path / "exp"
    docs, data, neg = tiny_data_dir(str(default))
    out = train_model(data, neg, tiny_config(), device="cpu")
    exp.mkdir()
    save_checkpoint(str(exp / "model.pt"), out["model"], out["config"], data)
    from cali_address.train import embed_documents
    torch.save(embed_documents(out["model"], data["d_tokens"][:, :40], device="cpu").cpu(), str(exp / "catastro_emb.pt"))
    monkeypatch.setattr(paths, "DEFAULT_ARTIFACTS_DIR", str(default))
    norm = AddressNormalizer(str(exp), device="cpu")                # catastro_docs.parquet only in default
    assert len(norm.docs) == len(docs) and norm.artifacts_dir == str(exp)
    assert norm.tuning_source.startswith("module defaults")        # tuning is per experiment, no fallback


# ---------------------------------------------------------------- overrides / tags / dirs
def test_parse_overrides_types_and_errors():
    got = ex.parse_overrides(["lr=1e-3", "epochs=2", "freeze-aux=true", "mode=finetune", "real_frac=0"])
    assert got == {"lr": 1e-3, "epochs": 2, "freeze_aux": True, "mode": "finetune", "real_frac": 0.0}
    assert ex.parse_overrides([]) == {} and ex.parse_overrides(None) == {}
    for bad in (["nope=1"], ["lr"], ["epochs=abc"], ["freeze_aux=maybe"]):
        with pytest.raises(ValueError):
            ex.parse_overrides(bad)


def test_every_documented_hyperparameter_is_exposed():
    d = ex.all_defaults()
    for key in ("lr", "weight_decay", "epochs", "batch_size", "init_temperature", "label_smoothing", "d_model",
                "n_layers", "n_heads", "d_out", "dropout", "w_coord", "w_comuna", "w_barrio", "real_frac",
                "real_oversample", "plate_neg_weight", "plate_neg_count", "seed", "pct_start"):
        assert key in d, key


@pytest.mark.parametrize("bad", ["", "a b", "../x", "x/y", "-x", "a" * 65, None])
def test_safe_tag_rejects_bad_tags(bad):
    with pytest.raises(ValueError):
        ex.safe_tag(bad)
    assert ex.safe_tag("mixed_r30-v2.1") == "mixed_r30-v2.1"


def test_prepare_out_dir_new_existing_nonempty_and_production(tmp_path):
    new = tmp_path / "a" / "b"
    assert ex.prepare_out_dir(str(new)) == str(new) and new.is_dir()      # does not exist -> created
    assert ex.prepare_out_dir(str(new)) == str(new)                       # exists but empty -> ok
    (new / "model.pt").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        ex.prepare_out_dir(str(new))                                      # would clobber a finished run
    assert ex.prepare_out_dir(str(new), overwrite=True) == str(new)
    with pytest.raises(ValueError):
        ex.prepare_out_dir(paths.DEFAULT_ARTIFACTS_DIR, overwrite=True)   # never the production dir


# ---------------------------------------------------------------- frozen guard
def test_guard_frozen():
    with pytest.raises(PermissionError):
        ex.guard_frozen("artifacts/splits/frozen_test.parquet", final=False)
    ex.guard_frozen("artifacts/splits/frozen_test.parquet", final=True)
    ex.guard_frozen("artifacts/splits/dev.parquet", final=False)
    assert ex.is_frozen_path("X/FROZEN_TEST.parquet") and not ex.is_frozen_path("dev.parquet")


def test_runner_plans_and_refuses_frozen_without_final():
    run = _load_script("run_experiment")
    assert [t for t, _, _ in run.plan_evals(False)] == ["dev_all", "dev_gold"]
    with pytest.raises(PermissionError):
        run.plan_evals(False, "frozen_test")
    with pytest.raises(PermissionError):
        run.plan_evals(False, "both")
    tags = [t for t, _, _ in run.plan_evals(True)]
    assert tags == ["dev_all", "dev_gold", "final_all", "final_gold"]


def test_runner_refuses_before_doing_any_work(tmp_path):
    run = _load_script("run_experiment")
    with pytest.raises(PermissionError):
        run.main(["--tag", "t1", "--eval-set", "frozen_test", "--root", str(tmp_path), "--no-hash"])
    assert not (tmp_path / "t1").exists()                                  # nothing created
    with pytest.raises(ValueError):
        run.main(["--tag", "t2", "--set", "bogus=1", "--root", str(tmp_path)])
    with pytest.raises(ValueError):
        run.main(["--tag", "../evil", "--root", str(tmp_path)])


def test_eval_strict_refuses_frozen_without_final(tmp_path, capsys):
    ev = _load_script("eval_strict")
    rc = ev.main(["--scored", "artifacts/splits/frozen_test.parquet", "--out-dir", str(tmp_path)])
    assert rc == 2 and "--final" in capsys.readouterr().err


# ---------------------------------------------------------------- leaderboard
def _summary(n_ok, n, man, pred, lost):
    return [
        {"dataset": "ALL", "estado": "OK", "n": str(n_ok), "manzana_precision": str(man),
         "predial_precision": str(pred), "lost_correct": "0"},
        {"dataset": "ALL", "estado": "SIN_MATCH", "n": str(n - n_ok), "manzana_precision": "",
         "predial_precision": "", "lost_correct": str(lost)},
        {"dataset": "fasecolda", "estado": "OK", "n": "999", "manzana_precision": "0.1",
         "predial_precision": "0.1", "lost_correct": "77"},          # per-dataset rows must not be double counted
    ]


def test_eval_metrics_and_row_math():
    m = ex.eval_metrics(_summary(30, 100, 0.9, 0.8, 5), "dev")
    assert m == {"dev_n": 100, "dev_ok": 30, "dev_manzana_precision": 0.9,
                 "dev_predial_precision": 0.8, "dev_lost_correct": 5}
    row = ex.leaderboard_row("t", "mixed", [{"recall@1": 0.5, "recall@5": 0.6, "recall@20": 0.7,
                                             "real_val_recall@1": 0.1}],
                             _summary(30, 100, 0.9, 0.8, 5), _summary(20, 25, 0.99, 0.95, 1), 12.34)
    assert row["dev_gold_ok_coverage"] == 0.8 and row["dev_gold_n"] == 25
    assert row["syn_recall@20"] == 0.7 and row["real_val_recall@1"] == 0.1 and row["real_val_recall@5"] is None


def test_eval_metrics_edge_cases():
    assert ex.eval_metrics([], "dev")["dev_ok"] == 0                       # no rows at all
    only_sin = [{"dataset": "ALL", "estado": "SIN_MATCH", "n": "10", "lost_correct": "2"}]
    m = ex.eval_metrics(only_sin, "dev")                                    # no OK row: precision unknown
    assert m["dev_ok"] == 0 and m["dev_manzana_precision"] is None and m["dev_n"] == 10
    nan = [{"dataset": "ALL", "estado": "OK", "n": "3", "manzana_precision": "nan", "predial_precision": "",
            "lost_correct": "0"}]
    assert ex.eval_metrics(nan, "dev")["dev_manzana_precision"] is None
    row = ex.leaderboard_row("t", "eval-only", [], [], [], 1.0)             # zero gold rows: no division
    assert row["dev_gold_ok_coverage"] is None and row["syn_recall@1"] is None


def test_leaderboard_append_creates_header_once_and_ignores_extras(tmp_path):
    lb = str(tmp_path / "sub" / "leaderboard.csv")                          # parent does not exist
    row = ex.leaderboard_row("a", "mixed", [], _summary(1, 2, 1.0, 1.0, 0), _summary(1, 1, 1.0, 1.0, 0), 1.0)
    ex.append_leaderboard(lb, {**row, "junk": 1})
    ex.append_leaderboard(lb, {**row, "tag": "b"})
    rows = list(csv.DictReader(open(lb, encoding="utf-8")))
    assert [r["tag"] for r in rows] == ["a", "b"] and list(rows[0]) == ex.LEADERBOARD_COLUMNS
    assert rows[0]["real_val_recall@1"] == ""                               # None -> empty cell
    open(lb, "w").close()                                                   # truncated to empty -> header again
    ex.append_leaderboard(lb, row)
    assert open(lb, encoding="utf-8").read().startswith("tag,")


def test_write_experiment_json_merges(tmp_path):
    p = ex.write_experiment_json(str(tmp_path / "e"), {"a": 1})
    ex.write_experiment_json(str(tmp_path / "e"), {"b": 2})
    assert json.load(open(p)) == {"a": 1, "b": 2}
    open(p, "w").write("{corrupt")                                           # malformed file is replaced, not fatal
    ex.write_experiment_json(str(tmp_path / "e"), {"c": 3})
    assert json.load(open(p)) == {"c": 3}


# ---------------------------------------------------------------- scripts/train.py end to end
def test_train_script_writes_only_into_out_dir_and_never_touches_production(tmp_path):
    train = _load_script("train")
    out = tmp_path / "exp"
    docs, data, neg = tiny_data_dir(str(out))
    pairs = write_pairs(str(tmp_path / "pairs.parquet"), docs, n=20)
    before = paths.sha256_serving_files()
    prod_files = set(os.listdir(paths.DEFAULT_ARTIFACTS_DIR))
    rc = train.main(["--out-dir", str(out), "--tag", "unit", "--device", "cpu", "--real-pairs", pairs,
                     "--set", "batch_size=16", "--set", "d_model=16", "--set", "n_layers=1",
                     "--set", "n_heads=2", "--set", "d_out=16", "--set", "real_frac=0.5",
                     "--set", "plate_neg_weight=0.5", "--set", "epochs=1"])
    assert rc == 0
    for name in ("model.pt", "catastro_emb.pt", "train_report.json", "experiment.json"):
        assert (out / name).exists(), name
    assert any(f.startswith("plate_negatives_c8_d20") for f in os.listdir(out))
    exp = json.load(open(out / "experiment.json"))
    assert exp["hyperparams"]["real_frac"] == 0.5 and exp["dataset_sizes"]["real"]["n_val"] > 0
    assert len(exp["code_fingerprint"]) == 64 and exp["metrics"]["real_val_recall@20"] is not None
    assert exp["dataset_sizes"]["n_docs"] == 48 and exp["saved"] is True
    assert paths.sha256_serving_files() == before                         # production byte-identical
    assert set(os.listdir(paths.DEFAULT_ARTIFACTS_DIR)) == prod_files      # not even a new file there


def test_train_script_finetune_realistic_noise_and_existing_out_dir(tmp_path):
    train = _load_script("train")
    out = tmp_path / "exp"
    docs, data, neg = tiny_data_dir(str(out))
    pairs = write_pairs(str(tmp_path / "pairs.parquet"), docs, n=20)
    base = train_model(data, neg, tiny_config(), device="cpu")
    init = str(tmp_path / "base.pt")
    save_checkpoint(init, base["model"], base["config"], data)
    rc = train.main(["--out-dir", str(out), "--device", "cpu", "--real-pairs", pairs, "--mode", "finetune",
                     "--init-from", init, "--realistic-noise", "--set", "batch_size=16", "--set", "epochs=1",
                     "--set", "n_variants=2"])
    assert rc == 0
    exp = json.load(open(out / "experiment.json"))
    assert exp["hyperparams"]["lr"] == 3e-4 and exp["hyperparams"]["real_frac"] == 0.5   # finetune presets
    table = exp["realistic_noise"]
    assert set(table) >= {"real", "synthetic", "op_scale"} and "raw_address" not in json.dumps(table)
    # re-running into an existing out dir works (train.py itself never refuses; the runner does)
    assert train.main(["--out-dir", str(out), "--device", "cpu", "--real-pairs", pairs,
                       "--set", "batch_size=16", "--set", "d_model=16", "--set", "n_layers=1",
                       "--set", "n_heads=2", "--set", "d_out=16", "--set", "epochs=1"]) == 0


def test_train_script_smoke_env_does_not_save_into_production(monkeypatch, tmp_path):
    train = _load_script("train")
    hp, _ = train.resolve_hparams(train.build_parser().parse_args([]), {"SMOKE": "1", "EPOCHS": "5"})
    assert hp["smoke"] is True and hp["epochs"] == 1                        # smoke wins over EPOCHS
    hp, explicit = train.resolve_hparams(train.build_parser().parse_args(["--lr", "2e-3"]), {"EPOCHS": "3"})
    assert hp["epochs"] == 3 and hp["lr"] == 2e-3 and hp["max_minutes"] == 24.0 and explicit == {"epochs", "lr"}
