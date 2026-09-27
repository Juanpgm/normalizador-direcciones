"""Train the dual encoder (baseline recipe, or an experiment variant).

Defaults reproduce the production recipe. Every hyper-parameter of
``TrainConfig`` is a CLI flag (``--lr 1e-3``) or a ``--set key=value`` override;
the legacy env vars ``EPOCHS``, ``MAX_MINUTES`` and ``SMOKE=1`` still work.

Output directory: ``--out-dir`` / ``--artifacts-dir`` (env ``CALI_ARTIFACTS_DIR``,
default ``artifacts/``). Model-independent inputs (``train_data.npz``,
``spatial_negatives.npy``, ``catastro_docs.parquet``) fall back to ``artifacts/``
when absent from the out-dir. With the default directory the script behaves as it
always did (and ``SMOKE=1`` still saves nothing there, to protect production).

Modes
-----
``mixed``     from scratch, synthetic queries plus (``--real-frac``) real pairs.
``finetune``  start from ``--init-from`` (default ``artifacts/model.pt``), low LR.
              Presets for keys you did not set: lr=3e-4, real_frac=0.5.

Every run writes ``experiment.json`` (hyper-parameters, code fingerprint, dataset
sizes, seconds, metrics) next to ``model.pt``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cali_address.experiment import (  # noqa: E402
    EXTRA_DEFAULTS, all_defaults, final_epoch_metrics, parse_overrides, write_experiment_json,
)
from cali_address.paths import (  # noqa: E402
    DEFAULT_ARTIFACTS_DIR, code_fingerprint, is_default_dir, resolve_artifacts_dir, shared_path,
)
from cali_address.train import (  # noqa: E402
    TrainConfig, embed_documents, resolve_device, save_checkpoint, train_model,
)

FINETUNE_PRESET = {"lr": 3e-4, "real_frac": 0.5}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", "--artifacts-dir", dest="out_dir", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--smoke", action="store_true", help="tiny run: 20k documents' queries, 1 epoch")
    ap.add_argument("--realistic-noise", action="store_true")
    ap.add_argument("--real-pairs", default=None, help="path of train_pairs.parquet")
    ap.add_argument("--init-from", default=None, help="finetune: model.pt to start from")
    ap.add_argument("--mode", choices=("mixed", "finetune"), default=None)
    defaults = all_defaults()
    for key, default in defaults.items():
        if key in ("mode", "realistic_noise", "real_pairs", "init_from", "smoke"):
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(default, bool):
            ap.add_argument(flag, dest=key, action="store_true", default=None)
        else:
            typ = float if isinstance(default, float) else int if isinstance(default, int) else str
            ap.add_argument(flag, dest=key, type=typ, default=None)
    return ap


def resolve_hparams(args: argparse.Namespace, env: dict | None = None) -> tuple[dict, set]:
    """Effective hyper-parameters and the set of keys the user set explicitly."""
    env = os.environ if env is None else env
    hp = all_defaults()
    hp["max_minutes"] = 24.0   # historical script default (TrainConfig's own is 25)
    explicit: set = set()
    if env.get("EPOCHS"):
        hp["epochs"] = int(env["EPOCHS"]); explicit.add("epochs")
    if env.get("MAX_MINUTES"):
        hp["max_minutes"] = float(env["MAX_MINUTES"]); explicit.add("max_minutes")
    if env.get("SMOKE", "0") == "1":
        hp["smoke"] = True
    special = ("mode", "realistic_noise", "real_pairs", "init_from", "smoke")
    for key in hp:
        val = None if key in special else getattr(args, key, None)
        if val is not None:
            hp[key] = val; explicit.add(key)
    for key, val in parse_overrides(args.sets).items():
        hp[key] = val; explicit.add(key)
    if args.smoke:
        hp["smoke"] = True
    if args.realistic_noise:
        hp["realistic_noise"] = True
    if args.mode:
        hp["mode"] = args.mode
    if args.real_pairs:
        hp["real_pairs"] = args.real_pairs
    if args.init_from:
        hp["init_from"] = args.init_from
    if hp["mode"] == "finetune":
        for k, v in FINETUNE_PRESET.items():
            if k not in explicit:
                hp[k] = v
    if hp["smoke"]:
        hp["epochs"], hp["max_minutes"] = 1, 3.0
    return hp, explicit


def _cfg_from(hp: dict) -> TrainConfig:
    fields = TrainConfig().to_dict()
    return TrainConfig(**{k: hp[k] for k in fields})


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = resolve_artifacts_dir(args.out_dir)
    hp, _ = resolve_hparams(args)
    cfg = _cfg_from(hp)
    cfg.validate()
    os.makedirs(out_dir, exist_ok=True)
    t_all = time.time()

    data = {k: v for k, v in np.load(shared_path("train_data.npz", out_dir), allow_pickle=True).items()}
    neg = np.load(shared_path("spatial_negatives.npy", out_dir))
    n_docs = len(data["d_tokens"])
    info: dict = {}

    if hp["realistic_noise"]:
        import pandas as pd
        from cali_address.dataset import build_training_arrays
        from cali_address.realistic_noise import estimate_op_scale
        from cali_address.realpairs import DEFAULT_PAIRS_PATH

        docs = pd.read_parquet(shared_path("catastro_docs.parquet", out_dir))
        splits = os.path.dirname(hp["real_pairs"] or DEFAULT_PAIRS_PATH)
        strings = []
        for name in ("train_pairs.parquet", "unlabeled_addresses.parquet"):
            path = hp["real_pairs"] if (name == "train_pairs.parquet" and hp["real_pairs"]) else os.path.join(splits, name)
            if os.path.exists(path):
                strings += pd.read_parquet(path, columns=["raw_address"])["raw_address"].dropna().astype(str).tolist()
        est = estimate_op_scale(strings, docs["direccion"].tolist(), seed=cfg.seed)
        info["realistic_noise"] = est
        print("realistic noise op_scale:", json.dumps(est["op_scale"]), flush=True)
        data.update(build_training_arrays(docs, n_variants=hp["n_variants"], seed=42,
                                          op_scale=est["op_scale"]))
        np.savez_compressed(os.path.join(out_dir, "train_data.npz"), **data)
        n_docs = len(data["d_tokens"])

    if hp["smoke"]:
        keep = np.flatnonzero(data["q_doc"] < 20000)
        for k in ("q_tokens", "q_tokens_raw", "q_doc", "q_is_val"):
            data[k] = data[k][keep]

    plate = None
    if cfg.plate_neg_weight > 0:
        import pandas as pd
        from cali_address.dataset import build_plate_negatives

        name = f"plate_negatives_c{cfg.plate_neg_count}_d{cfg.plate_neg_max_delta}.npy"
        cached = shared_path(name, out_dir)
        if os.path.exists(cached) and len(np.load(cached, mmap_mode="r")) == n_docs:
            plate = np.load(cached)
        else:
            docs = pd.read_parquet(shared_path("catastro_docs.parquet", out_dir))
            plate = build_plate_negatives(docs, count=cfg.plate_neg_count, max_delta=cfg.plate_neg_max_delta)
            np.save(os.path.join(out_dir, name), plate)
        info["plate_negatives"] = {"docs_with_negatives": int((plate[:, 0] >= 0).sum()), "n_docs": int(len(plate))}

    from cali_address.realpairs import load_real_pairs

    real = load_real_pairs(
        n_docs, comuna_codes=data.get("comuna_codes"), path=hp["real_pairs"] or None,
        val_frac=cfg.real_val_frac, seed=int(hp["real_split_seed"]), max_len=cfg.max_len,
    )
    print("real pairs:", json.dumps(real.stats), flush=True)

    init = None
    if cfg.mode == "finetune":
        init = hp["init_from"] or os.path.join(DEFAULT_ARTIFACTS_DIR, "model.pt")

    device = resolve_device(args.device)
    out = train_model(data, neg, cfg, device=device, real=real, plate_negatives=plate, init_checkpoint=init)
    model = out.pop("model")

    save = not (hp["smoke"] and is_default_dir(out_dir))   # SMOKE never touches production
    if save:
        save_checkpoint(os.path.join(out_dir, "model.pt"), model, out["config"], data)
        doc_emb = embed_documents(model, data["d_tokens"][:, : cfg.max_len], device=device)
        import torch
        torch.save(doc_emb.cpu(), os.path.join(out_dir, "catastro_emb.pt"))
        with open(os.path.join(out_dir, "train_report.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"saved model.pt, catastro_emb.pt, train_report.json in {out_dir}")

    write_experiment_json(out_dir, {
        "tag": args.tag or os.path.basename(out_dir),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": list(argv) if argv is not None else sys.argv[1:],
        "hyperparams": {**hp, **cfg.to_dict()},
        "code_fingerprint": code_fingerprint(),
        "device": device,
        "init_from": init,
        "dataset_sizes": {
            "n_docs": int(n_docs),
            "train_queries": int((~data["q_is_val"]).sum()),
            "val_queries": int(data["q_is_val"].sum()),
            "real": real.stats,
        },
        "train_seconds": round(out["seconds"], 1),
        "total_seconds": round(time.time() - t_all, 1),
        "saved": save,
        "batch_split": out["batch_split"],
        "steps_per_epoch": out["steps_per_epoch"],
        "initial_real_val": out["initial_real_val"],
        "metrics": final_epoch_metrics(out["history"]),
        **info,
    })
    print(json.dumps(out["history"][-1], default=str))
    print("total seconds", out["seconds"], "gpu", out["gpu"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
