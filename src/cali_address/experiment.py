"""Experiment bookkeeping: config overrides, guards, experiment.json and the leaderboard.

Pure functions only (no GPU, no training) so they are unit-testable; the heavy
lifting lives in ``scripts/train.py`` and is orchestrated by
``scripts/run_experiment.py``.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time

from .paths import DEFAULT_ARTIFACTS_DIR, code_fingerprint, is_default_dir
from .train import TrainConfig

__all__ = [
    "EXTRA_DEFAULTS",
    "LEADERBOARD_COLUMNS",
    "all_defaults",
    "parse_overrides",
    "safe_tag",
    "experiment_dir",
    "prepare_out_dir",
    "is_frozen_path",
    "guard_frozen",
    "final_epoch_metrics",
    "eval_metrics",
    "leaderboard_row",
    "append_leaderboard",
    "write_experiment_json",
]

#: Experiment-level knobs that are not fields of ``TrainConfig``.
EXTRA_DEFAULTS = {
    "realistic_noise": False,
    "real_pairs": "",        # path override for train_pairs.parquet ("" = default)
    "init_from": "",         # finetune: model.pt to start from ("" = production model.pt)
    "n_variants": 3,
    "real_split_seed": 42,   # fixed so every run reports on the same real validation fold
    "smoke": False,
}

LEADERBOARD_COLUMNS = [
    "tag", "timestamp", "mode", "smoke", "final",
    "syn_recall@1", "syn_recall@5", "syn_recall@20",
    "real_val_recall@1", "real_val_recall@5", "real_val_recall@20",
    "dev_ok", "dev_n", "dev_manzana_precision", "dev_predial_precision", "dev_lost_correct",
    "dev_gold_n", "dev_gold_ok", "dev_gold_ok_coverage",
    "dev_gold_manzana_precision", "dev_gold_predial_precision", "dev_gold_lost_correct",
    "seconds", "code_fingerprint",
]

_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def all_defaults() -> dict:
    d = TrainConfig().to_dict()
    d.update(EXTRA_DEFAULTS)
    return d


def _coerce(raw: str, default):
    if isinstance(default, bool):
        low = raw.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"not a boolean: {raw!r}")
    if default is None:
        return int(raw) if raw.lstrip("-").isdigit() else raw
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def parse_overrides(pairs, defaults: dict | None = None) -> dict:
    """``["lr=1e-3", "epochs=2"]`` -> ``{"lr": 0.001, "epochs": 2}``; unknown keys raise."""
    defaults = defaults if defaults is not None else all_defaults()
    out: dict = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, _, raw = item.partition("=")
        key = key.strip().replace("-", "_")
        if key not in defaults:
            raise ValueError(f"unknown hyper-parameter {key!r}; known: {sorted(defaults)}")
        out[key] = _coerce(raw.strip(), defaults[key])
    return out


def safe_tag(tag: str) -> str:
    if not tag or not _TAG_RE.match(tag):
        raise ValueError(f"invalid tag {tag!r}: use letters, digits, '_', '.', '-' (max 64)")
    return tag


def experiment_dir(tag: str, root: str | None = None) -> str:
    return os.path.join(root or os.path.join(DEFAULT_ARTIFACTS_DIR, "experiments"), safe_tag(tag))


def prepare_out_dir(out_dir: str, overwrite: bool = False) -> str:
    """Create ``out_dir``; refuse the production dir and (without ``overwrite``) a non-empty one."""
    if is_default_dir(out_dir):
        raise ValueError("refusing to use the production artifacts directory as an experiment directory")
    if os.path.isdir(out_dir) and os.listdir(out_dir) and not overwrite:
        raise FileExistsError(f"{out_dir} is not empty; pick another --tag or pass --overwrite")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def is_frozen_path(path: str) -> bool:
    return "frozen_test" in os.path.basename(str(path)).lower()


def guard_frozen(scored_path: str, final: bool) -> None:
    """Raise unless ``--final`` was passed when ``scored_path`` is the frozen test set."""
    if is_frozen_path(scored_path) and not final:
        raise PermissionError(
            "frozen_test is the final measurement only; pass --final to score it "
            "(never use it for tuning or model selection)"
        )


def final_epoch_metrics(history: list[dict]) -> dict:
    """Recalls of the last epoch (synthetic held-out and real validation fold)."""
    last = history[-1] if history else {}
    out = {}
    for k in (1, 5, 20):
        out[f"syn_recall@{k}"] = last.get(f"recall@{k}")
        out[f"real_val_recall@{k}"] = last.get(f"real_val_recall@{k}")
    return out


def _all_ok_row(summary_rows: list[dict]) -> dict | None:
    for r in summary_rows:
        if r.get("dataset") == "ALL" and r.get("estado") == "OK":
            return r
    return None


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def eval_metrics(summary_rows: list[dict], prefix: str) -> dict:
    """Headline numbers from ``strict_summary`` rows (dicts). ``n`` counts every ALL row."""
    total = sum(int(_num(r.get("n")) or 0) for r in summary_rows if r.get("dataset") == "ALL")
    lost = sum(int(_num(r.get("lost_correct")) or 0) for r in summary_rows if r.get("dataset") == "ALL")
    ok = _all_ok_row(summary_rows)
    return {
        f"{prefix}_n": total,
        f"{prefix}_ok": int(_num(ok.get("n")) or 0) if ok else 0,
        f"{prefix}_manzana_precision": _num(ok.get("manzana_precision")) if ok else None,
        f"{prefix}_predial_precision": _num(ok.get("predial_precision")) if ok else None,
        f"{prefix}_lost_correct": lost,
    }


def leaderboard_row(tag: str, mode: str, history: list[dict], dev_all: list[dict],
                    dev_gold: list[dict], seconds: float, smoke: bool = False,
                    final: bool = False, fingerprint: str | None = None) -> dict:
    row = {"tag": tag, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "mode": mode,
           "smoke": bool(smoke), "final": bool(final)}
    row.update(final_epoch_metrics(history))
    a = eval_metrics(dev_all, "dev")
    g = eval_metrics(dev_gold, "dev_gold")
    row.update({k: a[k] for k in ("dev_ok", "dev_n", "dev_manzana_precision", "dev_predial_precision",
                                  "dev_lost_correct")})
    row.update({
        "dev_gold_n": g["dev_gold_n"],
        "dev_gold_ok": g["dev_gold_ok"],
        "dev_gold_ok_coverage": (g["dev_gold_ok"] / g["dev_gold_n"]) if g["dev_gold_n"] else None,
        "dev_gold_manzana_precision": g["dev_gold_manzana_precision"],
        "dev_gold_predial_precision": g["dev_gold_predial_precision"],
        "dev_gold_lost_correct": g["dev_gold_lost_correct"],
    })
    row["seconds"] = round(float(seconds), 1)
    row["code_fingerprint"] = (fingerprint or code_fingerprint())[:12]
    return row


def append_leaderboard(path: str, row: dict) -> None:
    """Append one row, writing the header for a new/empty file; unknown keys are ignored."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LEADERBOARD_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in LEADERBOARD_COLUMNS})


def write_experiment_json(out_dir: str, payload: dict) -> str:
    """Merge ``payload`` into ``<out_dir>/experiment.json`` (later calls add sections)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "experiment.json")
    current: dict = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                current = json.load(fh)
        except (OSError, ValueError):
            current = {}
    current.update(payload)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(current, fh, indent=2, default=str)
    return path
