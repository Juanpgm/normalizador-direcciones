"""Training loop for the dual-encoder retrieval model."""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import AddressEncoder
from .realpairs import RealPairs, sample_real_batch

__all__ = [
    "TrainConfig", "train_model", "embed_documents", "set_seeds", "recall_at_k",
    "resolve_device", "save_checkpoint", "load_checkpoint", "AUX_HEADS",
]

#: Auxiliary head module names of :class:`AddressEncoder` (frozen by ``freeze_aux``).
AUX_HEADS = ("coord_head", "comuna_head", "barrio_head")


def resolve_device(requested: str | None = None) -> str:
    """``cuda`` only when it is both requested/auto and actually available; else ``cpu``."""
    want = requested or "cuda"
    if want.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return want


class TrainConfig:
    def __init__(
        self,
        epochs: int = 8,
        batch_size: int = 2048,
        max_len: int = 40,
        lr: float = 3e-3,
        weight_decay: float = 0.01,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_out: int = 256,
        dropout: float = 0.1,
        n_hard_neg: int = 1,
        w_coord: float = 0.5,
        w_comuna: float = 0.2,
        w_barrio: float = 0.2,
        p_raw_view: float = 0.25,
        grad_clip: float = 1.0,
        seed: int = 42,
        max_minutes: float = 25.0,
        label_smoothing: float = 0.02,
        init_temperature: float = 0.05,
        pct_start: float = 0.15,
        div_factor: float = 25.0,
        final_div_factor: float = 100.0,
        real_frac: float = 0.0,
        real_oversample: int = 1,
        real_val_frac: float = 0.10,
        plate_neg_weight: float = 0.0,
        plate_neg_count: int = 8,
        plate_neg_max_delta: int = 20,
        mode: str = "mixed",
        freeze_aux: bool = False,
        steps_per_epoch: int | None = None,
    ) -> None:
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def validate(self) -> None:
        if self.mode not in ("mixed", "finetune"):
            raise ValueError(f"mode must be 'mixed' or 'finetune', got {self.mode!r}")
        if not 0.0 <= self.real_frac <= 1.0:
            raise ValueError("real_frac must be in [0, 1]")
        if not 0.0 <= self.plate_neg_weight <= 1.0:
            raise ValueError("plate_neg_weight must be in [0, 1]")
        if self.real_oversample < 1:
            raise ValueError("real_oversample must be >= 1")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be >= 1")


def set_seeds(seed: int = 42) -> None:
    import random as _random

    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def embed_documents(
    model: AddressEncoder, d_tokens: np.ndarray, device: str = "cuda", batch: int = 4096, progress=None
) -> torch.Tensor:
    """Embed the whole cadastral index; returns an fp16 matrix on ``device``.

    ``progress(done, total)`` is called after every batch when given (used by ``cali-address setup``).
    """
    model.eval()
    # Token matrices are stored at MAX_LEN; the model may have been trained with a
    # shorter positional embedding, so always truncate to what it can address.
    if d_tokens.shape[1] > model.max_len:
        d_tokens = np.ascontiguousarray(d_tokens[:, : model.max_len])
    out = torch.empty((len(d_tokens), model.proj.out_features), dtype=torch.float16, device=device)
    for i in range(0, len(d_tokens), batch):
        ids = torch.from_numpy(d_tokens[i : i + batch].astype(np.int64)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            emb = model.embed(ids, role=1)
        out[i : i + batch] = F.normalize(emb.float(), dim=-1).half()
        if progress is not None:
            progress(min(i + batch, len(d_tokens)), len(d_tokens))
    return out


@torch.no_grad()
def recall_at_k(
    model: AddressEncoder,
    q_tokens: np.ndarray,
    q_doc: np.ndarray,
    doc_emb: torch.Tensor,
    ks=(1, 5, 20),
    device: str = "cuda",
    batch: int = 1024,
) -> dict:
    model.eval()
    if q_tokens.shape[1] > model.max_len:
        q_tokens = np.ascontiguousarray(q_tokens[:, : model.max_len])
    kmax = max(ks)
    hits = {k: 0 for k in ks}
    n = len(q_tokens)
    for i in range(0, n, batch):
        ids = torch.from_numpy(q_tokens[i : i + batch].astype(np.int64)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            emb = model.embed(ids, role=0)
        emb = F.normalize(emb.float(), dim=-1).half()
        sims = emb @ doc_emb.T
        top = sims.topk(kmax, dim=-1).indices.cpu().numpy()
        truth = q_doc[i : i + batch]
        for k in ks:
            hits[k] += int((top[:, :k] == truth[:, None]).any(axis=1).sum())
    return {f"recall@{k}": hits[k] / max(n, 1) for k in ks}


def save_checkpoint(path: str, model: AddressEncoder, cfg_dict: dict, data: dict) -> None:
    """Write a checkpoint in the exact format ``AddressNormalizer`` loads."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg_dict,
            "n_comuna": int(data["doc_comuna"].max()) + 1,
            "n_barrio": int(data["doc_barrio"].max()) + 1,
            "xy_mean": data["xy_mean"],
            "xy_std": data["xy_std"],
            "comuna_codes": data["comuna_codes"],
            "barrio_codes": data["barrio_codes"],
        },
        path,
    )


def load_checkpoint(path: str, device: str = "cpu", dropout: float | None = None) -> tuple[AddressEncoder, dict]:
    """Rebuild an :class:`AddressEncoder` from a ``model.pt``; returns ``(model, ckpt)``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    c = ckpt["config"]
    model = AddressEncoder(
        d_model=c["d_model"], n_layers=c["n_layers"], n_heads=c["n_heads"], d_out=c["d_out"],
        dropout=c.get("dropout", 0.1) if dropout is None else dropout,
        n_comuna=ckpt["n_comuna"], n_barrio=ckpt["n_barrio"], max_len=int(c.get("max_len", 40)),
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device), ckpt


def _recall_row(prefix: str, metrics: dict | None) -> dict:
    keys = ("recall@1", "recall@5", "recall@20")
    return {f"{prefix}{k}": (None if metrics is None else metrics.get(k)) for k in keys}


def train_model(
    data: dict,
    negatives: np.ndarray,
    cfg: TrainConfig,
    device: str = "cuda",
    real: RealPairs | None = None,
    plate_negatives: np.ndarray | None = None,
    init_checkpoint: str | None = None,
) -> dict:
    """Train the encoder; returns ``{'model', 'history', 'seconds', 'gpu', ...}``.

    * ``real``: real pairs mixed into every batch (``cfg.real_frac`` of the slots,
      cycling over ``real_oversample`` copies of the training pairs). Recall on its
      held-out validation fold is reported every epoch as ``real_val_recall@k``.
    * ``plate_negatives``: ``(n_docs, m)`` left-packed table from
      ``build_plate_negatives``; a share ``cfg.plate_neg_weight`` of the hard-negative
      slots is drawn from it instead of the spatial table.
    * ``init_checkpoint`` (``cfg.mode == 'finetune'``): start from an existing
      ``model.pt``; its architecture overrides the architecture fields of ``cfg``.
    """
    cfg.validate()
    device = resolve_device(device)
    set_seeds(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    n_comuna = int(data["doc_comuna"].max()) + 1
    n_barrio = int(data["doc_barrio"].max()) + 1
    if cfg.mode == "finetune":
        if not init_checkpoint:
            raise ValueError("finetune mode needs init_checkpoint")
        model, ckpt = load_checkpoint(init_checkpoint, device, dropout=cfg.dropout)
        if ckpt["n_comuna"] != n_comuna or ckpt["n_barrio"] != n_barrio:
            raise ValueError("checkpoint comuna/barrio heads do not match the training data")
        for k in ("d_model", "n_layers", "n_heads", "d_out", "max_len"):
            setattr(cfg, k, int(ckpt["config"].get(k, getattr(cfg, k))))
    else:
        model = AddressEncoder(
            d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads, d_out=cfg.d_out,
            dropout=cfg.dropout, n_comuna=n_comuna, n_barrio=n_barrio, max_len=cfg.max_len,
        ).to(device)
        with torch.no_grad():
            model.logit_scale.fill_(float(np.log(1.0 / cfg.init_temperature)))
    if cfg.freeze_aux:
        for name in AUX_HEADS:
            for prm in getattr(model, name).parameters():
                prm.requires_grad_(False)

    # Sequences are stored at MAX_LEN=48; p99 of the cadastral address length is
    # 31 characters, so training at 40 saves ~17% compute with no measurable loss.
    L = cfg.max_len
    q_tokens = np.ascontiguousarray(data["q_tokens"][:, :L])
    q_tokens_raw = np.ascontiguousarray(data["q_tokens_raw"][:, :L])
    q_doc = data["q_doc"]
    is_val = data["q_is_val"]
    d_tokens = np.ascontiguousarray(data["d_tokens"][:, :L])
    doc_comuna_np = np.asarray(data["doc_comuna"])

    train_idx = np.flatnonzero(~is_val)
    val_idx = np.flatnonzero(is_val)
    use_real = real is not None and real.n_train > 0 and cfg.real_frac > 0
    print(f"train queries {len(train_idx):,} | val queries {len(val_idx):,} | "
          f"real train {real.n_train if real else 0} val {real.n_val if real else 0} "
          f"| mode {cfg.mode} | device {device}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params/1e6:.2f} M", flush=True)

    doc_barrio = torch.from_numpy(data["doc_barrio"]).to(device)
    doc_xy = torch.from_numpy(data["doc_xy"]).to(device)
    neg = negatives
    plate_neg = plate_negatives if (plate_negatives is not None and cfg.plate_neg_weight > 0) else None
    plate_nvalid = (plate_neg >= 0).sum(1) if plate_neg is not None else None

    B = cfg.batch_size
    n_real = min(int(round(cfg.real_frac * B)), B) if use_real else 0
    n_syn = B - n_real
    if cfg.steps_per_epoch:
        steps_per_epoch = int(cfg.steps_per_epoch)
    elif n_syn == 0:
        steps_per_epoch = max(-(-real.n_train * cfg.real_oversample // B), 1)
    else:
        steps_per_epoch = max(len(train_idx) // B, 1)
    total_steps = steps_per_epoch * cfg.epochs
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay,
                            betas=(0.9, 0.98), eps=1e-8)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=total_steps, pct_start=cfg.pct_start,
        div_factor=cfg.div_factor, final_div_factor=cfg.final_div_factor,
    )

    rng = np.random.default_rng(cfg.seed)
    real_stream = None
    if use_real:
        real_stream = rng.permutation(np.tile(np.arange(real.n_train, dtype=np.int64), cfg.real_oversample))
    real_cursor = 0
    history: list[dict] = []
    t_start = time.time()
    seqs_seen = 0
    stopped_early = False

    def real_val_metrics(doc_emb):
        if real is None or real.n_val == 0:
            return None
        return recall_at_k(model, real.val_tokens[:, :L], real.val_doc, doc_emb, device=device)

    initial_real_val = None
    if cfg.mode == "finetune" and real is not None and real.n_val:
        initial_real_val = real_val_metrics(embed_documents(model, d_tokens, device=device))
        print("initial real-val recall:", json.dumps(initial_real_val), flush=True)

    empty = np.empty(0, dtype=np.int64)
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(train_idx) if n_syn > 0 else empty
        running = {"loss": 0.0, "nce": 0.0, "coord": 0.0, "comuna": 0.0, "barrio": 0.0, "acc": 0.0}
        t_epoch = time.time()
        step = 0
        for step in range(steps_per_epoch):
            # With real pairs the synthetic slice is B - n_real wide, so an epoch
            # may wrap around `order`; the modulo keeps the slice full.
            if n_syn > 0:
                start = (step * n_syn) % max(len(order), 1)
                qi = order[start : start + n_syn]
                if len(qi) < n_syn:
                    qi = np.concatenate([qi, order[: n_syn - len(qi)]])
            else:
                qi = empty
            if use_real:
                ri, real_cursor, real_stream = sample_real_batch(real_stream, real_cursor, n_real, rng)
            else:
                ri = empty
            if len(qi) + len(ri) == 0:
                continue
            pos = np.concatenate([q_doc[qi], real.train_doc[ri] if len(ri) else empty])
            # ---- hard negatives: spatial, with a share swapped for plate neighbours
            if cfg.n_hard_neg > 0:
                cols = rng.integers(0, neg.shape[1], size=(len(pos), cfg.n_hard_neg))
                hard = neg[pos[:, None], cols]
                if plate_neg is not None:
                    nv = plate_nvalid[pos][:, None]
                    pcols = (rng.random((len(pos), cfg.n_hard_neg)) * np.maximum(nv, 1)).astype(np.int64)
                    cand = plate_neg[pos[:, None], pcols]
                    use = (rng.random((len(pos), cfg.n_hard_neg)) < cfg.plate_neg_weight) & (nv > 0)
                    hard = np.where(use, cand, hard)
                hard = hard[hard >= 0]
            else:
                hard = empty
            doc_ids = np.concatenate([pos, hard])
            uniq, inverse = np.unique(doc_ids, return_inverse=True)
            target = torch.from_numpy(inverse[: len(pos)]).to(device, non_blocking=True)

            # ---- text views -------------------------------------------
            use_raw = rng.random(len(pos)) < cfg.p_raw_view
            q_ids = np.where(use_raw[: len(qi), None], q_tokens_raw[qi], q_tokens[qi])
            comuna_np = doc_comuna_np[pos].copy()
            if len(ri):
                real_ids = np.where(
                    use_raw[len(qi):, None],
                    real.train_tokens_raw[ri][:, :L], real.train_tokens[ri][:, :L],
                )
                q_ids = np.concatenate([q_ids, real_ids])
                rc = real.train_comuna[ri]
                comuna_np[len(qi):] = np.where(rc >= 0, rc, comuna_np[len(qi):])
            q = torch.from_numpy(q_ids.astype(np.int64)).to(device, non_blocking=True)
            d = torch.from_numpy(d_tokens[uniq].astype(np.int64)).to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                q_emb, coord_pred, comuna_logits, barrio_logits = model(q, role=0, with_aux=True)
                d_emb = model(d, role=1)
                scale = model.logit_scale.exp().clamp(max=100.0)
                logits = (q_emb.float() @ d_emb.float().T) * scale
                loss_nce = F.cross_entropy(logits, target, label_smoothing=cfg.label_smoothing)
                pos_t = torch.from_numpy(pos).to(device, non_blocking=True)
                comuna_t = torch.from_numpy(comuna_np).to(device, non_blocking=True)
                loss_coord = F.smooth_l1_loss(coord_pred.float(), doc_xy[pos_t], beta=0.2)
                loss_com = F.cross_entropy(comuna_logits.float(), comuna_t)
                loss_bar = F.cross_entropy(barrio_logits.float(), doc_barrio[pos_t])
                loss = (
                    loss_nce
                    + cfg.w_coord * loss_coord
                    + cfg.w_comuna * loss_com
                    + cfg.w_barrio * loss_bar
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            opt.step()
            sched.step()

            seqs_seen += len(pos) + len(uniq)
            running["loss"] += loss.item()
            running["nce"] += loss_nce.item()
            running["coord"] += loss_coord.item()
            running["comuna"] += loss_com.item()
            running["barrio"] += loss_bar.item()
            running["acc"] += (logits.argmax(-1) == target).float().mean().item()

            if step % 200 == 0:
                print(
                    f"  e{epoch} s{step}/{steps_per_epoch} loss={loss.item():.4f} "
                    f"nce={loss_nce.item():.4f} inbatch_acc={running['acc']/(step+1):.3f} "
                    f"lr={sched.get_last_lr()[0]:.2e} {time.time()-t_start:.0f}s",
                    flush=True,
                )
            if (time.time() - t_start) / 60 > cfg.max_minutes:
                print(f"  time budget of {cfg.max_minutes} min reached, stopping", flush=True)
                stopped_early = True
                break

        doc_emb = embed_documents(model, d_tokens, device=device)
        # A seeded RANDOM subsample: `val_idx[:20000]` would be a low-doc-id
        # prefix (queries are stored doc-major), which is a biased slice of the
        # city rather than a sample of it.
        val_rng = np.random.default_rng(cfg.seed + epoch)
        val_sub = (
            val_idx if len(val_idx) <= 20_000
            else val_rng.choice(val_idx, size=20_000, replace=False)
        )
        metrics = recall_at_k(model, q_tokens[val_sub], q_doc[val_sub], doc_emb, device=device)
        train_sub = rng.choice(train_idx, size=min(5_000, len(train_idx)), replace=False)
        metrics_train = recall_at_k(
            model, q_tokens[train_sub], q_doc[train_sub], doc_emb, ks=(1,), device=device
        )
        metrics_real = real_val_metrics(doc_emb)
        del doc_emb
        torch.cuda.empty_cache()

        n = max(step + 1, 1)
        row = {
            "epoch": epoch,
            "loss": running["loss"] / n,
            "loss_nce": running["nce"] / n,
            "loss_coord": running["coord"] / n,
            "loss_comuna": running["comuna"] / n,
            "loss_barrio": running["barrio"] / n,
            "inbatch_acc": running["acc"] / n,
            "train_recall@1": metrics_train["recall@1"],
            "seconds": time.time() - t_start,
            "epoch_seconds": time.time() - t_epoch,
            "seqs_per_s": seqs_seen / max(time.time() - t_start, 1e-9),
            "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0,
            **metrics,
            **_recall_row("real_val_", metrics_real),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if stopped_early:
            break

    seconds = time.time() - t_start
    gpu = {}
    if device == "cuda":
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            "max_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "cudnn": torch.backends.cudnn.is_available(),
        }
    return {
        "model": model,
        "history": history,
        "seconds": seconds,
        "gpu": gpu,
        "n_params": n_params,
        "config": cfg.to_dict(),
        "device": device,
        "initial_real_val": initial_real_val,
        "real_stats": (real.stats if real is not None else None),
        "steps_per_epoch": steps_per_epoch,
        "batch_split": {"synthetic": n_syn, "real": n_real},
    }
