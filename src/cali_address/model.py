"""Character-level Transformer dual encoder for Cali address retrieval.

One shared encoder embeds both the noisy query and the canonical cadastral
address, so retrieval is a cosine search in a single space. A ``role`` embedding
tells the encoder which side it is looking at.

Three training signals are combined (see ``train.py``):

1. **InfoNCE** query -> canonical address, with in-batch negatives **plus
   spatial hard negatives** (same cadastral block / nearest centroid), which is
   what forces the model to separate geographically confusable strings such as
   ``KR 26 H 1 # 73 - 10`` and ``KR 26 H # 73 - 10``.
2. **Coordinate regression** of the parcel centroid in a projected CRS
   (auxiliary head, SmoothL1) - injects the spatial prior into the embedding.
3. **Comuna and barrio classification** derived from the 30-digit predial code
   (auxiliary heads, cross entropy) - injects the administrative hierarchy.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CHARS", "VOCAB", "PAD", "UNK", "MAX_LEN", "encode_text", "encode_batch",
           "AddressEncoder"]

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 #-.,/'()|"
PAD = 0
UNK = 1
VOCAB = {ch: i + 2 for i, ch in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS) + 2
MAX_LEN = 48

_TABLE = np.full(256, UNK, dtype=np.uint8)
for _ch, _i in VOCAB.items():
    _TABLE[ord(_ch)] = _i


def encode_text(text: str, max_len: int = MAX_LEN) -> np.ndarray:
    """Encode one already-uppercased ASCII string into a fixed-length id array."""
    out = np.zeros(max_len, dtype=np.uint8)
    b = text.encode("ascii", "replace")[:max_len]
    if b:
        arr = np.frombuffer(b, dtype=np.uint8)
        out[: len(arr)] = _TABLE[arr]
    return out


def encode_batch(texts, max_len: int = MAX_LEN) -> np.ndarray:
    out = np.zeros((len(texts), max_len), dtype=np.uint8)
    for i, t in enumerate(texts):
        out[i] = encode_text(t, max_len)
    return out


class AddressEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 5,
        n_heads: int = 8,
        d_out: int = 256,
        max_len: int = MAX_LEN,
        dropout: float = 0.1,
        n_comuna: int = 24,
        n_barrio: int = 400,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model)
        self.role = nn.Embedding(2, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_out)
        self.coord_head = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 2)
        )
        self.comuna_head = nn.Linear(d_model, n_comuna)
        self.barrio_head = nn.Linear(d_model, n_barrio)
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1 / 0.05))))
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].fill_(0)

    def pool(self, ids: torch.Tensor, role: int) -> torch.Tensor:
        """Mean-pool the encoder output over non-pad positions."""
        b, t = ids.shape
        pad_mask = ids.eq(PAD)
        pos = torch.arange(t, device=ids.device).unsqueeze(0)
        x = self.tok(ids) + self.pos(pos) + self.role.weight[role].view(1, 1, -1)
        x = self.drop(x)
        # A fully padded row would make softmax produce NaNs; keep position 0 alive.
        safe_mask = pad_mask.clone()
        safe_mask[:, 0] = False
        x = self.encoder(x, src_key_padding_mask=safe_mask)
        w = (~safe_mask).to(x.dtype).unsqueeze(-1)
        pooled = (x * w).sum(1) / w.sum(1).clamp(min=1.0)
        return self.norm(pooled)

    def embed(self, ids: torch.Tensor, role: int = 0) -> torch.Tensor:
        """L2-normalized retrieval embedding."""
        return F.normalize(self.proj(self.pool(ids, role)), dim=-1)

    def forward(self, ids: torch.Tensor, role: int = 0, with_aux: bool = False):
        pooled = self.pool(ids, role)
        emb = F.normalize(self.proj(pooled), dim=-1)
        if not with_aux:
            return emb
        return emb, self.coord_head(pooled), self.comuna_head(pooled), self.barrio_head(pooled)
