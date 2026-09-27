"""Inference pipeline: free-form text -> canonical address + predio + confidence.

    rule-based canonicalization
      -> char-Transformer encoding on the GPU (two views: canonical + raw)
      -> top-k cosine search over the 330k cadastral embeddings (fp16 matmul)
      -> rerank with a fused score (model similarity + rapidfuzz + structure)
      -> best predio, canonical address, confidence, manzana, centroid, comuna
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rapidfuzz import fuzz

from .geo import haversine_m, to_metric
from .model import AddressEncoder, encode_batch
from .parser import _pre_normalize, canonical, match_key, parse_address
from .paths import DEFAULT_ARTIFACTS_DIR, model_path, shared_path
from .reliability import RELIABILITY_FILENAME, load_reliability

__all__ = [
    "AddressNormalizer",
    "FUSE_WEIGHTS",
    "CONFIDENCE_THRESHOLD",
    "GEO_TAU_M",
    "load_tuning",
    "SOFT_RULE_NAMES",
    "DECISION_KEYS",
    "normalize_soft_rules",
    "validate_decision",
]

#: Fallback weights of the reranking score, used when ``artifacts/tuning.json`` is
#: absent. The shipped values are the ones ``scripts/tune_threshold.py`` selected
#: on the **held-out cadastral validation split** (documents never seen as
#: positives during training); they were never fitted on the Excel evaluation
#: rows. ``AddressNormalizer`` reloads whatever that file contains, so the code
#: and the artifact cannot drift apart.
FUSE_WEIGHTS = {"sim": 0.70, "fuzz": 0.10, "struct": 0.10, "geo": 0.10}

#: Length scale (metres) of the geographic consistency term. A candidate sitting
#: `GEO_TAU_M` metres away from the coordinate predicted by the auxiliary head
#: keeps 1/e of its geographic credit. Fixed a priori, not tuned.
GEO_TAU_M = 600.0

#: Fallback high-confidence cutoff on the fused score; see ``load_tuning``.
#: Selected by maximizing F1 of "matched => correct document" on the held-out
#: cadastral split mixed with synthetic non-address negatives; stable across
#: assumed junk rates from 5 % to 40 %.
CONFIDENCE_THRESHOLD = 0.73

#: Name of the tuning artifact produced by ``scripts/tune_threshold.py``.
TUNING_FILENAME = "tuning.json"


#: Rule classes the decision layer may downgrade from hard to soft (see
#: ``service._tagged_violations``). A soft violation keeps the row OK but only vouches for the
#: manzana. The empty selection is today's behaviour.
SOFT_RULE_NAMES = (
    "letra_via",             # via letter differs on both sides (one-sided is already soft)
    "letra_cruce_cardinal",  # cross letter is a lone cardinal (N/O/W) on one side only
    "letra_cruce_una_cara",  # cross letter present on exactly one side
    "letra_cruce",           # any cross-letter mismatch
    "bis_via",
    "bis_cruce",
    "cuadrante_compl",       # quadrant word in the complement missing from the cadastre
)

#: Keys the optional ``chosen.decision`` block of ``tuning.json`` may carry.
DECISION_KEYS = ("soft_rules", "max_soft", "gate_fallback")


def normalize_soft_rules(value) -> frozenset:
    """Validate a ``soft_rules`` selection: ``None``/empty -> empty set; unknown name -> ``ValueError``.

    Accepts a comma-separated string or any iterable of names.
    """
    if value is None:
        return frozenset()
    if isinstance(value, str):
        names = [part.strip() for part in value.split(",") if part.strip()]
    else:
        try:
            names = list(value)
        except TypeError as exc:
            raise ValueError(f"soft_rules must be a string or an iterable of names, got {value!r}") from exc
    for name in names:
        if not isinstance(name, str) or name not in SOFT_RULE_NAMES:
            raise ValueError(f"unknown soft rule {name!r}; valid names: {', '.join(SOFT_RULE_NAMES)}")
    return frozenset(names)


def validate_max_soft(value) -> int:
    """``max_soft`` must be a non-negative ``int`` (``bool`` is rejected)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"max_soft must be an int >= 0, got {value!r}")
    return value


def validate_decision(decision) -> dict:
    """Validate the ``chosen.decision`` block of ``tuning.json`` and return a plain dict copy."""
    if not isinstance(decision, dict):
        raise ValueError(f"tuning decision block must be an object, got {type(decision).__name__}")
    unknown = sorted(set(decision) - set(DECISION_KEYS))
    if unknown:
        raise ValueError(f"unknown decision keys {unknown}; valid keys: {', '.join(DECISION_KEYS)}")
    out: dict = {}
    if "soft_rules" in decision:
        normalize_soft_rules(decision["soft_rules"])
        out["soft_rules"] = list(decision["soft_rules"]) if not isinstance(decision["soft_rules"], str) else [
            part.strip() for part in decision["soft_rules"].split(",") if part.strip()]
    if "max_soft" in decision:
        out["max_soft"] = validate_max_soft(decision["max_soft"])
    if "gate_fallback" in decision:
        if not isinstance(decision["gate_fallback"], bool):
            raise ValueError(f"gate_fallback must be a bool, got {decision['gate_fallback']!r}")
        out["gate_fallback"] = decision["gate_fallback"]
    return out


def load_tuning(artifacts_dir: str) -> dict:
    """Read the tuning artifact, falling back to the module defaults.

    Returns ``{"weights", "threshold", "thresholds", "decision", "source"}``. ``decision`` is the
    optional ``chosen.decision`` block (defaults of ``soft_rules`` / ``max_soft`` / ``gate_fallback``
    for the strict decision layer; ``{}`` reproduces the historical behaviour). ``thresholds``
    maps an ablation variant name to its own cutoff, because the variants score
    candidates on different scales (raw cosine vs fused score) and a single
    cutoff would make their coverage numbers incomparable.
    """
    path = os.path.join(artifacts_dir, TUNING_FILENAME)
    out = {
        "weights": dict(FUSE_WEIGHTS),
        "threshold": CONFIDENCE_THRESHOLD,
        "thresholds": {},
        "decision": {},
        "source": "module defaults (artifacts/tuning.json not found)",
    }
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    chosen = data.get("chosen", {})
    if chosen.get("weights"):
        out["weights"] = {k: float(v) for k, v in chosen["weights"].items()}
    if chosen.get("threshold") is not None:
        out["threshold"] = float(chosen["threshold"])
    out["thresholds"] = {k: float(v) for k, v in (chosen.get("thresholds") or {}).items()}
    if chosen.get("decision") is not None:
        out["decision"] = validate_decision(chosen["decision"])
    out["source"] = path
    return out

_STRUCT_WEIGHTS = {
    "via_type": 1.0,
    "via_number": 2.0,
    "via_letters": 1.0,
    "via_quadrant": 0.75,
    "cross_number": 2.0,
    "cross_letters": 1.0,
    "cross_quadrant": 0.75,
    "cross_type": 0.5,
    "plate": 1.5,
    "via_bis": 0.75,
    "cross_bis": 0.75,
}

#: Fields whose "no value" sentinel is ``False`` rather than ``None``. Two plain
#: addresses (``via_bis=cross_bis=False`` on both sides) must not get a bigger
#: denominator than a pair that never mentions these fields at all, or a
#: structural mismatch elsewhere would be diluted by a free, meaningless match.
_BOOLEAN_STRUCT_FIELDS = frozenset({"via_bis", "cross_bis"})


def load_manzana_nunique(docs: pd.DataFrame) -> np.ndarray:
    """Distinct manzanas behind each document's address.

    Older ``catastro_docs.parquet`` files predate the column; they fall back to
    1 for every doc (the address is assumed to sit in a single manzana).
    """
    if "manzana_nunique" in docs.columns:
        return docs["manzana_nunique"].fillna(1).to_numpy(dtype="int64")
    return np.ones(len(docs), dtype="int64")


def _structural_agreement(a, b) -> float:
    """Weighted share of structural fields that agree between two parses."""
    total = 0.0
    got = 0.0
    for field, w in _STRUCT_WEIGHTS.items():
        va, vb = getattr(a, field), getattr(b, field)
        if field in _BOOLEAN_STRUCT_FIELDS:
            if va is False and vb is False:
                continue
        elif va is None and vb is None:
            continue
        total += w
        if va == vb:
            got += w
        elif field == "plate" and va is not None and vb is not None:
            # plates are often mistyped by one digit; give partial credit
            if str(va).lstrip("0") == str(vb).lstrip("0"):
                got += w
            elif abs(int(va) - int(vb)) <= 2:
                got += 0.4 * w
    if total == 0:
        return 0.0
    return got / total


@dataclass
class _Doc:
    direccion: str
    predial: str
    manzana: str
    comuna: str
    barrio: str
    lat: float
    lon: float
    n_predios: int


class AddressNormalizer:
    """Loads the cached model + embeddings and normalizes addresses."""

    def __init__(
        self,
        artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
        device: str | None = None,
        docs: pd.DataFrame | None = None,
    ) -> None:
        self.artifacts_dir = artifacts_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tuning = load_tuning(artifacts_dir)
        self.weights = dict(tuning["weights"])
        self.threshold = float(tuning["threshold"])
        self.variant_thresholds = dict(tuning["thresholds"])
        #: Decision-layer defaults carried by the artifact (see ``load_tuning``).
        self.decision = dict(tuning["decision"])
        self.tuning_source = tuning["source"]
        #: Calibrated reliability model, or None (columns stay None) if the artifact is absent.
        self.reliability = load_reliability(os.path.join(artifacts_dir, RELIABILITY_FILENAME))
        ckpt = torch.load(model_path("model.pt", artifacts_dir), map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        self.max_len = int(cfg.get("max_len", 40))
        self.model = AddressEncoder(
            d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
            d_out=cfg["d_out"], dropout=0.0, n_comuna=ckpt["n_comuna"],
            n_barrio=ckpt["n_barrio"], max_len=self.max_len,
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(self.device)
        self.xy_mean = np.asarray(ckpt["xy_mean"], dtype="float64")
        self.xy_std = np.asarray(ckpt["xy_std"], dtype="float64")

        emb = torch.load(model_path("catastro_emb.pt", artifacts_dir), map_location="cpu")
        self.doc_emb = emb.to(self.device)
        if docs is None:
            # An experiment dir may omit the model-independent cadastral index.
            docs = pd.read_parquet(shared_path("catastro_docs.parquet", artifacts_dir))
        self.docs = docs.reset_index(drop=True)
        assert len(self.docs) == self.doc_emb.shape[0], "docs / embedding mismatch"
        self._direccion = self.docs["direccion"].to_numpy()
        self._predial = self.docs["numero_predial_nacional"].to_numpy()
        self._manzana = self.docs["manzana"].to_numpy()
        self._comuna = self.docs["comuna"].to_numpy()
        self._barrio = self.docs["barrio_code"].to_numpy()
        self._lat = self.docs["centroid_lat"].to_numpy()
        self._lon = self.docs["centroid_lon"].to_numpy()
        self._n_predios = self.docs["n_predios"].to_numpy()
        self._manzana_nunique = load_manzana_nunique(self.docs)
        if {"x_m", "y_m"}.issubset(self.docs.columns):
            self._x_m = self.docs["x_m"].to_numpy()
            self._y_m = self.docs["y_m"].to_numpy()
        else:
            self._x_m, self._y_m = to_metric(self._lon, self._lat)
        # exact-lookup table used by the rule-only baseline and as a fast path
        self._exact = {}
        for i, d in enumerate(self._direccion):
            self._exact.setdefault(match_key(str(d)), i)
        self._parsed_docs: dict[int, object] = {}

    # -- encoding ---------------------------------------------------------
    @torch.no_grad()
    def encode_queries(self, texts: list[str], role: int = 0, batch: int = 1024) -> torch.Tensor:
        out = torch.empty((len(texts), self.doc_emb.shape[1]), dtype=torch.float16, device=self.device)
        for i in range(0, len(texts), batch):
            ids = torch.from_numpy(
                encode_batch(texts[i : i + batch], self.max_len).astype(np.int64)
            ).to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(self.device == "cuda")):
                emb = self.model.embed(ids, role=role)
            out[i : i + batch] = F.normalize(emb.float(), dim=-1).half()
        return out

    @torch.no_grad()
    def predict_xy(self, texts: list[str], batch: int = 1024) -> np.ndarray:
        """Coordinate predicted by the auxiliary regression head, in metres (EPSG:3116).

        This is the spatial prior the model learned during training; the reranker
        uses it to damp candidates that match the text but sit in the wrong part
        of the city.
        """
        out = np.zeros((len(texts), 2), dtype=np.float64)
        for i in range(0, len(texts), batch):
            ids = torch.from_numpy(
                encode_batch(texts[i : i + batch], self.max_len).astype(np.int64)
            ).to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(self.device == "cuda")):
                pooled = self.model.pool(ids, 0)
                xy = self.model.coord_head(pooled)
            out[i : i + batch] = xy.float().cpu().numpy()
        return out * self.xy_std + self.xy_mean

    @torch.no_grad()
    def search(self, q_emb: torch.Tensor, k: int = 20, chunk: int = 512):
        """Top-k cosine search against the full index (chunked fp16 matmul)."""
        sims_out, idx_out = [], []
        for i in range(0, q_emb.shape[0], chunk):
            sims = q_emb[i : i + chunk] @ self.doc_emb.T
            top = sims.topk(min(k, sims.shape[1]), dim=-1)
            sims_out.append(top.values.float().cpu().numpy())
            idx_out.append(top.indices.cpu().numpy())
        return np.concatenate(sims_out), np.concatenate(idx_out)

    def _doc_parse(self, i: int):
        p = self._parsed_docs.get(i)
        if p is None:
            p = parse_address(str(self._direccion[i]))
            self._parsed_docs[i] = p
        return p

    # -- score components -------------------------------------------------
    def score_components(self, raws, k: int = 20) -> dict:
        """Per-candidate reranking components, without applying any weights.

        Returns ``{doc, sim, fuzz, struct, geo}`` arrays of shape ``(n, 2k)``
        (``doc == -1`` marks an empty slot) plus the parse results. Fusion weights
        can then be swept offline without recomputing the encoder or the search,
        which is what ``scripts/tune_threshold.py`` does.
        """
        raws = [None if r is None else str(r) for r in raws]
        parsed = [parse_address(r) for r in raws]
        canon_nocomp = [
            canonical(p, style="spaced", with_complement=False) if p.parse_ok else "" for p in parsed
        ]
        canon = [canonical(p, style="spaced", with_complement=True) if p.parse_ok else "" for p in parsed]
        prenorm = [_pre_normalize(r or "")[: self.max_len] for r in raws]
        q_text = [c or pn for c, pn in zip(canon, prenorm)]

        n = len(raws)
        e1 = self.encode_queries(q_text, role=0)
        e2 = self.encode_queries(prenorm, role=0)
        s1, i1 = self.search(e1, k=k)
        s2, i2 = self.search(e2, k=k)
        cand_idx = np.concatenate([i1, i2], axis=1)
        cand_sims = np.concatenate([s1, s2], axis=1)
        pred_xy = self.predict_xy(q_text)

        width = cand_idx.shape[1]
        out = {
            "doc": np.full((n, width), -1, dtype=np.int64),
            "sim": np.zeros((n, width), dtype=np.float32),
            "fuzz": np.zeros((n, width), dtype=np.float32),
            "struct": np.zeros((n, width), dtype=np.float32),
            "geo": np.zeros((n, width), dtype=np.float32),
        }
        for r in range(n):
            keep: dict[int, float] = {}
            for j, sim in zip(cand_idx[r], cand_sims[r]):
                keep[int(j)] = max(keep.get(int(j), -1.0), float(sim))
            px, py = pred_xy[r]
            text = canon_nocomp[r] or prenorm[r]
            for slot, (j, sim) in enumerate(keep.items()):
                target = str(self._direccion[j])
                out["doc"][r, slot] = j
                out["sim"][r, slot] = max(sim, 0.0)
                out["fuzz"][r, slot] = fuzz.token_set_ratio(text, target) / 100.0
                out["struct"][r, slot] = (
                    _structural_agreement(parsed[r], self._doc_parse(j)) if parsed[r].parse_ok else 0.0
                )
                dm = float(np.hypot(self._x_m[j] - px, self._y_m[j] - py))
                out["geo"][r, slot] = float(np.exp(-dm / GEO_TAU_M))
        out["parse_ok"] = np.array([p.parse_ok for p in parsed], dtype=bool)
        return out

    # -- main API ---------------------------------------------------------
    def normalize_batch(
        self, raws, k: int = 20, use_model: bool = True, rerank: bool = True,
        use_geo: bool = True, threshold: float | None = None,
        weights: dict | None = None,
    ) -> pd.DataFrame:
        cutoff = self.threshold if threshold is None else float(threshold)
        fuse_w = dict(self.weights if weights is None else weights)
        raws = [None if r is None else str(r) for r in raws]
        parsed = [parse_address(r) for r in raws]
        canon = [canonical(p, style="spaced", with_complement=True) if p.parse_ok else "" for p in parsed]
        canon_nocomp = [
            canonical(p, style="spaced", with_complement=False) if p.parse_ok else "" for p in parsed
        ]
        prenorm = [_pre_normalize(r or "")[: self.max_len] for r in raws]
        q_text = [c or pn for c, pn in zip(canon, prenorm)]

        n = len(raws)
        if use_model:
            e1 = self.encode_queries(q_text, role=0)
            e2 = self.encode_queries(prenorm, role=0)
            s1, i1 = self.search(e1, k=k)
            s2, i2 = self.search(e2, k=k)
            cand_sims = np.concatenate([s1, s2], axis=1)
            cand_idx = np.concatenate([i1, i2], axis=1)
        else:
            # rule-only baseline: exact key lookup, then rapidfuzz over the
            # addresses that share the same via/cross numbers.
            cand_idx, cand_sims = self._rule_candidates(parsed, canon_nocomp, k)

        pred_xy = (
            self.predict_xy(q_text) if (use_model and rerank and use_geo)
            else np.full((n, 2), np.nan)
        )

        rows = []
        for r in range(n):
            idxs, sims = cand_idx[r], cand_sims[r]
            keep: dict[int, float] = {}
            for j, s in zip(idxs, sims):
                if j < 0:
                    continue
                keep[int(j)] = max(keep.get(int(j), -1.0), float(s))
            if not keep:
                rows.append(self._empty_row(raws[r], canon[r], parsed[r]))
                continue
            cand = list(keep.items())
            if rerank:
                scored = []
                px, py = pred_xy[r]
                have_geo = np.isfinite(px) and np.isfinite(py)
                w = dict(fuse_w)
                if not have_geo:
                    # Redistribute the geographic weight over the other terms so
                    # the score stays on the same 0-1 scale.
                    share = w.pop("geo") / 3.0
                    for key in ("sim", "fuzz", "struct"):
                        w[key] += share
                for j, sim in cand:
                    target = str(self._direccion[j])
                    fz = fuzz.token_set_ratio(canon_nocomp[r] or prenorm[r], target) / 100.0
                    st = _structural_agreement(parsed[r], self._doc_parse(j)) if parsed[r].parse_ok else 0.0
                    fused = (
                        w["sim"] * max(sim, 0.0) + w["fuzz"] * fz + w["struct"] * st
                    )
                    geo = float("nan")
                    if have_geo:
                        dm = float(np.hypot(self._x_m[j] - px, self._y_m[j] - py))
                        geo = float(np.exp(-dm / GEO_TAU_M))
                        fused += w["geo"] * geo
                    scored.append((fused, sim, fz, st, j, geo))
                scored.sort(key=lambda t: t[0], reverse=True)
            else:
                scored = [
                    (max(sim, 0.0), sim, float("nan"), float("nan"), j, float("nan"))
                    for j, sim in cand
                ]
                scored.sort(key=lambda t: t[0], reverse=True)
            rows.append(self._row(raws[r], canon[r], parsed[r], scored, k, cutoff))
        return pd.DataFrame(rows)

    def normalize_address(self, raw: str, k: int = 20) -> dict:
        """Normalize a single free-form address.

        Always check ``matched`` / ``confidence`` before using the result: a
        low-confidence row still carries the best available candidate, including
        its coordinates, and for non-address input that candidate is meaningless.
        """
        return self.normalize_batch([raw], k=k).iloc[0].to_dict()

    # -- helpers ----------------------------------------------------------
    def _rule_candidates(self, parsed, canon_nocomp, k: int):
        """Baseline candidate generation without the neural model."""
        from rapidfuzz import process

        idx = np.full((len(parsed), k), -1, dtype=np.int64)
        sims = np.zeros((len(parsed), k), dtype=np.float32)
        if not hasattr(self, "_num_buckets"):
            # Light regex bucketing (`KR 26 ...` -> ('KR', '26')): a full parse of
            # all 330k cadastral addresses would cost ~20 s and 150 MB for nothing.
            import re as _re

            from .parser import VIA_TYPE_CANON

            pat = _re.compile(r"^([A-Z]+)\s+(\d+)")
            buckets: dict[tuple, list[int]] = {}
            for i, d in enumerate(self._direccion):
                m = pat.match(str(d))
                if m is None:
                    continue
                vt = VIA_TYPE_CANON.get(m.group(1))
                if vt is None:
                    continue
                buckets.setdefault((vt, str(int(m.group(2)))), []).append(i)
            self._num_buckets = buckets
        for r, p in enumerate(parsed):
            key = match_key(canon_nocomp[r])
            hit = self._exact.get(key)
            cands: list[int] = []
            if hit is not None:
                cands.append(hit)
            if p.parse_ok:
                cands.extend(self._num_buckets.get((p.via_type, p.via_number), [])[:4000])
            if not cands:
                continue
            choices = {i: str(self._direccion[i]) for i in dict.fromkeys(cands)}
            best = process.extract(
                canon_nocomp[r] or "", choices, scorer=fuzz.token_set_ratio, limit=k
            )
            for slot, (_, score, i) in enumerate(best):
                idx[r, slot] = i
                sims[r, slot] = score / 100.0
        return idx, sims

    def _empty_row(self, raw, canon, parsed) -> dict:
        return {
            "raw_address": raw,
            "parse_ok": parsed.parse_ok,
            "rule_canonical": canon,
            "canonical_address": canon,
            "numero_predial_nacional": None,
            "manzana": None,
            "comuna": None,
            "barrio_code": None,
            "lat": np.nan,
            "lon": np.nan,
            "confidence": 0.0,
            "model_similarity": np.nan,
            "fuzz_score": np.nan,
            "struct_score": np.nan,
            "geo_score": np.nan,
            "n_predios_at_address": 0,
            "matched": False,
            "top5_doc_ids": [],
            "topk_doc_ids": [],
            "parse_notes": ";".join(parsed.notes),
        }

    def _row(self, raw, canon, parsed, scored, k, cutoff: float | None = None) -> dict:
        cutoff = self.threshold if cutoff is None else cutoff
        fused, sim, fz, st, j, geo = scored[0]
        return {
            "raw_address": raw,
            "parse_ok": parsed.parse_ok,
            "rule_canonical": canon,
            "canonical_address": str(self._direccion[j]) if fused >= cutoff or not canon else canon,
            "matched_cadastral_address": str(self._direccion[j]),
            "numero_predial_nacional": str(self._predial[j]),
            "manzana": str(self._manzana[j]),
            "comuna": str(self._comuna[j]),
            "barrio_code": str(self._barrio[j]),
            "lat": float(self._lat[j]),
            "lon": float(self._lon[j]),
            "confidence": float(min(max(fused, 0.0), 1.0)),
            "model_similarity": float(sim),
            "fuzz_score": float(fz),
            "struct_score": float(st),
            "geo_score": float(geo),
            "n_predios_at_address": int(self._n_predios[j]),
            "matched": bool(fused >= cutoff),
            "top5_doc_ids": [int(s[4]) for s in scored[:5]],
            "topk_doc_ids": [int(s[4]) for s in scored[:k]],
            "parse_notes": ";".join(parsed.notes),
        }
