"""Tune the fusion weights and the confidence threshold on the HELD-OUT CADASTRAL
validation split, plus synthetic non-address negatives.

Why this script exists
----------------------
The reranking weights and the `matched` cutoff are decisions, and decisions must
be made on data that is not the test set. The evaluation rows from the five Excel
workbooks ARE the test set, so they are **never** read here. Instead:

* **Positives** come from the documents that `build_training_arrays` held out
  (4 % of the cadastral addresses, never a training positive), in two pools of
  equal weight:
    - *in-distribution*: the exact noisy variants generated for them, replayed
      deterministically, i.e. the corruption level the model was trained on;
    - *hard*: fresh corruptions of the same held-out documents at
      `HARD_INTENSITY` times the nominal corruption strength. Field text is
      messier than the training augmentation, and on the in-distribution pool
      alone retrieval is already ~94 % top-1, so the reranking terms have almost
      nothing left to explain and the grid search degenerates onto the cosine
      term. The hard pool restores a regime where the text and structure terms
      are informative, without ever looking at the test set.
* **Negatives** are synthetic non-address strings (numeric identifiers, random
  letter soup, generic Spanish place descriptions, blanks) built from a fixed
  vocabulary inside this file. They have **no** correct document, so any match is
  a false positive. They exist because the cadastral split contains no junk at
  all, and a confidence gate whose whole purpose is to reject junk cannot be
  calibrated on data that has none.

Two design decisions are declared up front, before any test-set number is looked
at, and their cost is reported:

1. **`geo` weight is constrained to >= MIN_GEO_WEIGHT.** On in-distribution
   queries the retrieval head alone is already ~94 % top-1, so a grid search that
   maximizes in-distribution top-1 drives the geographic term to zero: it has
   nothing left to fix. The term exists to suppress geographically absurd
   candidates on far messier field text, i.e. to shorten the error tail rather
   than to move in-distribution top-1. The constraint keeps the mechanism; the
   script prints what the unconstrained optimum would have been and what the
   constraint costs on this split.
2. **The junk rate is a stated prior, not a fitted quantity.** The threshold is
   chosen at `JUNK_RATE`, and a sensitivity table over several junk rates is
   printed so the reader can see how much the choice depends on it.
3. **Ties are broken towards a balanced weight vector.** The top of the grid is
   flat to within one standard error, so a bare `argmax` lands on an arbitrary
   corner of the simplex and can zero out a component for no measurable gain.
   Among combinations whose pooled top-1 is within `TIE_TOLERANCE_SE` standard
   errors of the best, and which keep every component strictly positive, the most
   balanced one (largest minimum weight) is selected. The rule is declared here,
   before the selection runs, and the bare argmax is reported alongside.

Procedure
---------
1. Replay the augmentation deterministically; keep the validation queries.
2. Take a seeded subsample of positives and synthesize negatives.
3. Compute the per-candidate score components once (`score_components`).
4. Sweep the fusion weights over a coarse simplex grid, scored by top-1 document
   accuracy on the positives. Pure numpy, no re-encoding.
5. At the chosen weights, sweep the threshold on 0.50-0.95 / 0.01 and pick the one
   maximizing F1 of "matched implies the document is correct"
   (TP = matched and correct, FP = matched and wrong-or-junk, FN = correct but not
   matched).
6. Repeat step 5 per ablation variant on its own score scale, so their coverage
   numbers are comparable.
7. Write `artifacts/threshold_sweep.csv`, `artifacts/weight_sweep.csv` and
   `artifacts/tuning.json`, which `cali_address.inference.load_tuning` reads back.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
ART = os.path.join(ROOT, "artifacts")

from cali_address.augment import corrupt, corrupt_many  # noqa: E402
from cali_address.inference import GEO_TAU_M, AddressNormalizer  # noqa: E402
from cali_address.paths import resolve_artifacts_dir, shared_path  # noqa: E402

TUNE_SEED = 1234
N_TUNE = int(os.environ.get("N_TUNE", "10000"))
THRESHOLD_GRID = np.round(np.arange(0.50, 0.9501, 0.01), 4)
WEIGHT_STEP = 0.05
WEIGHT_NAMES = ("sim", "fuzz", "struct", "geo")

#: A-priori floor on the geographic weight (decision 1 above).
MIN_GEO_WEIGHT = 0.10
#: Corruption strength of the `hard` positive pool, relative to nominal.
HARD_INTENSITY = 2.0
#: Weight of the hard pool in the fusion-weight objective (0.5 = equal footing).
HARD_POOL_WEIGHT = 0.5
#: Tie-breaking tolerance for the weight grid, in standard errors (decision 3).
TIE_TOLERANCE_SE = 1.0
#: Share of non-address junk assumed in a production stream (decision 2 above).
JUNK_RATE = 0.20
JUNK_RATE_SENSITIVITY = (0.05, 0.10, 0.20, 0.30, 0.40)

# --------------------------------------------------------------------------
# Synthetic non-address generator. Built from a fixed vocabulary in this file so
# it cannot encode anything learned from the evaluation workbooks.
# --------------------------------------------------------------------------
_JUNK_WORDS = (
    "sin", "especificar", "no", "reporta", "dato", "informacion", "pendiente",
    "casa", "lado", "tienda", "esquina", "parte", "alta", "baja", "arriba",
    "abajo", "frente", "iglesia", "escuela", "colegio", "parque", "cancha",
    "puente", "quebrada", "loma", "sector", "zona", "predio", "finca", "camino",
    "trocha", "entrada", "salida", "kilometro", "antena", "torre", "tanque",
)
_JUNK_TEMPLATES = (
    "{w} {w} {w}",
    "{w} {w} de la {w}",
    "al {w} de la {w}",
    "{w} {w} {n}",
    "{n}",
    "{n} {n}",
    "{W}",
    "{W} {W}",
    "",
    "   ",
    "-",
    "#",
    "n/a",
    "0",
)


def make_junk(n: int, rng: random.Random) -> list[str]:
    out = []
    while len(out) < n:
        tpl = rng.choice(_JUNK_TEMPLATES)
        text = tpl
        while "{w}" in text:
            text = text.replace("{w}", rng.choice(_JUNK_WORDS), 1)
        while "{n}" in text:
            text = text.replace("{n}", str(rng.randrange(10 ** rng.randint(4, 9))), 1)
        while "{W}" in text:
            text = text.replace(
                "{W}", "".join(rng.choice("abcdefghijklmnopqrstuvwxyz")
                                for _ in range(rng.randint(4, 12))), 1
            )
        out.append(text)
    return out


def replay_validation_queries(
    docs: pd.DataFrame, n_variants: int, seed: int, q_is_val: np.ndarray, q_doc: np.ndarray
) -> tuple[list[str], np.ndarray]:
    """Regenerate the exact noisy variants and return only the validation ones.

    `build_training_arrays` walks the documents in order and draws `n_variants`
    corruptions per document from a single seeded `random.Random`, so replaying
    the same loop reproduces the same strings. The assertion guards the ordering
    assumption.
    """
    assert np.array_equal(q_doc, np.repeat(np.arange(len(docs)), n_variants)), \
        "query ordering is not doc-major; cannot replay the augmentation"
    rng = random.Random(seed)
    keep_idx = np.flatnonzero(q_is_val)
    keep_set = set(keep_idx.tolist())
    texts: dict[int, str] = {}
    pos = 0
    for canon in docs["direccion"].tolist():
        for variant in corrupt_many(canon, n_variants, rng):
            if pos in keep_set:
                texts[pos] = variant
            pos += 1
    assert pos == len(q_doc), (pos, len(q_doc))
    return [texts[i] for i in keep_idx], keep_idx


def simplex_grid(step: float) -> list[dict]:
    """All 4-way weight combinations on a `step` grid that sum to 1."""
    n = int(round(1.0 / step))
    out = []
    for a, b, c in itertools.product(range(n + 1), repeat=3):
        d = n - a - b - c
        if d < 0:
            continue
        w = dict(zip(WEIGHT_NAMES, (a * step, b * step, c * step, d * step)))
        if w["sim"] <= 0:          # the retrieval score must carry some weight
            continue
        out.append({k: round(v, 4) for k, v in w.items()})
    return out


def fuse(comp: dict, w: dict) -> tuple[np.ndarray, np.ndarray]:
    """Top-1 document id and its fused score for every query."""
    fused = (
        w["sim"] * comp["sim"] + w["fuzz"] * comp["fuzz"]
        + w["struct"] * comp["struct"] + w["geo"] * comp["geo"]
    )
    fused = np.where(comp["doc"] >= 0, fused, -np.inf)
    best = fused.argmax(axis=1)
    rows = np.arange(fused.shape[0])
    return comp["doc"][rows, best], fused[rows, best]


def sweep_threshold(
    score: np.ndarray, correct: np.ndarray, weight: np.ndarray, grid=THRESHOLD_GRID
) -> pd.DataFrame:
    """Precision / recall / F1 of `matched => correct`, with per-row weights.

    `weight` lets positives and junk be mixed at an arbitrary rate without
    resampling: every count is a weighted count.
    """
    rows = []
    total_correct = float((correct * weight).sum())
    total_w = float(weight.sum())
    for t in grid:
        matched = score >= t
        tp = float((matched & correct) @ weight)
        fp = float((matched & ~correct) @ weight)
        fn = total_correct - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision) and (precision + recall) > 0 else 0.0
        )
        rows.append(
            {
                "threshold": float(t),
                "matched_rate": float((matched @ weight) / total_w),
                "precision": precision, "recall": recall, "f1": f1,
                "tp_w": tp, "fp_w": fp, "fn_w": fn,
            }
        )
    return pd.DataFrame(rows)


def mixture_weights(is_junk: np.ndarray, junk_rate: float) -> np.ndarray:
    """Row weights that make the junk share exactly `junk_rate`."""
    n_pos = int((~is_junk).sum())
    n_junk = int(is_junk.sum())
    w = np.empty(len(is_junk), dtype=np.float64)
    w[~is_junk] = (1.0 - junk_rate) / max(n_pos, 1)
    w[is_junk] = junk_rate / max(n_junk, 1)
    return w


def main(argv: list[str] | None = None) -> None:
    global ART
    ap = argparse.ArgumentParser(description="Tune fusion weights + confidence threshold.")
    ap.add_argument("--artifacts-dir", default=None,
                    help="experiment directory (env CALI_ARTIFACTS_DIR; default artifacts/). Holds model.pt, "
                         "catastro_emb.pt and receives tuning.json + sweep CSVs; shared inputs fall back to artifacts/.")
    args = ap.parse_args(argv)
    ART = resolve_artifacts_dir(args.artifacts_dir)
    os.makedirs(ART, exist_ok=True)
    t0 = time.time()
    docs = pd.read_parquet(shared_path("catastro_docs.parquet", ART))
    data = np.load(shared_path("train_data.npz", ART), allow_pickle=True)
    q_doc, q_is_val = data["q_doc"], data["q_is_val"]
    n_variants = len(q_doc) // len(docs)
    print(f"documents={len(docs):,} variants/doc={n_variants} "
          f"val queries={int(q_is_val.sum()):,}", flush=True)

    texts, keep_idx = replay_validation_queries(docs, n_variants, 42, q_is_val, q_doc)
    truth_all = q_doc[keep_idx]
    print(f"replayed {len(texts):,} held-out validation queries in {time.time()-t0:.0f}s")
    print("  examples:", [texts[i] for i in (0, 1, 2)], flush=True)

    rng_np = np.random.default_rng(TUNE_SEED)
    pick = np.sort(rng_np.choice(len(texts), size=min(N_TUNE, len(texts)), replace=False))
    pos_texts = [texts[i] for i in pick]
    pos_truth = truth_all[pick]

    rng_py = random.Random(TUNE_SEED)

    # Hard positive pool: the same held-out documents, corrupted harder.
    canon_by_doc = docs["direccion"].to_numpy()
    hard_texts = [
        corrupt(str(canon_by_doc[d]), rng_py, intensity=HARD_INTENSITY) for d in pos_truth
    ]
    hard_truth = pos_truth.copy()
    print(f"in-distribution positives={len(pos_texts):,}  "
          f"hard positives (intensity {HARD_INTENSITY})={len(hard_texts):,}")
    print("  hard examples:", [repr(t) for t in hard_texts[:4]], flush=True)

    n_junk = max(int(round(len(pos_texts) * 0.5)), 200)   # pooled, reweighted later
    junk_texts = make_junk(n_junk, rng_py)
    print(f"synthetic non-address negatives={len(junk_texts):,}")
    print("  junk examples:", [repr(j) for j in junk_texts[:6]], flush=True)

    all_texts = pos_texts + hard_texts + junk_texts
    truth = np.concatenate([
        pos_truth, hard_truth, np.full(len(junk_texts), -1, dtype=np.int64)
    ])
    is_junk = np.concatenate([
        np.zeros(len(pos_texts) + len(hard_texts), dtype=bool),
        np.ones(len(junk_texts), dtype=bool),
    ])
    pool = np.concatenate([
        np.zeros(len(pos_texts), dtype=np.int8),          # 0 = in-distribution
        np.ones(len(hard_texts), dtype=np.int8),           # 1 = hard
        np.full(len(junk_texts), 2, dtype=np.int8),        # 2 = junk
    ])

    norm = AddressNormalizer(ART, docs=docs)
    print(f"normalizer loaded (current tuning source: {norm.tuning_source})", flush=True)
    t1 = time.time()
    comp = norm.score_components(all_texts, k=20)
    print(f"score components computed in {time.time()-t1:.0f}s (shape {comp['doc'].shape})",
          flush=True)

    # ------------------ 1. fusion weights (positives only) -----------------
    grid = simplex_grid(WEIGHT_STEP)
    print(f"\nsweeping {len(grid):,} weight combinations on the positives", flush=True)
    in_dist = pool == 0
    hard = pool == 1
    rows = []
    for w in grid:
        pred, _ = fuse(comp, w)
        ok = pred == truth
        acc_in = float(ok[in_dist].mean())
        acc_hard = float(ok[hard].mean())
        rows.append({
            **w,
            "top1_in_distribution": acc_in,
            "top1_hard": acc_hard,
            "top1_positives": (1 - HARD_POOL_WEIGHT) * acc_in + HARD_POOL_WEIGHT * acc_hard,
        })
    weight_tbl = pd.DataFrame(rows).sort_values("top1_positives", ascending=False)
    weight_tbl.to_csv(os.path.join(ART, "weight_sweep.csv"), index=False)

    uncon = weight_tbl.iloc[0]
    uncon_w = {k: round(float(uncon[k]), 4) for k in WEIGHT_NAMES}
    best_score = float(uncon["top1_positives"])

    # Standard error of the pooled accuracy estimate, used as the tie tolerance.
    n_pos_total = int(in_dist.sum() + hard.sum())
    se = float(np.sqrt(best_score * (1 - best_score) / max(n_pos_total, 1)))
    tol = TIE_TOLERANCE_SE * se

    allowed = weight_tbl[weight_tbl["geo"] >= MIN_GEO_WEIGHT - 1e-9]
    argmax_constrained = allowed.iloc[0]
    tie = allowed[allowed["top1_positives"] >= float(argmax_constrained["top1_positives"]) - tol]
    active = tie[(tie[list(WEIGHT_NAMES)] > 1e-9).all(axis=1)]
    if len(active):
        balance = active[list(WEIGHT_NAMES)].min(axis=1)
        best = active.loc[balance.idxmax()]
        rule = (f"most balanced of {len(active)} combinations within "
                f"{TIE_TOLERANCE_SE:.0f} SE ({tol:.4f}) that keep every component active")
    else:
        best = argmax_constrained
        rule = "constrained argmax (no all-active combination within tolerance)"
    best_w = {k: float(best[k]) for k in WEIGHT_NAMES}

    print(f"unconstrained argmax      : {uncon_w} -> pooled top1 {best_score:.4f}")
    print(f"pooled accuracy SE        : {se:.4f}  (tie tolerance {tol:.4f})")
    print(f"constrained argmax        : "
          f"{ {k: round(float(argmax_constrained[k]), 4) for k in WEIGHT_NAMES} } -> "
          f"{float(argmax_constrained['top1_positives']):.4f}")
    print(f"CHOSEN ({rule}):")
    print(f"  {best_w} -> pooled top1 {float(best['top1_positives']):.4f}")
    print(f"cost vs the unconstrained argmax: "
          f"{float(best['top1_positives']) - best_score:+.4f} top-1 "
          f"({abs(float(best['top1_positives']) - best_score) / max(se, 1e-9):.2f} SE)")
    print(weight_tbl.head(6).to_string(index=False))

    shipped = {"sim": 0.40, "fuzz": 0.26, "struct": 0.22, "geo": 0.12}
    pred_s, _ = fuse(comp, shipped)
    ok_s = pred_s == truth
    top1_shipped = ((1 - HARD_POOL_WEIGHT) * float(ok_s[in_dist].mean())
                    + HARD_POOL_WEIGHT * float(ok_s[hard].mean()))
    print(f"previously shipped (untuned) weights {shipped} -> pooled top1 {top1_shipped:.4f} "
          f"(in-dist {ok_s[in_dist].mean():.4f} / hard {ok_s[hard].mean():.4f})")
    print(f"chosen weights breakdown: in-dist {best['top1_in_distribution']:.4f} / "
          f"hard {best['top1_hard']:.4f}")

    # ------------------ 2. threshold at the chosen weights ------------------
    pred, score = fuse(comp, best_w)
    correct = (pred == truth) & ~is_junk
    weights_main = mixture_weights(is_junk, JUNK_RATE)
    sweep = sweep_threshold(score, correct, weights_main)
    sweep.insert(0, "variant", "full (+ coord head)")
    best_row = sweep.loc[sweep["f1"].idxmax()]
    best_threshold = float(best_row["threshold"])
    print(f"\nchosen threshold = {best_threshold:.2f} at junk rate {JUNK_RATE:.0%} "
          f"(F1={best_row['f1']:.4f} precision={best_row['precision']:.4f} "
          f"recall={best_row['recall']:.4f} matched_rate={best_row['matched_rate']:.4f})")
    print(sweep.iloc[::5][["threshold", "matched_rate", "precision", "recall", "f1"]]
          .to_string(index=False))

    sens = []
    for jr in JUNK_RATE_SENSITIVITY:
        sw = sweep_threshold(score, correct, mixture_weights(is_junk, jr))
        r = sw.loc[sw["f1"].idxmax()]
        sens.append({"junk_rate": jr, "best_threshold": float(r["threshold"]),
                     "f1": float(r["f1"]), "precision": float(r["precision"]),
                     "recall": float(r["recall"]), "matched_rate": float(r["matched_rate"])})
    sens_tbl = pd.DataFrame(sens)
    print("\nsensitivity of the chosen threshold to the assumed junk rate:")
    print(sens_tbl.to_string(index=False))
    sens_tbl.to_csv(os.path.join(ART, "threshold_junk_sensitivity.csv"), index=False)

    # ------------------ 3. one threshold per ablation variant ---------------
    geo_free = {**best_w, "geo": 0.0}
    tot = sum(geo_free.values())
    geo_free = {k: round(v / tot, 4) for k, v in geo_free.items()}
    variants = {
        "rule (no model)": ({"sim": 0.0, "fuzz": 0.55, "struct": 0.45, "geo": 0.0},
                            "rapidfuzz + structure, model not used"),
        "neural only": ({"sim": 1.0, "fuzz": 0.0, "struct": 0.0, "geo": 0.0},
                        "raw cosine similarity"),
        "neural + rerank": (geo_free, "fused score without the coordinate term"),
        "full (+ coord head)": (best_w, "fused score"),
    }
    sweeps, thresholds, variant_rows = [sweep], {"full (+ coord head)": best_threshold}, []
    variant_weights = {}
    for name, (w, desc) in variants.items():
        variant_weights[name] = w
        p_v, s_v = fuse(comp, w)
        c_v = (p_v == truth) & ~is_junk
        if name == "full (+ coord head)":
            sw, t_v = sweep, best_threshold
        else:
            sw = sweep_threshold(s_v, c_v, weights_main)
            sw.insert(0, "variant", name)
            sweeps.append(sw)
            t_v = float(sw.loc[sw["f1"].idxmax(), "threshold"])
            thresholds[name] = t_v
        r = sw.loc[sw["threshold"] == t_v].iloc[0]
        variant_rows.append({"variant": name, "score": desc, "threshold": t_v,
                             "f1": float(r["f1"]), "precision": float(r["precision"]),
                             "recall": float(r["recall"]),
                             "top1_in_distribution": float((p_v == truth)[in_dist].mean()),
                             "top1_hard": float((p_v == truth)[hard].mean())})
    print("\nper-variant thresholds (each on its own score scale):")
    print(pd.DataFrame(variant_rows).to_string(index=False))

    pd.concat(sweeps, ignore_index=True).to_csv(
        os.path.join(ART, "threshold_sweep.csv"), index=False
    )

    payload = {
        "method": {
            "positives": "held-out cadastral validation documents (4% of addresses, "
                         "never a training positive); their exact noisy variants replayed",
            "negatives": "synthetic non-address strings generated from a fixed vocabulary "
                         "inside scripts/tune_threshold.py",
            "queries_replayed": int(len(texts)),
            "positives_in_distribution": int(len(pos_texts)),
            "positives_hard": int(len(hard_texts)),
            "hard_intensity": HARD_INTENSITY,
            "hard_pool_weight": HARD_POOL_WEIGHT,
            "negatives_used": int(len(junk_texts)),
            "junk_rate_assumed": JUNK_RATE,
            "subsample_seed": TUNE_SEED,
            "augmentation_seed": 42,
            "weight_grid_step": WEIGHT_STEP,
            "weight_combinations": len(grid),
            "weight_objective": "top-1 document accuracy on the positives",
            "min_geo_weight_constraint": MIN_GEO_WEIGHT,
            "tie_tolerance_se": TIE_TOLERANCE_SE,
            "selection_rule": rule,
            "threshold_grid": [float(THRESHOLD_GRID[0]), float(THRESHOLD_GRID[-1]), 0.01],
            "threshold_objective": "F1 of 'matched => correct document'",
            "geo_tau_m": GEO_TAU_M,
            "evaluation_rows_used": "none (the five Excel workbooks are the test set)",
        },
        "chosen": {
            "weights": best_w, "threshold": best_threshold, "thresholds": thresholds,
            "variant_weights": variant_weights,
        },
        "diagnostics": {
            "top1_at_chosen_weights": float(best["top1_positives"]),
            "top1_in_distribution_at_chosen": float(best["top1_in_distribution"]),
            "top1_hard_at_chosen": float(best["top1_hard"]),
            "unconstrained_best_weights": uncon_w,
            "top1_at_unconstrained_best": best_score,
            "pooled_accuracy_se": se,
            "tie_tolerance": tol,
            "selection_cost_top1": float(best["top1_positives"]) - best_score,
            "selection_cost_in_se": abs(float(best["top1_positives"]) - best_score) / max(se, 1e-9),
            "previously_shipped_weights": shipped,
            "top1_at_previously_shipped_weights": top1_shipped,
            "f1_at_chosen_threshold": float(best_row["f1"]),
            "precision_at_chosen_threshold": float(best_row["precision"]),
            "recall_at_chosen_threshold": float(best_row["recall"]),
            "matched_rate_at_chosen_threshold": float(best_row["matched_rate"]),
            "junk_rate_sensitivity": sens,
            "variants": variant_rows,
        },
    }
    with open(os.path.join(ART, "tuning.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {ART}/tuning.json, threshold_sweep.csv, weight_sweep.csv, "
          f"threshold_junk_sensitivity.csv in {time.time()-t0:.0f}s")
    print(json.dumps(payload["chosen"], indent=2))


if __name__ == "__main__":
    main()
