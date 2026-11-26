"""Analysis C - does the donor distance predict that the answer changes?

Per the pre-registration committed in a32347b, before any number of C existed.

The unit is one swapped row × checkpoint × seed:

    flip = 1[ prediction under the swap != prediction under C0 ]

compared with `answers_match`, the same ruler that decides `correct` behind
accuracy, VDD and CCR - so "yes" against "Yes." is not a flip. Using raw string
inequality would count the scoring rule's own tolerance as a change of answer.

`d` for a row is the distance of its (original, donor) image pair, taken from
the manifests and the C0 embedding cache. The derangement is over images, so
every row of an image carries the same `d`: the effective sample on the distance
axis is the image, not the row. That is why every confidence interval is
reported twice - resampling rows, and resampling images as clusters. Where the
two disagree, the row-wise one is the optimistic one and the image-wise one is
the honest one.

Rows of C2, C5 and C6 are pooled for the primary fit, because the unit is the
swapped row and `d` is its donor distance whichever condition produced it. That
pooling is what makes the analysis about distance rather than about condition;
the per-condition fits are reported beside it.

G3: the logistic coefficient of `flip ~ d` is positive with a confidence
interval excluding zero in at least 12 of the 15 configuration × benchmark
cells. Seeds are never pooled - a cell passes only if all three of its seeds do.

The logistic fit is IRLS with a Wald interval from the inverse Hessian; there is
no statsmodels in this environment and the model is two-parameter.
"""
from __future__ import annotations

import collections
import hashlib
import json
import socket
import sys
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.analysis.ccr import answers_match, load_predictions, question_key  # noqa: E402

CACHE = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "embeddings"
RES = REPO / "results"
OUT = REPO / "docs/v4/analysis_c.json"

SEED = 20260917
N_BOOT = 1000
N_BINS = 5
RULERS = ["medsiglip", "dinov3", "biomedclip", "dinov2", "colorhist"]
BENCH = ["KVASIR", "SLAKE", "VQARAD"]
CONFIGS = {"KVASIR": ["M3AE", "QWEN2VL_DORA", "QWEN2VL_LORA", "SMOLVLM_DORA",
                      "SMOLVLM_LORA"],
           "SLAKE": ["M3AE_SCRATCH", "QWEN2VL_DORA", "QWEN2VL_LORA", "SMOLVLM_DORA",
                     "SMOLVLM_LORA"],
           "VQARAD": ["M3AE_SCRATCH", "QWEN2VL_DORA", "QWEN2VL_LORA", "SMOLVLM_DORA",
                      "SMOLVLM_LORA"]}
REFERENCE = ["MEDVLTHINKER_ZEROSHOT", "M0_question_only"]   # drawn, never judged
# Kvasir names its first seed `_seed0`; SLAKE and VQA-RAD name theirs with no
# suffix at all. Resolving this per benchmark instead of assuming one convention
# is what keeps ten of the fifteen cells from silently losing a seed.
SEEDS = ["_seed0", "_seed1", "_seed2"]


def seed_suffixes(bench, cfg, res):
    out = []
    for s in SEEDS:
        if (res / ("%s_%s%s_C0.json" % (cfg, bench, s))).exists():
            out.append(s)
        elif s == "_seed0" and (res / ("%s_%s_C0.json" % (cfg, bench))).exists():
            out.append("")
    return out
CONDS = {"KVASIR": ["C2", "C5", "C6"], "SLAKE": ["C2", "C5", "C6"],
         "VQARAD": ["C2", "C6"]}
MANIFEST = {
    ("KVASIR", "C2"): "C2_manifest_KVASIR.json",
    ("KVASIR", "C5"): "C5_manifest_KVASIR_clean.json",
    ("KVASIR", "C6"): "C6_manifest_KVASIR_clean.json",
    ("SLAKE", "C2"): "C2_manifest_SLAKE.json",
    ("SLAKE", "C5"): "C5_manifest_SLAKE.json",
    ("SLAKE", "C6"): "C6_manifest_SLAKE.json",
    ("VQARAD", "C2"): "C2_manifest_VQARAD.json",
    ("VQARAD", "C6"): "C6_manifest_VQARAD.json",
}


def rng_for(*parts):
    h = hashlib.sha256("|".join(str(p) for p in (SEED,) + parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def label(i):
    p = [x for x in str(i).replace("//", "/").split("/") if x]
    if p and p[-1].lower() in ("source.jpg", "source.png"):
        return p[-2] if len(p) > 1 else p[-1]
    return str(i)


SM = json.loads((REPO / "data/swapmap.json").read_text())


def qid_meta(bench):
    return {int(k): v for k, v in SM[bench]["qid"].items()}


def donor_map(bench, cond):
    d = json.loads((RES / MANIFEST[(bench, cond)]).read_text())
    rows = d["pairs"] if isinstance(d, dict) else d
    return {label(e["qid_image"]): label(e["donor_image"]) for e in rows}


def distances(bench, ruler, dm):
    ids = [label(i) for i in json.loads((CACHE / "image_ids.json").read_text())
           ["%s__C0" % bench]]
    X = np.load(CACHE / ("%s__%s.npy" % (bench, ruler))).astype(np.float64)
    vec = dict(zip(ids, X))
    out = {}
    for a, b in dm.items():
        if a in vec and b in vec:
            va, vb = vec[a], vec[b]
            out[a] = 1.0 - float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
    return out


def buckets(bench):
    if bench == "KVASIR":
        b = json.loads((REPO / "docs/v4/kvasir_procedure_bucket.json").read_text())
        return {label(k): v for k, v in b["bucket"].items()}
    if bench == "VQARAD":
        out = {}
        for name, f in (("chest", "vqarad_bucket_chest.json"),
                        ("abdomen", "vqarad_bucket_abd.json"),
                        ("head", "vqarad_bucket_head.json")):
            for i in json.loads((RES / f).read_text()):
                out[label(i)] = name
        return out
    out = {}
    for f in sorted(RES.glob("C6_manifest_SLAKE_*.json")):
        tag = f.stem.replace("C6_manifest_SLAKE_", "")
        for e in json.loads(f.read_text()):
            out[label(e["qid_image"])] = tag
    return out


def logistic(x, y, ridge=1e-6, iters=60):
    """IRLS for flip ~ 1 + d, with a Wald interval from the inverse Hessian."""
    X = np.column_stack([np.ones(len(x)), x])
    b = np.zeros(2)
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-9, None)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(2)
        g = X.T @ (y - p) - ridge * b
        step = np.linalg.solve(H, g)
        b = b + step
        if np.abs(step).max() < 1e-10:
            break
    eta = np.clip(X @ b, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    W = np.clip(p * (1 - p), 1e-9, None)
    cov = np.linalg.inv(X.T @ (X * W[:, None]) + ridge * np.eye(2))
    se = float(np.sqrt(cov[1, 1]))
    return {"intercept": float(b[0]), "coef": float(b[1]), "se": se,
            "ci_low": float(b[1] - 1.96 * se), "ci_high": float(b[1] + 1.96 * se),
            "positive_ci_excludes_zero": bool(b[1] > 0 and b[1] - 1.96 * se > 0)}


def bin_curve(d, flip, img, rng, n_bins=N_BINS, n_boot=N_BOOT):
    """P(flip | d) in quantile bins, with CI by row and by image cluster."""
    edges = np.quantile(d, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-12
    idx = np.clip(np.digitize(d, edges[1:-1]), 0, n_bins - 1)
    uniq = np.unique(img)
    by_img = {u: np.where(img == u)[0] for u in uniq}
    out = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            out.append(None)
            continue
        rows, f = np.where(m)[0], flip[m]
        # row bootstrap
        br = [f[rng.integers(0, len(f), len(f))].mean() for _ in range(n_boot)]
        # cluster bootstrap over the images present in the bin
        imgs = np.unique(img[m])
        bi = []
        for _ in range(n_boot):
            pick = imgs[rng.integers(0, len(imgs), len(imgs))]
            sel = np.concatenate([by_img[u] for u in pick])
            sel = sel[m[sel]]
            bi.append(flip[sel].mean() if len(sel) else np.nan)
        bi = np.array(bi)
        bi = bi[~np.isnan(bi)]
        out.append({"bin": b, "n_rows": int(m.sum()), "n_images": int(len(imgs)),
                    "d_mean": float(d[m].mean()), "d_lo": float(edges[b]),
                    "d_hi": float(edges[b + 1]), "flip_rate": float(f.mean()),
                    "ci_row": [float(np.quantile(br, .025)), float(np.quantile(br, .975))],
                    "ci_image": [float(np.quantile(bi, .025)),
                                 float(np.quantile(bi, .975))] if len(bi) else None})
    return out


def rows_for(bench, prefix, ruler, restrict_bucket=False):
    """One row per (qid, condition) with its flip and its donor distance."""
    QM = qid_meta(bench)
    bk = buckets(bench) if restrict_bucket else None
    c0 = load_predictions(str(RES), prefix, "C0")
    if c0 is None:
        return None
    d_all, f_all, img_all, cond_all, qid_all = [], [], [], [], []
    for cond in CONDS[bench]:
        cx = load_predictions(str(RES), prefix, cond)
        if cx is None:
            continue
        dm = donor_map(bench, cond)
        dist = distances(bench, ruler, dm)
        for qid, meta in QM.items():
            im = label(meta["img"])
            if im not in dm or im not in dist or qid not in c0 or qid not in cx:
                continue
            if bk is not None:
                don = dm[im]
                if im not in bk or don not in bk or bk[im] != bk[don]:
                    continue
            flip = not answers_match(cx[qid]["predicted"], [c0[qid]["predicted"]])
            d_all.append(dist[im])
            f_all.append(float(flip))
            img_all.append(im)
            cond_all.append(cond)
            qid_all.append(qid)
    if not d_all:
        return None
    return (np.asarray(d_all), np.asarray(f_all), np.asarray(img_all),
            np.asarray(cond_all), np.asarray(qid_all))


def main():
    out = json.loads(OUT.read_text()) if OUT.exists() else {
        "host": os.environ.get("RUN_HOST", "unspecified"), "seed": SEED, "n_boot": N_BOOT,
        "n_bins": N_BINS, "flip_rule": "answers_match against the C0 prediction",
        "pooling": "rows of C2/C5/C6 pooled; per-condition fits reported too",
        "cells": {}}
    out.setdefault("cells", {})

    for bench in BENCH:
        for cfg in CONFIGS[bench] + REFERENCE:
            for s in seed_suffixes(bench, cfg, RES):
                prefix = "%s_%s%s" % (cfg, bench, s)
                for ruler in RULERS:
                    seed_name = s.strip("_") or "seed0"
                    key = "%s|%s|%s|%s" % (bench, cfg, seed_name, ruler)
                    if key in out["cells"]:
                        continue
                    r = rows_for(bench, prefix, ruler)
                    if r is None:
                        continue
                    d, f, img, cond, _ = r
                    rng = rng_for(bench, cfg, s, ruler)
                    rec = {"benchmark": bench, "config": cfg, "seed": seed_name,
                           "ruler": ruler, "reference_only": cfg in REFERENCE,
                           "n_rows": int(len(d)), "n_images": int(len(set(img))),
                           "flip_rate": float(f.mean()),
                           "fit": logistic(d, f),
                           "curve": bin_curve(d, f, img, rng),
                           "by_condition": {}}
                    for c in np.unique(cond):
                        m = cond == c
                        if m.sum() > 20:
                            rec["by_condition"][str(c)] = {
                                "n_rows": int(m.sum()), "flip_rate": float(f[m].mean()),
                                "fit": logistic(d[m], f[m])}
                    rb = rows_for(bench, prefix, ruler, restrict_bucket=True)
                    if rb is not None and len(rb[0]) > 20:
                        rec["within_bucket"] = {
                            "n_rows": int(len(rb[0])), "flip_rate": float(rb[1].mean()),
                            "fit": logistic(rb[0], rb[1])}
                    out["cells"][key] = rec
                    OUT.write_text(json.dumps(out, indent=1))
                    fit = rec["fit"]
                    print("  %-7s %-22s %-6s %-11s n %6d  flip %.4f  coef %+8.3f "
                          "[%+.3f, %+.3f] %s"
                          % (bench, cfg, seed_name, ruler, len(d), f.mean(),
                             fit["coef"], fit["ci_low"], fit["ci_high"],
                             "+" if fit["positive_ci_excludes_zero"] else ""),
                          flush=True)

    print("\n  gravado %s  (%d celulas)" % (OUT, len(out["cells"])))


if __name__ == "__main__":
    main()
