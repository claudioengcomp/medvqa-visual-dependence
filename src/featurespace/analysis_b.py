"""Analysis B — donor distance, per the pre-registration of 17/09 and amendment B.1.

Per swapped pair (original image, donor image) of C2, C5 and C6:

    d = 1 - cos( e(x), e(x') )

on five rulers, with MedSigLIP primary and the colour histogram as the
counterfactual of contribution 2. Vectors are resolved out of the C0 cache by
the sha256 of the image bytes (amendment 7.7) and never recomputed: a swap
carries exactly the images of C0, so every vector B needs already exists.

The derangement is over images, so d is one value per pair of images. Both
summaries are kept: per image, and weighted by the number of questions the image
carries, since an image with eleven questions weighs eleven times as much in any
accuracy the paper reports.

G2, on the benchmarks with domain metadata: Wasserstein-1 between the d of the
near group (donor in the same bucket) and of the far group, with 1,000 shuffles
of the group labels, significant at p < 0.01. The sliced Wasserstein between the
donors' feature clouds of the two groups is reported without a verdict.

Amendment 7.6 applies: every cell is written as it completes and skipped on
restart, and each cell's generator is seeded from (SEED, benchmark, condition,
ruler) so the result does not depend on how many cells ran before it.
"""
from __future__ import annotations

import hashlib
import json
import socket
import sys
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "embeddings"
HASHES = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "image_hashes.json"
OUT = REPO / "docs/v4/analysis_b.json"

SEED = 20260917
N_PERM = 1000
N_SLICES = 200
RULERS = ["medsiglip", "dinov3", "biomedclip", "dinov2", "colorhist"]

MANIFESTS = {
    ("KVASIR", "C2"): "C2_manifest_KVASIR.json",
    ("KVASIR", "C5"): "C5_manifest_KVASIR_clean.json",
    ("KVASIR", "C6"): "C6_manifest_KVASIR_clean.json",
    ("SLAKE", "C2"): "C2_manifest_SLAKE.json",
    ("SLAKE", "C5"): "C5_manifest_SLAKE.json",
    ("SLAKE", "C6"): "C6_manifest_SLAKE.json",
    ("VQARAD", "C2"): "C2_manifest_VQARAD.json",
    ("VQARAD", "C6"): "C6_manifest_VQARAD.json",
}


def cell_rng(bench, cond, ruler):
    """Amendment 7.6: independent of how many cells ran before this one."""
    h = hashlib.sha256(("%d|%s|%s|%s" % (SEED, bench, cond, ruler)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def label(i):
    parts = [p for p in str(i).replace("//", "/").split("/") if p]
    if parts and parts[-1].lower() in ("source.jpg", "source.png"):
        return parts[-2] if len(parts) > 1 else parts[-1]
    return str(i)


def load_json(p):
    return json.loads(Path(p).read_text())


IDS = load_json(CACHE / "image_ids.json")
H = load_json(HASHES)


def vectors(bench, ruler):
    """image_id -> vector, with the sha256 of the bytes as the check (7.7)."""
    X = np.load(CACHE / ("%s__%s.npy" % (bench, ruler))).astype(np.float64)
    ids = [str(i) for i in IDS["%s__C0" % bench]]
    assert len(ids) == len(X), "%s %s: %d ids contra %d linhas" % (bench, ruler, len(ids), len(X))
    sha = dict(zip([str(i) for i in H[bench]["test"]["ids"]], H[bench]["test"]["sha"]))
    by_id, by_sha = {}, {}
    for i, v in zip(ids, X):
        by_id[label(i)] = v
        s = sha.get(i)
        if s is not None:
            by_sha.setdefault(s, v)
    return by_id, by_sha, {label(k): v for k, v in sha.items()}


def pairs_of(bench, cond):
    d = load_json(REPO / "results" / MANIFESTS[(bench, cond)])
    rows = d["pairs"] if isinstance(d, dict) else d
    m, weight = {}, {}
    for e in rows:
        a, b = label(e["qid_image"]), label(e["donor_image"])
        m[a] = b
        weight[a] = weight.get(a, 0) + 1
    return m, weight


def buckets(bench):
    R = REPO / "results"
    if bench == "KVASIR":
        b = load_json(REPO / "docs/v4/kvasir_procedure_bucket.json")["bucket"]
        return {label(k): v for k, v in b.items()}
    if bench == "VQARAD":
        out = {}
        for name, f in (("chest", "vqarad_bucket_chest.json"),
                        ("abdomen", "vqarad_bucket_abd.json"),
                        ("head", "vqarad_bucket_head.json")):
            for i in load_json(R / f):
                out[label(i)] = name
        return out
    out = {}
    for f in sorted(R.glob("C6_manifest_SLAKE_*.json")):
        tag = f.stem.replace("C6_manifest_SLAKE_", "")
        for e in load_json(f):
            out[label(e["qid_image"])] = tag
    return out


def w1(a, b):
    """Wasserstein-1 between two empirical samples, by quantile coupling."""
    n = max(len(a), len(b))
    qa = np.quantile(np.sort(a), np.linspace(0, 1, n))
    qb = np.quantile(np.sort(b), np.linspace(0, 1, n))
    return float(np.abs(qa - qb).mean())


def w1_perm(a, b, rng, n_perm=N_PERM):
    obs = w1(a, b)
    pool = np.concatenate([a, b])
    na = len(a)
    hit = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pool))
        if w1(pool[p[:na]], pool[p[na:]]) >= obs:
            hit += 1
    return obs, (hit + 1) / (n_perm + 1)


def swd(X, Y, rng, n_slices=N_SLICES):
    d = X.shape[1]
    V = rng.normal(size=(d, n_slices))
    V /= np.linalg.norm(V, axis=0, keepdims=True)
    a, b = np.sort(X @ V, axis=0), np.sort(Y @ V, axis=0)
    n = min(len(a), len(b))
    qa = a[np.linspace(0, len(a) - 1, n).astype(int)]
    qb = b[np.linspace(0, len(b) - 1, n).astype(int)]
    return float(np.abs(qa - qb).mean())


def summarise(v, w=None):
    v = np.asarray(v, dtype=np.float64)
    if len(v) == 0:
        return None
    s = {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std()),
         "median": float(np.median(v)),
         "q25": float(np.quantile(v, .25)), "q75": float(np.quantile(v, .75)),
         "min": float(v.min()), "max": float(v.max())}
    if w is not None:
        w = np.asarray(w, dtype=np.float64)
        s["weighted_mean"] = float((v * w).sum() / w.sum())
        o = np.argsort(v)
        c = np.cumsum(w[o]) / w.sum()
        s["weighted_median"] = float(v[o][np.searchsorted(c, 0.5)])
        s["questions"] = int(w.sum())
    return s


def main(benches):
    done = json.loads(OUT.read_text()) if OUT.exists() else {
        "seed": SEED, "n_perm": N_PERM, "n_slices": N_SLICES,
        "host": os.environ.get("RUN_HOST", "unspecified"), "cells": {}}
    done.setdefault("cells", {})

    for bench in benches:
        bk = buckets(bench)
        for cond in ("C2", "C5", "C6"):
            if (bench, cond) not in MANIFESTS:
                continue
            m, weight = pairs_of(bench, cond)
            for ruler in RULERS:
                key = "%s|%s|%s" % (bench, cond, ruler)
                if key in done["cells"]:
                    print("  %s ja feito, pulando" % key, flush=True)
                    continue
                by_id, by_sha, sha = vectors(bench, ruler)
                rng = cell_rng(bench, cond, ruler)

                d, grp, w, miss_vec, miss_bk = [], [], [], 0, 0
                donors_near, donors_far = [], []
                for a, b in m.items():
                    if a not in by_id or b not in by_id:
                        miss_vec += 1
                        continue
                    va, vb = by_id[a], by_id[b]
                    dist = 1.0 - float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
                    if a not in bk or b not in bk:
                        miss_bk += 1
                        continue
                    g = "near" if bk[a] == bk[b] else "far"
                    d.append(dist)
                    grp.append(g)
                    w.append(weight[a])
                    (donors_near if g == "near" else donors_far).append(vb)

                d = np.asarray(d)
                grp = np.asarray(grp)
                w = np.asarray(w, dtype=np.float64)
                near, far = d[grp == "near"], d[grp == "far"]
                wn, wf = w[grp == "near"], w[grp == "far"]

                rec = {"benchmark": bench, "condition": cond, "ruler": ruler,
                       "pairs_in_manifest": len(m), "classifiable": int(len(d)),
                       "pairs_without_vector": miss_vec,
                       "pairs_without_bucket": miss_bk,
                       "near_fraction": float((grp == "near").mean()) if len(d) else None,
                       "near": summarise(near, wn), "far": summarise(far, wf),
                       "all": summarise(d, w)}

                if len(near) >= 2 and len(far) >= 2:
                    obs, p = w1_perm(near, far, rng)
                    rec["w1"] = obs
                    rec["w1_p"] = p
                    rec["separated"] = bool(p < 0.01)
                    A = np.asarray(donors_near, dtype=np.float64)
                    B = np.asarray(donors_far, dtype=np.float64)
                    rec["donor_swd"] = swd(A, B, rng)
                else:
                    rec["w1"] = None
                    rec["w1_p"] = None
                    rec["separated"] = None
                    rec["donor_swd"] = None
                    rec["why_no_test"] = "um dos grupos tem menos de 2 pares"

                done["cells"][key] = rec
                OUT.write_text(json.dumps(done, indent=1))   # amendment 7.6
                print("  %-7s %-3s %-11s pares %5d  perto %.4f  W1 %s p %s  %s"
                      % (bench, cond, ruler, len(d),
                         rec["near_fraction"] if rec["near_fraction"] is not None else -1,
                         "%.5f" % rec["w1"] if rec["w1"] is not None else "  --  ",
                         "%.4f" % rec["w1_p"] if rec["w1_p"] is not None else " -- ",
                         "SEPARA" if rec["separated"] else ""), flush=True)

    print("\n  gravado %s  (%d celulas)" % (OUT, len(done["cells"])))


if __name__ == "__main__":
    main(sys.argv[1:] or ["SLAKE", "VQARAD"])
