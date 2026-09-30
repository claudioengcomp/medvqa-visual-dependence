"""Analysis A: how far each condition pushes the image from the training bank.

Four quantities per benchmark x condition x ruler:

  knn      mean cosine distance from each test image to its k = 10 nearest
           neighbours in the TRAINING bank (Sun et al. 2022). Per item, then
           summarised.
  shift    per-item displacement, cosine distance between the same item under
           C0 and under the condition. Zero for a swap only if the item kept
           its own image, which a derangement forbids -- so `shift` is large
           for swaps and says nothing about the manifold. It is reported
           because it separates "the image changed" from "the distribution
           changed", which is the whole point of the 2x2 map.
  frechet  Frechet distance between the condition cloud and the training bank.
  swd      sliced Wasserstein between the same two clouds, with a permutation
           test.

G1 asks two things of these numbers:
  - the swaps (C2/C5/C6) must be indistinguishable from C0 in `knn`, because
    they are the same set of images permuted;
  - C1 > C3-75 > C3-50 > C3-25 in set distance.
A swap that departs from C0 in `knn` is a re-encode defect, not a finding.
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]

import numpy as np
from scipy import linalg

CACHE = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "embeddings"
K = 10
N_SLICES = 200
N_PERM = 1000
SEED = 20260917

RULERS = ["medsiglip", "dinov3", "biomedclip", "dinov2", "colorhist"]
# paper numbering; the arrow names are in the manifest
CONDS = ["C0", "C1", "C2", "C4_OCC_25", "C4_OCC_50", "C4_OCC_75",
         "C4_NOISE_25", "C4_NOISE_50", "C4_NOISE_75", "C5", "C6"]
PAPER = {"C4_OCC_25": "C3-25", "C4_OCC_50": "C3-50", "C4_OCC_75": "C3-75",
         "C4_NOISE_25": "C4-25", "C4_NOISE_50": "C4-50", "C4_NOISE_75": "C4-75"}
BENCH = ["KVASIR", "SLAKE", "VQARAD"]


def load(bench, cond, ruler):
    name = ("%s__%s.npy" % (bench, ruler)) if cond == "C0" else \
           ("%s__%s__%s.npy" % (bench, cond, ruler))
    p = CACHE / name
    return np.load(p).astype(np.float64) if p.exists() else None


def unit(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def knn_to_bank(X, bank, k=K, block=512):
    """Mean cosine distance to the k nearest neighbours of the bank."""
    Z, B = unit(X), unit(bank)
    out = np.empty(len(Z))
    for i in range(0, len(Z), block):
        S = Z[i:i + block] @ B.T
        part = np.partition(-S, k - 1, axis=1)[:, :k]
        out[i:i + block] = 1.0 - (-part).mean(1)
    return out


def frechet(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    cx, cy = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)
    covmean, _ = linalg.sqrtm(cx @ cy, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mx - my) ** 2).sum() + np.trace(cx + cy - 2 * covmean))


def swd(X, Y, rng, n_slices=N_SLICES):
    d = X.shape[1]
    V = rng.normal(size=(d, n_slices))
    V /= np.linalg.norm(V, axis=0, keepdims=True)
    a, b = np.sort(X @ V, axis=0), np.sort(Y @ V, axis=0)
    n = min(len(a), len(b))
    qa = a[np.linspace(0, len(a) - 1, n).astype(int)]
    qb = b[np.linspace(0, len(b) - 1, n).astype(int)]
    return float(np.abs(qa - qb).mean())


def swd_perm(X, Y, rng, n_perm=N_PERM):
    obs = swd(X, Y, rng)
    Z = np.vstack([X, Y])
    nx = len(X)
    count = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(Z))
        if swd(Z[idx[:nx]], Z[idx[nx:]], rng, n_slices=32) >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def main():
    rng = np.random.default_rng(SEED)
    out = {"k": K, "n_slices": N_SLICES, "n_perm": N_PERM, "seed": SEED,
           "paper_names": PAPER, "results": {}}

    for bench in BENCH:
        for ruler in RULERS:
            bank = load(bench, "TRAIN", ruler)
            c0 = load(bench, "C0", ruler)
            if bank is None or c0 is None:
                continue
            c0u = unit(c0)
            for cond in CONDS:
                X = load(bench, cond, ruler)
                if X is None:
                    continue
                kn = knn_to_bank(X, bank)
                shift = 1.0 - (unit(X) * c0u).sum(1)
                sw, p = swd_perm(X, bank, rng)
                key = "%s|%s|%s" % (bench, cond, ruler)
                out["results"][key] = {
                    "benchmark": bench, "condition": cond,
                    "paper_condition": PAPER.get(cond, cond), "ruler": ruler,
                    "n": int(len(X)),
                    "knn_mean": float(kn.mean()), "knn_sd": float(kn.std()),
                    "knn_median": float(np.median(kn)),
                    "shift_mean": float(shift.mean()), "shift_median": float(np.median(shift)),
                    "frechet": frechet(X, bank), "swd": sw, "swd_p": p}
                print("  %-7s %-12s %-11s knn=%.4f shift=%.4f frechet=%.1f swd=%.4f p=%.4f"
                      % (bench, PAPER.get(cond, cond), ruler, kn.mean(),
                         shift.mean(), out["results"][key]["frechet"], sw, p), flush=True)

    p = REPO / "docs/v4/analysis_a.json"
    p.write_text(json.dumps(out, indent=1))
    print("\n  gravado %s  (%d celulas)" % (p, len(out["results"])))


if __name__ == "__main__":
    main()
