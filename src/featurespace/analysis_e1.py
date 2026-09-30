"""E1c — where C7 lands, against the two predictions fixed before it ran.

C7 keeps every pixel and destroys the arrangement. The pre-registration bet two
things on that, and this script reports the measured value beside each:

  (i)  under the four content rulers, knn(C7) falls **between C3-50 and C1**;
  (ii) under the colour histogram, the per-item distance d(C0, C7) is **exactly
       zero** and knn_colour(C7) equals knn_colour(C0).

Prediction (ii) is a test of the generation, not a finding. A histogram counts
pixels and ignores where they are, so a non-zero value means the arrows were not
written as specified — a resize, an interpolation, a lossy re-encode — and the
defect is fixed before anything else is read. Reporting a non-zero colour
distance as a result about colour is the error this framing exists to prevent.

Computed per benchmark and ruler: knn to the training bank, the sliced
Wasserstein against the bank with the permutation test, and the per-item
displacement C0 to C7.
"""
from __future__ import annotations

import hashlib
import json
import socket
import sys
import os
from pathlib import Path

import numpy as np
from scipy import linalg   # noqa: F401  (kept for parity with analysis_a)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
CACHE = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "embeddings"
OUT = REPO / "docs/v4/analysis_e1.json"
RULERS = ["medsiglip", "dinov3", "biomedclip", "dinov2", "colorhist"]
CONTENT = ["medsiglip", "dinov3", "biomedclip", "dinov2"]
BENCH = ["KVASIR", "SLAKE", "VQARAD"]
K, N_SLICES, N_PERM, SEED = 10, 200, 1000, 20260918

G1 = json.loads((REPO / "docs/v4/g1_results.json").read_text())
REF = {(r["benchmark"], r["condition"], r["ruler"]): r
       for r in G1["figure3_relative_to_c0"]}
IDS = json.loads((CACHE / "image_ids.json").read_text())


def rng_for(*p):
    h = hashlib.sha256("|".join(str(x) for x in (SEED,) + p).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def load(b, cond, r):
    name = ("%s__%s.npy" % (b, r)) if cond == "C0" else ("%s__%s__%s.npy" % (b, cond, r))
    p = CACHE / name
    return np.load(p).astype(np.float64) if p.exists() else None


def canon(b, cond, r):
    """Rows ordered by image_id (amendment 7.7)."""
    X = load(b, cond, r)
    key = "%s__%s" % (b, cond)
    i = IDS.get(key) or IDS.get("%s__C0" % b)
    if X is None or i is None:
        return None
    o = np.argsort(np.asarray(i, dtype=str), kind="stable")
    return X[o]


def unit(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def knn_mean(X, bank, k=K, block=512):
    Z, B = unit(X), unit(bank)
    out = np.empty(len(Z))
    for i in range(0, len(Z), block):
        S = Z[i:i + block] @ B.T
        part = np.partition(-S, k - 1, axis=1)[:, :k]
        out[i:i + block] = 1.0 - (-part).mean(1)
    return float(out.mean())


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
    hit = 0
    for _ in range(n_perm):
        i = rng.permutation(len(Z))
        if swd(Z[i[:nx]], Z[i[nx:]], rng, n_slices=32) >= obs:
            hit += 1
    return obs, (hit + 1) / (n_perm + 1)


out = json.loads(OUT.read_text()) if OUT.exists() else {
    "host": os.environ.get("RUN_HOST", "unspecified"), "k": K, "n_perm": N_PERM, "seed": SEED,
    "prediction_i": "knn(C7) between C3-50 and C1 under the four content rulers",
    "prediction_ii": ("colour: per-item d(C0,C7) exactly 0 and knn(C7) = knn(C0); "
                      "a generation test, not a finding"),
    "cells": {}}
out.setdefault("cells", {})

print("%-7s %-11s %9s %9s %9s %9s  %s"
      % ("base", "regua", "knn C7", "knn C0", "knn C3-50", "knn C1", "previsao (i)"))
for b in BENCH:
    for r in RULERS:
        key = "%s|%s" % (b, r)
        if key in out["cells"]:
            continue
        bank = load(b, "TRAIN", r)
        c0, c7 = canon(b, "C0", r), canon(b, "C7", r)
        if bank is None or c0 is None or c7 is None:
            print("  %-7s %-11s ausente" % (b, r))
            continue
        rng = rng_for(b, r)
        kc7 = knn_mean(c7, bank)
        shift = 1.0 - (unit(c7) * unit(c0)).sum(1)
        sw, p = swd_perm(c7, bank, rng)
        k0 = REF.get((b, "C0", r), {}).get("knn_abs")
        k50 = REF.get((b, "C3-50", r), {}).get("knn_abs")
        k1 = REF.get((b, "C1", r), {}).get("knn_abs")
        rec = {"benchmark": b, "ruler": r, "knn_c7": kc7,
               "knn_c0": k0, "knn_c3_50": k50, "knn_c1": k1,
               "shift_mean": float(shift.mean()),
               "shift_max": float(np.abs(shift).max()),
               "swd": sw, "swd_p": p, "n": int(len(c7))}
        if r in CONTENT and None not in (k50, k1):
            rec["prediction_i_holds"] = bool(k50 <= kc7 <= k1)
        if r == "colorhist":
            # the test is whether the VECTORS are identical, not whether the
            # computed cosine is exactly 1: a cosine of a vector with itself
            # lands on 1 +/- 1e-16 in float64, so testing the arithmetic would
            # report a generation defect that is not there
            rec["prediction_ii_vectors_identical"] = bool(np.array_equal(c0, c7))
            rec["prediction_ii_max_abs_vector_diff"] = float(
                np.abs(c0.astype(np.float64) - c7.astype(np.float64)).max())
            rec["prediction_ii_shift_is_zero"] = rec["prediction_ii_vectors_identical"]
            rec["prediction_ii_knn_equals_c0"] = bool(k0 is not None
                                                      and abs(kc7 - k0) == 0.0)
            rec["prediction_ii_knn_rel_diff"] = (abs(kc7 - k0) / max(abs(k0), 1e-30)
                                                 if k0 is not None else None)
        out["cells"][key] = rec
        OUT.write_text(json.dumps(out, indent=1))
        verdict = ("SIM" if rec.get("prediction_i_holds") else
                   "nao" if r in CONTENT else "--")
        print("%-7s %-11s %9.4f %9s %9s %9s  %s"
              % (b, r, kc7,
                 "%.4f" % k0 if k0 is not None else "--",
                 "%.4f" % k50 if k50 is not None else "--",
                 "%.4f" % k1 if k1 is not None else "--", verdict), flush=True)

print("\n=== previsao (ii): teste de GERACAO, sob o histograma de cor ===")
for b in BENCH:
    c = out["cells"].get("%s|colorhist" % b)
    if not c:
        continue
    print("  %-7s deslocamento por item max %.3e  |  knn C7 %.6f vs C0 %.6f  (rel %.2e)  %s"
          % (b, c["shift_max"], c["knn_c7"], c["knn_c0"] or float("nan"),
             c["prediction_ii_knn_rel_diff"] or 0,
             "OK" if c.get("prediction_ii_vectors_identical") else
             "FALHA: defeito de geracao"))

ok_i = [c for c in out["cells"].values() if c.get("prediction_i_holds") is True]
no_i = [c for c in out["cells"].values() if c.get("prediction_i_holds") is False]
print("\n  previsao (i): %d de %d celulas de conteudo confirmam"
      % (len(ok_i), len(ok_i) + len(no_i)))
for c in no_i:
    print("    %-7s %-11s knn %.4f fora de [%.4f, %.4f]"
          % (c["benchmark"], c["ruler"], c["knn_c7"], c["knn_c3_50"], c["knn_c1"]))
out["prediction_i_confirmed"] = "%d of %d" % (len(ok_i), len(ok_i) + len(no_i))
OUT.write_text(json.dumps(out, indent=1))
print("\n  gravado %s" % OUT)
