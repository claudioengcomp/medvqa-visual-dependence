"""Under canonical order, is a swap the same matrix as C0?

A swap moves the image and its image_id together into another item's row. The
set of images is therefore unchanged and only the row order differs, so sorting
both arrays by image_id should line them up element for element. If it does,
every set statistic is bit-identical by construction, the permutation noise
disappears rather than being tolerated, and any departure at all is a re-encode.

Checked with array_equal, which admits no tolerance, and with the maximum
absolute element-wise difference for the cases where it fails.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

import numpy as np

CACHE = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "embeddings"
IDS = json.loads((CACHE / "image_ids.json").read_text())
RULERS = ["medsiglip", "dinov3", "biomedclip", "dinov2", "colorhist"]
BENCH = ["KVASIR", "SLAKE", "VQARAD"]
SWAPS = ["C2", "C5", "C6"]


def arr(b, c, r):
    n = ("%s__%s.npy" % (b, r)) if c == "C0" else ("%s__%s__%s.npy" % (b, c, r))
    p = CACHE / n
    return np.load(p) if p.exists() else None


def ids(b, c):
    return IDS.get("%s__%s" % (b, c)) or IDS.get("%s" % b)


def canon(X, i):
    o = np.argsort(np.array(i, dtype=object).astype(str), kind="stable")
    return X[o], [i[k] for k in o]


print("%-8s %-3s %-11s %-10s %s" % ("base", "cond", "regua", "identico?", "max |dif|"))
bad = []
for b in BENCH:
    i0 = ids(b, "C0")
    for r in RULERS:
        X0 = arr(b, "C0", r)
        if X0 is None or i0 is None:
            continue
        A, ia = canon(X0, i0)
        for c in SWAPS:
            X = arr(b, c, r)
            ic = ids(b, c)
            if X is None or ic is None:
                continue
            B, ib = canon(X, ic)
            same_ids = ia == ib
            eq = same_ids and np.array_equal(A, B)
            d = float(np.abs(A.astype(np.float64) - B.astype(np.float64)).max()) \
                if same_ids else float("nan")
            print("%-8s %-3s %-11s %-10s %.3e%s"
                  % (b, c, r, "sim" if eq else "NAO", d,
                     "" if same_ids else "   (ids nao batem)"))
            if not eq:
                bad.append((b, c, r, d))

print("\n  %s" % ("toda troca e bit-identica ao C0 sob ordem canonica"
                  if not bad else "%d celulas divergem" % len(bad)))
