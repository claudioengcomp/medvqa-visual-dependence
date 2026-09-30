"""Distance to the single nearest training image, per test image.

The Frechet of 0.7 between the VQA-RAD test cloud and its training bank is the
duplication the content audit measured, seen through a ruler. It depends on k
and on the shape of both clouds, so it makes the point only to a reader who
already believes it. The 1-nearest-neighbour distance does not: if a test image
has a byte-identical copy in the training split, its embedding is that copy's
embedding and the distance is zero, for every ruler, with nothing to tune.

The count of exact zeros is therefore the duplicate audit run on a second
instrument. The expectation is set by sha256 against the TRAINING split alone --
the embedding bank is train only, while the audit compared test against train
AND validation, so the two headline numbers are not the same quantity and the
sha-against-train count is computed here rather than quoted.

Reported per benchmark, condition and ruler: mean and median 1-NN distance, the
number of images at distance zero, and whether that number matches sha256.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]

import numpy as np
import pyarrow as pa

CACHE = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "embeddings"
M3 = Path(os.environ.get("M3AE_ARROWS", str(REPO / "data/finetune_arrows")))
P0A = REPO / "data/finetune_arrows"

RULERS = ["medsiglip", "dinov3", "biomedclip", "dinov2", "colorhist"]
CONDS = ["C0", "C1", "C2", "C4_OCC_25", "C4_OCC_50", "C4_OCC_75",
         "C4_NOISE_25", "C4_NOISE_50", "C4_NOISE_75", "C5", "C6"]
PAPER = {"C4_OCC_25": "C3-25", "C4_OCC_50": "C3-50", "C4_OCC_75": "C3-75",
         "C4_NOISE_25": "C4-25", "C4_NOISE_50": "C4-50", "C4_NOISE_75": "C4-75"}
BENCH = ["KVASIR", "SLAKE", "VQARAD"]
ARROWS = {"KVASIR": (P0A, "vqa_kvasir_train_clean", "vqa_kvasir_test_clean"),
          "SLAKE": (M3, "vqa_slack_train", "vqa_slack_test"),
          "VQARAD": (M3, "vqa_vqa_rad_train", "vqa_vqa_rad_test")}
# a cosine distance below this is zero up to float64 accumulation in the matmul
ZERO = 1e-9


def load(bench, cond, ruler):
    name = ("%s__%s.npy" % (bench, ruler)) if cond in ("C0",) else \
           ("%s__%s__%s.npy" % (bench, cond, ruler))
    p = CACHE / name
    return np.load(p).astype(np.float64) if p.exists() else None


def unit(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def nn1(X, bank, block=256):
    Z, B = unit(X), unit(bank)
    out = np.empty(len(Z))
    for i in range(0, len(Z), block):
        out[i:i + block] = 1.0 - (Z[i:i + block] @ B.T).max(1)
    return out


def shas(path):
    with pa.memory_map(str(path), "rb") as s:
        t = pa.ipc.open_file(s).read_all()
    return [hashlib.sha256(b).hexdigest() for b in t.column("image").to_pylist()]


out = {"zero_tolerance": ZERO, "bank": "training split only", "benchmarks": {}}

for bench in BENCH:
    d, tr, te = ARROWS[bench]
    tr_sha, te_sha = set(shas(d / (tr + ".arrow"))), shas(d / (te + ".arrow"))
    expected = sum(1 for s in te_sha if s in tr_sha)
    out["benchmarks"][bench] = {"n_test": len(te_sha),
                                "sha256_copies_in_train": expected, "cells": {}}
    print("\n=== %s   sha256: %d de %d imagens de teste tem copia exata no treino"
          % (bench, expected, len(te_sha)))
    for ruler in RULERS:
        bank = load(bench, "TRAIN", ruler)
        if bank is None:
            continue
        for cond in CONDS:
            X = load(bench, cond, ruler)
            if X is None:
                continue
            v = nn1(X, bank)
            z = int((v <= ZERO).sum())
            rec = {"n": int(len(v)), "nn1_mean": float(v.mean()),
                   "nn1_median": float(np.median(v)), "nn1_min": float(v.min()),
                   "nn1_zero_count": z}
            if cond == "C0":
                rec["matches_sha256"] = z == expected
            out["benchmarks"][bench]["cells"]["%s|%s" % (PAPER.get(cond, cond), ruler)] = rec
            if cond == "C0":
                print("  C0  %-11s 1-NN medio %.4f  mediana %.4f  zeros %4d  %s"
                      % (ruler, v.mean(), np.median(v), z,
                         "confere com sha256" if z == expected else "NAO confere (sha %d)" % expected))

print("\n=== zeros no C0, por regua ===")
print("  %-8s %s" % ("base", "  ".join("%-11s" % r for r in RULERS)))
for bench in BENCH:
    c = out["benchmarks"][bench]["cells"]
    print("  %-8s %s   (sha256: %d)"
          % (bench, "  ".join("%-11d" % c["C0|%s" % r]["nn1_zero_count"]
                              if "C0|%s" % r in c else "%-11s" % "--" for r in RULERS),
             out["benchmarks"][bench]["sha256_copies_in_train"]))

(REPO / "docs/v4/nn1_to_bank.json").write_text(json.dumps(out, indent=1))
print("\n  gravado docs/v4/nn1_to_bank.json")
