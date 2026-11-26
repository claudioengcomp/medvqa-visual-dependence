"""Analysis D - does standardising by donor distance change what the ladder says?

A VDD compares benchmarks that were never asked the same question. Part of what
separates them is how far their swaps go: a benchmark whose donors are far will
drop more, whatever its models do. D asks whether correcting for that changes
the ordering, or only the numbers.

The construction, from §2.5 of the plan, fixed before any of it was computed:

  bins      common across benchmarks, by quantiles of the **pooled** distribution
            of d under MedSigLIP over the five benchmarks, so a bin means the
            same distance everywhere
  VDD_std   VDD computed inside each bin and averaged with **uniform weights**,
            which is what removes each benchmark's own distance profile
  W1        between the d distributions of every pair of benchmarks, to say how
            far apart the profiles were to begin with

  rule      the standardisation is useful if it **reorders at least one pair of
            benchmarks in at least 3 of the 5 configurations**; otherwise it is
            reported as a limitation and the raw VDD stands

The bins come from all five benchmarks, including PathVQA and HEAL, because the
scale of d is what they contribute. VDD_std is computed only where predictions
exist -- Kvasir-VQA-x1, SLAKE and VQA-RAD -- since a VDD needs a model.

Runs on host-A: HEAL's embeddings stay there under the licence of section 4, and
what leaves this script is the bin edges, the per-benchmark fractions and the
W1 table -- aggregates, which the licence allows.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.analysis.ccr import load_predictions   # noqa: E402
from src.featurespace import paths                        # noqa: E402

CACHE = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "embeddings"
OUT = REPO / "docs/v4/analysis_d.json"
RULER = "medsiglip"
N_BINS = 5
ALL_BENCH = ["KVASIR", "SLAKE", "VQARAD", "PATHVQA", "HEAL"]
MANIFEST = {
    ("KVASIR", "C2"): "C2_manifest_KVASIR.json",
    ("KVASIR", "C5"): "C5_manifest_KVASIR_clean.json",
    ("KVASIR", "C6"): "C6_manifest_KVASIR_clean.json",
    ("SLAKE", "C2"): "C2_manifest_SLAKE.json",
    ("SLAKE", "C5"): "C5_manifest_SLAKE.json",
    ("SLAKE", "C6"): "C6_manifest_SLAKE.json",
    ("VQARAD", "C2"): "C2_manifest_VQARAD.json",
    ("VQARAD", "C6"): "C6_manifest_VQARAD.json",
    ("PATHVQA", "C2"): "C2_manifest_PATHVQA.json",
    ("HEAL", "C2"): "C2_manifest_HEAL.json",
}
SM = json.loads((REPO / "data/swapmap.json").read_text())
IDS = json.loads((CACHE / "image_ids.json").read_text())


def label(i):
    p = [x for x in str(i).replace("//", "/").split("/") if x]
    if p and p[-1].lower() in ("source.jpg", "source.png"):
        return p[-2] if len(p) > 1 else p[-1]
    return str(i)


def pair_distances(b, cond):
    d = json.loads((paths.RESULTS / MANIFEST[(b, cond)]).read_text())
    rows = d["pairs"] if isinstance(d, dict) else d
    dm = {label(e["qid_image"]): label(e["donor_image"]) for e in rows}
    ids = [label(i) for i in IDS["%s__C0" % b]]
    # two naming conventions in the cache: the first three benchmarks were
    # embedded as BENCH__ruler.npy and PathVQA and HEAL as BENCH__C0__ruler.npy
    f = CACHE / ("%s__C0__%s.npy" % (b, RULER))
    if not f.exists():
        f = CACHE / ("%s__%s.npy" % (b, RULER))
    X = np.load(f).astype(np.float64)
    v = dict(zip(ids, X))
    return {a: 1.0 - float(v[a] @ v[c] / (np.linalg.norm(v[a]) * np.linalg.norm(v[c])))
            for a, c in dm.items() if a in v and c in v}


# the pooled distribution that defines the bins: C2 of every benchmark, the one
# unrestricted swap they all have
per_bench_c2 = {b: np.fromiter(pair_distances(b, "C2").values(), float)
                for b in ALL_BENCH}
pooled = np.concatenate([per_bench_c2[b] for b in ALL_BENCH])
edges = np.quantile(pooled, np.linspace(0, 1, N_BINS + 1))
edges[0] -= 1e-12
edges[-1] += 1e-12

out = {"ruler": RULER, "n_bins": N_BINS, "rule": "section 2.5 of the plan",
       "bin_edges": edges.tolist(), "pooled_n": int(len(pooled)),
       "bench_in_bins": {}, "w1": {}, "cells": {}}

print("=== bins comuns, quantis da distribuicao agrupada de d nas cinco bases ===")
print("  cortes: %s" % "  ".join("%.4f" % e for e in edges))
print("\n  %-8s %6s %s" % ("base", "pares", "fracao por bin"))
for b in ALL_BENCH:
    x = per_bench_c2[b]
    t = np.clip(np.digitize(x, edges[1:-1]), 0, N_BINS - 1)
    frac = [float((t == i).mean()) for i in range(N_BINS)]
    out["bench_in_bins"][b] = {"n": int(len(x)), "fraction": frac,
                               "d_mean": float(x.mean())}
    print("  %-8s %6d %s   d medio %.4f"
          % (b, len(x), "  ".join("%.3f" % f for f in frac), x.mean()))


def w1(a, b):
    n = max(len(a), len(b))
    qa = np.quantile(np.sort(a), np.linspace(0, 1, n))
    qb = np.quantile(np.sort(b), np.linspace(0, 1, n))
    return float(np.abs(qa - qb).mean())


print("\n=== W1 entre as distribuicoes de d, por par de bases ===")
for a, b in combinations(ALL_BENCH, 2):
    v = w1(per_bench_c2[a], per_bench_c2[b])
    out["w1"]["%s|%s" % (a, b)] = v
    print("  %-8s %-8s %.4f" % (a, b, v))

print("\n=== VDD bruta e VDD padronizada, por celula ===")
print("%-7s %-16s %-6s %9s %9s %s"
      % ("base", "config", "seed", "VDD", "VDD_std", "bins usados"))
for c in paths.cells():
    b, cfg, s = c["benchmark"], c["config"], c["seed"]
    QM = {int(k): v for k, v in SM[b]["qid"].items()}
    c0 = load_predictions(str(paths.RESULTS), c["prefix"], "C0")
    cx = load_predictions(str(paths.RESULTS), c["prefix"], "C2")
    if c0 is None or cx is None:
        continue
    dist = pair_distances(b, "C2")
    hit = [[0, 0] for _ in range(N_BINS)]
    for q, m in QM.items():
        im = label(m["img"])
        if im not in dist or q not in c0 or q not in cx:
            continue
        i = int(np.clip(np.digitize(dist[im], edges[1:-1]), 0, N_BINS - 1))
        hit[i][0] += c0[q]["correct"]
        hit[i][1] += cx[q]["correct"]
    per_bin = [((h[0] - h[1]) / h[0]) if h[0] else None for h in hit]
    used = [v for v in per_bin if v is not None]
    tot = [sum(h[0] for h in hit), sum(h[1] for h in hit)]
    vdd = (tot[0] - tot[1]) / tot[0] if tot[0] else None
    std = float(np.mean(used)) if used else None
    out["cells"]["%s|%s|%s" % (b, cfg, s)] = {
        "vdd": vdd, "vdd_std": std, "per_bin": per_bin,
        "bins_used": len(used)}
    print("%-7s %-16s %-6s %9s %9s %d"
          % (b, cfg, s, "%.4f" % vdd if vdd is not None else "--",
             "%.4f" % std if std is not None else "--", len(used)))

print("\n=== a regra: a padronizacao reordena algum par de bases? ===")
CONFIG_ALIAS = {"M3AE": "classifier", "M3AE_SCRATCH": "classifier"}
reorders, detail = 0, {}
configs = sorted({CONFIG_ALIAS.get(c["config"], c["config"]) for c in paths.cells()})
for cfg in configs:
    raw, std = {}, {}
    for b in ["KVASIR", "SLAKE", "VQARAD"]:
        v = [out["cells"][k] for k in out["cells"]
             if k.startswith("%s|" % b)
             and CONFIG_ALIAS.get(k.split("|")[1], k.split("|")[1]) == cfg]
        v = [x for x in v if x["vdd"] is not None and x["vdd_std"] is not None]
        if v:
            raw[b] = float(np.mean([x["vdd"] for x in v]))
            std[b] = float(np.mean([x["vdd_std"] for x in v]))
    swapped = [(a, c) for a, c in combinations(raw, 2)
               if (raw[a] > raw[c]) != (std[a] > std[c])]
    reorders += bool(swapped)
    detail[cfg] = {"raw": raw, "std": std,
                   "pairs_reordered": ["%s|%s" % p for p in swapped]}
    print("  %-14s bruta %s" % (cfg, "  ".join("%s %.4f" % (b, raw[b]) for b in raw)))
    print("  %-14s padr. %s   %s"
          % ("", "  ".join("%s %.4f" % (b, std[b]) for b in std),
             "REORDENA %s" % swapped if swapped else "mesma ordem"))
out["reordering"] = detail
out["configs_with_reordering"] = reorders
out["useful"] = bool(reorders >= 3)
print("\n  reordena em %d de %d configuracoes  ->  %s"
      % (reorders, len(configs),
         "padronizacao UTIL" if out["useful"] else "padronizacao nao util: limitacao"))

OUT.write_text(json.dumps(out, indent=1, default=float))
print("\n  gravado %s" % OUT)
