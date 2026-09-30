"""Analysis D, with the support of each bin and both weightings.

The first pass reported that standardising by donor distance reverses Kvasir and
SLAKE in five of five configurations. It did not report how many images hold up
each bin, and SLAKE puts 0.010 of its 96 images in the second bin — one image.
Under uniform weights that bin carries a fifth of the standardised VDD. A
reversal resting on a bin of one is not a result, so the support is counted here
and the rule is declared before the numbers are read.

**Two weightings**, as §2.5 requires. The bins are quantiles of the pooled
distribution of d **over image pairs**, so each bin holds exactly a fifth of the
pooled pairs by construction and pooled *image* weights would be identical to
uniform ones. The weighting that differs is the pooled distribution **over
rows**: Kvasir carries 10,779 questions against SLAKE's 1,061, so the aggregate
question-level profile leans toward the near bins. That is the second weighting
computed here, and the ambiguity is recorded rather than resolved silently.

**Rule, declared here:** a bin holding fewer than 10 images of a benchmark is
marked, and a reordering counts only if it survives **both** weightings and the
bootstrap interval. Bootstrap is over images, 1,000 resamples, because rows of
one image share its distance.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
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
RULER, N_BINS, N_BOOT, MIN_IMAGES = "medsiglip", 5, 1000, 10
SEED = 20260917
ALL_BENCH = ["KVASIR", "SLAKE", "VQARAD", "PATHVQA", "HEAL"]
FIT = ["KVASIR", "SLAKE", "VQARAD"]
MANIFEST = {b: "C2_manifest_%s.json" % b for b in ALL_BENCH}
SM = json.loads((REPO / "data/swapmap.json").read_text())
IDS = json.loads((CACHE / "image_ids.json").read_text())


def label(i):
    p = [x for x in str(i).replace("//", "/").split("/") if x]
    if p and p[-1].lower() in ("source.jpg", "source.png"):
        return p[-2] if len(p) > 1 else p[-1]
    return str(i)


def dists(b):
    d = json.loads((paths.RESULTS / MANIFEST[b]).read_text())
    rows = d["pairs"] if isinstance(d, dict) else d
    dm = {label(e["qid_image"]): label(e["donor_image"]) for e in rows}
    ids = [label(i) for i in IDS["%s__C0" % b]]
    f = CACHE / ("%s__C0__%s.npy" % (b, RULER))
    if not f.exists():
        f = CACHE / ("%s__%s.npy" % (b, RULER))
    X = np.load(f).astype(np.float64)
    v = dict(zip(ids, X))
    return {a: 1.0 - float(v[a] @ v[c] / (np.linalg.norm(v[a]) * np.linalg.norm(v[c])))
            for a, c in dm.items() if a in v and c in v}


D = {b: dists(b) for b in ALL_BENCH}
edges = np.quantile(np.concatenate([np.fromiter(D[b].values(), float)
                                    for b in ALL_BENCH]),
                    np.linspace(0, 1, N_BINS + 1))
edges[0] -= 1e-12
edges[-1] += 1e-12


def binof(x):
    return int(np.clip(np.digitize(x, edges[1:-1]), 0, N_BINS - 1))


# support: images and rows per bin, per benchmark
support = {}
row_pool = np.zeros(N_BINS)
for b in ALL_BENCH:
    QM = {int(k): v for k, v in SM[b]["qid"].items()} if b in SM else {}
    qcount = defaultdict(int)
    for m in QM.values():
        qcount[label(m["img"])] += 1
    im, rw = np.zeros(N_BINS, int), np.zeros(N_BINS, int)
    for a, dv in D[b].items():
        i = binof(dv)
        im[i] += 1
        rw[i] += qcount.get(a, 0)
    support[b] = {"images": im.tolist(), "rows": rw.tolist(),
                  "thin_bins": [i for i in range(N_BINS) if im[i] < MIN_IMAGES]}
    row_pool += rw
w_pooled_rows = row_pool / row_pool.sum()

print("=== suporte de cada bin (bins sao quintis da distribuicao agrupada por par) ===")
print("  cortes: %s\n" % "  ".join("%.4f" % e for e in edges))
print("  %-8s %-30s %-30s %s" % ("base", "imagens por bin", "linhas por bin", "bins finos"))
for b in ALL_BENCH:
    s = support[b]
    print("  %-8s %-30s %-30s %s"
          % (b, " ".join("%5d" % x for x in s["images"]),
             " ".join("%6d" % x for x in s["rows"]),
             s["thin_bins"] if s["thin_bins"] else "-"))
print("\n  pesos uniformes:          %s" % "  ".join("%.3f" % (1 / N_BINS)
                                                     for _ in range(N_BINS)))
print("  pesos agregados por linha: %s" % "  ".join("%.3f" % x for x in w_pooled_rows))
print("  (pesos agregados por IMAGEM seriam 0.200 em todos, por construcao dos quintis)")

rng = np.random.default_rng(SEED)
out = json.loads(OUT.read_text()) if OUT.exists() else {}
out.update({"min_images_per_bin": MIN_IMAGES, "n_boot": N_BOOT,
            "weights_uniform": [1 / N_BINS] * N_BINS,
            "weights_pooled_rows": w_pooled_rows.tolist(),
            "note_pooled_image_weights": "identical to uniform: the bins are "
                                         "quintiles of the pooled pair distribution",
            "support": support, "cells_weighted": {}})

print("\n=== VDD padronizada sob as duas ponderacoes, com IC por imagem ===")
print("%-7s %-16s %-6s %9s %20s %20s"
      % ("base", "config", "seed", "VDD", "uniforme [IC]", "agregada linha [IC]"))
for c in paths.cells():
    b, cfg, s = c["benchmark"], c["config"], c["seed"]
    QM = {int(k): v for k, v in SM[b]["qid"].items()}
    c0 = load_predictions(str(paths.RESULTS), c["prefix"], "C0")
    cx = load_predictions(str(paths.RESULTS), c["prefix"], "C2")
    if c0 is None or cx is None:
        continue
    by_img = defaultdict(lambda: [0, 0, 0])      # bin, c0 hits, cx hits
    for q, m in QM.items():
        im = label(m["img"])
        if im not in D[b] or q not in c0 or q not in cx:
            continue
        r = by_img[im]
        r[0] = binof(D[b][im])
        r[1] += c0[q]["correct"]
        r[2] += cx[q]["correct"]
    imgs = list(by_img)

    def std_vdd(sample, w):
        tot = np.zeros((N_BINS, 2))
        for im in sample:
            r = by_img[im]
            tot[r[0], 0] += r[1]
            tot[r[0], 1] += r[2]
        v, ww = [], []
        for i in range(N_BINS):
            if tot[i, 0]:
                v.append((tot[i, 0] - tot[i, 1]) / tot[i, 0])
                ww.append(w[i])
        return float(np.average(v, weights=ww)) if v else None

    wu = np.full(N_BINS, 1 / N_BINS)
    obs_u, obs_p = std_vdd(imgs, wu), std_vdd(imgs, w_pooled_rows)
    tot = np.array([[by_img[i][1], by_img[i][2]] for i in imgs]).sum(0)
    vdd = float((tot[0] - tot[1]) / tot[0]) if tot[0] else None
    bu, bp = [], []
    for _ in range(N_BOOT):
        pick = [imgs[k] for k in rng.integers(0, len(imgs), len(imgs))]
        a, p = std_vdd(pick, wu), std_vdd(pick, w_pooled_rows)
        if a is not None:
            bu.append(a)
        if p is not None:
            bp.append(p)
    ciu = [float(np.quantile(bu, .025)), float(np.quantile(bu, .975))]
    cip = [float(np.quantile(bp, .025)), float(np.quantile(bp, .975))]
    out["cells_weighted"]["%s|%s|%s" % (b, cfg, s)] = {
        "vdd": vdd, "std_uniform": obs_u, "ci_uniform": ciu,
        "std_pooled_rows": obs_p, "ci_pooled_rows": cip,
        "thin_bins": support[b]["thin_bins"]}
    print("%-7s %-16s %-6s %9.4f  %.4f [%.4f,%.4f]  %.4f [%.4f,%.4f]"
          % (b, cfg, s, vdd, obs_u, ciu[0], ciu[1], obs_p, cip[0], cip[1]), flush=True)

print("\n=== a inversao Kvasir x SLAKE sobrevive as duas ponderacoes? ===")
ALIAS = {"M3AE": "classifier", "M3AE_SCRATCH": "classifier"}
cfgs = sorted({ALIAS.get(c["config"], c["config"]) for c in paths.cells()})
surv = {"uniform": 0, "pooled_rows": 0, "both_and_ci": 0}
for cfg in cfgs:
    g = {}
    for b in FIT:
        v = [out["cells_weighted"][k] for k in out["cells_weighted"]
             if k.startswith("%s|" % b)
             and ALIAS.get(k.split("|")[1], k.split("|")[1]) == cfg]
        if v:
            g[b] = v
    if "KVASIR" not in g or "SLAKE" not in g:
        continue
    raw = {b: np.mean([x["vdd"] for x in g[b]]) for b in g}
    m = lambda b, f: float(np.mean([x[f] for x in g[b]]))
    inv_u = (raw["KVASIR"] < raw["SLAKE"]) != (m("KVASIR", "std_uniform")
                                               < m("SLAKE", "std_uniform"))
    inv_p = (raw["KVASIR"] < raw["SLAKE"]) != (m("KVASIR", "std_pooled_rows")
                                               < m("SLAKE", "std_pooled_rows"))
    # the interval test: the two must not overlap under the weighting
    lo_k = min(x["ci_uniform"][0] for x in g["KVASIR"])
    hi_s = max(x["ci_uniform"][1] for x in g["SLAKE"])
    sep_u = lo_k > hi_s
    lo_k2 = min(x["ci_pooled_rows"][0] for x in g["KVASIR"])
    hi_s2 = max(x["ci_pooled_rows"][1] for x in g["SLAKE"])
    sep_p = lo_k2 > hi_s2
    surv["uniform"] += inv_u
    surv["pooled_rows"] += inv_p
    surv["both_and_ci"] += bool(inv_u and inv_p and sep_u and sep_p)
    print("  %-14s bruta K %.4f S %.4f | unif K %.4f S %.4f %s | agreg K %.4f S %.4f %s"
          % (cfg, raw["KVASIR"], raw["SLAKE"],
             m("KVASIR", "std_uniform"), m("SLAKE", "std_uniform"),
             "INVERTE" if inv_u else "mantem",
             m("KVASIR", "std_pooled_rows"), m("SLAKE", "std_pooled_rows"),
             "INVERTE" if inv_p else "mantem"))
out["reversal_survival"] = surv
print("\n  inverte sob pesos uniformes: %d de %d" % (surv["uniform"], len(cfgs)))
print("  inverte sob pesos agregados por linha: %d de %d" % (surv["pooled_rows"], len(cfgs)))
print("  inverte nas duas E com ICs separados: %d de %d" % (surv["both_and_ci"], len(cfgs)))
out["useful_strict"] = bool(surv["both_and_ci"] >= 3)
print("\n  regra estrita (as duas ponderacoes + IC, >= 3 de 5): %s"
      % ("padronizacao UTIL" if out["useful_strict"] else "NAO sustentada: limitacao"))
OUT.write_text(json.dumps(out, indent=1, default=float))
print("\n  gravado %s" % OUT)
