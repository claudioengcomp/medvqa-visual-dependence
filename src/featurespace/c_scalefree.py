"""Comparing rulers needs a measure without units; the raw coefficient has them.

The logistic coefficient of `flip ~ d` carries the scale of `d`, and `d` is not
on the same scale across rulers: on Kvasir its mean is 0.24 under MedSigLIP and
0.37 under the colour histogram, with different spreads. Saying "MedSigLIP is
thirty times colour" from the raw coefficients confounds the effect with the
unit, so the counterfactual sentence cannot rest on it.

Two scale-free readings are added, both per cell and ruler:

  coef_z   the coefficient on d standardised within (benchmark, ruler). No
           refit is needed: with d_z = (d - mu)/sigma, a + b*d = a' + (b*sigma)*d_z,
           so coef_z = coef * sigma exactly.
  auc      the probability that a flipped row has a larger d than a row that did
           not flip -- the Mann-Whitney statistic of `flip ~ d`. Bounded in
           [0, 1], 0.5 when distance says nothing, and comparable across rulers
           and benchmarks because it depends only on the ordering of d.

G3 is untouched: the sign and the confidence interval of the coefficient are
invariant to scale, so the gate reads the same. Only the comparison **between**
rulers moves onto these two, in Figure 5 and in the text.
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.analysis.ccr import answers_match, load_predictions  # noqa: E402
from src.featurespace import paths                                     # noqa: E402

CACHE = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "embeddings"
OUT = REPO / "docs/v4/analysis_c.json"
RULERS = ["medsiglip", "dinov3", "biomedclip", "dinov2", "colorhist"]
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
SM = json.loads((REPO / "data/swapmap.json").read_text())
IDS = json.loads((CACHE / "image_ids.json").read_text())


def label(i):
    p = [x for x in str(i).replace("//", "/").split("/") if x]
    if p and p[-1].lower() in ("source.jpg", "source.png"):
        return p[-2] if len(p) > 1 else p[-1]
    return str(i)


def donor_map(bench, cond):
    d = json.loads((paths.RESULTS / MANIFEST[(bench, cond)]).read_text())
    rows = d["pairs"] if isinstance(d, dict) else d
    return {label(e["qid_image"]): label(e["donor_image"]) for e in rows}


def distances(bench, ruler, dm):
    ids = [label(i) for i in IDS["%s__C0" % bench]]
    X = np.load(CACHE / ("%s__%s.npy" % (bench, ruler))).astype(np.float64)
    vec = dict(zip(ids, X))
    return {a: 1.0 - float(vec[a] @ vec[b] /
                           (np.linalg.norm(vec[a]) * np.linalg.norm(vec[b])))
            for a, b in dm.items() if a in vec and b in vec}


def auc(d, y):
    """Mann-Whitney AUC with midranks, so ties in d count as half."""
    pos, neg = d[y > .5], d[y < .5]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    s = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


out = json.loads(OUT.read_text())
cells = out["cells"]
sd_cache = {}
n_done = 0

for c in paths.cells(include_reference=True):
    b, cfg, s, prefix = c["benchmark"], c["config"], c["seed"], c["prefix"]
    c0 = load_predictions(str(paths.RESULTS), prefix, "C0")
    if c0 is None:
        continue
    QM = {int(k): v for k, v in SM[b]["qid"].items()}
    for ruler in RULERS:
        key = "%s|%s|%s|%s" % (b, cfg, s, ruler)
        if key not in cells or "auc" in cells[key]:
            continue
        d_all, f_all = [], []
        for cond in paths.SWAP_CONDITIONS[b]:
            cx = load_predictions(str(paths.RESULTS), prefix, cond)
            if cx is None:
                continue
            dm = donor_map(b, cond)
            dist = distances(b, ruler, dm)
            for qid, meta in QM.items():
                im = label(meta["img"])
                if im in dist and qid in c0 and qid in cx:
                    d_all.append(dist[im])
                    f_all.append(float(not answers_match(cx[qid]["predicted"],
                                                         [c0[qid]["predicted"]])))
        if not d_all:
            continue
        d = np.asarray(d_all)
        y = np.asarray(f_all)
        if (b, ruler) not in sd_cache:
            sd_cache[(b, ruler)] = float(d.std())
        sd = sd_cache[(b, ruler)]
        cells[key]["d_mean"] = float(d.mean())
        cells[key]["d_sd"] = sd
        cells[key]["coef_z"] = cells[key]["fit"]["coef"] * sd
        cells[key]["coef_z_ci"] = [cells[key]["fit"]["ci_low"] * sd,
                                   cells[key]["fit"]["ci_high"] * sd]
        cells[key]["auc"] = auc(d, y)
        n_done += 1
        if n_done % 10 == 0:
            OUT.write_text(json.dumps(out, indent=1))
            print("  %d celulas" % n_done, flush=True)

OUT.write_text(json.dumps(out, indent=1))
print("\n=== sem escala, media sobre configuracoes e seeds ===")
print("%-7s %s" % ("base", "  ".join("%-22s" % r for r in RULERS)))
for b in paths.BENCHMARKS:
    line = []
    for r in RULERS:
        v = [cells[k] for k in cells
             if k.startswith("%s|" % b) and k.endswith("|%s" % r)
             and not cells[k].get("reference_only") and "auc" in cells[k]]
        if not v:
            line.append("%-22s" % "--")
            continue
        cz = sum(x["coef_z"] for x in v) / len(v)
        au = sum(x["auc"] for x in v) / len(v)
        line.append("%-22s" % ("coef_z %+.3f  AUC %.3f" % (cz, au)))
    print("%-7s %s" % (b, "  ".join(line)))
print("\n  %d celulas acrescidas de coef_z e AUC" % n_done)
