"""The two pieces of C that tie it to Equation 3 and to v3.

**Item 3 — where the answer goes.** A flip says the answer changed; it does not
say it changed toward the donor. On the verifiable set V, where the donor image
carries the same question and its reference y' differs from y, the swapped
prediction can match y' (the model read the image it was given), match y (a
textual or domain prior prevailed), or neither. Reported per bin of d, so the
question is whether a far donor is followed more often than a near one — which
is what "the model uses the image" would mean, as opposed to "the model is
disturbed".

This is the CCR of v3 conditioned on distance. The three buckets are the same
ones `compute_ccr` defines, and the scoring is `answers_match`.

**Item 4 — VDD by tercile of d, against VDD(C5).** In percentage points,
Acc(C0) - Acc(T) on the rows of the tercile, as Equation 1 defines it. Until
2026-09-20 this block divided by the C0 correct count, which is the relative
drop; the field was called `tercile_vdd` and read as VDD, and the comparison
with the prediction of `phi_by_tercile.py` mixed a relative observation with an
absolute prediction. C5 is the swap restricted to
the endoscopic modality on Kvasir and to the bucket on SLAKE, so if the near
tercile of the pooled distances reproduces VDD(C5), then "a near donor" and
"the C5 construction" are the same thing measured twice, and the two rungs of
the ladder are not independent. On SLAKE the comparison is made inside
XRAY_LUNG, its largest bucket.

Both are read on MedSigLIP, with the colour histogram beside them.
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.analysis.ccr import answers_match, load_predictions, question_key  # noqa: E402
from src.featurespace import paths                                                    # noqa: E402

CACHE = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "embeddings"
OUT = REPO / "docs/v4/analysis_c_items34.json"
RULERS = ["medsiglip", "colorhist"]
N_BINS = 5
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


def donor_map(b, cond):
    d = json.loads((paths.RESULTS / MANIFEST[(b, cond)]).read_text())
    rows = d["pairs"] if isinstance(d, dict) else d
    return {label(e["qid_image"]): label(e["donor_image"]) for e in rows}


def distances(b, ruler, dm):
    ids = [label(i) for i in IDS["%s__C0" % b]]
    X = np.load(CACHE / ("%s__%s.npy" % (b, ruler))).astype(np.float64)
    v = dict(zip(ids, X))
    return {a: 1.0 - float(v[a] @ v[c] / (np.linalg.norm(v[a]) * np.linalg.norm(v[c])))
            for a, c in dm.items() if a in v and c in v}


def slake_xray():
    out = set()
    f = paths.RESULTS / "C6_manifest_SLAKE_XRAY_LUNG.json"
    for e in json.loads(f.read_text()):
        out.add(label(e["qid_image"]))
    return out


out = {"rulers": RULERS, "n_bins": N_BINS, "benchmarks": {}}

for b in paths.BENCHMARKS:
    QM = {int(k): v for k, v in SM[b]["qid"].items()}
    by_image = {}
    for qid, m in QM.items():
        by_image.setdefault(label(m["img"]), {})[question_key(m["q"])] = m["a"]
    rec = {"item3": {}, "item4": {}}

    for ruler in RULERS:
        rows_d, rows_qid, rows_don, rows_cond = [], [], [], []
        for cond in paths.SWAP_CONDITIONS[b]:
            dm = donor_map(b, cond)
            dist = distances(b, ruler, dm)
            for qid, m in QM.items():
                im = label(m["img"])
                if im in dist:
                    rows_d.append(dist[im])
                    rows_qid.append(qid)
                    rows_don.append(dm[im])
                    rows_cond.append(cond)
        d = np.asarray(rows_d)
        edges = np.quantile(d, np.linspace(0, 1, N_BINS + 1))
        edges[0] -= 1e-12
        binof = np.clip(np.digitize(d, edges[1:-1]), 0, N_BINS - 1)

        # item 3, averaged over the fifteen runs
        acc = [{"followed": 0, "kept": 0, "neither": 0, "n": 0} for _ in range(N_BINS)]
        ter = {}
        cells = [c for c in paths.cells(b) if not c["reference"]]
        for c in cells:
            c0 = load_predictions(str(paths.RESULTS), c["prefix"], "C0")
            preds = {cd: load_predictions(str(paths.RESULTS), c["prefix"], cd)
                     for cd in paths.SWAP_CONDITIONS[b]}
            for k in range(len(d)):
                qid, don, cond = rows_qid[k], rows_don[k], rows_cond[k]
                cx = preds.get(cond)
                if cx is None or qid not in cx or qid not in c0:
                    continue
                y = QM[qid]["a"]
                yq = by_image.get(don, {}).get(question_key(QM[qid]["q"]))
                if yq is None or any(answers_match(x, yq) for x in y):
                    continue                     # not verifiable / uninformative
                p = cx[qid]["predicted"]
                bkt = acc[binof[k]]
                bkt["n"] += 1
                if answers_match(p, yq):
                    bkt["followed"] += 1
                elif answers_match(p, y):
                    bkt["kept"] += 1
                else:
                    bkt["neither"] += 1
        rec["item3"][ruler] = [
            {"bin": i, "n": a["n"], "d_lo": float(edges[i]), "d_hi": float(edges[i + 1]),
             "followed_donor": a["followed"] / a["n"] if a["n"] else None,
             "kept_original": a["kept"] / a["n"] if a["n"] else None,
             "neither": a["neither"] / a["n"] if a["n"] else None}
            for i, a in enumerate(acc)]

        # item 4: VDD by tercile of d, and VDD(C5) on the same rows
        tedges = np.quantile(d, [0, 1 / 3, 2 / 3, 1])
        tedges[0] -= 1e-12
        tof = np.clip(np.digitize(d, tedges[1:-1]), 0, 2)
        restrict = slake_xray() if b == "SLAKE" else None
        vdd_t = [[] for _ in range(3)]
        vdd_c5 = []
        for c in cells:
            c0 = load_predictions(str(paths.RESULTS), c["prefix"], "C0")
            preds = {cd: load_predictions(str(paths.RESULTS), c["prefix"], cd)
                     for cd in paths.SWAP_CONDITIONS[b]}
            hit = [[0, 0, 0] for _ in range(3)]      # c0 correct, cx correct, rows
            c5 = [0, 0, 0]
            for k in range(len(d)):
                qid, cond = rows_qid[k], rows_cond[k]
                if restrict is not None and label(QM[qid]["img"]) not in restrict:
                    continue
                cx = preds.get(cond)
                if cx is None or qid not in cx or qid not in c0:
                    continue
                t = tof[k]
                hit[t][0] += c0[qid]["correct"]
                hit[t][1] += cx[qid]["correct"]
                hit[t][2] += 1
                if cond == "C5":
                    c5[0] += c0[qid]["correct"]
                    c5[1] += cx[qid]["correct"]
                    c5[2] += 1
            for t in range(3):
                if hit[t][2]:
                    vdd_t[t].append(100.0 * (hit[t][0] - hit[t][1]) / hit[t][2])
            if c5[2]:
                vdd_c5.append(100.0 * (c5[0] - c5[1]) / c5[2])
            n_rows_t = [h[2] for h in hit]
            n_rows_c5 = c5[2]
        rec["item4"][ruler] = {
            "unit": "percentage points, Acc(C0) - Acc(T) on the rows of the tercile",
            "n_rows_per_tercile": n_rows_t, "n_rows_c5": n_rows_c5,
            "restricted_to": "XRAY_LUNG" if restrict else "all",
            "tercile_vdd": [float(np.mean(v)) if v else None for v in vdd_t],
            "tercile_sd": [float(np.std(v)) if v else None for v in vdd_t],
            "vdd_c5": float(np.mean(vdd_c5)) if vdd_c5 else None,
            "n_cells": len(cells)}
    out["benchmarks"][b] = rec
    print("  %s pronto" % b, flush=True)

print("\n=== item 3: para onde vai a resposta, sob MedSigLIP ===")
print("%-7s %-4s %8s %10s %10s %10s" % ("base", "bin", "n", "segue doadora",
                                        "mantem orig", "nenhuma"))
for b in paths.BENCHMARKS:
    for r in out["benchmarks"][b]["item3"]["medsiglip"]:
        if not r["n"]:
            continue
        print("%-7s %-4d %8d %10.4f %10.4f %10.4f"
              % (b, r["bin"], r["n"], r["followed_donor"], r["kept_original"],
                 r["neither"]))
    print()

print("=== item 4: VDD por tercil de d contra VDD(C5), sob MedSigLIP ===")
print("%-7s %-10s %9s %9s %9s %10s" % ("base", "escopo", "perto", "meio", "longe",
                                       "VDD(C5)"))
print("(pontos percentuais)")
for b in paths.BENCHMARKS:
    i4 = out["benchmarks"][b]["item4"]["medsiglip"]
    t = i4["tercile_vdd"]
    print("%-7s %-10s %9s %9s %9s %10s"
          % (b, i4["restricted_to"],
             *["%.2f" % x if x is not None else "--" for x in t],
             "%.2f" % i4["vdd_c5"] if i4["vdd_c5"] is not None else "--"))

OUT.write_text(json.dumps(out, indent=1))
print("\n  gravado %s" % OUT)
