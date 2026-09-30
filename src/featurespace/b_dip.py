"""Analysis B without domain metadata: Hartigan's dip on the C2 distances.

PathVQA and HEAL publish no modality or organ label, so there is no near group
and no far group to compare. The pre-registration therefore tests the shape of
the distance distribution itself: if donors fall into domains, the distances
they produce should be bimodal even with no label to say which domain.

  PathVQA  significant at p < 0.05 -> bimodal -> counts toward G2.
  HEAL     the negative control: it must come out UNIMODAL, p >= 0.05. HEAL has
           no domain metadata to separate near from far, so a clean split of its
           donor distances would mean something other than domain is producing
           the bimodality, and the reading of every other benchmark would be in
           doubt. A bimodal HEAL fails G2.

Distances come from the C0 cache by image_id, as in the rest of B; nothing is
recomputed. The dip is reported per ruler, with the colour histogram beside the
content rulers as the counterfactual of amendment 7.4.
"""
from __future__ import annotations

import json
import socket
import sys
import os
from pathlib import Path

import diptest
import numpy as np

REPO = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "embeddings"
OUT = REPO / "docs/v4/analysis_b_dip.json"
RULERS = ["medsiglip", "dinov3", "biomedclip", "dinov2", "colorhist"]
MANIFEST = {"PATHVQA": "C2_manifest_PATHVQA.json", "HEAL": "C2_manifest_HEAL.json"}
# HEAL is the negative control: unimodal is the passing outcome
EXPECT = {"PATHVQA": "bimodal", "HEAL": "unimodal"}
ALPHA = 0.05


def label(i):
    p = [x for x in str(i).replace("//", "/").split("/") if x]
    if p and p[-1].lower() in ("source.jpg", "source.png"):
        return p[-2] if len(p) > 1 else p[-1]
    return str(i)


def main(benches):
    ids_all = json.loads((CACHE / "image_ids.json").read_text())
    # HEAL runs on host-A and the rest on host-B: the declared limitation keeps VinDr
    # embeddings on host-A, and the dip needs no cross-host comparison to hold
    out = json.loads(OUT.read_text()) if OUT.exists() else {
        "alpha": ALPHA,
        "diptest_version": diptest.__version__, "cells": {}}
    out.setdefault("cells", {})

    for bench in benches:
        f = REPO / "results" / MANIFEST[bench]
        if not f.exists():
            print("  %s: sem manifesto de C2 (%s)" % (bench, f.name))
            continue
        d = json.loads(f.read_text())
        rows = d["pairs"] if isinstance(d, dict) else d
        pair = {label(e["qid_image"]): label(e["donor_image"]) for e in rows}
        weight = {}
        for e in rows:
            k = label(e["qid_image"])
            weight[k] = weight.get(k, 0) + int(e.get("questions", 1))

        ids = [label(i) for i in ids_all["%s__C0" % bench]]
        for ruler in RULERS:
            p = CACHE / ("%s__C0__%s.npy" % (bench, ruler))
            if not p.exists():
                p = CACHE / ("%s__%s.npy" % (bench, ruler))
            if not p.exists():
                print("  %s %s: embedding ausente" % (bench, ruler))
                continue
            X = np.load(p).astype(np.float64)
            vec = dict(zip(ids, X))
            dist, miss = [], 0
            for a, b in pair.items():
                if a not in vec or b not in vec:
                    miss += 1
                    continue
                va, vb = vec[a], vec[b]
                dist.append(1.0 - float(va @ vb /
                                        (np.linalg.norm(va) * np.linalg.norm(vb))))
            v = np.asarray(dist)
            stat, pval = diptest.diptest(v)
            bimodal = bool(pval < ALPHA)
            passes = (bimodal if EXPECT[bench] == "bimodal" else not bimodal)
            out["cells"]["%s|%s" % (bench, ruler)] = {
                "benchmark": bench, "ruler": ruler, "n_pairs": int(len(v)),
                "pairs_without_vector": miss,
                "mean": float(v.mean()), "sd": float(v.std()),
                "median": float(np.median(v)),
                "q25": float(np.quantile(v, .25)), "q75": float(np.quantile(v, .75)),
                "host": os.environ.get("RUN_HOST", "unspecified"),
                "dip": float(stat), "dip_p": float(pval),
                "bimodal_at_005": bimodal,
                "expected": EXPECT[bench], "passes": bool(passes)}
            OUT.write_text(json.dumps(out, indent=1))
            print("  %-8s %-11s n %4d  media %.4f  dip %.5f  p %.4f  %-9s %s"
                  % (bench, ruler, len(v), v.mean(), stat, pval,
                     "bimodal" if bimodal else "unimodal",
                     "OK" if passes else "CONTRA O ESPERADO (%s)" % EXPECT[bench]),
                  flush=True)

    print("\n  gravado %s  (%d celulas)" % (OUT, len(out["cells"])))


if __name__ == "__main__":
    main(sys.argv[1:] or ["PATHVQA"])
