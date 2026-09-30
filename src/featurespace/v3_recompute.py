"""What the v3 numbers become when the donor map comes from the manifests.

`data/swapmap.json` resolves the donor through `md5` of the image bytes with a
`setdefault`, so Kvasir's byte-identical duplicates collapse onto one
identifier. Two consequences, both measured in commit a92f1f1:

  C5 and C6   six rows enter the map that never swapped. Their image_id and
              their bytes are unchanged between C0 and the condition; they are
              duplicates, so the hash returns the twin's identifier and
              `donor != original` becomes true.
  C2          seven donors are named as their byte-identical twin. The swap is
              real; the identifier is wrong.

Everything v3 derived from that map inherits the defect, because
`eligible_qids` selects the questions of the images the map contains, and the
informative/uninformative split asks what the *named* donor answers.

Recomputed here from the manifests, which were rebuilt by row index and checked
against the arrows by G1, and reported beside the swapmap values so the
difference is visible rather than asserted:

  |S_T|  questions of the swapped images
  |I|    informative: the donor carries the same question with a different answer
  |U|    uninformative: the donor carries it with the same answer
  |N|    the donor does not carry the question at all
  |V|    verifiable = I + U
  phi    |I| / |V|
  CCR    of the altered answers on I, the share matching the donor's reference
  VDD    conditional, restricted to S_T

Five configurations by three seeds, Kvasir, C2/C5/C6. Table 3 precision (0.1 pp).
"""
from __future__ import annotations

import json
import socket
import sys
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.analysis.ccr import (answers_match, compute_ccr, load_predictions,  # noqa: E402
                                  question_key)
from src.stats.conditional_vdd import conditional_vdd, eligible_qids  # noqa: E402

RES = REPO / "results"
CONFIGS = ["M3AE", "QWEN2VL_DORA", "QWEN2VL_LORA", "SMOLVLM_DORA", "SMOLVLM_LORA"]
SEEDS = ["_seed0", "_seed1", "_seed2"]
CONDS = ["C2", "C5", "C6"]
MANIFEST = {"C2": "C2_manifest_KVASIR.json", "C5": "C5_manifest_KVASIR_clean.json",
            "C6": "C6_manifest_KVASIR_clean.json"}

sm = json.loads((REPO / "data/swapmap.json").read_text())["KVASIR"]
QID = {int(k): v for k, v in sm["qid"].items()}


def manifest_map(cond):
    d = json.loads((REPO / "results" / MANIFEST[cond]).read_text())
    rows = d["pairs"] if isinstance(d, dict) else d
    return {str(e["qid_image"]): str(e["donor_image"]) for e in rows}


def partition(donor_map):
    """I, U and N over the questions of the swapped images.

    The three tests mirror `compute_ccr` exactly, including `answers_match` and
    `question_key`, so the informative count here cannot drift from the one the
    published CCR was computed on.
    """
    by_image = {}
    for qid, meta in QID.items():
        by_image.setdefault(meta["img"], {})[question_key(meta["q"])] = meta["a"]
    I = U = N = 0
    for qid, meta in QID.items():
        src = meta["img"]
        if src not in donor_map:
            continue
        ref_donor = by_image.get(donor_map[src], {}).get(question_key(meta["q"]))
        if ref_donor is None:
            N += 1                                    # donor lacks the question
        elif any(answers_match(x, ref_donor) for x in meta["a"]):
            U += 1                                    # same reference on both
        else:
            I += 1
    return I, U, N


def cell(prefix, cond, donor_map):
    c0 = load_predictions(str(RES), prefix, "C0")
    cx = load_predictions(str(RES), prefix, cond)
    if c0 is None or cx is None:
        return None
    elig = eligible_qids(QID, donor_map)
    vdd = conditional_vdd(c0, cx, restrict_to=elig)
    ccr = compute_ccr(c0, cx, QID, donor_map)
    I, U, N = partition(donor_map)
    V = I + U
    return {"S_T": len(elig), "I": I, "U": U, "N": N, "V": V,
            "phi": I / V if V else None,
            "ccr": ccr["ccr"], "informative_n": ccr["informative"],
            "vdd": vdd.get("vdd"), "acc_c0": vdd.get("acc_c0"),
            "acc_cx": vdd.get("acc_cx")}


out = {"host": os.environ.get("RUN_HOST", "unspecified"), "benchmark": "KVASIR", "cells": {}}
print("%-16s %-7s %-4s %7s %6s %6s %5s %7s %7s %7s"
      % ("config", "seed", "cond", "S_T", "I", "U", "N", "phi", "CCR", "VDD"))
for cond in CONDS:
    mm = manifest_map(cond)
    sw = sm["swap"][cond]
    for cfg in CONFIGS:
        for s in SEEDS:
            prefix = "%s_KVASIR%s" % (cfg, s)
            a = cell(prefix, cond, mm)
            b = cell(prefix, cond, sw)
            if a is None or b is None:
                print("  %-16s %-7s %-4s AUSENTE" % (cfg, s.strip("_"), cond))
                continue
            out["cells"]["%s|%s|%s" % (cfg, s.strip("_"), cond)] = {
                "corrected_from_manifest": a, "v3_from_swapmap": b}
            print("%-16s %-7s %-4s %7d %6d %6d %5d %7.4f %7.4f %7.4f   <- manifesto"
                  % (cfg, s.strip("_"), cond, a["S_T"], a["I"], a["U"], a["N"],
                     a["phi"] or 0, a["ccr"] or 0, a["vdd"] or 0))
            print("%-16s %-7s %-4s %7d %6d %6d %5d %7.4f %7.4f %7.4f   <- swapmap (v3)"
                  % ("", "", "", b["S_T"], b["I"], b["U"], b["N"],
                     b["phi"] or 0, b["ccr"] or 0, b["vdd"] or 0))

p = REPO / "docs/v4/v3_recompute_kvasir.json"
p.write_text(json.dumps(out, indent=1))
print("\n  gravado %s  (%d celulas)" % (p, len(out["cells"])))
