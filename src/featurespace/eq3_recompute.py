"""Equation 3 of v3, recomputed from the manifests.

Definitions transcribed from main_v3.tex §3.5 and §5.2, not reconstructed:

    a_I   accuracy under C0 on the original items of I
    k     the rate at which the prediction under the swap coincides with the
          original reference on I, so Acc_I(T) = k
    t_U   Acc_U(C0) - Acc_U(T)
    d_N   Acc_N(C0) - Acc_N(T)

    VDD(T) = (|I|/|S_T|)(a_I - k) + (|U|/|S_T|) t_U + (|N|/|S_T|) d_N

The identity is checked per cell rather than assumed: the three terms must sum
to the conditional VDD that `conditional_vdd` measures on the same restriction.
A mismatch would mean the partition and the VDD are not being taken over the
same set, which is precisely the kind of drift the swapmap defect introduced.

Spread prediction: |I|/|S_T| times the sum of the spreads of a_I and k, where a
spread is the range across the five configurations of each configuration's mean
over its three seeds. Observed spread is the range of the conditional VDD across
the same five configurations.

Expected k: v3 defines it as the rate expected "from each model's own
distribution of predictions on the same question template". `harness/analysis`
has no routine for it, so the operational reading is used and marked as such in
the output: for each i in I, how often the model predicts y_i under C0 among the
items of the same template, averaged over I.
"""
from __future__ import annotations

import collections
import json
import socket
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.analysis.ccr import (answers_match, load_predictions,  # noqa: E402
                                  question_key)
from src.stats.conditional_vdd import conditional_vdd, eligible_qids  # noqa: E402

RES = REPO / "results"
CONFIGS = ["M3AE", "QWEN2VL_DORA", "QWEN2VL_LORA", "SMOLVLM_DORA", "SMOLVLM_LORA"]
SEEDS = ["_seed0", "_seed1", "_seed2"]
CONDS = ["C5", "C6"]
MANIFEST = {"C5": "C5_manifest_KVASIR_clean.json", "C6": "C6_manifest_KVASIR_clean.json"}

sm = json.loads((REPO / "data/swapmap.json").read_text())["KVASIR"]
QID = {int(k): v for k, v in sm["qid"].items()}
BY_IMAGE = {}
for _q, _m in QID.items():
    BY_IMAGE.setdefault(_m["img"], {})[question_key(_m["q"])] = _m["a"]


def manifest_map(cond):
    d = json.loads((REPO / "results" / MANIFEST[cond]).read_text())
    rows = d["pairs"] if isinstance(d, dict) else d
    return {str(e["qid_image"]): str(e["donor_image"]) for e in rows}


def partition(donor_map):
    I, U, N = [], [], []
    for qid, meta in QID.items():
        src = meta["img"]
        if src not in donor_map:
            continue
        ref = BY_IMAGE.get(donor_map[src], {}).get(question_key(meta["q"]))
        if ref is None:
            N.append(qid)
        elif any(answers_match(x, ref) for x in meta["a"]):
            U.append(qid)
        else:
            I.append(qid)
    return I, U, N


def acc(pred, qids):
    q = [x for x in qids if x in pred]
    return (sum(pred[x]["correct"] for x in q) / len(q)) if q else None


def expected_k(c0, I):
    """Operational reading of v3's expected k; no harness routine exists.

    Cached by (template, reference): the same reference recurs across thousands
    of items of a template, and scanning every peer for each of them turns a
    5,216-item set into millions of string comparisons.
    """
    by_t = collections.defaultdict(list)
    for qid, meta in QID.items():
        if qid in c0:
            by_t[question_key(meta["q"])].append(qid)
    cache = {}
    vals = []
    for qid in I:
        if qid not in c0:
            continue
        t = question_key(QID[qid]["q"])
        y = QID[qid]["a"]
        key = (t, json.dumps(y, sort_keys=True))
        if key not in cache:
            peers = by_t[t]
            cache[key] = (sum(1 for p in peers
                              if answers_match(c0[p]["predicted"], y)) / len(peers)
                          if peers else None)
        if cache[key] is not None:
            vals.append(cache[key])
    return sum(vals) / len(vals) if vals else None


def cell(prefix, cond, donor_map):
    c0 = load_predictions(str(RES), prefix, "C0")
    cx = load_predictions(str(RES), prefix, cond)
    if c0 is None or cx is None:
        return None
    I, U, N = partition(donor_map)
    S = len(I) + len(U) + len(N)
    a_I, k = acc(c0, I), acc(cx, I)
    t_U = acc(c0, U) - acc(cx, U)
    d_N = acc(c0, N) - acc(cx, N)
    term_I = len(I) / S * (a_I - k)
    term_U = len(U) / S * t_U
    term_N = len(N) / S * d_N
    vdd = conditional_vdd(c0, cx, restrict_to=eligible_qids(QID, donor_map))["vdd"]
    V = len(I) + len(U)
    phi = len(I) / V
    return {"S_T": S, "I": len(I), "U": len(U), "N": len(N), "V": V, "phi": phi,
            "a_I": a_I, "k": k, "t_U": t_U, "d_N": d_N,
            "term_I": term_I, "term_U": term_U, "term_N": term_N,
            "eq3_sum": term_I + term_U + term_N, "vdd_measured": vdd,
            "identity_gap": term_I + term_U + term_N - vdd,
            "vdd_over_phi": vdd / phi, "k_expected": expected_k(c0, I)}


def spread(d):
    """Range across configurations of each configuration's mean over seeds."""
    per = [sum(v) / len(v) for v in d.values() if v]
    return max(per) - min(per) if per else None


out = {"host": os.environ.get("RUN_HOST", "unspecified"), "benchmark": "KVASIR",
       "definitions": "main_v3.tex 3.5 and 5.2, transcribed",
       "k_expected": "operational reading; no harness routine exists",
       "cells": {}, "summary": {}}

for cond in CONDS:
    for source, dm in (("corrected_from_manifest", manifest_map(cond)),
                       ("v3_from_swapmap", sm["swap"][cond])):
        aI, kk, vd = collections.defaultdict(list), collections.defaultdict(list), \
            collections.defaultdict(list)
        worst = 0.0
        for cfg in CONFIGS:
            for s in SEEDS:
                r = cell("%s_KVASIR%s" % (cfg, s), cond, dm)
                if r is None:
                    continue
                out["cells"].setdefault("%s|%s|%s" % (cfg, s.strip("_"), cond), {})[source] = r
                aI[cfg].append(r["a_I"])
                kk[cfg].append(r["k"])
                vd[cfg].append(r["vdd_measured"])
                worst = max(worst, abs(r["identity_gap"]))
        sI = spread(aI)
        sK = spread(kk)
        frac = r["I"] / r["S_T"]
        out["summary"]["%s|%s" % (cond, source)] = {
            "I_over_S_T": frac, "spread_a_I": sI, "spread_k": sK,
            "spread_predicted": frac * (sI + sK),
            "spread_observed": spread(vd),
            "max_identity_gap": worst}

print("=== identidade da Eq. 3, por celula (soma dos tres termos - VDD medida) ===")
gaps = [abs(v[s]["identity_gap"]) for v in out["cells"].values() for s in v]
print("  maior desvio absoluto em %d celulas: %.3e" % (len(gaps), max(gaps)))

print("\n=== a_I, k, t_U, d_N  (corrigido vs v3) ===")
print("%-14s %-6s %-4s %8s %8s %8s %8s %9s %9s"
      % ("config", "seed", "cond", "a_I", "k", "t_U", "d_N", "VDD", "VDD/phi"))
for key in sorted(out["cells"]):
    cfg, seed, cond = key.split("|")
    a = out["cells"][key]["corrected_from_manifest"]
    b = out["cells"][key]["v3_from_swapmap"]
    print("%-14s %-6s %-4s %8.4f %8.4f %8.4f %8.4f %9.4f %9.4f"
          % (cfg, seed, cond, a["a_I"], a["k"], a["t_U"], a["d_N"],
             a["vdd_measured"], a["vdd_over_phi"]))
    d = [("a_I", a["a_I"] - b["a_I"]), ("k", a["k"] - b["k"]),
         ("t_U", a["t_U"] - b["t_U"]), ("d_N", a["d_N"] - b["d_N"]),
         ("VDD", a["vdd_measured"] - b["vdd_measured"]),
         ("VDD/phi", a["vdd_over_phi"] - b["vdd_over_phi"])]
    print("%-14s %-6s %-4s %s" % ("", "", "dif",
                                  "  ".join("%s %+.5f" % (n, v) for n, v in d)))

print("\n=== espalhamento previsto contra observado ===")
print("%-4s %-24s %10s %10s %10s %10s"
      % ("cond", "fonte", "|I|/|S_T|", "spr(a_I)", "spr(k)", "previsto"))
for key in sorted(out["summary"]):
    cond, source = key.split("|")
    s = out["summary"][key]
    print("%-4s %-24s %10.4f %9.2fpp %9.2fpp %9.2fpp   observado %.2fpp"
          % (cond, source, s["I_over_S_T"], 100 * s["spread_a_I"],
             100 * s["spread_k"], 100 * s["spread_predicted"],
             100 * s["spread_observed"]))

print("\n=== k esperado (leitura operacional) contra k observado ===")
for cond in CONDS:
    v = [(out["cells"][x]["corrected_from_manifest"]["k_expected"],
          out["cells"][x]["corrected_from_manifest"]["k"])
         for x in out["cells"] if x.endswith("|" + cond)]
    v = [(e, k) for e, k in v if e is not None]
    if v:
        print("  %s  esperado %.1f-%.1f%%   observado %.1f-%.1f%%"
              % (cond, 100 * min(e for e, _ in v), 100 * max(e for e, _ in v),
                 100 * min(k for _, k in v), 100 * max(k for _, k in v)))

p = REPO / "docs/v4/eq3_recompute_kvasir.json"
p.write_text(json.dumps(out, indent=1))
print("\n  gravado %s" % p)
