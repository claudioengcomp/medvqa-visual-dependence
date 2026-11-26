"""Three loose ends before submission. No GPU, no new inference.

1  |I| of C5 on Kvasir-VQA-x1: 5,216 or 5,218. The informative set is defined by
   references alone, so it cannot depend on the model -- but it does depend on
   *which donor map* is read. The swap map of v3 kept six rows that never
   swapped; the corrected manifests dropped them, and two of their questions
   stopped being verifiable. Both numbers are computed here, the two items that
   differ are named, and every quantity that hangs on the choice is recomputed
   under both.

2  The 1,008 prediction files: recounted on the current tree, with the
   negative/positive tally per model.

3  What a cell is in the 30-of-30 identity check, and whether the zero-shot is
   one of them.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.analysis.ccr import (answers_match, load_predictions,   # noqa: E402
                                  load_swapmap, question_key)
from src.analysis.ccr_slices import seed_prefixes                # noqa: E402

OUT = REPO / "docs/v4/closeout_three.json"
RES = str(REPO / "results")
SM = load_swapmap(str(REPO / "data/swapmap.json"))
SYN, CLOSED = {"negative": "no", "positive": "yes"}, {"yes", "no"}
out = {}


def label(i):
    p = [x for x in str(i).replace("//", "/").split("/") if x]
    if p and p[-1].lower() in ("source.jpg", "source.png"):
        return p[-2] if len(p) > 1 else p[-1]
    return str(i)


def manifest_donors(name):
    d = json.loads((Path(RES) / name).read_text())
    rows = d["pairs"] if isinstance(d, dict) else d
    return {label(e["qid_image"]): label(e["donor_image"]) for e in rows}


def partition(bench, donors):
    """I, U, N by the definitions of compute_ccr, from references only."""
    meta = {int(k): v for k, v in SM[bench]["qid"].items()}
    by_image = {}
    for q, m in meta.items():
        by_image.setdefault(label(m["img"]), {})[question_key(m["q"])] = m["a"]
    I, U, N = set(), set(), set()
    for q, m in meta.items():
        im = label(m["img"])
        if im not in donors:
            continue
        ref = by_image.get(donors[im], {}).get(question_key(m["q"]))
        if ref is None:
            N.add(q)
        elif any(answers_match(x, ref) for x in m["a"]):
            U.add(q)
        else:
            I.add(q)
    return I, U, N, meta, by_image


# ----------------------------------------------------------------- item 1
sw = {label(k): label(v) for k, v in SM["KVASIR"]["swap"]["C5"].items()}
mf = manifest_donors("C5_manifest_KVASIR_clean.json")
res = {}
for tag, donors in (("swap_map", sw), ("corrected_manifest", mf)):
    I, U, N, meta, by_image = partition("KVASIR", donors)
    res[tag] = {"donor_assignments": len(donors), "S_T": len(I | U | N),
                "I": len(I), "U": len(U), "N": len(N), "V": len(I) + len(U),
                "phi": len(I) / (len(I) + len(U)), "_I": I}
a, b = res["swap_map"], res["corrected_manifest"]
only_sw = sorted(a["_I"] - b["_I"])
print("item 1 - |I| do C5 no Kvasir")
for tag in ("swap_map", "corrected_manifest"):
    r = res[tag]
    print("  %-20s doadoras %d | S_T %d  I %d  U %d  N %d  V %d  phi %.4f"
          % (tag, r["donor_assignments"], r["S_T"], r["I"], r["U"], r["N"],
             r["V"], r["phi"]))
print("  itens em I so pelo swap map: %s" % only_sw)
for q in only_sw:
    m = {int(k): v for k, v in SM["KVASIR"]["qid"].items()}[q]
    im = label(m["img"])
    print("     qid %d  imagem %s  pergunta %r" % (q, im, m["q"][:60]))
    print("        doadora no swap map %s | no manifesto %s"
          % (sw.get(im), mf.get(im, "AUSENTE (linha fantasma)")))

# every quantity that hangs on the choice
ccr = {"swap_map": {}, "corrected_manifest": {}}
for tag, donors in (("swap_map", sw), ("corrected_manifest", mf)):
    I, U, N, meta, by_image = partition("KVASIR", donors)
    for cfg in ("M3AE", "QWEN2VL_LORA", "QWEN2VL_DORA", "SMOLVLM_LORA", "SMOLVLM_DORA",
                "MEDVLTHINKER_ZEROSHOT"):
        vals, e7v = [], []
        for prefix in seed_prefixes(RES, "%s_KVASIR" % cfg, "C5"):
            c0 = load_predictions(RES, prefix, "C0")
            cx = load_predictions(RES, prefix, "C5")
            if c0 is None or cx is None:
                continue
            hit = e7h = n = 0
            for q in I:
                if q not in cx:
                    continue
                ref = by_image[donors[label(meta[q]["img"])]][question_key(meta[q]["q"])]
                p = cx[q]["predicted"]
                n += 1
                ok = answers_match(p, ref)
                hit += ok
                if not ok and str(p).strip().lower() in SYN and \
                        any(str(r).strip().lower() in CLOSED for r in ref):
                    ok = answers_match(SYN[str(p).strip().lower()], ref)
                e7h += ok
            if n:
                vals.append(100 * hit / n)
                e7v.append(100 * e7h / n)
        if vals:
            ccr[tag][cfg] = {"ccr": sum(vals) / len(vals), "n_I": n,
                             "ccr_e7": sum(e7v) / len(e7v), "runs": len(vals)}
print("\n  CCR sob os dois denominadores (media sobre seeds):")
print("  %-22s %12s %12s %10s" % ("config", "swap map", "manifesto", "diferenca"))
for cfg in ccr["swap_map"]:
    x, y = ccr["swap_map"][cfg]["ccr"], ccr["corrected_manifest"][cfg]["ccr"]
    extra = ""
    if cfg == "MEDVLTHINKER_ZEROSHOT":
        extra = "   | sob E7: %.2f -> %.2f" % (ccr["swap_map"][cfg]["ccr_e7"],
                                               ccr["corrected_manifest"][cfg]["ccr_e7"])
    print("  %-22s %12.2f %12.2f %+10.2f%s" % (cfg, x, y, y - x, extra))
ft = [ccr["swap_map"][c]["ccr"] for c in ccr["swap_map"] if "MEDVL" not in c]
ftm = [ccr["corrected_manifest"][c]["ccr"] for c in ccr["corrected_manifest"] if "MEDVL" not in c]
print("  faixa das afinadas: swap map %.1f-%.1f | manifesto %.1f-%.1f"
      % (min(ft), max(ft), min(ftm), max(ftm)))
for r in res.values():
    r.pop("_I")
out["item1_informative_set"] = {
    "definition": "I: the donor carries the same question and its reference differs; "
                  "references only, no model",
    "by_donor_map": res, "items_only_in_swap_map": only_sw,
    "ccr_kvasir_c5": ccr,
    "fine_tuned_range": {"swap_map": [min(ft), max(ft)],
                         "corrected_manifest": [min(ftm), max(ftm)]}}

# ----------------------------------------------------------------- item 2
print("\nitem 2 - arquivos de predicao e os tokens negative/positive")
files = sorted(Path(RES).glob("*.json"))
pred_files, tally, items = [], defaultdict(int), defaultdict(int)
for f in files:
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    if not (isinstance(d, dict) and isinstance(d.get("predictions"), list)):
        continue
    pred_files.append(f.name)
    model = d.get("model") or f.name.split("_")[0]
    fam = ("M0" if "M0_question_only" in f.name else
           "M3AE" if f.name.startswith("M3AE") else
           "QWEN2VL" if f.name.startswith("QWEN2VL") else
           "SMOLVLM" if f.name.startswith("SMOLVLM") else
           "MEDVLTHINKER" if f.name.startswith("MEDVLTHINKER") else model)
    n = sum(1 for p in d["predictions"]
            if str(p.get("predicted", "")).strip().lower() in SYN)
    items[fam] += n
    tally[fam] += 1
print("  arquivos varridos com campo predictions: %d" % len(pred_files))
for fam in sorted(tally):
    print("     %-14s %4d arquivos | %6d itens com negative/positive"
          % (fam, tally[fam], items[fam]))
out["item2_prediction_files"] = {"files_with_predictions": len(pred_files),
                                 "files_by_family": dict(tally),
                                 "items_with_negative_or_positive": dict(items),
                                 "tokens": list(SYN)}

# ----------------------------------------------------------------- item 3
print("\nitem 3 - o que e uma celula na identidade")
ci = json.loads((REPO / "docs/v4/ccr_identity_bytes.json").read_text())
cells = [k for k in ci["cells"]]
byb = defaultdict(list)
for k in cells:
    byb[k.split("|")[0]].append(k.split("|")[1])
print("  celulas: %d" % len(cells))
for b in byb:
    print("     %-7s %d celulas = %d configuracoes x 3 seeds: %s"
          % (b, len(byb[b]), len(set(byb[b])), sorted(set(byb[b]))))
print("  zero-shot entre elas: %s" % any("MEDVL" in k for k in cells))
# does the identity hold for the zero-shot too? one run, so one cell per benchmark
zs = {}
for bench, cond, mfile in (("KVASIR", "C5", "C5_manifest_KVASIR_clean.json"),
                           ("SLAKE", "C6", "C6_manifest_SLAKE.json")):
    donors = {label(k): label(v) for k, v in SM[bench]["swap"][cond].items()}
    I, U, N, meta, by_image = partition(bench, donors)
    prefix = "MEDVLTHINKER_ZEROSHOT_%s" % bench
    c0 = load_predictions(RES, prefix, "C0")
    cx = load_predictions(RES, prefix, cond)
    n = agree = 0
    for q in I:
        if q not in cx:
            continue
        donor = donors[label(meta[q]["img"])]
        tw = [t for t, m in meta.items()
              if label(m["img"]) == donor
              and question_key(m["q"]) == question_key(meta[q]["q"]) and t in c0]
        if not tw:
            continue
        n += 1
        agree += str(cx[q]["predicted"]) in {str(c0[t]["predicted"]) for t in tw}
    zs[bench] = {"condition": cond, "n": n, "agree": agree, "rate": agree / n if n else None}
    print("  zero-shot %s %s: identidade %.4f sobre %d itens" % (bench, cond, agree / n, n))
out["item3_identity_cells"] = {
    "cell": "one fine-tuned checkpoint: configuration x seed, on the benchmark "
            "whose CCR Table 3 reports (Kvasir-VQA-x1 under C5, SLAKE under C6)",
    "n_cells": len(cells), "per_benchmark": {b: len(v) for b, v in byb.items()},
    "zero_shot_included": any("MEDVL" in k for k in cells),
    "zero_shot_measured_apart": zs}

OUT.write_text(json.dumps(out, indent=1, default=float))
print("\n  gravado %s" % OUT)
