"""The four decisions of 21/09, measured.

1  CCR on Kvasir-VQA-x1 under C5 over the informative set of the **corrected
   manifest** (|I| = 5,216), mean and sample sd over the three seeds, which is
   what Table 3 prints. The zero-shot is one run and has no sd; it is reported
   under the shared ruler and under E7.

4  Why the identity does not close on the zero-shot. Under C1 every image is the
   same grey frame, so two rows with the same question receive the same input;
   if decoding is deterministic within a run, they must receive the same answer.
   The test needs no second run and no other host: it is inside one file.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.analysis.ccr import (answers_match, load_predictions,   # noqa: E402
                                  load_swapmap, question_key)
from src.analysis.ccr_slices import seed_prefixes                # noqa: E402

OUT = REPO / "docs/v4/closeout_four.json"
RES = str(REPO / "results")
SM = load_swapmap(str(REPO / "data/swapmap.json"))
SYN, CLOSED = {"negative": "no", "positive": "yes"}, {"yes", "no"}
out = {}


def label(i):
    p = [x for x in str(i).replace("//", "/").split("/") if x]
    if p and p[-1].lower() in ("source.jpg", "source.png"):
        return p[-2] if len(p) > 1 else p[-1]
    return str(i)


# ---------------------------------------------------------------- item 1
d = json.loads((Path(RES) / "C5_manifest_KVASIR_clean.json").read_text())
rows = d["pairs"] if isinstance(d, dict) else d
donors = {label(e["qid_image"]): label(e["donor_image"]) for e in rows}
meta = {int(k): v for k, v in SM["KVASIR"]["qid"].items()}
ref_of = {}
for q, m in meta.items():
    ref_of.setdefault(label(m["img"]), {})[question_key(m["q"])] = m["a"]
I = []
for q, m in meta.items():
    im = label(m["img"])
    if im not in donors:
        continue
    r = ref_of.get(donors[im], {}).get(question_key(m["q"]))
    if r is not None and not any(answers_match(x, r) for x in m["a"]):
        I.append(q)
print("item 1 — CCR sobre |I| = %d (manifesto corrigido)" % len(I))
ccr = {}
for cfg in ("M3AE", "QWEN2VL_LORA", "QWEN2VL_DORA", "SMOLVLM_LORA", "SMOLVLM_DORA",
            "MEDVLTHINKER_ZEROSHOT"):
    vals, e7v = [], []
    for prefix in seed_prefixes(RES, "%s_KVASIR" % cfg, "C5"):
        cx = load_predictions(RES, prefix, "C5")
        if cx is None:
            continue
        hit = e7h = n = 0
        for q in I:
            if q not in cx:
                continue
            r = ref_of[donors[label(meta[q]["img"])]][question_key(meta[q]["q"])]
            p = str(cx[q]["predicted"])
            n += 1
            ok = answers_match(p, r)
            hit += ok
            if not ok and p.strip().lower() in SYN and \
                    any(str(x).strip().lower() in CLOSED for x in r):
                ok = answers_match(SYN[p.strip().lower()], r)
            e7h += ok
        vals.append(100 * hit / n)
        e7v.append(100 * e7h / n)
    ccr[cfg] = {"runs": len(vals), "mean": st.mean(vals),
                "sd": st.stdev(vals) if len(vals) > 1 else 0.0,
                "mean_e7": st.mean(e7v), "n_I": len(I)}
    print("   %-22s %d execucoes  CCR %.2f +- %.2f%s"
          % (cfg, len(vals), ccr[cfg]["mean"], ccr[cfg]["sd"],
             "   (sob E7 %.2f)" % ccr[cfg]["mean_e7"] if "MEDVL" in cfg else ""))
ft = [v["mean"] for k, v in ccr.items() if "MEDVL" not in k]
print("   faixa das afinadas: %.1f a %.1f" % (min(ft), max(ft)))
out["item1_ccr_corrected_manifest"] = {
    "n_I": len(I), "per_config": ccr,
    "fine_tuned_range": [min(ft), max(ft)],
    "table3_form": {k: "%.1f +- %.1f" % (v["mean"], v["sd"]) for k, v in ccr.items()}}

# ---------------------------------------------------------------- item 4
print("\nitem 4 — determinismo dentro da execucao, sob o quadro cinza do C1")
det = {}
for bench in ("KVASIR", "SLAKE"):
    m = {int(k): v for k, v in SM[bench]["qid"].items()}
    names = [("MEDVLTHINKER_ZEROSHOT_%s" % bench, "zero-shot")]
    for cfg in ("M3AE" if bench == "KVASIR" else "M3AE_SCRATCH",
                "QWEN2VL_LORA", "SMOLVLM_LORA"):
        pr = seed_prefixes(RES, "%s_%s" % (cfg, bench), "C1")
        if pr:
            names.append((pr[0], cfg))
    for prefix, lab in names:
        p = load_predictions(RES, prefix, "C1")
        if p is None:
            continue
        g = defaultdict(list)
        for q, mm in m.items():
            if q in p:
                g[question_key(mm["q"])].append(str(p[q]["predicted"]))
        multi = {k: v for k, v in g.items() if len(set(v)) > 1}
        # rows that depart from the modal answer of their question
        off = 0
        for k, v in g.items():
            mode = max(set(v), key=v.count)
            off += sum(1 for x in v if x != mode)
        rowsn = sum(len(v) for v in g.values())
        det["%s|%s" % (bench, lab)] = {
            "prefix": prefix, "rows": rowsn, "questions": len(g),
            "questions_with_more_than_one_answer": len(multi),
            "rows_off_the_modal_answer": off, "fraction_off": off / rowsn}
        print("   %-7s %-20s %5d linhas | %3d perguntas | %3d com mais de uma resposta | "
              "%4d linhas fora da resposta modal (%.2f%%)"
              % (bench, lab, rowsn, len(g), len(multi), off, 100 * off / rowsn))
out["item4_within_run_determinism"] = {
    "test": "under C1 every image is the same grey frame, so rows sharing a "
            "question share the input; more than one answer means the decoding "
            "is not deterministic within the run",
    "cells": det,
    "generation_config": {
        "MedVLThinker-7B-RL_PMC": {"do_sample": True, "temperature": 1e-06,
                                   "top_k": None, "top_p": None, "num_beams": None},
        "Qwen2-VL-2B-Instruct": {"do_sample": True, "temperature": 0.01,
                                 "top_p": 0.001, "top_k": 1},
        "SmolVLM2-2.2B-Instruct": {"do_sample": None, "note": "falls back to do_sample=False"},
        "source": "generation_config of each checkpoint, read on 2026-09-07 "
                  "(visapp/analise_2026-09-05/generation_config.txt); the call site "
                  "passes no sampling argument"}}
OUT.write_text(json.dumps(out, indent=1, default=float))
print("\n  gravado %s" % OUT)
