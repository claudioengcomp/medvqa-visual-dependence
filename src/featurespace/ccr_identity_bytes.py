"""The CCR identity, paired by image bytes rather than by image identifier.

The identity is a construction check: under a swap the model sees an existing
test item, so its prediction on the swapped row must equal its prediction under
C0 on the donor item -- same image, same question. What makes it delicate is
that the C5/C6 arrow replaces the whole image row, `image_id` together with the
bytes, keeping the questions of the original item. A check that pairs rows by
`image_id` therefore compares the donor's own questions, not the ones asked of
it after the swap.

Here the pairing is by **sha256 of the image plus identical question text**. The
ids travel with the bytes in every swap (`swap_bytes_identity.json`: the
image_id-to-sha map of each swap equals C0's, 8 of 8), so the two pairings
coincide -- except on the seven byte-identical duplicate groups of
Kvasir-VQA-x1, where several ids carry the same picture and the sha is the
honest key.

Three cells need files that are not the default ones, for the reasons below:

  M3AE_SCRATCH SLAKE          C0 and C6 re-run in one session (`_sameenv`),
                              because the C6 of 27/08 came from another
                              environment and differed in 5.94% of its rows;
  SmolVLM2 LoRA Kvasir seed2  C0 and C5 re-run with batch size 1 (`_bs1`),
  SmolVLM2 DoRA Kvasir seed0  idem,
                              because decoding is not invariant to how a batch
                              is composed; the same host and the same arrow give
                              0.71% and 0.63% different predictions under C0
                              between batch 4 and batch 1.

Also measured here, for the record: those two divergences, since the paper
distinguishes them.
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.analysis.ccr import (answers_match, load_predictions,   # noqa: E402
                                  load_swapmap, question_key)
from src.analysis.ccr_slices import seed_prefixes                 # noqa: E402

CACHE = Path(os.environ.get("V4_CACHE", str(REPO / "cache")))
OUT = REPO / "docs/v4/ccr_identity_bytes.json"
SM = load_swapmap(str(REPO / "data/swapmap.json"))
HASH = json.loads((CACHE / "image_hashes.json").read_text())
RES = str(REPO / "results")
CONFIGS = ["M3AE", "QWEN2VL_LORA", "QWEN2VL_DORA", "SMOLVLM_LORA", "SMOLVLM_DORA"]
SEEDS = ["seed0", "seed1", "seed2"]
CELLS = {"KVASIR": "C5", "SLAKE": "C6"}
# cell -> (suffix for C0, suffix for the swap), and why
OVERRIDE = {
    ("SLAKE", "M3AE_SCRATCH", 0): ("_sameenv", "_sameenv",
                                   "C0 and C6 re-run in one session"),
    ("KVASIR", "SMOLVLM_LORA", 2): ("_bs1", "_bs1", "batch size 1"),
    ("KVASIR", "SMOLVLM_DORA", 0): ("_bs1", "_bs1", "batch size 1"),
}
FORBIDDEN = ["M3AE_SLAKE_C0_sameenv", "M3AE_SLAKE_C6_sameenv",
             "_C5_sameenv"]           # never used: wrong checkpoint / superseded


def sha_of(bench):
    t = HASH[bench]["test"]
    return dict(zip(t["ids"], t["sha"]))


def prefixes_of(bench, cfg, cond):
    """The three runs of a configuration, in seed order, under this directory's
    convention: seed 0 carries no suffix, seeds 1 and 2 do."""
    name = "M3AE_SCRATCH" if (cfg == "M3AE" and bench != "KVASIR") else cfg
    return seed_prefixes(RES, "%s_%s" % (name, bench), cond), name


def load(prefix, cond, suffix=""):
    if suffix:
        p = Path(RES) / ("%s_%s%s.json" % (prefix, cond, suffix))
        if not p.exists():
            return None, str(p)
        d = json.loads(p.read_text())
        return {int(x["qid"]): x for x in d["predictions"]}, p.name
    return load_predictions(RES, prefix, cond), "%s_%s.json" % (prefix, cond)


out = {"pairing": "sha256 of the image plus identical question text",
       "forbidden_files": FORBIDDEN, "cells": {}, "files_used": {}}
agree_all = []
for bench, cond in CELLS.items():
    meta = {int(k): v for k, v in SM[bench]["qid"].items()}
    donors = SM[bench]["swap"][cond]
    sha = sha_of(bench)
    # (sha, question) -> qids of C0 that carry this picture and this question
    by_sha_q = {}
    for q, m in meta.items():
        s = sha.get(m["img"])
        if s is None:
            continue
        by_sha_q.setdefault((s, question_key(m["q"])), []).append(q)
    by_image = {}
    for q, m in meta.items():
        by_image.setdefault(m["img"], {})[question_key(m["q"])] = m["a"]

    for cfg in CONFIGS:
        prefixes, name = prefixes_of(bench, cfg, cond)
        for seed, prefix in enumerate(prefixes):
            s0, s1, why = OVERRIDE.get((bench, name, seed), ("", "", ""))
            c0, f0 = load(prefix, "C0", s0)
            cx, fx = load(prefix, cond, s1)
            key = "%s|%s|seed%d" % (bench, name, seed)
            if c0 is None or cx is None:
                out["cells"][key] = {"status": "missing", "files": [f0, fx]}
                continue
            out["files_used"][key] = {"C0": f0, cond: fx, "why": why or "default files"}
            n = agree = amb = 0
            for q, m in meta.items():
                if m["img"] not in donors or q not in cx or q not in c0:
                    continue
                donor = donors[m["img"]]
                ref = by_image.get(donor, {}).get(question_key(m["q"]))
                if ref is None or any(answers_match(x, ref) for x in m["a"]):
                    continue                       # not verifiable / not informative
                twins = by_sha_q.get((sha.get(donor), question_key(m["q"])), [])
                twins = [t for t in twins if t in c0]
                if not twins:
                    continue
                n += 1
                preds = {str(c0[t]["predicted"]) for t in twins}
                amb += len(preds) > 1
                if str(cx[q]["predicted"]) in preds:
                    agree += 1
            r = {"n_informative": n, "agree": agree,
                 "rate": agree / n if n else None, "ambiguous_twins": amb,
                 "exact": agree == n and n > 0}
            out["cells"][key] = r
            agree_all.append(r["exact"])
            print("  %-28s n=%-5d identidade %.4f %s%s"
                  % (key, n, r["rate"], "EXATA" if r["exact"] else "NAO EXATA",
                     ("   [%s]" % why) if why else ""))

out["exact_cells"] = sum(agree_all)
out["total_cells"] = len(agree_all)
print("\n  identidade exata em %d de %d celulas" % (out["exact_cells"], out["total_cells"]))

# --- the two sources of non-determinism, measured
def divergence(a, b):
    pa, pb = a[0], b[0]
    shared = [q for q in pa if q in pb]
    d = sum(str(pa[q]["predicted"]) != str(pb[q]["predicted"]) for q in shared)
    return d, len(shared), d / len(shared)

nd = {}
old, fo = load("M3AE_SCRATCH_SLAKE", "C6")
new, fn = load("M3AE_SCRATCH_SLAKE", "C6", "_sameenv")
if old and new:
    d, n, r = divergence((old,), (new,))
    nd["across_environments"] = {"cell": "M3AE_SCRATCH SLAKE C6", "files": [fo, fn],
                                 "differing": d, "n": n, "fraction": r}
    print("  entre ambientes  %-28s %d de %d = %.4f" % ("M3AE_SCRATCH SLAKE C6", d, n, r))
nd["batch_composition"] = {}
for cfg, seed in (("SMOLVLM_LORA", 2), ("SMOLVLM_DORA", 0)):
    p = seed_prefixes(RES, "%s_KVASIR" % cfg, "C5")[seed]
    a, fa = load(p, "C0")
    b_, fb = load(p, "C0", "_bs1")
    if a and b_:
        d, n, r = divergence((a,), (b_,))
        # the same comparison on the scored flag rather than on the text, since
        # a changed answer need not change whether it is right
        dc = sum(a[q]["correct"] != b_[q]["correct"] for q in a if q in b_)
        nd["batch_composition"]["%s seed%d" % (cfg, seed)] = {
            "condition": "C0", "files": [fa, fb], "differing_predictions": d,
            "differing_correct": dc, "n": n, "fraction": r, "fraction_correct": dc / n}
        print("  lote 4 x lote 1 %-28s %d de %d = %.4f  (correct: %d = %.4f)"
              % (p, d, n, r, dc, dc / n))
out["non_determinism"] = nd
OUT.write_text(json.dumps(out, indent=1))
print("\n  gravado %s" % OUT)
