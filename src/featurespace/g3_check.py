"""G3, against the criterion registered in a32347b before any number of C existed.

    the logistic coefficient of flip ~ d is positive with a confidence interval
    excluding zero in at least 12 of the 15 configuration × benchmark cells,
    with the three seeds never pooled

"Never pooled" is read as the strict rule: a cell passes only if all three of
its seeds pass on their own. Pooling them would let one strong seed carry two
weak ones, which is the thing the multi-seed protocol exists to prevent.

The verdict is read on MedSigLIP. The colour histogram is reported beside it as
the counterfactual of amendment 7.4 and decides nothing; the reading fixed in
amendment B.3, before the curves existed, was that MedSigLIP rising while colour
does not means content, and both rising equally means colour.

MedVLThinker zero-shot and the question-only floor are drawn as references and
never judged. M0 is the floor by construction: it never sees the image, so its
prediction cannot change under a swap and its flip rate is zero in every bin.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.featurespace import paths            # noqa: E402  the single path resolver

C = json.loads((REPO / "docs/v4/analysis_c.json").read_text())["cells"]
BENCH, CONFIGS, SEEDS = paths.BENCHMARKS, paths.CONFIGS, paths.SEEDS
PRIMARY = "medsiglip"
RULERS = ["medsiglip", "dinov3", "biomedclip", "dinov2", "colorhist"]

out = {"criterion": "coef > 0 with CI excluding zero in >= 12 of 15 cells; "
                    "a cell passes only if all three seeds pass",
       "verdict_read_on": PRIMARY, "cells": {}}


def get(b, cfg, s, r, where="fit"):
    c = C.get("%s|%s|%s|%s" % (b, cfg, s, r))
    if c is None:
        return None
    return c.get(where) if where != "fit" else c["fit"]


print("=== G3: coeficiente de flip ~ d, regua primaria (%s) ===" % PRIMARY)
print("%-7s %-16s %s" % ("base", "config", "  ".join("%-26s" % s for s in SEEDS)))
n_pass = 0
for b in BENCH:
    for cfg in CONFIGS[b]:
        row, ok_all = [], True
        for s in SEEDS:
            f = get(b, cfg, s, PRIMARY)
            if f is None:
                row.append("%-26s" % "ausente")
                ok_all = False
                continue
            ok = f["positive_ci_excludes_zero"]
            ok_all &= ok
            row.append("%+7.3f [%+6.3f,%+6.3f]%s"
                       % (f["coef"], f["ci_low"], f["ci_high"], " +" if ok else "  "))
        n_pass += ok_all
        out["cells"]["%s|%s" % (b, cfg)] = {
            "seeds_pass": ok_all,
            "per_seed": {s: get(b, cfg, s, PRIMARY) for s in SEEDS}}
        print("%-7s %-16s %s  %s" % (b, cfg, "  ".join(row),
                                     "PASSA" if ok_all else "nao"))
print("\n  %d de 15 celulas com as tres seeds positivas e IC excluindo zero" % n_pass)
out["cells_passing"] = n_pass
out["passed"] = bool(n_pass >= 12)

def cellv(b, cfg, s, r, field):
    c = C.get("%s|%s|%s|%s" % (b, cfg, s, r))
    return None if c is None else c.get(field)


print("\n=== comparacao entre reguas, SEM ESCALA (a bruta confunde efeito e unidade) ===")
print("%-7s %-12s %9s %9s %9s   %s" % ("base", "regua", "coef", "coef_z", "AUC", "d medio"))
for b in BENCH:
    for r in RULERS:
        co = [get(b, cfg, s, r) for cfg in CONFIGS[b] for s in SEEDS]
        co = [x["coef"] for x in co if x]
        cz = [cellv(b, cfg, s, r, "coef_z") for cfg in CONFIGS[b] for s in SEEDS]
        cz = [x for x in cz if x is not None]
        au = [cellv(b, cfg, s, r, "auc") for cfg in CONFIGS[b] for s in SEEDS]
        au = [x for x in au if x is not None]
        dm = [cellv(b, cfg, s, r, "d_mean") for cfg in CONFIGS[b] for s in SEEDS]
        dm = [x for x in dm if x is not None]
        if not au:
            continue
        mark = "  <- primaria" if r == PRIMARY else (
            "  <- contrafactual" if r == "colorhist" else "")
        print("%-7s %-12s %+9.3f %+9.3f %9.3f   %.3f%s"
              % (b, r, sum(co) / len(co), sum(cz) / len(cz), sum(au) / len(au),
                 sum(dm) / len(dm), mark))
        out.setdefault("scale_free", {})["%s|%s" % (b, r)] = {
            "coef": sum(co) / len(co), "coef_z": sum(cz) / len(cz),
            "auc": sum(au) / len(au), "d_mean": sum(dm) / len(dm)}
    print()

print("\n=== controle: dentro do bucket (mesmo dominio) ===")
print("%-7s %-16s %12s %12s %10s" % ("base", "config", "todas", "dentro", "n dentro"))
for b in BENCH:
    for cfg in CONFIGS[b]:
        a = [get(b, cfg, s, PRIMARY) for s in SEEDS]
        w = [get(b, cfg, s, PRIMARY, "within_bucket") for s in SEEDS]
        a = [x["coef"] for x in a if x]
        w2 = [x["fit"]["coef"] for x in w if x]
        n = [x["n_rows"] for x in w if x]
        if not a:
            continue
        print("%-7s %-16s %12.3f %12s %10s"
              % (b, cfg, sum(a) / len(a),
                 "%+.3f" % (sum(w2) / len(w2)) if w2 else "--",
                 "%d" % (sum(n) // len(n)) if n else "--"))
        out["cells"]["%s|%s" % (b, cfg)]["within_bucket_coef"] = \
            (sum(w2) / len(w2)) if w2 else None

print("\n=== referencias, nunca julgadas ===")
for b in BENCH:
    for cfg in ("MEDVLTHINKER_ZEROSHOT", "M0_question_only"):
        f = get(b, cfg, "seed0", PRIMARY)
        c = C.get("%s|%s|seed0|%s" % (b, cfg, PRIMARY))
        if f is None:
            continue
        print("  %-7s %-22s flip %.4f  coef %+8.3f [%+.3f, %+.3f]"
              % (b, cfg, c["flip_rate"], f["coef"], f["ci_low"], f["ci_high"]))

print("\n=== VEREDITO G3 ===")
print("  %d de 15 celulas   ->   %s" % (n_pass, "PASSA" if out["passed"] else "FALHA"))
p = REPO / "docs/v4/g3_results.json"
p.write_text(json.dumps(out, indent=1))
print("\n  gravado %s" % p)
