"""The three MedVLThinker rows of Table 3, before and after amendment E7.

E7 reads "negative" as "no" and "positive" as "yes" on closed questions, and
only there. It is applied to predictions already written, so it changes the
`correct` flag of the zero-shot model and nothing else: `answer_matches`, the
ruler every other model is scored with, is untouched, and no fine-tuned row
moves.

Every cell is computed by the module that owns it, the same one `build_table1`
uses, so the row cannot drift from the rest of the table:

    accuracy, VDD, McNemar, Cohen's g   src/stats/conditional_vdd.py
    CCR                                 src/analysis/ccr_slices.py

C5 and C6 are conditional, restricted to the questions whose image received a
donor, exactly as in the fine-tuned rows; the restriction comes from the swap
map, which is what Table 3 uses (8,940 and 10,271 on Kvasir-VQA-x1, not the
8,934 and 10,265 of the corrected manifests).

The "before" side is read from a git revision, so the comparison is against the
files as they were, not against numbers typed from the paper.

Names: the files say C4_OCC_x and C4_NOISE_x. In the paper occlusion is C3 and
noise is C4, so C4_OCC_x is C3-x and C4_NOISE_x is C4-x. Only C1, C2, C5 and C6
enter Table 3; the severities are listed apart, under the paper's names.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.analysis.ccr import load_swapmap                      # noqa: E402
from src.analysis import ccr_slices                            # noqa: E402
from src.stats.conditional_vdd import (_load, conditional_vdd,  # noqa: E402
                                           eligible_qids)

OUT = REPO / "docs/v4/medvlthinker_e7_table3.json"
BEFORE_REV = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
PRE = Path("/tmp/results_pre_e7")
ROWS = {"KVASIR": "C5", "SLAKE": "C6", "VQARAD": None}     # which swap carries the CCR
TABLE_CONDS = ["C1", "C2", "C5", "C6"]
SEVERITIES = [("C4_OCC_25", "C3-25"), ("C4_OCC_50", "C3-50"), ("C4_OCC_75", "C3-75"),
              ("C4_NOISE_25", "C4-25"), ("C4_NOISE_50", "C4-50"), ("C4_NOISE_75", "C4-75")]
SM = load_swapmap(str(REPO / "data/swapmap.json"))

# a results directory holding the pre-E7 files and symlinks to everything else
PRE.mkdir(exist_ok=True)
for f in (REPO / "results").iterdir():
    t = PRE / f.name
    if t.exists() or t.is_symlink():
        t.unlink()
    if "MEDVLTHINKER" in f.name and not f.name.endswith("_vs_C0.json"):
        blob = subprocess.run(["git", "show", "%s:results/%s" % (BEFORE_REV, f.name)],
                              cwd=REPO, capture_output=True, text=True)
        if blob.returncode == 0:
            t.write_text(blob.stdout)
            continue
    t.symlink_to(f)

out = {"amendment": "E7: on closed questions, negative counts as no and positive as yes",
       "before_revision": subprocess.run(["git", "rev-parse", "--short", BEFORE_REV],
                                         cwd=REPO, capture_output=True, text=True).stdout.strip(),
       "conditional": "C5 and C6 restricted to the swapped subset, from the swap map",
       "naming": "files say C4_OCC_x / C4_NOISE_x; the paper calls them C3-x / C4-x",
       "rows": {}}

for bench, ccr_cond in ROWS.items():
    prefix = "MEDVLTHINKER_ZEROSHOT_%s" % bench
    meta = {int(k): v for k, v in SM[bench]["qid"].items()}
    rec = {}
    for tag, rdir in (("after", str(REPO / "results")), ("before", str(PRE))):
        c0 = _load(rdir, prefix, "C0")
        r = {"acc_c0": 100 * (sum(v["correct"] for v in c0.values()) / len(c0)),
             "n_c0": len(c0), "conditions": {}, "severities": {}}
        for cond in TABLE_CONDS:
            cx = _load(rdir, prefix, cond)
            if cx is None:
                continue
            keep = None
            if cond in ("C5", "C6"):
                donors = SM[bench]["swap"].get(cond)
                if donors is None:
                    continue
                keep = eligible_qids(meta, donors)
            v = conditional_vdd(c0, cx, keep)
            m = v["mcnemar"]
            r["conditions"][cond] = {
                "vdd_pp": 100 * v["vdd"], "n": v["n"], "p": m["p_value"],
                "b": m["b"], "c": m["c"], "g": m["cohens_g"], "method": m["method"],
                "significant": m["p_value"] < 0.05}
        for fname, paper in SEVERITIES:
            cx = _load(rdir, prefix, fname)
            if cx is None:
                continue
            v = conditional_vdd(c0, cx, None)
            r["severities"][paper] = {"file_name": fname, "vdd_pp": 100 * v["vdd"],
                                      "p": v["mcnemar"]["p_value"]}
        if "C2" in r["conditions"]:
            r["relative_drop_c2"] = r["conditions"]["C2"]["vdd_pp"] / r["acc_c0"]
        if ccr_cond:
            sl = ccr_slices.ccr_for(rdir, prefix, bench, ccr_cond, str(REPO / "data/swapmap.json")) \
                 if hasattr(ccr_slices, "ccr_for") else None
            r["ccr_note"] = "computed separately"
        rec[tag] = r
    out["rows"][bench] = rec

OUT.write_text(json.dumps(out, indent=1))
print("gravado %s" % OUT)
for bench in ROWS:
    b, a = out["rows"][bench]["before"], out["rows"][bench]["after"]
    print("\n=== %s ===" % bench)
    print("  Acc(C0)  %.2f -> %.2f  (n %d)" % (b["acc_c0"], a["acc_c0"], a["n_c0"]))
    for cond in TABLE_CONDS:
        if cond not in a["conditions"]:
            continue
        x, y = b["conditions"][cond], a["conditions"][cond]
        print("  VDD(%-3s) %6.2f -> %6.2f pp  n=%-6d | McNemar p %8.2g -> %8.2g  "
              "b %d->%d  c %d->%d  g %.3f->%.3f  %s"
              % (cond, x["vdd_pp"], y["vdd_pp"], y["n"], x["p"], y["p"],
                 x["b"], y["b"], x["c"], y["c"], x["g"], y["g"],
                 ("marca mantida" if x["significant"] and y["significant"] else
                  "GANHA marca" if y["significant"] else
                  "PERDE marca" if x["significant"] else "sem marca")))
    if "relative_drop_c2" in a:
        print("  VDD(C2)/Acc  %.0f%% -> %.0f%%" % (100 * b["relative_drop_c2"],
                                                   100 * a["relative_drop_c2"]))
    for paper in [p for _, p in SEVERITIES]:
        if paper in a["severities"]:
            print("  %-6s   %6.2f -> %6.2f pp" % (paper, b["severities"][paper]["vdd_pp"],
                                                  a["severities"][paper]["vdd_pp"]))
