"""The CCR cell of the MedVLThinker rows, with and without amendment E7.

E7 is applied when a prediction is scored against its gold answer, and it
rewrites nothing: the files keep `predicted` = "negative" and flip only
`correct`. Accuracy and VDD read `correct` and therefore move; the CCR does not,
because it compares the prediction with the *donor's* reference through
`answer_matches`, the shared ruler, which never saw E7.

So the published CCR of the zero-shot row is on a different ruler from the Acc
and VDD cells beside it. This script reports the cell both ways and leaves the
choice to the writing: `ccr_shared_ruler` is what Table 3 carries today,
`ccr_e7` applies the same synonym rule to the comparison with the donor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.analysis.ccr import (answers_match, load_predictions,   # noqa: E402
                                  load_swapmap, question_key)

OUT = REPO / "docs/v4/medvlthinker_ccr_e7.json"
SM = load_swapmap(str(REPO / "data/swapmap.json"))
RES = str(REPO / "results")
SYN = {"negative": "no", "positive": "yes"}
CLOSED = {"yes", "no"}


def e7(pred, refs):
    """answers_match, plus: on a closed reference, negative is no and positive is yes."""
    if answers_match(pred, refs):
        return True
    p = str(pred).strip().lower()
    if p in SYN and any(str(r).strip().lower() in CLOSED for r in refs):
        return answers_match(SYN[p], refs)
    return False


out = {}
for bench, cond in (("KVASIR", "C5"), ("SLAKE", "C6")):
    prefix = "MEDVLTHINKER_ZEROSHOT_%s" % bench
    meta = {int(k): v for k, v in SM[bench]["qid"].items()}
    donors = SM[bench]["swap"][cond]
    by_image = {}
    for q, m in meta.items():
        by_image.setdefault(m["img"], {})[question_key(m["q"])] = m["a"]
    c0 = load_predictions(RES, prefix, "C0")
    cx = load_predictions(RES, prefix, cond)
    n = shared = e7hit = 0
    for q, m in meta.items():
        if m["img"] not in donors or q not in cx:
            continue
        ref = by_image.get(donors[m["img"]], {}).get(question_key(m["q"]))
        if ref is None or any(answers_match(x, ref) for x in m["a"]):
            continue
        n += 1
        p = cx[q]["predicted"]
        shared += answers_match(p, ref)
        e7hit += e7(p, ref)
    out[bench] = {"condition": cond, "informative": n,
                  "ccr_shared_ruler": 100 * shared / n,
                  "ccr_e7": 100 * e7hit / n,
                  "note": "the row's Acc and VDD cells use E7; this CCR cell does not"}
    print("  %-7s %s  |I| %d  CCR regua compartilhada %.2f  CCR sob E7 %.2f"
          % (bench, cond, n, out[bench]["ccr_shared_ruler"], out[bench]["ccr_e7"]))
OUT.write_text(json.dumps(out, indent=1))
print("  gravado %s" % OUT)
