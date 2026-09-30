"""The only comparison the official split allows: inside the one shared template.

The 237 unseen test images of the official Kvasir split carry exactly one
question each, always `does this image contain any finding?`, always answered
"no". So the seen/unseen accuracy gap of -9.1 pp is not an effect of having seen
the image: it is the trivial template, the same one whose VDD is 0.00 and whose
answer is negative by construction of the benchmark.

The honest comparison is within that template, which is the only one both
groups share. Whether it has any power is itself a question, and is reported.
"""
from __future__ import annotations

import json
from collections import Counter
import os
from pathlib import Path

import numpy as np
import pyarrow as pa

REPO = Path(__file__).resolve().parents[2]
A = REPO / "data/finetune_arrows"
PRED = Path(os.environ.get("V4_RESTORE", str(REPO / "cache/restore"))) / "M3AE_KVASIR_C0.json"
OUT = REPO / "docs/v4/seen_vs_unseen_kvasir.json"
Q = "does this image contain any finding?"
N_BOOT, SEED = 1000, 20260917


def read(p):
    with pa.memory_map(str(A / p), "rb") as s:
        return pa.ipc.open_file(s).read_all()


bank = set()
for f in ("vqa_kvasir_train.arrow", "vqa_kvasir_val.arrow"):
    bank |= {str(x) for x in read(f).column("image_id").to_pylist()}

t = read("vqa_kvasir_test.arrow")
rows = []
for im, qs, ans, qids in zip(t.column("image_id").to_pylist(),
                             t.column("questions").to_pylist(),
                             t.column("answers").to_pylist(),
                             t.column("question_id").to_pylist()):
    for q, a, qid in zip(qs, ans, qids):
        if " ".join(str(q).strip().lower().split()) == Q:
            v = a[0] if isinstance(a, list) else a
            rows.append((int(qid), str(im), str(im) in bank, str(v)))

pred = {int(p["qid"]): bool(p["correct"])
        for p in json.loads(PRED.read_text())["predictions"]}
grp = {True: [], False: []}
ans = {True: Counter(), False: Counter()}
for qid, im, seen, a in rows:
    if qid in pred:
        grp[seen].append(pred[qid])
        ans[seen][a] += 1

print("=== o unico template compartilhado: %r ===" % Q)
res = {}
for seen, tag in ((True, "vistas"), (False, "nao vistas")):
    v = grp[seen]
    acc = (sum(v) / len(v)) if v else None
    res[tag] = {"n": len(v), "accuracy": acc, "answers": dict(ans[seen].most_common(3))}
    print("  %-11s n=%5d  acuracia %s   respostas %s"
          % (tag, len(v), "%.4f" % acc if acc is not None else "--",
             dict(ans[seen].most_common(3))))

if grp[True] and grp[False]:
    rng = np.random.default_rng(SEED)
    a, b = np.array(grp[True], float), np.array(grp[False], float)
    d = a.mean() - b.mean()
    boot = [a[rng.integers(0, len(a), len(a))].mean()
            - b[rng.integers(0, len(b), len(b))].mean() for _ in range(N_BOOT)]
    ci = [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
    print("  diferenca dentro do template %+.4f  [%+.4f, %+.4f]  %s"
          % (d, ci[0], ci[1], "exclui zero" if ci[0] > 0 or ci[1] < 0 else "inclui zero"))
    res["within_template_difference"] = {"diff": float(d), "ci": ci,
                                         "excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}
else:
    print("  sem comparacao possivel: um dos grupos nao tem este template")
    res["within_template_difference"] = None

prev = json.loads(OUT.read_text())
prev["within_template_control"] = {
    "template": Q, "note": ("the 237 unseen images carry only this template, "
                            "always answered no; the naive gap is composition"),
    **res}
prev["naive_gap_is_composition"] = True
OUT.write_text(json.dumps(prev, indent=1))
print("\n  gravado %s" % OUT)
