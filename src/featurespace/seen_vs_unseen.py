"""Seen against unseen images, on the official Kvasir split (§5).

The official Kvasir-VQA-x1 split puts the same image in training and in test:
3,821 of its 4,058 test images have an image_id that also appears in train or
validation, and 237 do not. The question is what that buys a model.

No inference was needed. The predictions of M3AE on the official split are the
run the repository kept as `_DESCARTADO_split_vazado_kvasir_prefix` - discarded
for the paper's main results precisely because the split leaks, which is exactly
what makes it the right run for this measurement. Restored from Drive with its
md5 checked; the checkpoints themselves are gone from host-A and were not needed.

Accuracy is compared on the questions of seen images against those of unseen
images, with a bootstrap over **images**: the questions of one image share its
seen/unseen status, so resampling questions would report an interval the design
does not support. The difference is bootstrapped directly rather than read off
two overlapping intervals.
"""
from __future__ import annotations

import json
import socket
import os
from pathlib import Path

import numpy as np
import pyarrow as pa

REPO = Path(__file__).resolve().parents[2]
A = REPO / "data/finetune_arrows"
PRED = Path(os.environ.get("V4_RESTORE", str(REPO / "cache/restore"))) / "M3AE_KVASIR_C0.json"
OUT = REPO / "docs/v4/seen_vs_unseen_kvasir.json"
N_BOOT, SEED = 1000, 20260917


def read(p):
    with pa.memory_map(str(A / p), "rb") as s:
        return pa.ipc.open_file(s).read_all()


bank = set()
for f in ("vqa_kvasir_train.arrow", "vqa_kvasir_val.arrow"):
    bank |= {str(x) for x in read(f).column("image_id").to_pylist()}

test = read("vqa_kvasir_test.arrow")
img_of_q, seen_of_img = {}, {}
for im, qids in zip(test.column("image_id").to_pylist(),
                    test.column("question_id").to_pylist()):
    im = str(im)
    seen_of_img[im] = im in bank
    for q in qids:
        img_of_q[int(q)] = im

pred = {int(p["qid"]): bool(p["correct"])
        for p in json.loads(PRED.read_text())["predictions"]}
per_img = {}
for q, ok in pred.items():
    im = img_of_q.get(q)
    if im is None:
        continue
    a = per_img.setdefault(im, [0, 0])
    a[0] += 1
    a[1] += ok

imgs = sorted(per_img)
seen = [i for i in imgs if seen_of_img[i]]
unseen = [i for i in imgs if not seen_of_img[i]]


def acc(group):
    n = sum(per_img[i][0] for i in group)
    c = sum(per_img[i][1] for i in group)
    return (c / n) if n else None, n


a_s, n_s = acc(seen)
a_u, n_u = acc(unseen)
rng = np.random.default_rng(SEED)
bs, bu, bd = [], [], []
for _ in range(N_BOOT):
    ps = [seen[k] for k in rng.integers(0, len(seen), len(seen))]
    pu = [unseen[k] for k in rng.integers(0, len(unseen), len(unseen))]
    x, _ = acc(ps)
    y, _ = acc(pu)
    bs.append(x)
    bu.append(y)
    bd.append(x - y)


def ci(v):
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


out = {"host": os.environ.get("RUN_HOST", "unspecified"), "n_boot": N_BOOT, "seed": SEED,
       "predictions": PRED.name,
       "predictions_note": ("M3AE on the official split, the run the repository "
                            "kept as _DESCARTADO_split_vazado_kvasir_prefix; "
                            "restored from Drive, md5 verified"),
       "definition_seen": "image_id present in train or validation of the official split",
       "images": {"total": len(imgs), "seen": len(seen), "unseen": len(unseen)},
       "questions": {"seen": n_s, "unseen": n_u, "total": n_s + n_u},
       "accuracy_seen": a_s, "ci_seen": ci(bs),
       "accuracy_unseen": a_u, "ci_unseen": ci(bu),
       "difference": a_s - a_u, "ci_difference": ci(bd),
       "excludes_zero": bool(ci(bd)[0] > 0 or ci(bd)[1] < 0)}

print("=== vistas x nao vistas, split oficial do Kvasir ===")
print("  imagens   %5d total, %5d vistas, %4d nao vistas"
      % (len(imgs), len(seen), len(unseen)))
print("  perguntas %5d total, %5d vistas, %4d nao vistas" % (n_s + n_u, n_s, n_u))
print()
print("  acuracia nas VISTAS      %.4f  [%.4f, %.4f]" % (a_s, *out["ci_seen"]))
print("  acuracia nas NAO VISTAS  %.4f  [%.4f, %.4f]" % (a_u, *out["ci_unseen"]))
print("  diferenca                %+.4f  [%+.4f, %+.4f]  %s"
      % (out["difference"], *out["ci_difference"],
         "exclui zero" if out["excludes_zero"] else "inclui zero"))
print()
print("  para comparacao, a mesma arquitetura no split LIMPO: 0.8723")
print("  (media das tres seeds do M3AE_KVASIR_seed*_C0, 10.779 perguntas)")

OUT.write_text(json.dumps(out, indent=1))
print("\n  gravado %s" % OUT)
