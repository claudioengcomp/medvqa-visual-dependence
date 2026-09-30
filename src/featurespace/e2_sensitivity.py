"""E2d — the same search under the restriction the method should have used.

CORAL requires `label(x_j) != y_i`, where label is the donor's answer to its own
question. What makes a swap informative is different: the reference of **q_i**
on the donor must differ from y_i. The first is a property of the donor's own
annotation, the second of the pair.

Same encoders, same pool, same exact search — only the acceptance test changes,
to: the donor carries q_i, and its reference for q_i differs from y_i. Reported
is how many items find a donor at all under it, and what the resulting map looks
like beside the original.

This is C5 built by neighbourhood rather than by label, and it says what the
hard negative would have selected had the restriction matched the intent.
"""
from __future__ import annotations

import json
import socket
import sys
from collections import defaultdict
import os
from pathlib import Path

import numpy as np
import pyarrow as pa

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.analysis.ccr import answers_match, question_key   # noqa: E402

CACHE = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "embeddings"
M3 = Path(os.environ.get("M3AE_ARROWS", str(REPO / "data/finetune_arrows")))
P0A = REPO / "data/finetune_arrows"
OUT = REPO / "docs/v4/analysis_e2.json"
BENCH = {"KVASIR": (P0A, "vqa_kvasir_test_clean"), "SLAKE": (M3, "vqa_slack_test"),
         "PATHVQA": (P0A, "vqa_path_vqa_test")}
ENCODERS = ["biomedclip", "clipL14"]
IDS = json.loads((CACHE / "image_ids.json").read_text())


def label(i):
    p = [x for x in str(i).replace("//", "/").split("/") if x]
    if p and p[-1].lower() in ("source.jpg", "source.png"):
        return p[-2] if len(p) > 1 else p[-1]
    return str(i)


def load(bench):
    d, stem = BENCH[bench]
    with pa.memory_map(str(d / ("%s.arrow" % stem)), "rb") as s:
        t = pa.ipc.open_file(s).read_all()
    items, by_image = [], defaultdict(dict)
    for im, qs, ans in zip(t.column("image_id").to_pylist(),
                           t.column("questions").to_pylist(),
                           t.column("answers").to_pylist()):
        im = label(im)
        for q, a in zip(qs, ans):
            items.append({"img": im, "qk": question_key(q), "y": a})
            by_image[im][question_key(q)] = a
    return items, by_image


def arr_path(bench, ruler):
    """Both cache naming conventions; see analysis_e2.py."""
    a = CACHE / ("%s__%s.npy" % (bench, ruler))
    b = CACHE / ("%s__C0__%s.npy" % (bench, ruler))
    return a if a.exists() else b


def vectors(bench, ruler):
    X = np.load(arr_path(bench, ruler)).astype(np.float64)
    ids = [label(i) for i in IDS["%s__C0" % bench]]
    return ids, X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


out = json.loads(OUT.read_text())
print("=== E2d: a restricao correta -- a referencia de q_i na doadora difere de y_i ===")
print("%-8s %-11s %8s %10s %12s %10s %10s"
      % ("base", "encoder", "itens", "com doadora", "%", "cos medio", "d_med"))
for bench in BENCH:
    items, by_image = load(bench)
    ids_med, Xmed = vectors(bench, "medsiglip")
    pos_med = {i: k for k, i in enumerate(ids_med)}
    # which images carry a given question with which reference
    carriers = defaultdict(list)
    for im, qs in by_image.items():
        for qk in qs:
            carriers[qk].append(im)
    for enc in ENCODERS:
        f = arr_path(bench, enc)
        if not f.exists():
            continue
        ids, X = vectors(bench, enc)
        pos = {i: k for k, i in enumerate(ids)}
        S = X @ X.T
        np.fill_diagonal(S, -np.inf)
        order = np.argsort(-S, axis=1)

        found, cos, dmed = 0, [], []
        for it in items:
            k = pos[it["img"]]
            ok = None
            for j in order[k]:
                cand = ids[j]
                if cand == it["img"]:
                    continue
                ref = by_image.get(cand, {}).get(it["qk"])
                if ref is None:
                    continue
                if not any(answers_match(x, ref) for x in it["y"]):
                    ok = (cand, float(S[k, j]))
                    break
            if ok is None:
                continue
            found += 1
            cos.append(ok[1])
            a, b = pos_med.get(it["img"]), pos_med.get(ok[0])
            if a is not None and b is not None:
                dmed.append(1.0 - float(Xmed[a] @ Xmed[b]))
        cos = np.asarray(cos)
        dmed = np.asarray(dmed)
        rec = {"encoder": enc, "items": len(items), "items_with_donor": found,
               "fraction_with_donor": found / len(items),
               "cosine_mean": float(cos.mean()) if len(cos) else None,
               "cosine_min": float(cos.min()) if len(cos) else None,
               "d_medsiglip_mean": float(dmed.mean()) if len(dmed) else None,
               "phi_by_construction": 1.0,
               "note": ("under this restriction every accepted pair is "
                        "informative by construction, so phi is 1 and the "
                        "question becomes how many items find a donor at all")}
        out["benchmarks"][bench].setdefault("correct_restriction", {})[enc] = rec
        print("%-8s %-11s %8d %10d %11.1f%% %10s %10s"
              % (bench, enc, len(items), found, 100 * found / len(items),
                 "%.3f" % cos.mean() if len(cos) else "--",
                 "%.4f" % dmed.mean() if len(dmed) else "--"), flush=True)

out["e2d_note"] = ("the correct restriction makes phi 1 by construction; what it "
                   "costs is coverage, and that is the number reported")
OUT.write_text(json.dumps(out, indent=1))
print("\n  gravado %s" % OUT)
