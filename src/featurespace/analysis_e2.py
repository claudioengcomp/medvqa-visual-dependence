"""E2 — what CORAL's hard negative actually selects, measured without a model.

For each test item (x_i, q_i, y_i), the hard negative is the pool image nearest
to x_i by cosine, subject to `label(x_j) != y_i`, where `label` is the gold
answer of item j to **its own** question and each (x_j, q_j) pair is a separate
candidate, as their pool is built. The search is exact; x_j = x_i is excluded.

The restriction can only ever exclude an image whose every question carries
`y_i`. So the **fraction of pool images holding two or more distinct labels is
the ceiling on what the restriction can filter**: an image with two different
answers among its questions survives the restriction for any y_i whatsoever. On
a closed benchmark, where the labels are yes and no, that fraction is the share
of images carrying both — and if it is high, the restriction is not choosing a
donor, it is choosing a question.

Measured per benchmark and per search encoder, with no model run at any point:
|V|_HN, phi_HN, the share whose donor lacks q_i, the share of donors
byte-identical to the original and within pHash <= 4, the cosine and its range,
and the distance under MedSigLIP against the terciles of C2.

The reading rule was fixed in the pre-registration, before these numbers
existed: phi_HN >= 0.9 with |V|_HN >= 20 in more than half the items makes the
HN-Gap interpretable; |V|_HN below that, or phi_HN < 0.7, gives the criticism of
the restriction a number.

VQA-RAD is excluded (one informative question) and PathVQA is excluded (no
question-level gold structure here); both were declared in advance.
"""
from __future__ import annotations

import hashlib
import json
import socket
import sys
from collections import Counter, defaultdict
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
         # PathVQA joins under amendment E.1: the exclusion's stated reason
         # stopped holding once E1 built its arrow
         "PATHVQA": (P0A, "vqa_path_vqa_test")}
ENCODERS = ["biomedclip", "clipL14"]
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
MAN = {"KVASIR": "C2_manifest_KVASIR.json", "SLAKE": "C2_manifest_SLAKE.json",
       "PATHVQA": "C2_manifest_PATHVQA.json"}
C5MAN = {"KVASIR": "C5_manifest_KVASIR_clean.json", "SLAKE": "C5_manifest_SLAKE.json"}
IDS = json.loads((CACHE / "image_ids.json").read_text())
HASH = json.loads((Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "image_hashes.json").read_text())


def label(i):
    p = [x for x in str(i).replace("//", "/").split("/") if x]
    if p and p[-1].lower() in ("source.jpg", "source.png"):
        return p[-2] if len(p) > 1 else p[-1]
    return str(i)


def norm_ans(a):
    v = a[0] if isinstance(a, list) else a
    return str(v).strip().lower()


def load_items(bench):
    d, stem = BENCH[bench]
    with pa.memory_map(str(d / ("%s.arrow" % stem)), "rb") as s:
        t = pa.ipc.open_file(s).read_all()
    items, by_image = [], defaultdict(dict)
    for im, qs, ans in zip(t.column("image_id").to_pylist(),
                           t.column("questions").to_pylist(),
                           t.column("answers").to_pylist()):
        im = label(im)
        for q, a in zip(qs, ans):
            items.append({"img": im, "q": str(q), "qk": question_key(q), "y": a,
                          "ylab": norm_ans(a)})
            by_image[im][question_key(q)] = a
    return items, by_image


def arr_path(bench, ruler):
    """Two naming conventions live in this cache: the first three benchmarks
    were written as BENCH__ruler.npy and PathVQA and HEAL as
    BENCH__C0__ruler.npy. CLIP was written under the first for every
    benchmark, so both have to be tried."""
    a = CACHE / ("%s__%s.npy" % (bench, ruler))
    b = CACHE / ("%s__C0__%s.npy" % (bench, ruler))
    return a if a.exists() else b


def vectors(bench, ruler):
    f = arr_path(bench, ruler)
    X = np.load(f).astype(np.float64)
    ids = [label(i) for i in IDS["%s__C0" % bench]]
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    return ids, X


def donor_map_of(bench, which):
    """PathVQA has no C5 -- it was never built for it -- so the lookup returns
    None instead of raising, the way VQA-RAD's missing C5 is handled elsewhere."""
    table = MAN if which == "C2" else C5MAN
    if bench not in table:
        return None
    f = REPO / "results" / table[bench]
    d = json.loads(f.read_text())
    rows = d["pairs"] if isinstance(d, dict) else d
    return {label(e["qid_image"]): label(e["donor_image"]) for e in rows}


out = json.loads(OUT.read_text()) if OUT.exists() else {
    "host": os.environ.get("RUN_HOST", "unspecified"),
    "excluded": {"VQARAD": "one informative question; phi has no denominator"},
    "pathvqa_note": ("excluded in the pre-registration for want of question-level "
                     "gold structure; that reason stopped holding when E1 built "
                     "the arrow, and amendment E.1 declares the lift"),
    "reading_rule": ("phi_HN >= 0.9 and |V|_HN >= 20 in more than half the items "
                     "-> HN-Gap interpretable; |V|_HN below the floor or "
                     "phi_HN < 0.7 -> the restriction's criticism gains a number"),
    "benchmarks": {}}

for bench in BENCH:
    items, by_image = load_items(bench)
    images = sorted({it["img"] for it in items})
    labels_of = defaultdict(set)
    for it in items:
        labels_of[it["img"]].add(it["ylab"])
    multi = [i for i in images if len(labels_of[i]) >= 2]
    binary = [i for i in images if {"yes", "no"} <= labels_of[i]]
    rec = out["benchmarks"].setdefault(bench, {})
    rec["pool"] = {
        "images": len(images), "items": len(items),
        "items_per_image_mean": round(len(items) / len(images), 3),
        "images_with_two_or_more_labels": len(multi),
        "filter_ceiling": round(1 - len(multi) / len(images), 4),
        "images_with_both_yes_and_no": len(binary),
        "fraction_yes_no": round(len(binary) / len(images), 4),
        "note": ("an image with two distinct labels survives the restriction for "
                 "any y_i, so the ceiling is the share the restriction could "
                 "ever exclude")}
    print("\n=== %s  pool: %d imagens, %d itens (%.2f por imagem)"
          % (bench, len(images), len(items), len(items) / len(images)))
    print("  imagens com >= 2 rotulos distintos: %d de %d  ->  a restricao pode "
          "excluir no maximo %.1f%% do pool"
          % (len(multi), len(images), 100 * (1 - len(multi) / len(images))))
    print("  imagens que carregam yes E no:      %d de %d (%.1f%%)"
          % (len(binary), len(images), 100 * len(binary) / len(images)))

    # PathVQA has no entry in the pHash cache, which was built for the three
    # original benchmarks; its byte and pHash columns are reported as absent
    # rather than silently filled with zeros
    H = HASH.get(bench, {}).get("test")
    sha_of = ({label(i): s for i, s in zip(H["ids"], H["sha"])} if H else {})
    ph_of = ({label(i): np.asarray(p, np.uint8)
              for i, p in zip(H["ids"], H["phash"])} if H else {})
    ids_med, Xmed = vectors(bench, "medsiglip")
    pos_med = {i: k for k, i in enumerate(ids_med)}

    for enc in ENCODERS:
        f = arr_path(bench, enc)
        if not f.exists():
            print("  %s: embedding %s ausente" % (bench, enc))
            continue
        ids, X = vectors(bench, enc)
        pos = {i: k for k, i in enumerate(ids)}
        S = X @ X.T
        np.fill_diagonal(S, -np.inf)
        order = np.argsort(-S, axis=1)

        donors, cos, miss = [], [], 0
        for it in items:
            k = pos[it["img"]]
            pick = None
            for j in order[k]:
                cand = ids[j]
                if cand == it["img"]:
                    continue
                if labels_of[cand] - {it["ylab"]}:      # has some other label
                    pick = (cand, float(S[k, j]))
                    break
            if pick is None:
                miss += 1
                donors.append(None)
                cos.append(np.nan)
            else:
                donors.append(pick[0])
                cos.append(pick[1])

        I = U = N = 0
        same_bytes = near_ph = 0
        dmed = []
        for it, don in zip(items, donors):
            if don is None:
                continue
            ref = by_image.get(don, {}).get(it["qk"])
            if ref is None:
                N += 1
            elif any(answers_match(x, ref) for x in it["y"]):
                U += 1
            else:
                I += 1
            if sha_of and sha_of.get(don) == sha_of.get(it["img"]):
                same_bytes += 1
            if don in ph_of and it["img"] in ph_of:
                if int(POP[np.bitwise_xor(ph_of[don], ph_of[it["img"]])].sum()) <= 4:
                    near_ph += 1
            a, b = pos_med.get(it["img"]), pos_med.get(don)
            if a is not None and b is not None:
                dmed.append(1.0 - float(Xmed[a] @ Xmed[b]))
        V = I + U
        n = len(items)
        cosa = np.asarray([c for c in cos if not np.isnan(c)])
        dmed = np.asarray(dmed)
        r = {"encoder": enc, "items": n, "no_donor_found": miss,
             "V_HN": V, "I": I, "U": U, "N": N,
             "phi_HN": (I / V) if V else None,
             "fraction_without_question": N / n,
             "donor_byte_identical": (same_bytes / n) if sha_of else None,
             "donor_phash_le_4": (near_ph / n) if ph_of else None,
             "cosine_mean": float(cosa.mean()), "cosine_min": float(cosa.min()),
             "cosine_max": float(cosa.max()),
             "d_medsiglip_mean": float(dmed.mean()),
             "d_medsiglip_median": float(np.median(dmed))}
        rec.setdefault("hard_negative", {})[enc] = r
        print("  %-9s |V| %5d  phi %s  N %.3f  bytes %s  pHash<=4 %s  "
              "cos %.3f [%.3f, %.3f]  d_med %.4f"
              % (enc, V, "%.4f" % r["phi_HN"] if r["phi_HN"] else "--",
                 r["fraction_without_question"],
                 "%.3f" % r["donor_byte_identical"]
                 if r["donor_byte_identical"] is not None else " -- ",
                 "%.3f" % r["donor_phash_le_4"]
                 if r["donor_phash_le_4"] is not None else " -- ",
                 r["cosine_mean"], r["cosine_min"],
                 r["cosine_max"], r["d_medsiglip_mean"]), flush=True)
        OUT.write_text(json.dumps(out, indent=1))

    # C2 and C5 beside it, on the same quantities
    for which in ("C2", "C5"):
        try:
            dm = donor_map_of(bench, which)
        except FileNotFoundError:
            dm = None
        if dm is None:
            print("  %-9s ausente por construcao nesta base" % which)
            continue
        I = U = N = 0
        dmed = []
        for it in items:
            don = dm.get(it["img"])
            if don is None:
                continue
            ref = by_image.get(don, {}).get(it["qk"])
            if ref is None:
                N += 1
            elif any(answers_match(x, ref) for x in it["y"]):
                U += 1
            else:
                I += 1
            a, b = pos_med.get(it["img"]), pos_med.get(don)
            if a is not None and b is not None:
                dmed.append(1.0 - float(Xmed[a] @ Xmed[b]))
        V = I + U
        dmed = np.asarray(dmed)
        rec.setdefault("reference", {})[which] = {
            "V": V, "I": I, "U": U, "N": N, "phi": (I / V) if V else None,
            "d_medsiglip_mean": float(dmed.mean()) if len(dmed) else None}
        print("  %-9s |V| %5d  phi %s  d_med %s   <- referencia"
              % (which, V, "%.4f" % (I / V) if V else "--",
                 "%.4f" % dmed.mean() if len(dmed) else "--"))

OUT.write_text(json.dumps(out, indent=1))
print("\n  gravado %s" % OUT)
