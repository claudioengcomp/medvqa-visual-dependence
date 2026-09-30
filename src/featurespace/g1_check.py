"""G1, in the order fixed with the advisor, under amendment 7.7.

  1. swaps equal to C0, in three layers;
  2. C1 above everything;
  3. monotonicity within C3 (25 -> 50 -> 75) and within C4, separately.
     C3 is never compared against C4: the severities are not commensurable
     between occlusion and noise.

**Provenance.** The pass of 17/09 that fills `analysis_a.json` began before
amendment 7.7 and does not sort by image_id, so its set statistics carry the
reduction-order noise a permutation produces -- up to 1.59e-6 relative in the
Frechet, above the gate's own tolerance, with no image having changed. Check 1
is therefore recomputed here from the cache under canonical order, where that
noise is zero by construction; checks 2 and 3 come from the 17/09 pass, both
comparing quantities that a permutation does not reorder between conditions.
Each check records which source it used.

**The three layers of check 1.** A swap moves images between rows and changes
nothing else, so the question "was an image re-encoded" has an exact answer in
the bytes and only an approximate one in the vectors.

  layer 1  sha256 multiset and the image_id-to-sha256 map against C0. Exact,
           no tolerance. This is the re-encode criterion: a failure here means
           an image changed.
  layer 2  the canonically ordered embedding matrices, compared with
           array_equal. Differences here are non-determinism of the embedding
           step -- the same image, embedded at a different row position, comes
           out slightly different in MedSigLIP, DINOv3 and BiomedCLIP and
           identical in DINOv2 and the colour histogram. Reported, not judged.
  layer 3  knn and Frechet on the canonically ordered arrays, against 1e-6
           relative. With layer 1 passing, whatever appears here traces to the
           rows layer 2 names.

C1 is one point: every row carries the same image, the covariance is degenerate,
and a Frechet over it is a number the definition does not support. C1 is judged
on knn and the sliced Wasserstein, and Figure 3 carries it as a separate marker.

Figure 3 uses distance RELATIVE to C0. C0 is not at zero: it is the distance
from the test set to the training bank, which on SLAKE is a Frechet of 22.2 over
96 images. Plotting absolute distance would have a reader asking why the intact
condition is off-distribution.
"""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import numpy as np
from scipy import linalg

REPO = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "embeddings"
A = json.loads((REPO / "docs/v4/analysis_a.json").read_text())["results"]
UNIQ = json.loads((REPO / "docs/v4/unique_images_per_condition.json").read_text())
NN1 = json.loads((REPO / "docs/v4/nn1_to_bank.json").read_text())["benchmarks"]
BYTES = json.loads((REPO / "docs/v4/swap_bytes_identity.json").read_text())
IDS = json.loads((CACHE / "image_ids.json").read_text())

RULERS = ["medsiglip", "dinov3", "biomedclip", "dinov2", "colorhist"]
# section 1 of the pre-registration lists four rulers; the colour histogram was
# added by amendment 7.4 as the counterfactual of contribution 2, not as a
# ruler. Amendment 7.9 scopes check 3 accordingly.
CONTENT_RULERS = ["medsiglip", "dinov3", "biomedclip", "dinov2"]
BENCH = ["KVASIR", "SLAKE", "VQARAD"]
SWAPS = ["C2", "C5", "C6"]
OCC = ["C4_OCC_25", "C4_OCC_50", "C4_OCC_75"]
NOISE = ["C4_NOISE_25", "C4_NOISE_50", "C4_NOISE_75"]
PAPER = {"C4_OCC_25": "C3-25", "C4_OCC_50": "C3-50", "C4_OCC_75": "C3-75",
         "C4_NOISE_25": "C4-25", "C4_NOISE_50": "C4-50", "C4_NOISE_75": "C4-75"}
REL_TOL = 1e-6
K = 10
HAS_DISP = "disp" in inspect.signature(linalg.sqrtm).parameters


def g(b, c, r, k):
    return A.get("%s|%s|%s" % (b, c, r), {}).get(k)


def uniq(b, c):
    return UNIQ.get(b, {}).get(PAPER.get(c, c), {}).get("unique_image_id")


def nn1(b, c, r, k="nn1_mean"):
    return NN1.get(b, {}).get("cells", {}).get("%s|%s" % (PAPER.get(c, c), r), {}).get(k)


def arr(b, c, r):
    n = ("%s__%s.npy" % (b, r)) if c == "C0" else ("%s__%s__%s.npy" % (b, c, r))
    p = CACHE / n
    return np.load(p) if p.exists() else None


def canon(b, c, r):
    """Embeddings of condition c, rows ordered by image_id (amendment 7.7)."""
    X = arr(b, c, r)
    i = IDS.get("%s__%s" % (b, c))
    if X is None or i is None:
        return None, None
    o = np.argsort(np.asarray(i, dtype=str), kind="stable")
    return X[o], [i[k] for k in o]


def unit(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def knn_mean(X, bank, k=K, block=512):
    Z, B = unit(X), unit(bank)
    out = np.empty(len(Z))
    for i in range(0, len(Z), block):
        S = Z[i:i + block] @ B.T
        part = np.partition(-S, k - 1, axis=1)[:, :k]
        out[i:i + block] = 1.0 - (-part).mean(1)
    return float(out.mean())


def frechet(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    cx, cy = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)
    cm = linalg.sqrtm(cx @ cy, disp=False)[0] if HAS_DISP else linalg.sqrtm(cx @ cy)
    if np.iscomplexobj(cm):
        cm = cm.real
    return float(((mx - my) ** 2).sum() + np.trace(cx + cy - 2 * cm))


out = {"tolerance": "exact equality up to %g relative" % REL_TOL,
       "amendment": "7.7 -- set statistics over embeddings ordered by image_id",
       "c1_frechet": "undefined: one distinct image, covariance is degenerate",
       "provenance": {
           "check_1": "recomputed from the embedding cache under canonical order",
           "check_2": "analysis_a.json, pass of 17/09",
           "check_3": "analysis_a.json, pass of 17/09",
           "check_4": "nn1_to_bank.json",
           "figure3": "analysis_a.json, pass of 17/09"},
       "checks": {}}

print("=== 0. IMAGENS UNICAS POR CONDICAO (A e sobre imagens, nao linhas) ===")
rows0, fail0 = [], []
for b in BENCH:
    n0 = uniq(b, "C0")
    parts = ["%-7s C0 %4d" % (b, n0 or -1)]
    for c in SWAPS:
        n = uniq(b, c)
        if n is None:
            parts.append("%s   --" % c)
            continue
        ok = n == n0
        parts.append("%s %4d%s" % (c, n, "" if ok else " DIFERE"))
        rows0.append({"benchmark": b, "condition": c, "unique_image_id": n,
                      "c0": n0, "equal": ok})
        if not ok:
            fail0.append((b, c, n, n0))
    print("  " + "   ".join(parts))
print("  %s" % ("todas as trocas com o mesmo numero de imagens unicas do C0"
                if not fail0 else "DIFERENCA em %d trocas" % len(fail0)))
out["checks"]["swap_image_count_equals_c0"] = {"passed": not fail0, "rows": rows0,
                                               "failures": fail0}

print("\n=== 1a. REENCODE, PERGUNTADO AOS BYTES (exato) ===")
rows1a, fail1a = [], []
for b in BENCH:
    for c in SWAPS:
        rec = BYTES.get(b, {}).get(c)
        if rec is None:
            continue
        ok = rec["same_sha_multiset"] and rec["same_id_to_sha_map"]
        rows1a.append({"benchmark": b, "condition": c, **rec, "passed": ok})
        print("  %-7s %-3s multiconjunto %-4s mapa id->sha %-4s  linhas com id trocado %4d de %4d"
              % (b, c, "sim" if rec["same_sha_multiset"] else "NAO",
                 "sim" if rec["same_id_to_sha_map"] else "NAO",
                 rec["rows_whose_image_id_changed"], rec["n_rows"]))
        if not ok:
            fail1a.append((b, c))
print("  %s" % ("nenhuma imagem foi reencodada em nenhuma troca" if not fail1a
                else "REENCODE em %d trocas: %s" % (len(fail1a), fail1a)))
out["checks"]["swaps_no_reencode_bytes"] = {"passed": not fail1a, "rows": rows1a,
                                            "failures": fail1a, "layer": 1}

print("\n=== 1b. REPRODUTIBILIDADE DO EMBEDDING, ORDEM CANONICA (relatado) ===")
rows1b = []
for b in BENCH:
    for r in RULERS:
        A0, i0 = canon(b, "C0", r)
        if A0 is None:
            continue
        for c in SWAPS:
            B0, ic = canon(b, c, r)
            if B0 is None:
                continue
            if i0 != ic:
                rows1b.append({"benchmark": b, "condition": c, "ruler": r,
                               "id_sets_match": False})
                print("  %-7s %-3s %-11s conjuntos de id NAO batem" % (b, c, r))
                continue
            d = np.abs(A0.astype(np.float64) - B0.astype(np.float64))
            n = int((d > 0).any(1).sum())
            rows1b.append({"benchmark": b, "condition": c, "ruler": r,
                           "id_sets_match": True, "rows": int(len(A0)),
                           "rows_differing": n, "max_abs_diff": float(d.max())})
            if n:
                print("  %-7s %-3s %-11s %4d de %4d linhas diferem, max %.3e"
                      % (b, c, r, n, len(A0), d.max()))
tot = sum(x.get("rows_differing", 0) for x in rows1b)
cells = sum(1 for x in rows1b if x.get("rows_differing"))
print("  %d de %d celulas bit-identicas; %d linhas divergentes ao todo"
      % (len(rows1b) - cells, len(rows1b), tot))
out["checks"]["swap_embedding_reproducibility"] = {
    "reported_not_judged": True, "layer": 2, "cells_bit_identical": len(rows1b) - cells,
    "cells": len(rows1b), "rows": rows1b,
    "note": "non-determinism of the embedding step; layer 1 proves the images are identical"}

print("\n=== 1c. TROCAS IGUAIS AO C0 EM knn E FRECHET, ORDEM CANONICA ===")
rows1c, fail1c = [], []
for b in BENCH:
    for r in RULERS:
        bank = arr(b, "TRAIN", r)
        A0, _ = canon(b, "C0", r)
        if bank is None or A0 is None:
            continue
        bank = bank.astype(np.float64)
        A0 = A0.astype(np.float64)
        k0, f0 = knn_mean(A0, bank), frechet(A0, bank)
        for c in SWAPS:
            X, _ = canon(b, c, r)
            if X is None:
                continue
            X = X.astype(np.float64)
            k, f = knn_mean(X, bank), frechet(X, bank)
            dk = abs(k - k0) / max(abs(k0), 1e-30)
            df = abs(f - f0) / max(abs(f0), 1e-30)
            bad = dk > REL_TOL or df > REL_TOL
            rows1c.append({"benchmark": b, "ruler": r, "condition": c,
                           "knn": k, "knn_c0": k0, "rel_knn": dk,
                           "frechet": f, "frechet_c0": f0, "rel_frechet": df,
                           "over_tolerance": bad})
            if bad:
                fail1c.append((b, c, r, dk, df))
                print("  %-7s %-3s %-11s knn rel %.2e | frechet rel %.2e  <-- ACIMA DA TOLERANCIA"
                      % (b, c, r, dk, df))
n1 = len(rows1c)
mk = max((x["rel_knn"] for x in rows1c), default=0.0)
mf = max((x["rel_frechet"] for x in rows1c), default=0.0)
print("  %d de %d celulas dentro de %g" % (n1 - len(fail1c), n1, REL_TOL))
print("  maior desvio relativo observado: knn %.3e   frechet %.3e" % (mk, mf))
out["checks"]["swaps_equal_c0_canonical"] = {
    "passed": not fail1c, "layer": 3, "cells": n1, "failures": len(fail1c),
    "max_rel_knn": mk, "max_rel_frechet": mf, "rows": rows1c}

print("\n=== 2. C1 ACIMA DE TUDO (knn e sliced Wasserstein; Frechet indefinido) ===")
rows2, fail2 = [], []
for b in BENCH:
    for r in RULERS:
        for metric in ("knn_mean", "swd"):
            c1 = g(b, "C1", r, metric)
            if c1 is None:
                continue
            others = [(PAPER.get(c, c), g(b, c, r, metric)) for c in SWAPS + OCC + NOISE
                      if g(b, c, r, metric) is not None]
            if not others:
                continue
            top = max(others, key=lambda x: x[1])
            ok = c1 >= top[1]
            rows2.append({"benchmark": b, "ruler": r, "metric": metric, "c1": c1,
                          "runner_up": top[0], "runner_up_value": top[1],
                          "c1_highest": ok})
            if not ok:
                fail2.append((b, r, metric, c1, top[0], top[1]))
                print("  %-7s %-11s %-9s C1 %.4f < %s %.4f  FALHA"
                      % (b, r, metric, c1, top[0], top[1]))
print("  %d de %d celulas com C1 no topo" % (len(rows2) - len(fail2), len(rows2)))
out["checks"]["c1_highest"] = {"passed": not fail2, "metrics": ["knn_mean", "swd"],
                               "frechet_excluded": "single point", "rows": rows2,
                               "failures": fail2}

print("\n=== 3. MONOTONIA EM C3, QUATRO REGUAS DE CONTEUDO (criterio) ===")
rows3, fail3 = [], []
for b in BENCH:
    for r in CONTENT_RULERS:
        v = [g(b, c, r, "knn_mean") for c in OCC]
        if any(x is None for x in v):
            continue
        ok = all(v[i] <= v[i + 1] for i in range(len(v) - 1))
        rows3.append({"benchmark": b, "ruler": r, "family": "C3", "knn": v,
                      "monotonic": ok})
        if not ok:
            fail3.append((b, r, "C3", v))
            print("  %-7s %-11s C3  %.4f -> %.4f -> %.4f  NAO MONOTONA"
                  % (b, r, v[0], v[1], v[2]))
print("  %d de %d sequencias monotonas" % (len(rows3) - len(fail3), len(rows3)))
out["checks"]["monotonic_c3_content_rulers"] = {
    "passed": not fail3, "rulers": CONTENT_RULERS, "rows": rows3, "failures": fail3,
    "scope": "amendment 7.9: the four rulers of section 1; C3 only"}

print("\n=== 3b. CONTRAFACTUAL DE COR E MONOTONIA EM C4 (relatado, sem veredito) ===")
rows3b = []
for b in BENCH:
    for r in RULERS:
        for name, seq in (("C3", OCC), ("C4", NOISE)):
            if r in CONTENT_RULERS and name == "C3":
                continue  # already a criterion above
            v = [g(b, c, r, "knn_mean") for c in seq]
            if any(x is None for x in v):
                continue
            inc = all(v[i] <= v[i + 1] for i in range(len(v) - 1))
            dec = all(v[i] >= v[i + 1] for i in range(len(v) - 1))
            rows3b.append({"benchmark": b, "ruler": r, "family": name, "knn": v,
                           "increasing": inc, "decreasing": dec})
            if r == "colorhist":
                print("  %-7s cor %s  %.4f -> %.4f -> %.4f   %s"
                      % (b, name, v[0], v[1], v[2],
                         "cresce" if inc else "DECRESCE" if dec else "nao monotona"))
out["checks"]["colour_counterfactual_and_c4"] = {
    "reported_not_judged": True, "rows": rows3b,
    "reading": ("in radiology the colour distance DECREASES with occlusion -- the "
                "border is black and occluded radiographs converge on where the "
                "radiological bank already sits; in endoscopy it increases, the "
                "frame being informative. The ladder under the content rulers is "
                "therefore not a colour artefact: colour does the opposite.")}

print("\n=== 4. 1-VIZINHO NO BANCO (diagnostico, sem veredito -- emenda 7.9) ===")
rows4, fail4 = [], []
for b in BENCH:
    exp = NN1.get(b, {}).get("sha256_copies_in_train")
    got = {r: nn1(b, "C0", r, "nn1_zero_count") for r in RULERS}
    got = {r: v for r, v in got.items() if v is not None}
    print("  %-7s sha256 no treino %4d   zeros: %s"
          % (b, exp, "  ".join("%s %d" % (r, v) for r, v in got.items())))
    for r, v in got.items():
        ok = v == exp
        rows4.append({"benchmark": b, "ruler": r, "zero_count": v,
                      "sha256_copies_in_train": exp, "equal": ok,
                      "nn1_mean": nn1(b, "C0", r)})
        if not ok:
            fail4.append((b, r, v, exp))
print("  %d de %d reguas concordam exatamente com o sha256"
      % (len(rows4) - len(fail4), len(rows4)))
for b, r, v, exp in fail4:
    print("    %-7s %-11s %d contra %d  (mesma foto recodificada: pHash 0, sha diferente)"
          % (b, r, v, exp))
out["checks"]["nn1_zero_vs_sha256"] = {
    "reported_not_judged": True, "rows": rows4, "disagreements": fail4,
    "note": ("bank is the training split only; the audit compared against train and "
             "validation. The two Kvasir disagreements are one image whose pHash "
             "distance to its match is 0 and whose sha256 differs -- the same "
             "picture re-encoded. The ruler finds one duplicate more than byte "
             "identity does, not one fewer.")}

missing = [(b, PAPER.get(c, c), r) for b in BENCH
           for c in ["C0", "C1"] + SWAPS + OCC + NOISE for r in RULERS
           if g(b, c, r, "knn_mean") is None]
if missing:
    byc = {}
    for b, c, r in missing:
        byc.setdefault((b, c), []).append(r)
    print("\n=== celulas ausentes do A ===")
    for (b, c), rs in sorted(byc.items()):
        print("  %-7s %-6s em %d reguas" % (b, c, len(rs)))
ABSENT_BY_CONSTRUCTION = {("VQARAD", "C5"): "absent by construction (paper section 4.1)"}
out["missing_cells"] = [{"benchmark": b, "condition": c, "ruler": r,
                         "reason": ABSENT_BY_CONSTRUCTION.get((b, c), "not computed")}
                        for b, c, r in missing]

fig = []
for b in BENCH:
    for r in RULERS:
        c0k, c0f = g(b, "C0", r, "knn_mean"), g(b, "C0", r, "frechet")
        if c0k is None:
            continue
        for c in ["C0", "C1"] + SWAPS + OCC + NOISE:
            k, f = g(b, c, r, "knn_mean"), g(b, c, r, "frechet")
            if k is None:
                continue
            single = c == "C1"
            fig.append({"benchmark": b, "ruler": r, "condition": PAPER.get(c, c),
                        "knn_minus_c0": k - c0k,
                        "frechet_minus_c0": None if single else f - c0f,
                        "knn_abs": k, "frechet_abs": None if single else f,
                        "swd_abs": g(b, c, r, "swd"), "swd_p": g(b, c, r, "swd_p"),
                        "shift_mean": g(b, c, r, "shift_mean"),
                        "nn1_mean": nn1(b, c, r), "unique_images": uniq(b, c),
                        "marker": "single point" if single else "ladder",
                        "note": "Frechet undefined: one distinct image" if single else None,
                        "c0_reference_knn": c0k, "c0_reference_frechet": c0f})
out["figure3_relative_to_c0"] = fig

print("\n=== VEREDITO G1 ===")
for k, v in out["checks"].items():
    if "passed" in v:
        print("  %-32s %s" % (k, "PASSA" if v["passed"] else "FALHA"))
    else:
        print("  %-32s (relatado, nao julgado)" % k)
(REPO / "docs/v4/g1_results.json").write_text(json.dumps(out, indent=1, default=str))
print("\n  gravado docs/v4/g1_results.json  (%d pontos para a Figura 3)" % len(fig))
