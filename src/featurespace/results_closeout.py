"""Five closing questions of the Results section. No GPU, no new inference.

Everything here reads the predictions already on disk, the swap manifests and
the swapmap; the rulers are not touched. Each block prints its denominator,
because the point of the exercise is that three rates of §5.2 have three
different ones.

1  |V| and phi of C6 on SLAKE, beside C2 and C5 under one routine. Model-free:
   a question is verifiable when the donor image carries the same question, and
   informative when that question's reference differs there.

2  Where the altered answer goes on Kvasir under C5, over the rows of I whose
   prediction actually changed -- an exhaustive partition into the donor's
   reference, the original reference and neither, so the three sum to one. `k`
   of Equation 2 is reported beside it with its own denominator (all of I),
   because it is not the same quantity.

4  Acc(C1) on the closed subset of SLAKE and VQA-RAD, the same questions the
   question-only floor was trained and evaluated on, read off the M0 files.

5  What the 188 pairs of Figure 5 are on SLAKE, and how many of them sit at
   colour distance below 0.01.

6  Whether the within-bucket coefficients and the all-pairs ones they are
   compared with are on the same scale and under the same aggregation.
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.analysis.ccr import answers_match, load_predictions, question_key  # noqa: E402
from src.featurespace import paths                                                    # noqa: E402

CACHE = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "embeddings"
V4 = REPO / "docs/v4"
OUT = V4 / "results_closeout.json"
SM = json.loads((REPO / "data/swapmap.json").read_text())
IDS = json.loads((CACHE / "image_ids.json").read_text())
MANIFEST = {
    ("KVASIR", "C2"): "C2_manifest_KVASIR.json",
    ("KVASIR", "C5"): "C5_manifest_KVASIR_clean.json",
    ("KVASIR", "C6"): "C6_manifest_KVASIR_clean.json",
    ("SLAKE", "C2"): "C2_manifest_SLAKE.json",
    ("SLAKE", "C5"): "C5_manifest_SLAKE.json",
    ("SLAKE", "C6"): "C6_manifest_SLAKE.json",
    ("VQARAD", "C2"): "C2_manifest_VQARAD.json",
    ("VQARAD", "C6"): "C6_manifest_VQARAD.json",
}
SEEDS = ["seed0", "seed1", "seed2"]
out = {}


def label(i):
    p = [x for x in str(i).replace("//", "/").split("/") if x]
    if p and p[-1].lower() in ("source.jpg", "source.png"):
        return p[-2] if len(p) > 1 else p[-1]
    return str(i)


def donor_map(b, cond):
    d = json.loads((paths.RESULTS / MANIFEST[(b, cond)]).read_text())
    rows = d["pairs"] if isinstance(d, dict) else d
    return {label(e["qid_image"]): label(e["donor_image"]) for e in rows}


def meta_of(b):
    QM = {int(k): v for k, v in SM[b]["qid"].items()}
    by_image = {}
    for q, m in QM.items():
        by_image.setdefault(label(m["img"]), {})[question_key(m["q"])] = m["a"]
    return QM, by_image


def classify(QM, by_image, dm, qid):
    """N, U or I for one swapped row, by the definitions of compute_ccr."""
    im = label(QM[qid]["img"])
    if im not in dm:
        return None                      # the condition does not touch this row
    ref = by_image.get(dm[im], {}).get(question_key(QM[qid]["q"]))
    if ref is None:
        return "N"
    return "U" if any(answers_match(x, ref) for x in QM[qid]["a"]) else "I"


# ------------------------------------------------------------------ item 1
print("=== 1. |V| e phi no SLAKE, uma rotina para C2, C5 e C6 "
      "(denominador: linhas trocadas; V: doadora carrega a pergunta) ===")
print("%-5s %8s %8s %8s %8s %9s" % ("cond", "|S_T|", "|I|", "|U|", "|V|", "phi"))
item1 = {}
QM, BY = meta_of("SLAKE")
for cond in ("C2", "C5", "C6"):
    dm = donor_map("SLAKE", cond)
    cls = [classify(QM, BY, dm, q) for q in QM]
    cls = [c for c in cls if c is not None]
    nI, nU, nN = cls.count("I"), cls.count("U"), cls.count("N")
    V = nI + nU
    item1[cond] = {"S_T": len(cls), "I": nI, "U": nU, "N": nN, "V": V,
                   "phi": nI / V if V else None}
    print("%-5s %8d %8d %8d %8d %9.4f" % (cond, len(cls), nI, nU, V, nI / V))
out["item1_slake_V_phi"] = {
    "definition": "S_T: swapped rows; V: donor carries the same question; "
                  "I: and its reference differs; phi = |I|/|V|",
    "model_free": True, "by_condition": item1}

# ------------------------------------------------------------------ item 2
print("\n=== 2. destino da resposta alterada, KVASIR C5, sobre I ===")
QM, BY = meta_of("KVASIR")
dm = donor_map("KVASIR", "C5")
I = [q for q in QM if classify(QM, BY, dm, q) == "I"]
print("  |I| = %d linhas (denominador de k e da variante 'sobre todo I')" % len(I))
per_cfg = {}
for cfg in paths.CONFIGS["KVASIR"]:
    rows = [c for c in paths.cells("KVASIR") if c["config"] == cfg]
    acc = []
    for c in rows:
        c0 = load_predictions(str(paths.RESULTS), c["prefix"], "C0")
        cx = load_predictions(str(paths.RESULTS), c["prefix"], "C5")
        if c0 is None or cx is None:
            continue
        n_ch = fo = ke = ne = 0
        n_all = k_all = fo_all = 0
        for q in I:
            if q not in c0 or q not in cx:
                continue
            y = QM[q]["a"]
            yq = BY[dm[label(QM[q]["img"])]][question_key(QM[q]["q"])]
            p = cx[q]["predicted"]
            n_all += 1
            k_all += int(answers_match(p, y))          # Eq. 2: over ALL of I
            fo_all += int(answers_match(p, yq))
            if answers_match(p, [c0[q]["predicted"]]):
                continue                               # the answer did not change
            n_ch += 1
            if answers_match(p, yq):
                fo += 1
            elif answers_match(p, y):
                ke += 1
            else:
                ne += 1
        acc.append({"n_I": n_all, "n_changed": n_ch,
                    "neither_over_I": 1.0 - (fo_all + k_all) / n_all,
                    "changed_followed": fo / n_ch, "changed_kept": ke / n_ch,
                    "changed_neither": ne / n_ch, "changed_share_of_I": n_ch / n_all,
                    "k_over_I": k_all / n_all, "followed_over_I": fo_all / n_all})
    per_cfg[cfg] = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]} if acc else None
    a = per_cfg[cfg]
    print("  %-14s I=%d  mudaram %d (%.1f%% de I) | segue %.4f  mantem %.4f  nenhuma %.4f"
          " | k sobre I %.4f  segue sobre I %.4f"
          % (cfg, a["n_I"], round(a["n_changed"]), 100 * a["changed_share_of_I"],
             a["changed_followed"], a["changed_kept"], a["changed_neither"],
             a["k_over_I"], a["followed_over_I"]))
rng = lambda k: (min(v[k] for v in per_cfg.values()), max(v[k] for v in per_cfg.values()))
print("  min-max sobre as cinco configuracoes:")
for k, lab in (("changed_followed", "segue a doadora (das mudadas)"),
               ("changed_kept", "mantem a original (das mudadas)"),
               ("changed_neither", "nenhuma das duas (das mudadas)"),
               ("changed_share_of_I", "fracao de I que mudou"),
               ("k_over_I", "k, Eq. 2 (sobre todo I)"),
               ("followed_over_I", "segue a doadora (sobre todo I) = CCR"),
               ("neither_over_I", "nenhuma das duas (sobre todo I)")):
    lo, hi = rng(k)
    print("    %-34s %.4f a %.4f" % (lab, lo, hi))
out["item2_kvasir_c5_destination"] = {
    "denominators": {"changed_*": "rows of I whose prediction changed; the three sum to 1",
                     "k_over_I": "all rows of I (Equation 2)",
                     "followed_over_I": "all rows of I (the CCR of Table 3)"},
    "n_I": len(I), "per_config": per_cfg,
    "min_max": {k: rng(k) for k in ("changed_followed", "changed_kept", "changed_neither",
                                    "changed_share_of_I", "k_over_I", "followed_over_I",
                                    "neither_over_I")}}

# ------------------------------------------------------------------ item 4
print("\n=== 4. Acc(C1) nas perguntas fechadas (as mesmas do piso M0) ===")
item4 = {}
KVASIR_FLOOR = 0.657          # v3, all 10,779 questions are categorical there
for b in ("SLAKE", "VQARAD", "KVASIR"):
    m0 = paths.RESULTS / ("M0_question_only_%s_C0.json" % b)
    if m0.exists():
        d = json.loads(m0.read_text())
        closed = {int(p["qid"]) for p in d["predictions"]}
        fl = d["accuracy"]
    else:
        # no closed subset on Kvasir-VQA-x1: every answer is categorical, and the
        # floor is the published v3 one over all 10,779 questions
        c0 = load_predictions(str(paths.RESULTS), paths.cells(b)[0]["prefix"], "C0")
        closed = set(c0)
        fl = KVASIR_FLOOR
    rec = {"n_closed": len(closed), "floor": fl, "by_config": {}}
    print("  %-7s piso %.4f sobre %d perguntas" % (b, fl, len(closed)))
    for cfg in paths.CONFIGS[b]:
        vals = []
        for c in [c for c in paths.cells(b) if c["config"] == cfg]:
            cx = load_predictions(str(paths.RESULTS), c["prefix"], "C1")
            c0 = load_predictions(str(paths.RESULTS), c["prefix"], "C0")
            if cx is None:
                continue
            q = [x for x in cx if x in closed]
            qa = list(cx)
            vals.append({"acc_c1_closed": sum(cx[x]["correct"] for x in q) / len(q),
                         "n_closed_seen": len(q),
                         "acc_c1_all": sum(cx[x]["correct"] for x in qa) / len(qa),
                         "acc_c0_closed": (sum(c0[x]["correct"] for x in q) / len(q))
                                          if c0 else None})
        if not vals:
            continue
        rec["by_config"][cfg] = {k: float(np.mean([v[k] for v in vals]))
                                 for k in vals[0] if vals[0][k] is not None}
        r = rec["by_config"][cfg]
        print("     %-16s Acc(C1) fechadas %.4f (%+.1f pp vs piso) | todas %.4f | Acc(C0) fechadas %.4f"
              % (cfg, r["acc_c1_closed"], 100 * (r["acc_c1_closed"] - fl),
                 r["acc_c1_all"], r.get("acc_c0_closed", float("nan"))))
    item4[b] = rec
out["item4_c1_on_closed"] = {
    "denominator": "the qids of the M0 question-only file of each benchmark",
    "benchmarks": item4}

# ------------------------------------------------------------------ item 5
print("\n=== 5. os pares do SLAKE na Figura 5 ===")
rows = {}
for cond in paths.SWAP_CONDITIONS["SLAKE"]:
    dm = donor_map("SLAKE", cond)
    rows[cond] = {a: c for a, c in dm.items()}
ids = [label(i) for i in IDS["SLAKE__C0"]]
item5 = {"rows_per_condition": {c: len(v) for c, v in rows.items()},
         "self_pairs_per_condition": {c: sum(1 for a, d in v.items() if a == d)
                                      for c, v in rows.items()}}
for ruler in ("medsiglip", "colorhist"):
    X = np.load(CACHE / ("SLAKE__%s.npy" % ruler)).astype(np.float64)
    v = dict(zip(ids, X))
    def dist(a, c):
        return 1.0 - float(v[a] @ v[c] / (np.linalg.norm(v[a]) * np.linalg.norm(v[c])))
    pairs = {}                       # (image, rounded d) -> set of conditions
    imgpair = {}                     # (image, donor) -> d
    for cond, dm in rows.items():
        for a, c in dm.items():
            if a in v and c in v:
                d = dist(a, c)
                pairs.setdefault((a, round(d, 12)), set()).add(cond)
                imgpair[(a, c)] = d
    below = [k for k in pairs if k[1] < 0.01]
    item5[ruler] = {"distinct_image_distance_entries": len(pairs),
                    "distinct_image_donor_pairs": len(imgpair),
                    "entries_below_0.01": len(below),
                    "distinct_images_among_those": len({k[0] for k in below}),
                    "image_donor_pairs_below_0.01":
                        sum(1 for (a, c), d in imgpair.items() if d < 0.01)}
    print("  %-10s entradas (imagem, d) %d | pares (imagem, doadora) %d | "
          "entradas d<0.01 %d em %d imagens | pares d<0.01 %d"
          % (ruler, len(pairs), len(imgpair), len(below),
             len({k[0] for k in below}), item5[ruler]["image_donor_pairs_below_0.01"]))
print("  linhas por condicao: %s ; auto-pares: %s"
      % (item5["rows_per_condition"], item5["self_pairs_per_condition"]))
out["item5_slake_pairs"] = item5

# ------------------------------------------------------------------ item 6
print("\n=== 6. escala e agregacao dos coeficientes ===")
G3 = json.loads((V4 / "g3_results.json").read_text())["cells"]
item6 = {}
for b in paths.BENCHMARKS:
    mean_cfg, seed0, within = [], [], []
    for k, c in G3.items():
        if not k.startswith(b + "|"):
            continue
        cs = [s["coef"] for s in c["per_seed"].values()]
        mean_cfg.append(sum(cs) / len(cs))
        seed0.append(c["per_seed"]["seed0"]["coef"])
        if c.get("within_bucket_coef") is not None:
            within.append(c["within_bucket_coef"])
    item6[b] = {"all_pairs_per_config_means": [min(mean_cfg), max(mean_cfg)],
                "all_pairs_seed0": [min(seed0), max(seed0)],
                "within_bucket_per_config_means": [min(within), max(within)] if within else None,
                "scale": "raw logistic coefficient on d, both",
                "within_aggregation": "mean over the three seeds of the within-bucket fit"}
    print("  %-7s todas: medias por config %.3f a %.3f | seed0 %.3f a %.3f || "
          "dentro do balde (medias por config) %.3f a %.3f"
          % (b, min(mean_cfg), max(mean_cfg), min(seed0), max(seed0),
             min(within), max(within)))
out["item6_coefficient_scale"] = item6

OUT.write_text(json.dumps(out, indent=1, default=float))
print("\n  gravado %s" % OUT)
