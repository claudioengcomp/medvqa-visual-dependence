"""Analyses A-D over the model-derived rulers, and the same code over the five
external rulers as its own check: run with `--rulers medsiglip` it must give
back g0_results.json and the C2 cells of analysis_b.json to the digit, or it is
not to be trusted on the new arrays.

A  Table 1: leave-one-out kNN (k=10) on intact images against the same labels
   as G0 (src/featurespace/g0_embed.py build_labels/knn_loo, imported, not copied).
   Per line in the JSON; a view aggregated per configuration (mean +- sd over
   seeds) for the table.
B  Figure 3, lower panel: d = 1 - cos over the C2 pairs, split near/far by the
   same buckets as analysis_b.py (imported), with the same W1 permutation test.
C  Agreement: Kendall's tau between d under a model ruler and d under each
   external ruler, over all C2 pairs, per benchmark. Primary = MedSigLIP.
D  fp16 against fp32: median item-wise cosine >= 0.999 and no Table-1 accuracy
   moving in the second decimal, the rule of amendment 7.5.

Reports, does not decide: nothing here chooses which ruler enters the paper.
Output: docs/v4/model_rulers.json and a text summary on stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.featurespace import paths as P                              # noqa: E402
from src.featurespace import analysis_b as B                         # noqa: E402
from src.featurespace.g0_embed import build_labels, knn_loo, load_images, ARROWS  # noqa: E402

CACHE = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "embeddings"
OUT = REPO / "docs/v4/model_rulers.json"
EXTERNAL = ["medsiglip", "dinov3", "biomedclip", "dinov2", "colorhist"]
TASKS = ["slake_modality", "vqarad_organ", "kvasir_procedure", "kvasir_finding",
         "kvasir_finding_in_colonoscopy"]
IDS = json.loads((CACHE / "image_ids.json").read_text())
MAN = json.loads((CACHE / "manifest.json").read_text())


def new_rulers():
    """Every model-derived ruler present in the cache, fp32 only, per benchmark."""
    out = defaultdict(list)
    for key in MAN["arrays"]:
        bench, _, r = key.partition("__")
        if "ids_sha256" in MAN["arrays"][key] and not r.endswith(("_fp16", "_bf16")):
            out[bench].append(r)
    rb = {b: sorted(v) for b, v in out.items()}
    only_fp32(rb)
    return rb


EXPECTED_DTYPE = "float32"      # set by --suffix; fp32 is canonical unless a set is asked for


def only_fp32(rb):
    """fp32 is canonical -- the paper declares every ruler in fp32. A, B and C
    read fp32 arrays only; an fp16 twin reaching them is a bug, not a choice.
    A *_bf16 set may be analysed on request (--suffix _bf16), and then every
    array of the set must be bf16: no mixing."""
    for b, rs in rb.items():
        for r in rs:
            assert not r.endswith("_fp16"), "fp16 em A/B/C: %s %s" % (b, r)
            got = MAN["arrays"].get("%s__%s" % (b, r), {}).get("dtype", "float32")
            assert got == EXPECTED_DTYPE, "array %s em A/B/C, esperado %s: %s %s" % (got, EXPECTED_DTYPE, b, r)


def arr(bench, ruler):
    p = CACHE / ("%s__%s.npy" % (bench, ruler))
    return np.load(p).astype(np.float64) if p.exists() else None


def config_of(ruler):
    """`smolvlm_lora_s0_tower` -> ('smolvlm_lora', 's0', 'tower'); base -> seed None."""
    tag = next((t for t in ("_fp16", "_bf16") if ruler.endswith(t)), "")
    ruler = ruler[:-len(tag)] if tag else ruler
    if ruler.rsplit("_", 1)[-1] not in ("tower", "proj"):
        return ruler + tag, None, ""           # an external ruler or a pooling variant: one row each
    base, variant = ruler.rsplit("_", 1)
    variant += tag
    parts = base.split("_")
    if parts[-1].startswith("s") and parts[-1][1:].isdigit():
        return "_".join(parts[:-1]), parts[-1], variant
    return base, None, variant


# ------------------------------------------------------------------ A
def analysis_a(rulers_by_bench):
    only_fp32(rulers_by_bench)
    ids = {b: load_images(p)[0] for b, p in ARROWS.items()}
    L = build_labels(ids)
    L.pop("_composite_dropped", None)
    res = {}
    for task in TASKS:
        bench, lab = L[task]
        keep = [j for j, i in enumerate(ids[bench]) if i in lab]
        y = [lab[ids[bench][j]] for j in keep]
        res[task] = {"benchmark": bench, "n": len(keep), "acc": {}}
        for r in rulers_by_bench.get(bench, []):
            X = arr(bench, r)
            if X is None:
                continue
            a, _ = knn_loo(X[keep], y)
            res[task]["acc"][r] = a
    # aggregated view per configuration x variant: mean +- sd over seeds
    agg = {}
    for task, rec in res.items():
        g = defaultdict(list)
        for r, a in rec["acc"].items():
            cfg, seed, var = config_of(r)
            g["%s_%s" % (cfg, var)].append(a)
        agg[task] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                         "n_seeds": len(v)} for k, v in g.items()}
    return {"per_line": res, "per_config": agg}


# ------------------------------------------------------------------ B, C
def d_pairs(bench, ruler):
    """d over the C2 pairs of a benchmark, keyed by original image label."""
    by_id, _, _ = B.vectors(bench, ruler)
    m, weight = B.pairs_of(bench, "C2")
    out = {}
    for a, b in m.items():
        va, vb = by_id.get(a), by_id.get(b)
        if va is None or vb is None:
            continue
        out[a] = 1.0 - float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
    return out, weight


def analysis_b(rulers_by_bench):
    only_fp32(rulers_by_bench)
    res = {}
    for bench in P.BENCHMARKS:
        bk = B.buckets(bench)
        for r in rulers_by_bench.get(bench, []):
            d, w = d_pairs(bench, r)
            m, _ = B.pairs_of(bench, "C2")
            grp = np.array(["near" if bk.get(a) is not None and bk.get(a) == bk.get(m[a]) else
                            ("far" if bk.get(a) is not None and bk.get(m[a]) is not None else "none")
                            for a in d])
            v = np.array(list(d.values()))
            near, far = v[grp == "near"], v[grp == "far"]
            rec = {"n_pairs": len(v), "near_fraction": float((grp == "near").mean()) if len(v) else None,
                   "near": B.summarise(near) if len(near) else None,
                   "far": B.summarise(far) if len(far) else None,
                   "quantiles_near": np.quantile(near, [.05, .25, .5, .75, .95]).tolist() if len(near) else None,
                   "quantiles_far": np.quantile(far, [.05, .25, .5, .75, .95]).tolist() if len(far) else None}
            if len(near) >= 2 and len(far) >= 2:
                obs, p = B.w1_perm(near, far, B.cell_rng(bench, "C2", r))
                rec.update({"w1": float(obs), "w1_p": float(p), "separated": bool(p < 0.01)})
            res["%s|%s" % (bench, r)] = rec
    return res


def analysis_c(rulers_by_bench):
    only_fp32(rulers_by_bench)
    from scipy.stats import kendalltau
    res = {}
    for bench in P.BENCHMARKS:
        ext = {e: d_pairs(bench, e)[0] for e in EXTERNAL}
        for r in rulers_by_bench.get(bench, []):
            d, _ = d_pairs(bench, r)
            rec = {"n_pairs": len(d)}
            for e, de in ext.items():
                keys = [k for k in d if k in de]
                t, p = kendalltau([d[k] for k in keys], [de[k] for k in keys])
                rec[e] = {"tau": float(t), "p": float(p), "n": len(keys)}
            res["%s|%s" % (bench, r)] = rec
    return res


# ------------------------------------------------------------------ D
def analysis_d(rulers_by_bench, table_a):
    res = {}
    ids = {b: load_images(p)[0] for b, p in ARROWS.items()}
    L = build_labels(ids)
    L.pop("_composite_dropped", None)
    for bench in P.BENCHMARKS:
        for r in rulers_by_bench.get(bench, []):
            X, H = arr(bench, r), arr(bench, r + "_fp16")
            if X is None or H is None:
                continue
            cos = np.einsum("ij,ij->i", X, H) / (np.linalg.norm(X, axis=1) * np.linalg.norm(H, axis=1) + 1e-12)
            moved = []
            for task in TASKS:
                tb, lab = L[task]
                if tb != bench:
                    continue
                keep = [j for j, i in enumerate(ids[bench]) if i in lab]
                y = [lab[ids[bench][j]] for j in keep]
                a32 = table_a["per_line"][task]["acc"].get(r)
                a16, _ = knn_loo(H[keep], y)
                if a32 is not None and a16 is not None and round(a32, 2) != round(a16, 2):
                    moved.append({"task": task, "fp32": a32, "fp16": a16})
            res["%s|%s" % (bench, r)] = {
                "cos_min": float(cos.min()), "cos_median": float(np.median(cos)),
                "cos_mean": float(cos.mean()), "cos_rule_ok": bool(np.median(cos) >= 0.999),
                "table1_stable_2dp": not moved, "moved": moved}
    return res


# ---------------------------------------------------------------- report
def report(A, Bres, C, D, rulers_by_bench):
    print("\n=== A. Tabela 1 (kNN LOO k=10), por configuracao: media +- dp sobre seeds ===")
    cols = sorted({k for t in A["per_config"].values() for k in t})
    for task in TASKS:
        rec = A["per_line"][task]
        print("--- %-30s n=%d" % (task, rec["n"]))
        for k in cols:
            if k in A["per_config"][task]:
                v = A["per_config"][task][k]
                print("    %-28s %.3f +- %.3f  (%d seed%s)" % (
                    k, v["mean"], v["sd"], v["n_seeds"], "" if v["n_seeds"] == 1 else "s"))
    print("\n=== C. tau de Kendall, d(C2) regua-do-modelo x regua neutra, por base ===")
    print("%-8s %-26s %6s  " % ("base", "regua", "pares") + "  ".join("%9s" % e for e in EXTERNAL))
    for key, rec in sorted(C.items()):
        b, r = key.split("|")
        print("%-8s %-26s %6d  " % (b, r, rec["n_pairs"]) +
              "  ".join("%9.3f" % rec[e]["tau"] for e in EXTERNAL))
    print("\n=== B. near/far em d(C2): mediana near | far, W1, p ===")
    for key, rec in sorted(Bres.items()):
        b, r = key.split("|")
        if rec["near"] and rec["far"]:
            print("%-8s %-26s near %.4f (n=%d) | far %.4f (n=%d)  W1 %.4f p %s %s" % (
                b, r, rec["near"]["median"], rec["near"]["n"], rec["far"]["median"], rec["far"]["n"],
                rec.get("w1", float("nan")), "%.4f" % rec["w1_p"] if "w1_p" in rec else "--",
                "SEPARA" if rec.get("separated") else ""))
        else:
            print("%-8s %-26s sem particao near/far (n=%d)" % (b, r, rec["n_pairs"]))
    print("\n=== D. fp16 x fp32: cos mediano (regra >= 0.999) e Tabela 1 estavel a 2 casas ===")
    for key, rec in sorted(D.items()):
        b, r = key.split("|")
        print("%-8s %-26s cos med %.6f min %.6f  regra %s  tabela %s%s" % (
            b, r, rec["cos_median"], rec["cos_min"], "OK" if rec["cos_rule_ok"] else "FALHA",
            "estavel" if rec["table1_stable_2dp"] else "MOVEU", "" if not rec["moved"] else " " + str(rec["moved"])))


def main():
    global EXPECTED_DTYPE
    ap = argparse.ArgumentParser()
    ap.add_argument("--rulers", nargs="*", default=None,
                    help="run over these existing rulers instead of the model-derived fp32 set")
    ap.add_argument("--suffix", default=None, help="analyse the set of arrays ending in this suffix, e.g. _bf16")
    ap.add_argument("--write-key", default=None,
                    help="store A and C (and B) of a --rulers/--suffix run under this key in model_rulers.json")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    if a.suffix:
        EXPECTED_DTYPE = {"_bf16": "bfloat16"}[a.suffix]
        rb = {}
        for key, ent in MAN["arrays"].items():
            b, _, r = key.partition("__")
            if "ids_sha256" in ent and r.endswith(a.suffix):
                rb.setdefault(b, []).append(r)
        rb = {b: sorted(v) for b, v in rb.items()}
        only_fp32(rb)
    elif a.rulers:
        rb = {b: [r for r in a.rulers if (CACHE / ("%s__%s.npy" % (b, r))).exists()] for b in P.BENCHMARKS}
    else:
        rb = new_rulers()
    A = analysis_a(rb)
    Bres = analysis_b(rb)
    C = analysis_c(rb)
    D = analysis_d(rb, A) if not (a.rulers or a.suffix) else {}
    report(A, Bres, C, D, rb)
    if a.no_write:
        return
    if a.write_key:
        cur = json.loads(OUT.read_text())
        cur[a.write_key] = {"rulers": rb, "A_table1": A, "B_near_far_c2": Bres, "C_kendall_tau_c2": C}
        OUT.write_text(json.dumps(cur, indent=1))
        print("\n  gravado em %s sob %s" % (OUT, a.write_key))
    elif not (a.rulers or a.suffix):
        cur = json.loads(OUT.read_text()) if OUT.exists() else {}
        cur.update({"rulers": rb, "A_table1": A, "B_near_far_c2": Bres,
                    "C_kendall_tau_c2": C, "D_fp16_vs_fp32": D})
        OUT.write_text(json.dumps(cur, indent=1))
        print("\n  gravado %s" % OUT)


if __name__ == "__main__":
    main()
