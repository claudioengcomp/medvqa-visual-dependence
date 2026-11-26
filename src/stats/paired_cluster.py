"""Paired comparisons against C0 with image-clustered uncertainty and Holm's
correction (Sections 3.3 and 4.3 of the paper), and VDD(C5) on Kvasir-VQA-x1
over the 8,934 rows of the released per-row manifest.

Reads the stored predictions only; no model and no GPU.

    python -m src.stats.paired_cluster

Writes docs/camera_ready/paired_cluster.json.
"""
import hashlib, json, statistics, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.stats.compute_vdd import mcnemar_test  # noqa: E402

RES = REPO / "results"
OUT = REPO / "docs/camera_ready/paired_cluster.json"

SEED, N_BOOT = 20261125, 2000
CONFIGS = {"KVASIR": ["M3AE", "QWEN2VL_LORA", "QWEN2VL_DORA", "SMOLVLM_LORA", "SMOLVLM_DORA"],
           "SLAKE": ["M3AE_SCRATCH", "QWEN2VL_LORA", "QWEN2VL_DORA", "SMOLVLM_LORA", "SMOLVLM_DORA"],
           "VQARAD": ["M3AE_SCRATCH", "QWEN2VL_LORA", "QWEN2VL_DORA", "SMOLVLM_LORA", "SMOLVLM_DORA"]}
ZS = "MEDVLTHINKER_ZEROSHOT"
FULL = ["C1", "C2", "C4_OCC_25", "C4_OCC_50", "C4_OCC_75", "C4_NOISE_25", "C4_NOISE_50", "C4_NOISE_75"]
CONDS = {"KVASIR": FULL + ["C5", "C6"], "SLAKE": FULL + ["C5", "C6"], "VQARAD": FULL + ["C6"]}
SM = json.loads((REPO / "data/swapmap.json").read_text())


def label(i):
    p = [x for x in str(i).replace("//", "/").split("/") if x]
    if p and p[-1].lower() in ("source.jpg", "source.png"):
        return p[-2] if len(p) > 1 else p[-1]
    return str(i)


def load(prefix, cond):
    f = RES / ("%s_%s.json" % (prefix, cond))
    if not f.exists():
        return None
    return {p["qid"]: bool(p["correct"]) for p in json.loads(f.read_text())["predictions"]}


def prefixes(bench, cfg):
    out = []
    for s in ("_seed0", "_seed1", "_seed2"):
        if (RES / ("%s_%s%s_C0.json" % (cfg, bench, s))).exists():
            out.append((s.strip("_"), "%s_%s%s" % (cfg, bench, s)))
        elif s == "_seed0" and (RES / ("%s_%s_C0.json" % (cfg, bench))).exists():
            out.append(("seed0", "%s_%s" % (cfg, bench)))
    return out


def swapped_qids(bench, cond):
    f = RES / "manifests" / ("%s_%s_rows.json" % (bench, cond))
    return {int(r["qid"]) for r in json.loads(f.read_text())["rows"]}


def one(bench, prefix, cond, img_of, subset):
    c0, cx = load(prefix, "C0"), load(prefix, cond)
    if c0 is None or cx is None:
        return None
    qids = sorted(q for q in c0 if q in cx and (subset is None or q in subset))
    a = np.array([c0[q] for q in qids], float); b_ = np.array([cx[q] for q in qids], float)
    diff = a - b_
    b = int(((a == 1) & (b_ == 0)).sum()); c = int(((a == 0) & (b_ == 1)).sum())
    mc = mcnemar_test([bool(v) for v in a], [bool(v) for v in b_])
    imgs = np.array([img_of[q] for q in qids])
    uniq, inv = np.unique(imgs, return_inverse=True)
    s = np.bincount(inv, weights=diff, minlength=len(uniq)); n = np.bincount(inv, minlength=len(uniq)).astype(float)
    h = hashlib.sha256(("%d|%s|%s" % (SEED, prefix, cond)).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    idx = rng.integers(0, len(uniq), (N_BOOT, len(uniq)))
    boot = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.quantile(boot, [.025, .975])
    return {"n": len(qids), "n_images": int(len(uniq)), "acc_c0": float(a.mean()), "acc_t": float(b_.mean()),
            "vdd_pp": float(100 * diff.mean()), "b": b, "c": c, "mcnemar_p": float(mc["p_value"]),
            "mcnemar_method": mc.get("method"), "cohens_g": mc.get("cohens_g"),
            "cluster_ci_pp": [float(100 * lo), float(100 * hi)],
            "cluster_excludes_zero": bool(lo > 0 or hi < 0)}


def main():
    cells = {}
    for bench in CONFIGS:
        img_of = {int(k): label(v["img"]) for k, v in SM[bench]["qid"].items()}
        subs = {c: (swapped_qids(bench, c) if c in ("C5", "C6") else None) for c in CONDS[bench]}
        runs = [(cfg, sn, pf) for cfg in CONFIGS[bench] for sn, pf in prefixes(bench, cfg)]
        runs += [(ZS, "run", "%s_%s" % (ZS, bench))]
        for cfg, sn, pf in runs:
            for cond in CONDS[bench]:
                r = one(bench, pf, cond, img_of, subs[cond])
                if r is not None:
                    cells["%s|%s|%s|%s" % (bench, cfg, sn, cond)] = r
    ft = [k for k in cells if "|%s|" % ZS not in k]
    # Holm over the fine-tuned family, and over the family with the zero-shot model
    def holm(keys):
        order = sorted(keys, key=lambda k: cells[k]["mcnemar_p"]); m = len(order); out = {}; run = 0.0
        for i, k in enumerate(order):
            run = max(run, min(1.0, (m - i) * cells[k]["mcnemar_p"])); out[k] = run
        return out
    hf, ha = holm(ft), holm(list(cells))
    for k in cells:
        cells[k]["holm_p_all464"] = ha[k]
        if k in hf:
            cells[k]["holm_p_finetuned435"] = hf[k]
    summ = {"n_finetuned": len(ft), "n_all": len(cells),
            "finetuned_raw_p_lt_05": sum(cells[k]["mcnemar_p"] < .05 for k in ft),
            "finetuned_holm_lt_05": sum(hf[k] < .05 for k in ft),
            "finetuned_cluster_ci_excludes_zero": sum(cells[k]["cluster_excludes_zero"] for k in ft),
            "all_raw_p_lt_05": sum(cells[k]["mcnemar_p"] < .05 for k in cells),
            "all_holm_lt_05": sum(ha[k] < .05 for k in cells),
            "all_cluster_ci_excludes_zero": sum(cells[k]["cluster_excludes_zero"] for k in cells),
            "raw_sig_but_not_holm": sorted(k for k in ft if cells[k]["mcnemar_p"] < .05 and hf[k] >= .05),
            "raw_sig_but_cluster_includes_zero": sorted(k for k in ft if cells[k]["mcnemar_p"] < .05 and not cells[k]["cluster_excludes_zero"]),
            "cluster_sig_but_raw_not": sorted(k for k in ft if cells[k]["mcnemar_p"] >= .05 and cells[k]["cluster_excludes_zero"])}
    # Table 3 cells: mean and sd over seeds, and the dagger under each rule (all three seeds)
    table = {}
    for bench in CONFIGS:
        for cfg in CONFIGS[bench]:
            for cond in ("C1", "C2", "C5", "C6"):
                ks = [k for k in ft if k.startswith("%s|%s|" % (bench, cfg)) and k.endswith("|" + cond)]
                if len(ks) != 3:
                    continue
                v = [cells[k]["vdd_pp"] for k in ks]
                table["%s|%s|%s" % (bench, cfg, cond)] = {
                    "n": cells[ks[0]]["n"], "mean": statistics.mean(v), "sd": statistics.stdev(v),
                    "dagger_raw": all(cells[k]["mcnemar_p"] < .05 for k in ks),
                    "dagger_holm": all(hf[k] < .05 for k in ks),
                    "dagger_cluster": all(cells[k]["cluster_excludes_zero"] for k in ks)}
    spread = {}
    for cond in ("C1", "C2", "C5", "C6"):
        m = [table["KVASIR|%s|%s" % (cfg, cond)]["mean"] for cfg in CONFIGS["KVASIR"]]
        spread[cond] = {"min": min(m), "max": max(m), "spread_pp": max(m) - min(m)}
    out = {"seed": SEED, "n_boot": N_BOOT, "cluster": "original image", "summary": summ,
           "table3": table, "kvasir_spread": spread, "cells": cells}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(summ, indent=1)); print(json.dumps(spread, indent=1))
    for k, v in table.items():
        print("%-32s n %5d  %6.2f +- %4.2f  raw %d holm %d cluster %d" % (k, v["n"], v["mean"], v["sd"], v["dagger_raw"], v["dagger_holm"], v["dagger_cluster"]))


if __name__ == "__main__":
    main()
