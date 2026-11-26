"""Logistic fit flip ~ d of analysis C with image-clustered bootstrap intervals
(Sections 3.3 and 5.3 of the paper).

Rebuilds the rows of analysis C from the per-item predictions, the swap map and
the C0 test embeddings (MedSigLIP and colour histogram), checks the Wald fit
against docs/v4/analysis_c.json, and adds a percentile interval from a bootstrap
that resamples original images as clusters. Needs the embedding cache
(V4_CACHE, as analysis_c.py); no model and no GPU.

    python -m src.featurespace.logistic_cluster

Writes docs/camera_ready/logistic_cluster.json.
"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.analysis.ccr import answers_match, load_predictions  # noqa: E402

CACHE = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "embeddings"
RES = REPO / "results"
OUT = REPO / "docs/camera_ready/logistic_cluster.json"

SEED, N_BOOT = 20261125, 2000
BENCH = ["KVASIR", "SLAKE", "VQARAD"]
CONFIGS = {"KVASIR": ["M3AE", "QWEN2VL_DORA", "QWEN2VL_LORA", "SMOLVLM_DORA", "SMOLVLM_LORA"],
           "SLAKE": ["M3AE_SCRATCH", "QWEN2VL_DORA", "QWEN2VL_LORA", "SMOLVLM_DORA", "SMOLVLM_LORA"],
           "VQARAD": ["M3AE_SCRATCH", "QWEN2VL_DORA", "QWEN2VL_LORA", "SMOLVLM_DORA", "SMOLVLM_LORA"]}
SEEDS = ["_seed0", "_seed1", "_seed2"]
CONDS = {"KVASIR": ["C2", "C5", "C6"], "SLAKE": ["C2", "C5", "C6"], "VQARAD": ["C2", "C6"]}
IDS = json.loads((CACHE / "image_ids.json").read_text())
SM = json.loads((REPO / "data/swapmap.json").read_text())
STORED = json.loads((REPO / "docs/v4/analysis_c.json").read_text())["cells"]


def label(i):
    p = [x for x in str(i).replace("//", "/").split("/") if x]
    if p and p[-1].lower() in ("source.jpg", "source.png"):
        return p[-2] if len(p) > 1 else p[-1]
    return str(i)


def seed_suffixes(bench, cfg):
    out = []
    for s in SEEDS:
        if (RES / ("%s_%s%s_C0.json" % (cfg, bench, s))).exists():
            out.append(s)
        elif s == "_seed0" and (RES / ("%s_%s_C0.json" % (cfg, bench))).exists():
            out.append("")
    return out


def donor_map(bench, cond):
    return {label(a): label(b) for a, b in SM[bench]["swap"][cond].items()}


def distances(bench, ruler, dm):
    ids = [label(i) for i in IDS["%s__C0" % bench]]
    X = np.load(CACHE / ("%s__%s.npy" % (bench, ruler))).astype(np.float64)
    vec = dict(zip(ids, X))
    out = {}
    for a, b in dm.items():
        if a in vec and b in vec:
            va, vb = vec[a], vec[b]
            out[a] = 1.0 - float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
    return out


def logistic(x, y, w=None, ridge=1e-6, iters=60):
    X = np.column_stack([np.ones(len(x)), x])
    w = np.ones(len(x)) if w is None else w
    b = np.zeros(2)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))
        W = np.clip(p * (1 - p), 1e-9, None) * w
        H = X.T @ (X * W[:, None]) + ridge * np.eye(2)
        step = np.linalg.solve(H, X.T @ (w * (y - p)) - ridge * b)
        b = b + step
        if np.abs(step).max() < 1e-10:
            break
    p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))
    W = np.clip(p * (1 - p), 1e-9, None) * w
    cov = np.linalg.inv(X.T @ (X * W[:, None]) + ridge * np.eye(2))
    return b, float(np.sqrt(cov[1, 1]))


def rows_for(bench, prefix, ruler):
    QM = {int(k): v for k, v in SM[bench]["qid"].items()}
    c0 = load_predictions(str(RES), prefix, "C0")
    d_all, f_all, img_all = [], [], []
    for cond in CONDS[bench]:
        cx = load_predictions(str(RES), prefix, cond)
        if cx is None:
            continue
        dm = donor_map(bench, cond)
        dist = distances(bench, ruler, dm)
        for qid, meta in QM.items():
            im = label(meta["img"])
            if im not in dm or im not in dist or qid not in c0 or qid not in cx:
                continue
            flip = not answers_match(cx[qid]["predicted"], [c0[qid]["predicted"]])
            d_all.append(dist[im]); f_all.append(float(flip)); img_all.append(im)
    return np.asarray(d_all), np.asarray(f_all), np.asarray(img_all)


def main():
    out = {"seed": SEED, "n_boot": N_BOOT, "cluster": "original image",
           "interval": "percentile 2.5-97.5 of the coefficient over image resamples", "cells": {}}
    for ruler in ("medsiglip", "colorhist"):
        for bench in BENCH:
            for cfg in CONFIGS[bench]:
                for s in seed_suffixes(bench, cfg):
                    sn = s.strip("_") or "seed0"
                    key = "%s|%s|%s|%s" % (bench, cfg, sn, ruler)
                    d, f, img = rows_for(bench, "%s_%s%s" % (cfg, bench, s), ruler)
                    b, se = logistic(d, f)
                    st = STORED[key]["fit"]
                    uniq, inv = np.unique(img, return_inverse=True)
                    h = hashlib.sha256(("%d|%s" % (SEED, key)).encode()).digest()
                    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
                    coefs = np.empty(N_BOOT)
                    for i in range(N_BOOT):
                        cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
                        coefs[i] = logistic(d, f, w=cnt[inv].astype(float))[0][1]
                    lo, hi = np.quantile(coefs, [.025, .975])
                    rec = {"n_rows": int(len(d)), "n_images": int(len(uniq)), "coef": float(b[1]),
                           "wald_ci": [float(b[1] - 1.96 * se), float(b[1] + 1.96 * se)],
                           "stored_coef": st["coef"], "stored_n_rows": STORED[key]["n_rows"],
                           "abs_diff_vs_stored": abs(float(b[1]) - st["coef"]),
                           "cluster_ci": [float(lo), float(hi)],
                           "wald_pass": bool(b[1] > 0 and b[1] - 1.96 * se > 0),
                           "cluster_pass": bool(b[1] > 0 and lo > 0)}
                    out["cells"][key] = rec
                    print("%-38s n %6d img %4d coef %+7.3f (stored %+7.3f) wald [%+.3f,%+.3f]%s cluster [%+.3f,%+.3f]%s"
                          % (key, len(d), len(uniq), b[1], st["coef"], rec["wald_ci"][0], rec["wald_ci"][1],
                             "+" if rec["wald_pass"] else " ", lo, hi, "+" if rec["cluster_pass"] else " "), flush=True)
        # verdict per ruler: a configuration x benchmark cell passes when all three seeds pass
        for rule in ("wald_pass", "cluster_pass"):
            n = 0; fails = []
            for bench in BENCH:
                for cfg in CONFIGS[bench]:
                    ks = [k for k in out["cells"] if k.startswith("%s|%s|" % (bench, cfg)) and k.endswith("|" + ruler)]
                    ok = len(ks) == 3 and all(out["cells"][k][rule] for k in ks)
                    n += ok
                    if not ok:
                        fails.append("%s|%s: %s" % (bench, cfg, [k.split("|")[2] for k in ks if not out["cells"][k][rule]]))
            out.setdefault("verdict", {})["%s|%s" % (ruler, rule)] = {"cells_passing": n, "of": 15, "failing": fails}
            print("VERDICT", ruler, rule, n, "of 15", fails, flush=True)
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
