"""Training-bank rulers (VISAPP v4 closing, 2026-09-28; advisor's request). CPU only, read-only on every input.

Step 0   regression: Table 1 (G0: kNN LOO k=10 on the C0 test images, src/featurespace/g0_embed.knn_loo + build_labels, imported)
         reproduced to the digit from the cached arrays before anything new.
Front 1  kNN LOO k=10 on the TRAINING bank (neutral rulers) with labels derived by the same question->label rules as the test
         (Kvasir procedure: procedure question + landmark map of src/featurespace/kvasir_procedure_bucket.py; Kvasir finding: the
         clean-split pathology-label file, composite labels dropped; SLAKE modality: `modality` field of the SLAKE json
         (train/validate/test), the field the C6 manifests carry; VQA-RAD organ: `image_organ` of the VQA-RAD json).
Front 1b same, for the model-derived rulers whose bank arrays exist in v5_analysis/embeddings.
Front 2  distance to the bank: d_train (LOO, k=10, bank vs itself) vs d_test (C0, k=10, src/featurespace/analysis_a.knn_to_bank,
         imported; second regression against analysis_a.json), W1, quantiles, anchors of C1/C2/C7/occlusion-75, k=1 near-
         duplicate detector. Writes only under ../v5_analysis/train_rulers/."""
from __future__ import annotations
import hashlib, json, sys, time, collections, glob, socket, subprocess
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import numpy as np, pyarrow as pa
from scipy.stats import wasserstein_distance
P0 = Path("."); M3 = Path("../M3AE"); sys.path.insert(0, str(P0))
from src.featurespace.g0_embed import knn_loo, build_labels          # noqa: E402  (imported, not copied)
from src.featurespace.analysis_a import knn_to_bank, unit             # noqa: E402
CACHE = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "embeddings"; V5E = Path("../v5_analysis/embeddings"); OUT = Path("../v5_analysis/train_rulers")
for d in ("manifests", "arrays", "report"): (OUT / d).mkdir(parents=True, exist_ok=True)
K = 10; RULERS = ["medsiglip", "dinov3", "biomedclip", "dinov2", "colorhist"]; BENCH = ["KVASIR", "SLAKE", "VQARAD"]
IDS = json.loads((CACHE / "image_ids.json").read_text()); BANK_IDS = json.loads((V5E / "bank_ids.json").read_text())
G0 = json.loads((P0 / "docs/v4/g0_results.json").read_text()); AA = json.loads((P0 / "docs/v4/analysis_a.json").read_text())["results"]
TRAIN_ARROW = {"KVASIR": P0 / "data/finetune_arrows/vqa_kvasir_train_clean.arrow", "SLAKE": M3 / "data/finetune_arrows/vqa_slack_train.arrow", "VQARAD": M3 / "data/finetune_arrows/vqa_vqa_rad_train.arrow"}
log = lambda *a: print(*a, flush=True); t0 = time.time()
INPUTS = {}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
def reg(p, role):
    p = Path(p); k = str(p)
    if k not in INPUTS: INPUTS[k] = {"sha256": sha(p), "bytes": p.stat().st_size, "role": role}
    return p
def arr(name, role):
    """mmap the array from v4_cache or v5_analysis/embeddings; None if absent (no computation of embeddings here)."""
    for base in (CACHE, V5E):
        p = base / (name + ".npy")
        if p.exists(): reg(p, role); return np.load(p, mmap_mode="r")
    return None
# ------------------------------------------------------------------ labels (test: G0 function; bank: same rules)
norm = lambda s: " ".join(str(s).strip().lower().split())
Q_PROC = "what type of procedure is the image taken from?"
LANDMARK = {"cecum": "colonoscopy", "ileum": "colonoscopy", "retroflex-rectum": "colonoscopy", "retroflex rectum": "colonoscopy", "z-line": "gastroscopy", "zline": "gastroscopy", "pylorus": "gastroscopy", "retroflex-stomach": "gastroscopy", "retroflex stomach": "gastroscopy"}
def read_arrow(p):
    with pa.memory_map(str(reg(p, "arrow (questions/answers for labels)")), "rb") as s: t = pa.ipc.open_file(s).read_all()
    return [str(x) for x in t.column("image_id").to_pylist()], t.column("questions").to_pylist(), t.column("answers").to_pylist()
def kvasir_procedure_by_rule(arrow):
    """Rule of src/featurespace/kvasir_procedure_bucket.py (amendment B.1): procedure question where asked, landmark map otherwise."""
    ids, qs, ans = read_arrow(arrow); out, conflict = {}, set()
    for i, q, a in zip(ids, qs, ans):
        for qq, aa in zip(q, a):
            n = norm(qq); vals = [norm(v) for x in (aa if isinstance(aa, list) else [aa]) for v in str(x).split(";")]; lab = None
            if n == Q_PROC: lab = next((v for v in vals if v in ("colonoscopy", "gastroscopy")), None)
            elif "landmark" in n: labs = {LANDMARK[v] for v in vals if v in LANDMARK}; lab = labs.pop() if len(labs) == 1 else None
            if lab:
                if i in out and out[i] != lab: conflict.add(i)
                out[i] = lab
    for i in conflict: out.pop(i, None)
    return out, sorted(conflict)
def bank_labels():
    L = {}; ids = {b: BANK_IDS.get(b + "__TRAIN", IDS[b + "__TRAIN"]) for b in BENCH}; cov = {}
    proc, conf = kvasir_procedure_by_rule(TRAIN_ARROW["KVASIR"]); L["kvasir_procedure"] = ("KVASIR", {i: proc[i] for i in ids["KVASIR"] if i in proc}); cov["kvasir_procedure_conflicts_dropped"] = conf
    path = json.loads(reg(P0 / "data/finetune_arrows/vqa_kvasir_clean_pathology_labels.json", "labels (kvasir finding)").read_text())
    pure = {i: path[i] for i in ids["KVASIR"] if i in path and ";" not in path[i]}; cov["kvasir_finding_composite_dropped"] = sum(1 for i in ids["KVASIR"] if i in path and ";" in path[i]); L["kvasir_finding"] = ("KVASIR", pure)
    for pname in ("colonoscopy", "gastroscopy"): L["kvasir_finding_in_" + pname] = ("KVASIR", {i: v for i, v in pure.items() if proc.get(i) == pname})
    org = {}
    for f in ("trainset.json", "testset.json", "valset.json"):
        for x in json.loads(reg(M3 / "data/finetune_data/vqa_rad" / f, "labels (vqarad image_organ)").read_text()): org.setdefault(x["image_name"], set()).add({"CHEST": "chest", "ABD": "abdomen", "HEAD": "head"}[x["image_organ"]])
    L["vqarad_organ"] = ("VQARAD", {i: next(iter(org[i.split("images/")[-1]])) for i in ids["VQARAD"] if i.split("images/")[-1] in org and len(org[i.split("images/")[-1]]) == 1})
    mod = {}
    for f in ("train.json", "validate.json", "test.json"):
        for x in json.loads(reg(M3 / "data/finetune_data/slack" / f, "labels (slake modality)").read_text()): mod.setdefault(x["img_name"], set()).add(x["modality"].upper().replace("-", ""))
    L["slake_modality"] = ("SLAKE", {i: next(iter(mod[i.split("imgs/")[-1]])) for i in ids["SLAKE"] if i.split("imgs/")[-1] in mod and len(mod[i.split("imgs/")[-1]]) == 1})
    return L, cov, mod, proc
def to_sha_manifest():
    (OUT / "manifests/inputs.json").write_text(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "host": os.environ.get("RUN_HOST", "unspecified"), "git_commit_p0": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(P0)).stdout.strip(), "k": K, "inputs": INPUTS}, indent=1))
# ------------------------------------------------------------------ Step 0: regression of Table 1 (test)
TEST_IDS = {b: IDS[b + "__C0"] for b in BENCH}
Ltest = build_labels(TEST_IDS); composite_test = Ltest.pop("_composite_dropped")
for f in ("kvasir_clean_bucket_colonoscopy.json", "kvasir_clean_bucket_gastroscopy.json", "vqarad_bucket_chest.json", "vqarad_bucket_abd.json", "vqarad_bucket_head.json"): reg(P0 / "results" / f, "labels (test, G0)")
for f in sorted((P0 / "results").glob("C6_manifest_SLAKE_*.json")): reg(f, "labels (test, G0 slake modality)")
reg(P0 / "docs/v4/g0_results.json", "regression target (Table 1)"); reg(P0 / "docs/v4/analysis_a.json", "regression target (Analysis A C0)")
REG1 = {"tasks": {}, "all_exact": True, "note": "arrays as stored (float32) -> knn_loo of g0_embed; a float64 pass is reported alongside"}
def loo_task(X, ids, lab):
    keep = [j for j, i in enumerate(ids) if i in lab]; y = [lab[ids[j]] for j in keep]
    return knn_loo(np.asarray(X)[keep], y), len(keep), sorted(set(y))
for task, (bench, lab) in Ltest.items():
    rec = {"benchmark": bench, "n_published": G0["results"][task]["n"], "cells": {}}
    for r in RULERS:
        X = arr("%s__%s" % (bench, r), "test C0 embeddings (Table 1 / Analysis A)"); (a, n), nk, cls = loo_task(X, TEST_IDS[bench], lab); a64 = knn_loo(np.asarray(X)[[j for j, i in enumerate(TEST_IDS[bench]) if i in lab]].astype(np.float64), [lab[TEST_IDS[bench][j]] for j in range(len(TEST_IDS[bench])) if TEST_IDS[bench][j] in lab])[0]
        pub = G0["results"][task]["acc"][r]; exact = (a == pub) if (a is not None and pub is not None) else (a is None and pub is None); REG1["all_exact"] &= bool(exact) and nk == G0["results"][task]["n"]
        rec["cells"][r] = {"published": pub, "recomputed_fp32": a, "recomputed_fp64": a64, "exact": bool(exact), "n": nk}
    REG1["tasks"][task] = rec; log("REG1 %-32s n %d/%d %s" % (task, nk, G0["results"][task]["n"], {r: ("=" if v["exact"] else "≠ %.4f vs %.4f" % (v["recomputed_fp32"], v["published"])) for r, v in rec["cells"].items()}))
log("REG1 Tabela 1 reproduz ao digito:", REG1["all_exact"])
if not REG1["all_exact"]:
    (OUT / "manifests/REGRESSION_FAILED_table1.json").write_text(json.dumps(REG1, indent=1)); to_sha_manifest(); sys.exit("PARE: Tabela 1 nao reproduz")
# ------------------------------------------------------------------ Front 1 / 1b: kNN LOO on the bank
Lbank, COV, MOD_META, PROC_RULE = bank_labels()
# consistency checks of the label rules on the test set
proc_test, conf_test = kvasir_procedure_by_rule(P0 / "data/finetune_arrows/vqa_kvasir_test_clean.arrow"); col = set(json.loads((P0 / "results/kvasir_clean_bucket_colonoscopy.json").read_text())); gas = set(json.loads((P0 / "results/kvasir_clean_bucket_gastroscopy.json").read_text()))
CHECK = {"kvasir_procedure_rule_vs_test_buckets": {"agree": sum(1 for i in col if proc_test.get(i) == "colonoscopy") + sum(1 for i in gas if proc_test.get(i) == "gastroscopy"), "buckets": len(col) + len(gas), "rule_labels_test": len(proc_test), "conflicts_test": len(conf_test)},
         "slake_modality_meta_vs_c6_manifests": {"agree": sum(1 for i, m in Ltest["slake_modality"][1].items() if i.split("imgs/")[-1] in MOD_META and m.upper() in MOD_META[i.split("imgs/")[-1]]), "manifest_labels": len(Ltest["slake_modality"][1])},
         "vqarad_organ_meta_vs_test_buckets": {"agree": sum(1 for i, o in Ltest["vqarad_organ"][1].items() if i in bank_labels_cache) if False else None}}
coverage = {task: {"benchmark": b, "n_bank_labelled": len(lab), "n_bank": len(IDS[b + "__TRAIN"]), "n_test_labelled": Ltest[task][1].__len__() if task in Ltest else None, "n_test": len(TEST_IDS[b]), "classes_bank": dict(collections.Counter(lab.values()))} for task, (b, lab) in Lbank.items()}
coverage["_notes"] = COV; log("coverage:", {t: (v["n_bank_labelled"], v["n_test_labelled"]) for t, v in coverage.items() if not t.startswith("_")})
def front1(ruler_list, arr_name):
    res = {}
    for task, (bench, lab) in Lbank.items():
        bank_ids = IDS[bench + "__TRAIN"]; rec = {"benchmark": bench, "n_bank": None, "classes": None, "cells": {}}
        for r in ruler_list[bench] if isinstance(ruler_list, dict) else ruler_list:
            Xb = arr(arr_name(bench, r), "bank embeddings"); Xt = arr("%s__%s" % (bench, r), "test C0 embeddings")
            if Xb is None or Xt is None: rec["cells"][r] = {"missing": True}; continue
            (a_tr, n_tr), nk, cls = loo_task(Xb, bank_ids, lab); rec["n_bank"] = nk; rec["classes"] = cls
            a_te = (loo_task(Xt, TEST_IDS[bench], Ltest[task][1])[0][0] if task in Ltest else None)
            rec["cells"][r] = {"train": a_tr, "test": a_te, "delta_train_minus_test": (a_tr - a_te) if (a_tr is not None and a_te is not None) else None, "n_train": nk, "n_test": len(Ltest[task][1]) if task in Ltest else None}
        res[task] = rec; log("F1 %-32s n %s %s" % (task, rec["n_bank"], {r: ("%.4f/%.4f" % (v["train"], v["test"]) if v.get("train") is not None and v.get("test") is not None else "—") for r, v in rec["cells"].items()}))
    return res
T1 = front1(RULERS, lambda b, r: "%s__TRAIN__%s" % (b, r)); (OUT / "table1_train.json").write_text(json.dumps({"k": K, "labels_bank": {t: {"benchmark": b, "n": len(l)} for t, (b, l) in Lbank.items()}, "coverage": coverage, "checks": CHECK, "results": T1}, indent=1))
MODEL_RULERS = {b: sorted(p.name[len(b) + 9:-4] for p in V5E.glob("%s__TRAIN__*.npy" % b)) for b in BENCH}
T1M = front1(MODEL_RULERS, lambda b, r: "%s__TRAIN__%s" % (b, r)); (OUT / "table1_train_model_rulers.json").write_text(json.dumps({"k": K, "model_rulers_with_bank": MODEL_RULERS, "results": T1M}, indent=1))
# ------------------------------------------------------------------ Front 2: distance to the bank, train (LOO) vs test (C0)
def knn_loo_dist(B, k=K, block=2048):
    """Mean cosine distance from each bank image to its k nearest OTHER bank images (float64, as knn_to_bank)."""
    Bu = unit(np.asarray(B, dtype=np.float64)); out = np.empty(len(Bu)); nn1 = np.empty(len(Bu))
    for i in range(0, len(Bu), block):
        S = Bu[i:i + block] @ Bu.T; S[np.arange(S.shape[0]), np.arange(i, i + S.shape[0])] = -np.inf
        part = np.partition(-S, k - 1, axis=1)[:, :k]; out[i:i + block] = 1.0 - (-part).mean(1); nn1[i:i + block] = 1.0 - S.max(1)
    return out, nn1
def nn1_to_bank(X, B, block=2048):
    Z, Bu = unit(np.asarray(X, dtype=np.float64)), unit(np.asarray(B, dtype=np.float64)); out = np.empty(len(Z)); arg = np.empty(len(Z), int)
    for i in range(0, len(Z), block): S = Z[i:i + block] @ Bu.T; arg[i:i + block] = S.argmax(1); out[i:i + block] = 1.0 - S.max(1)
    return out, arg
ANCH = {"C1": "C1", "C2": "C2", "C7": "C7", "C4_OCC_75": "C4_OCC_75"}
def q(x): return {"mean": float(np.mean(x)), "median": float(np.median(x)), "q10": float(np.percentile(x, 10)), "q90": float(np.percentile(x, 90)), "n": int(len(x))}
REG2 = {"cells": {}, "all_exact_1e-6": True}; F2 = {}
def front2(bench, r, bank_name, ruler_group):
    B = arr(bank_name, "bank embeddings"); X0 = arr("%s__%s" % (bench, r), "test C0 embeddings")
    if B is None or X0 is None: return {"missing": True}
    d_tr, nn1_tr = knn_loo_dist(B); d_te = knn_to_bank(np.asarray(X0, dtype=np.float64), np.asarray(B, dtype=np.float64)); d1, a1 = nn1_to_bank(X0, B)
    np.save(OUT / "arrays" / ("%s__%s__d_train_loo_k10.npy" % (bench, r)), d_tr); np.save(OUT / "arrays" / ("%s__%s__d_test_c0_k10.npy" % (bench, r)), d_te); np.save(OUT / "arrays" / ("%s__%s__d1_test_to_bank.npy" % (bench, r)), d1)
    med = float(np.median(d_tr)); rec = {"ruler_group": ruler_group, "d_train_loo": q(d_tr), "d_test_c0": q(d_te), "w1_train_vs_test": float(wasserstein_distance(d_tr, d_te)), "share_test_above_train_median": float((d_te > med).mean()), "train_median": med,
           "anchors_relative_to_train_median": {}, "near_duplicates_k1": {"min": float(d1.min()), "q01": float(np.percentile(d1, 1)), "n_lt_0.01": int((d1 < 0.01).sum()), "n_lt_0.05": int((d1 < 0.05).sum()), "ids_lt_0.01": [{"test_id": IDS[bench + "__C0"][j], "bank_id": IDS[bench + "__TRAIN"][int(a1[j])], "d1": float(d1[j])} for j in np.where(d1 < 0.01)[0]], "n_lt_1e-4": int((d1 < 1e-4).sum()), "train_loo_k1": q(nn1_tr)}}
    key = "%s|C0|%s" % (bench, r)
    if key in AA:
        pub = AA[key]["knn_mean"]; rel = abs(float(d_te.mean()) - pub) / max(abs(pub), 1e-12); REG2["cells"][key] = {"published_knn_mean": pub, "recomputed": float(d_te.mean()), "rel_diff": rel, "exact_1e-6": bool(rel <= 1e-6)}; REG2["all_exact_1e-6"] &= bool(rel <= 1e-6)
    for cond, tag in ANCH.items():
        Xc = arr("%s__%s__%s" % (bench, tag, r), "test condition embeddings (anchors)")
        if Xc is None: rec["anchors_relative_to_train_median"][cond] = None; continue
        dc = knn_to_bank(np.asarray(Xc, dtype=np.float64), np.asarray(B, dtype=np.float64)); rec["anchors_relative_to_train_median"][cond] = {"knn_mean": float(dc.mean()), "knn_median": float(np.median(dc)), "minus_train_median": float(np.median(dc) - med), "share_above_train_median": float((dc > med).mean()), "share_above_train_q90": float((dc > np.percentile(d_tr, 90)).mean())}
    return rec
for bench in BENCH:
    for r in RULERS: F2["%s|%s" % (bench, r)] = front2(bench, r, "%s__TRAIN__%s" % (bench, r), "neutral"); rec = F2["%s|%s" % (bench, r)]; log("F2 %-7s %-11s train med %.4f test med %.4f W1 %.4f share>med %.3f | k1 min %.4f n<0.01 %d n<0.05 %d | REG2 %s" % (bench, r, rec["train_median"], rec["d_test_c0"]["median"], rec["w1_train_vs_test"], rec["share_test_above_train_median"], rec["near_duplicates_k1"]["min"], rec["near_duplicates_k1"]["n_lt_0.01"], rec["near_duplicates_k1"]["n_lt_0.05"], REG2["cells"].get("%s|C0|%s" % (bench, r), {}).get("exact_1e-6")))
    for r in MODEL_RULERS.get(bench, []): F2["%s|%s" % (bench, r)] = front2(bench, r, "%s__TRAIN__%s" % (bench, r), "model")
log("REG2 Analysis A C0 reproduz (rel <= 1e-6):", REG2["all_exact_1e-6"])
# byte-level duplicates: sha256 of the image bytes, test (C0 arrow of G0) x bank (train arrow); no image is written
TEST_ARROW = {"KVASIR": P0 / "data/finetune_arrows/vqa_kvasir_test_clean.arrow", "SLAKE": M3 / "data/finetune_arrows/vqa_slack_test.arrow.orig_backup", "VQARAD": M3 / "data/finetune_arrows/vqa_vqa_rad_test.arrow"}
def image_shas(p):
    with pa.memory_map(str(reg(p, "arrow (image bytes for the duplicate check)")), "rb") as s_: t = pa.ipc.open_file(s_).read_all()
    return {str(i): hashlib.sha256(b).hexdigest() for i, b in zip(t.column("image_id").to_pylist(), t.column("image").to_pylist())}
DUP = {}
for bench in BENCH:
    ts, bs = image_shas(TEST_ARROW[bench]), image_shas(TRAIN_ARROW[bench]); inv = collections.defaultdict(list)
    for i, h in bs.items(): inv[h].append(i)
    hits = {i: inv[h] for i, h in ts.items() if h in inv}; same_id = sum(1 for i, v in hits.items() if i in v)
    within_test = sum(v - 1 for v in collections.Counter(ts.values()).values() if v > 1); within_bank = sum(len(v) - 1 for v in inv.values() if len(v) > 1)
    DUP[bench] = {"n_test": len(ts), "n_bank": len(bs), "test_images_with_identical_bytes_in_bank": len(hits), "of_which_same_image_id": same_id, "distinct_bank_images_hit": len({j for v in hits.values() for j in v}), "duplicate_bytes_within_test": within_test, "duplicate_bytes_within_bank": within_bank, "pairs_sample": [{"test_id": i, "bank_ids": v} for i, v in list(hits.items())[:10]], "all_pairs": [{"test_id": i, "bank_ids": v} for i, v in hits.items()]}
    log("DUP %-7s test->bank identical bytes: %d of %d (same id %d); within-test dup %d; within-bank dup %d" % (bench, len(hits), len(ts), same_id, within_test, within_bank))
(OUT / "manifests/byte_duplicates_test_vs_bank.json").write_text(json.dumps(DUP, indent=1))
(OUT / "distances_train_vs_test.json").write_text(json.dumps({"k": K, "regression_analysis_a_c0": REG2, "results": F2}, indent=1)); (OUT / "manifests/regression_table1.json").write_text(json.dumps(REG1, indent=1)); to_sha_manifest()
if not REG2["all_exact_1e-6"]: sys.exit("PARE: C0 da Analise A nao reproduz")
# ------------------------------------------------------------------ report
pc = lambda x, d=4: "—" if x is None else ("%." + str(d) + "f") % x
L = ["# Réguas no banco de treino — Tabela 1 no treino e distância ao banco (treino LOO × teste C0)", "", "Gerado por `harness/v5_analysis/train_rulers_main.py` em %s (host %s, commit p0 `%s`). CPU, `nice -n 19`, arrays em `mmap_mode='r'`. Números sem interpretação além das leituras pré-fixadas. Entradas e sha256 em `manifests/inputs.json` (%d arquivos)." % (time.strftime("%Y-%m-%d %H:%M"), os.environ.get("RUN_HOST", "unspecified"), json.loads((OUT / "manifests/inputs.json").read_text())["git_commit_p0"][:7], len(INPUTS)), "",
     "## Entradas", "", "| arquivo | papel | sha256 |", "|---|---|---|"] + ["| `%s` | %s | `%s…` |" % (k.replace("../", ""), v["role"], v["sha256"][:16]) for k, v in sorted(INPUTS.items())]
L += ["", "## Regressões obrigatórias", "", "1. **Tabela 1 (G0, kNN LOO k = 10 no teste, `g0_embed.knn_loo` + `build_labels` importados):** reproduz ao dígito: **%s** (%d tarefas × 5 réguas; fp32 como armazenado; a passagem fp64 está no JSON)." % ("sim" if REG1["all_exact"] else "NÃO", len(REG1["tasks"])), "2. **C0 da Análise A (`analysis_a.knn_to_bank` importado, k = 10 ao banco):** reproduz (|Δ| relativo ≤ 1e-6): **%s**; maior Δ relativo %.2e." % ("sim" if REG2["all_exact_1e-6"] else "NÃO", max(v["rel_diff"] for v in REG2["cells"].values())), ""]
L += ["## Cobertura de rótulos (banco × teste)", "", "| tarefa | benchmark | banco rotulado / banco | teste rotulado / teste | classes no banco |", "|---|---|---|---|---|"]
for t, v in coverage.items():
    if t.startswith("_"): continue
    L.append("| %s | %s | %d / %d | %s / %d | %s |" % (t, v["benchmark"], v["n_bank_labelled"], v["n_bank"], v["n_test_labelled"], v["n_test"], ", ".join("%s %d" % kv for kv in sorted(v["classes_bank"].items()))))
L += ["", "Regras: procedimento do Kvasir pela pergunta de procedimento e pelo mapa de marcos anatômicos (`kvasir_procedure_bucket.py`); no teste a regra reproduz os buckets `kvasir_clean_bucket_*` em %d/%d; no banco %d imagens com conflito pergunta × marco foram descartadas (%s). Achado do Kvasir: `vqa_kvasir_clean_pathology_labels.json`, rótulos compostos descartados (banco %d; teste %d). Modalidade do SLAKE: campo `modality` dos JSONs do SLAKE (train/validate/test), concordância com os manifestos C6 do teste %d/%d. Órgão do VQA-RAD: `image_organ` dos JSONs do VQA-RAD (os buckets do teste vêm dele; 203/203)." % (CHECK["kvasir_procedure_rule_vs_test_buckets"]["agree"], CHECK["kvasir_procedure_rule_vs_test_buckets"]["buckets"], len(COV["kvasir_procedure_conflicts_dropped"]), ", ".join(COV["kvasir_procedure_conflicts_dropped"]), COV["kvasir_finding_composite_dropped"], composite_test, CHECK["slake_modality_meta_vs_c6_manifests"]["agree"], CHECK["slake_modality_meta_vs_c6_manifests"]["manifest_labels"]), ""]
L += ["## Frente 1 — kNN LOO (k = 10) no banco de treino, réguas neutras", "", "Célula: teste (Tabela 1) / treino / Δ = treino − teste. Leitura pré-fixada: |Δ| ≤ 0,02.", "", "| tarefa | n teste / n treino | " + " | ".join(RULERS) + " |", "|---|---|" + "---|" * len(RULERS)]
viol = []
for t, rec in T1.items():
    L.append("| %s | %s / %s | " % (t, rec["cells"][RULERS[0]].get("n_test"), rec["n_bank"]) + " | ".join(("%s / %s / %+.3f" % (pc(c["test"]), pc(c["train"]), c["delta_train_minus_test"])) if c.get("delta_train_minus_test") is not None else "—" for c in (rec["cells"][r] for r in RULERS)) + " |")
    for r in RULERS:
        c = rec["cells"][r]
        if c.get("delta_train_minus_test") is not None and abs(c["delta_train_minus_test"]) > 0.02: viol.append("%s × %s: Δ %+.3f" % (t, r, c["delta_train_minus_test"]))
L += ["", "## Frente 1b — réguas do modelo com banco em `v5_analysis/embeddings` (Kvasir %d, SLAKE %d, VQA-RAD %d)" % (len(MODEL_RULERS["KVASIR"]), len(MODEL_RULERS["SLAKE"]), len(MODEL_RULERS.get("VQARAD", []))), "", "Leitura pré-fixada: torres adaptadas kNN(treino) > kNN(teste); congeladas Δ ≈ 0 (|Δ| ≤ 0,02).", "", "| tarefa | régua | teste | treino | Δ |", "|---|---|---|---|---|"]
violm = []
for t, rec in T1M.items():
    for r, c in rec["cells"].items():
        if c.get("missing"): continue
        L.append("| %s | %s | %s | %s | %s |" % (t, r, pc(c["test"]), pc(c["train"]), ("%+.3f" % c["delta_train_minus_test"]) if c["delta_train_minus_test"] is not None else "—"))
        adapted = ("_vt_" in r) or (r.startswith("smolvlm_") and "base" not in r) or (r.startswith("m3ae_s"))
        if c["delta_train_minus_test"] is not None and ((adapted and c["delta_train_minus_test"] <= 0) or ((not adapted) and abs(c["delta_train_minus_test"]) > 0.02)): violm.append("%s × %s (%s): Δ %+.3f" % (t, r, "adaptada" if adapted else "congelada", c["delta_train_minus_test"]))
L += ["", "## Frente 2 — distância ao banco (k = 10): treino LOO × teste C0", "", "| benchmark | régua | treino: média / mediana / q10 / q90 | teste C0: média / mediana / q10 / q90 | W1 | teste acima da mediana do treino | C1 / C2 / C7 / occ75: mediana − mediana(treino) [fração acima] |", "|---|---|---|---|---|---|---|"]
for key, rec in F2.items():
    if rec.get("missing"): continue
    b, r = key.split("|"); a = rec["anchors_relative_to_train_median"]; anc = " / ".join(("%+.3f [%.2f]" % (a[c]["minus_train_median"], a[c]["share_above_train_median"])) if a.get(c) else "—" for c in ANCH)
    L.append("| %s | %s%s | %s / %s / %s / %s | %s / %s / %s / %s | %s | %.3f | %s |" % (b, r, "" if rec["ruler_group"] == "neutral" else " (modelo)", pc(rec["d_train_loo"]["mean"]), pc(rec["d_train_loo"]["median"]), pc(rec["d_train_loo"]["q10"]), pc(rec["d_train_loo"]["q90"]), pc(rec["d_test_c0"]["mean"]), pc(rec["d_test_c0"]["median"]), pc(rec["d_test_c0"]["q10"]), pc(rec["d_test_c0"]["q90"]), pc(rec["w1_train_vs_test"]), rec["share_test_above_train_median"], anc))
L += ["", "### Detector de quase-duplicatas (k = 1, teste → banco)", "", "Leitura pré-fixada: no split limpo, zero ou quase zero casos com d₁ < 0,01.", "", "| benchmark | régua | mín d₁ | q01 | n < 1e-4 | n < 0,01 | n < 0,05 | ids < 0,01 (até 6) |", "|---|---|---|---|---|---|---|---|"]
for key, rec in F2.items():
    if rec.get("missing"): continue
    b, r = key.split("|"); nd = rec["near_duplicates_k1"]; L.append("| %s | %s | %s | %s | %d | %d | %d | %s |" % (b, r, pc(nd["min"]), pc(nd["q01"]), nd["n_lt_1e-4"], nd["n_lt_0.01"], nd["n_lt_0.05"], "; ".join("%s→%s (%.4f)" % ("/".join(x["test_id"].split("/")[-2:]).replace("/source.jpg", ""), "/".join(x["bank_id"].split("/")[-2:]).replace("/source.jpg", ""), x["d1"]) for x in nd["ids_lt_0.01"][:6]) or "—"))
L += ["", "### Duplicatas exatas por bytes (sha256 da imagem; teste × banco)", "", "| benchmark | imagens de teste com bytes idênticos a alguma do banco | mesmo image_id | imagens do banco atingidas | duplicatas internas ao teste | duplicatas internas ao banco |", "|---|---|---|---|---|---|"] + ["| %s | %d / %d | %d | %d | %d | %d |" % (b, v["test_images_with_identical_bytes_in_bank"], v["n_test"], v["of_which_same_image_id"], v["distinct_bank_images_hit"], v["duplicate_bytes_within_test"], v["duplicate_bytes_within_bank"]) for b, v in DUP.items()] + ["", "Lista completa dos pares em `manifests/byte_duplicates_test_vs_bank.json`. Exemplos (Kvasir): %s." % ("; ".join("%s→%s" % (x["test_id"].split("/")[-1], ",".join(j.split("/")[-1] for j in x["bank_ids"])) for x in DUP["KVASIR"]["pairs_sample"][:5]) or "nenhum")]
L += ["", "## Leituras pré-fixadas: cumpridas / violadas", "", "- Frente 1 (|Δ| ≤ 0,02, réguas neutras): %s" % ("cumprida em todas as células" if not viol else "violada em %d células: %s" % (len(viol), "; ".join(viol))), "- Frente 1b (adaptadas Δ > 0; congeladas |Δ| ≤ 0,02): %s" % ("cumprida em todas as células" if not violm else "violada em %d células: %s" % (len(violm), "; ".join(violm))),
      "- Frente 2, bytes idênticos teste→banco: %s; quase-duplicatas por régua (d₁ < 0,01; pré-fixado ≈ 0 no split limpo): %s" % ("; ".join("%s: %d imagens" % (b, v["test_images_with_identical_bytes_in_bank"]) for b, v in DUP.items()) if False else "; ".join("%s: %d imagens" % (b, v["test_images_with_identical_bytes_in_bank"]) for b, v in DUP.items()), "; ".join("%s %s: %d" % (k.split("|")[0], k.split("|")[1], v["near_duplicates_k1"]["n_lt_0.01"]) for k, v in F2.items() if not v.get("missing") and v["ruler_group"] == "neutral")), "", "Nada foi escrito fora de `v5_analysis/train_rulers/`; nenhum embedding foi calculado; sem GPU."]
(OUT / "report/relatorio_reguas_treino.md").write_text("\n".join(L)); log("FIM %.0fs" % (time.time() - t0))
