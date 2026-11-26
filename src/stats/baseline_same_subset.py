"""Accuracy under C1 of every fine-tuned checkpoint on the questions the question-only baseline is
evaluated on (its own question set), against the baseline's accuracy on that set (Section 5.1 of the
paper). Reads the stored predictions only; no model and no GPU.

    python -m src.stats.baseline_same_subset

Writes docs/camera_ready/baseline_same_subset.json.
"""
import json, statistics
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
OUT = REPO / "docs/camera_ready/baseline_same_subset.json"
CONFIGS = {"KVASIR": ["M3AE", "QWEN2VL_LORA", "QWEN2VL_DORA", "SMOLVLM_LORA", "SMOLVLM_DORA"],
           "SLAKE": ["M3AE_SCRATCH", "QWEN2VL_LORA", "QWEN2VL_DORA", "SMOLVLM_LORA", "SMOLVLM_DORA"],
           "VQARAD": ["M3AE_SCRATCH", "QWEN2VL_LORA", "QWEN2VL_DORA", "SMOLVLM_LORA", "SMOLVLM_DORA"]}
def load(name):
    f = RES / (name + ".json")
    return {p["qid"]: bool(p["correct"]) for p in json.loads(f.read_text())["predictions"]} if f.exists() else None
out = {}
for b, cfgs in CONFIGS.items():
    m0 = load("M0_question_only_%s_C0" % b); q = set(m0)
    base = 100 * sum(m0.values()) / len(m0)
    row = {"baseline_n": len(q), "baseline_acc": base, "configs": {}}
    for c in cfgs:
        accs, alls = [], []
        for s in ("_seed0", "_seed1", "_seed2"):
            p = load("%s_%s%s_C1" % (c, b, s)) or (load("%s_%s_C1" % (c, b)) if s == "_seed0" else None)
            assert p is not None, (c, b, s)
            assert q <= set(p), (c, b, s, len(q - set(p)))
            accs.append(100 * sum(p[i] for i in q) / len(q)); alls.append(100 * sum(p.values()) / len(p))
        row["configs"][c] = {"n_all": len(p), "c1_closed_per_seed": accs, "c1_closed_mean": statistics.mean(accs),
                             "c1_all_mean": statistics.mean(alls), "below_in_all_seeds": all(a < base for a in accs),
                             "mean_minus_baseline": statistics.mean(accs) - base}
    out[b] = row
    print(b, "baseline %.2f on %d" % (base, len(q)))
    for c, v in row["configs"].items():
        print("  %-14s closed %.2f (%s) all %.2f (n=%d) diff %+.2f below_all_seeds=%s" % (c, v["c1_closed_mean"], ", ".join("%.2f" % a for a in v["c1_closed_per_seed"]), v["c1_all_mean"], v["n_all"], v["mean_minus_baseline"], v["below_in_all_seeds"]))
OUT.write_text(json.dumps(out, indent=1))
