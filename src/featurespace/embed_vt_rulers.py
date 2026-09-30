"""Model-derived rulers of the v5 Qwen2-VL visual-tower lines, SLAKE and VQA-RAD, C0.

Same machinery as src/featurespace/embed_model_rulers.py (its functions are imported,
not copied), pointed at a separate cache: ../v5_analysis/
embeddings. The v4 cache is never written; image_ids.json is copied from it so
the order contract is the same one. fp32 only (the canonical pass); tower =
before PatchMerger, proj = after; mean over visual tokens; position control
exactly zero before anything is written.

Rulers: qwen2vl_{lora,dora}_vt_s{0,1,2}, one per line, adapters read from
../v5_qwen_vt/models (verified sha256 in its manifests).
"""
import json, os, shutil, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from src.featurespace import embed_model_rulers as E
NEW = Path("../v5_analysis/embeddings"); NEW.mkdir(parents=True, exist_ok=True)
V4 = Path("cache/embeddings"); V5 = Path("../v5_qwen_vt")
if not (NEW / "image_ids.json").exists():
    shutil.copy(V4 / "image_ids.json", NEW / "image_ids.json")
if not (NEW / "manifest.json").exists():
    (NEW / "manifest.json").write_text(json.dumps({"arrays": {}, "rulers": {}, "position_control": {},
        "note": "v5 analysis cache: rulers of the visual-tower-adapted Qwen2-VL lines; external rulers are read from the v4 cache by symlink, never rewritten"}, indent=1))
E.CACHE = NEW
adapters = json.loads((V5 / "manifests/adapters.json").read_text())
BENCHES = sys.argv[1:] or ["SLAKE", "VQARAD"]
t0 = time.time()
for b in BENCHES:
    ids, imgs = E.load(E.ARROWS[b]); sha_ids = E.order_contract(b, ids)
    for peft in ("lora", "dora"):
        for s in (0, 1, 2):
            line = "qwen2vl_%s_vt_%s_seed%d" % (peft, b.lower(), s)
            rec = adapters.get(line)
            if not rec or not rec.get("verified"):
                print("  %-7s %-30s adapter nao verificado: pulado" % (b, line), flush=True); continue
            w = Path(rec["adapter"])
            E.run_line(b, "qwen2vl_%s_vt_s%d" % (peft, s), "QWEN2VL", w, ids, imgs, sha_ids, passes=("fp32",))
    del imgs
# the role text run_line writes assumes a frozen Qwen tower; these lines are the opposite
def fix(man):
    for k, v in man.get("rulers", {}).items():
        if "_vt_s" in k:
            v["role"] = "model-derived ruler, visual tower ADAPTED (LoRA/DoRA on visual.blocks.*.attn.qkv+proj and LLM q/k/v/o_proj); v5 line"
            v["adapter_sha256"] = adapters["qwen2vl_%s_vt_%s_seed%s" % (v["adapter"].split("qwen2vl_")[1].split("_vt_")[0], v["adapter"].rsplit("_vt_", 1)[1].rsplit("_seed", 1)[0], v["adapter"][-1])]["sha256"] if v.get("adapter") else None
E.update_manifest(fix)
print("\n  concluido em %.1f min" % ((time.time() - t0) / 60))
