"""Rulers derived from the models the paper evaluates, C0 only.

The five external rulers say how far an intervention moves an image. Whether
that is the displacement each model actually feels is the supervisor's
question, and it needs the encoders of the evaluated models themselves, run
over the same intact images, into the same cache, under the same rules.

Gate 0 decided what "the model's encoder" means per family, and it is not one
answer:

  Qwen2-VL   the adapters never touched the tower or the merger -- the bare
             suffix list q/k/v/o_proj matches nothing in a tower that names
             its attention `qkv`/`proj` -- so the ruler is the base encoder,
             one per benchmark: `qwen2vl_base`.
  SmolVLM2   q/k/v of all 27 SigLIP layers carry LoRA/DoRA (not out_proj), so
             the tower differs per configuration and seed: one ruler per line.
  M3AE       full fine-tune; 138 of 142 tower tensors moved: one per line.
             `M3AE` (Kvasir, authors' backbone) and `M3AE_SCRATCH` (SLAKE and
             VQA-RAD, from scratch) are different lines and are named apart.

Plus the untouched towers as the before/after reference: `smolvlm_base`,
`m3ae_base`. Weights come from `src.featurespace.weights.resolve`, which refuses to
choose between rivals; a cell it cannot resolve is skipped and named, never
guessed.

Two variants per ruler, one pooling for all of them:
  tower  output of the vision tower, before the projector
  proj   after the projector, the tokens the LLM receives
  mean over visual tokens (M3AE's CLS is not a visual token and is dropped)

Preprocessing is the model's own processor with the caps the evaluated models
ran under (med_vlm/models/loader.py: Qwen2-VL max_pixels 512*512, SmolVLM2
longest_edge 768; M3AE clip_transform at image_size 384 as in
harness/extract_predictions_m3ae.py). One image per forward for the two VLMs
-- dynamic resolution gives every image its own token count -- and batches of
32 for M3AE, as for the rest of the cache.

Amendment 7.7 as everywhere in the cache: fp32, TF32 off, deterministic
algorithms, and the position control (same images in two orders) equal to
exactly zero before any array is written. An fp16 twin of every array is
written beside it for analysis D, as `*_fp16.npy`, evidence and not a ruler.

Order contract: the ids of the arrow being embedded must equal
image_ids.json["<BENCH>__C0"] item by item, checked before the first save; the
sha256 of that id list is stored with every array so the check can be repeated
without the arrow.

Naming follows {BENCH}__{ruler}.npy with ruler = <name>_<variant>:
  KVASIR__qwen2vl_base_tower.npy   KVASIR__smolvlm_lora_s0_proj.npy
  SLAKE__m3ae_scratch_s1_tower.npy  KVASIR__m3ae_base_proj.npy

Nothing existing in the cache is overwritten: an array on disk is skipped and
manifest entries are only added. The M3AE lines run in the `lapa` environment
(pytorch_lightning, sacred), one subprocess per checkpoint; the VLM lines in
`circle`. `--plan` lists every line and runs the order contract with no model
loaded and no GPU touched.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.featurespace import paths as P          # noqa: E402
from src.featurespace import weights as W        # noqa: E402

ROOT = REPO.parent
M3AE_DIR = ROOT / "M3AE"
CACHE = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "embeddings"
LAPA = "../miniconda3/envs/lapa/bin/python"
DEV = os.environ.get("MR_DEV", "cuda")   # MR_DEV=cpu only for the smoke test
CONTROL_N = 32
M3AE_BS = 32
M3AE_IMAGE_SIZE = 384                      # harness/extract_predictions_m3ae.py
QWEN_MAX_PIXELS, QWEN_MIN_PIXELS = 512 * 512, 4 * 28 * 28   # med_vlm/models/loader.py
SMOLVLM_LONGEST_EDGE = 768                                  # med_vlm/models/loader.py

# the C0 arrows of the G0 cache (src/featurespace/g0_embed.py ARROWS)
ARROWS = {
    "KVASIR": REPO / "data/finetune_arrows/vqa_kvasir_test_clean.arrow",
    "SLAKE": M3AE_DIR / "data/finetune_arrows/vqa_slack_test.arrow",
    "VQARAD": M3AE_DIR / "data/finetune_arrows/vqa_vqa_rad_test.arrow",
}
BASE_ID = {"QWEN2VL": "Qwen/Qwen2-VL-2B-Instruct",
           "SMOLVLM": "HuggingFaceTB/SmolVLM2-2.2B-Instruct"}   # med_vlm/enums.py
M3AE_PRETRAINED = M3AE_DIR / "downloaded/pretrained/m3ae.ckpt"
M3AE_KEY = {"KVASIR": "vqa_kvasir", "SLAKE": "vqa_slack", "VQARAD": "vqa_vqa_rad"}
VARIANTS = ("tower", "proj")


# ------------------------------------------------------------------ lines
def lines():
    """Every (bench, ruler, family, weights) to embed. Unresolvable cells are
    returned apart, with the reason, and never embedded."""
    out, skipped = [], []
    for b in P.BENCHMARKS:
        out.append((b, "qwen2vl_base", "QWEN2VL", None))
        out.append((b, "smolvlm_base", "SMOLVLM", None))
        out.append((b, "m3ae_base", "M3AE", None))
        for cfg in P.CONFIGS[b]:
            if cfg.startswith("QWEN2VL"):
                continue                      # tower frozen: the base ruler covers it
            for s in P.SEEDS:
                try:
                    w = W.resolve(b, cfg, s)
                except (W.AmbiguousWeights, W.MissingWeights) as e:
                    skipped.append((b, cfg, s, type(e).__name__))
                    continue
                fam = "M3AE" if cfg.startswith("M3AE") else cfg.split("_")[0]
                out.append((b, W.ruler_name(cfg, s), fam, w))
    return out, skipped


PASSES = {"fp32": "", "fp16": "_fp16", "bf16": "_bf16"}      # pass name -> array suffix


def arr_path(bench, ruler, variant, tag=""):
    """tag is the array suffix: "" (fp32, the ruler), "_fp16" or "_bf16" (twins)."""
    if tag is True or tag is False:                 # older callers passed a bool
        tag = "_fp16" if tag else ""
    return CACHE / ("%s__%s_%s%s.npy" % (bench, ruler, variant, tag))


# ------------------------------------------------------------------- data
def load(path):
    import pyarrow as pa
    from PIL import Image
    with pa.memory_map(str(path), "rb") as src:
        t = pa.ipc.open_file(src).read_all()
    ids = [str(x) for x in t.column("image_id").to_pylist()]
    imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in t.column("image").to_pylist()]
    return ids, imgs


def ids_sha(ids):
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


def order_contract(bench, ids):
    """image_ids.json defines the order. Anything else and nothing is written."""
    ref = json.loads((CACHE / "image_ids.json").read_text())["%s__C0" % bench]
    if list(ids) != list(ref):
        n = sum(1 for a, b in zip(ids, ref) if a != b) + abs(len(ids) - len(ref))
        raise RuntimeError("contrato de ordem FALHOU em %s: %d posicoes divergem "
                           "de image_ids.json" % (bench, n))
    return ids_sha(ref)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def determinism(torch):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:                    # torch 1.9 has no warn_only
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            print("  aviso: algoritmos deterministicos nao forcados: %s" % e)


# ------------------------------------------------------------ manifest io
def update_manifest(fn):
    """Read-modify-write under an exclusive lock: four processes share the
    manifest, one per GPU, and an unlocked write would drop entries."""
    import fcntl
    with open(CACHE / "manifest.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        man = json.loads((CACHE / "manifest.json").read_text())
        fn(man)
        tmp = CACHE / "manifest.json.tmp"
        tmp.write_text(json.dumps(man, indent=1))
        os.replace(tmp, CACHE / "manifest.json")


def record(bench, ruler, variant, fp16, a, sha_ids, extra):
    f = arr_path(bench, ruler, variant, fp16)
    assert not f.exists(), "nunca sobrescrever: %s" % f
    np.save(f, a)
    key = f.stem

    def add(man):
        assert key not in man["arrays"], "entrada ja existe no manifesto: %s" % key
        man["arrays"][key] = {"shape": list(a.shape), "sha256": sha_file(f),
                              "ids_sha256": sha_ids, **extra}
    update_manifest(add)


def control(embed, imgs, tag):
    """Same images, two orders: exactly zero per variant, or nothing is written."""
    s = imgs[:CONTROL_N]
    a, b = embed(s), embed(s[::-1])
    out = {}
    for v in VARIANTS:
        d = float(np.abs(a[v].astype(np.float64) - b[v][::-1].astype(np.float64)).max())
        out[v] = d
        print("     controle de posicao %-34s max|dif| %.3e  %s"
              % ("%s %s" % (tag, v), d, "OK" if d == 0.0 else "FALHA"), flush=True)
    return out


# ------------------------------------------------------------------- VLMs
def vlm_embedder(family, weights, dtype):
    """Returns (embed(imgs) -> {'tower': (n,d1), 'proj': (n,d2)}, meta)."""
    import torch
    from transformers import AutoProcessor
    determinism(torch)
    if family == "QWEN2VL":
        from transformers import Qwen2VLForConditionalGeneration as Cls
    else:
        from transformers import SmolVLMForConditionalGeneration as Cls
    mid = BASE_ID[family]
    model = Cls.from_pretrained(mid, dtype=dtype)
    if weights is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(Path(weights).parent))
        model = model.get_base_model()        # adapters stay injected, unmerged
    model = model.to(dtype).to(DEV).eval()
    proc = AutoProcessor.from_pretrained(mid)
    ip = proc.image_processor
    if family == "QWEN2VL":
        ip.max_pixels, ip.min_pixels = QWEN_MAX_PIXELS, QWEN_MIN_PIXELS
    else:
        ip.size = {"longest_edge": SMOLVLM_LONGEST_EDGE}
    inner = model.model                       # Qwen2VLModel / SmolVLMModel

    if family == "QWEN2VL":
        def one(img):
            px = ip(images=[img], return_tensors="pt")
            with torch.no_grad():
                o = inner.visual(px["pixel_values"].to(DEV, dtype),
                                 grid_thw=px["image_grid_thw"].to(DEV))
            return (o.last_hidden_state.float().mean(0).cpu().numpy(),
                    o.pooler_output.float().mean(0).cpu().numpy())
        meta = {"tower": "visual blocks, before PatchMerger",
                "proj": "PatchMerger output (the LLM's visual tokens)",
                "max_pixels": QWEN_MAX_PIXELS, "min_pixels": QWEN_MIN_PIXELS}
    else:
        cap = {}
        h = inner.connector.register_forward_hook(
            lambda m, i, o: cap.update(inp=i[0], out=o))

        def one(img):
            px = ip(images=[img], return_tensors="pt")
            with torch.no_grad():
                inner.get_image_features(px["pixel_values"].to(DEV, dtype),
                                         px["pixel_attention_mask"].to(DEV))
            t = cap["inp"].float().reshape(-1, cap["inp"].shape[-1]).mean(0)
            p = cap["out"].float().reshape(-1, cap["out"].shape[-1]).mean(0)
            return t.cpu().numpy(), p.cpu().numpy()
        meta = {"tower": "SigLIP last_hidden_state over all tiles, before connector",
                "proj": "connector output (pixel-shuffle + MLP), all tiles",
                "longest_edge": SMOLVLM_LONGEST_EDGE}

    def embed(imgs):
        T, Pj = zip(*[one(im) for im in imgs])
        return {"tower": np.stack(T).astype(np.float32),
                "proj": np.stack(Pj).astype(np.float32)}
    meta.update({"batch_size": 1, "batch_note": "dynamic resolution: one image per forward",
                 "pooling": "mean over visual tokens", "base": mid,
                 "adapter": str(Path(weights).parent) if weights else None})
    return embed, meta


# ------------------------------------------------------------------- M3AE
def m3ae_embedder(bench, weights, dtype):
    """Runs only inside `lapa`. Builds the model exactly as inference does
    (build_m3ae_config + M3AETransformerSS with test_only/load_path)."""
    import torch
    from src.extract_predictions_m3ae import _enter_m3ae_dir, build_m3ae_config
    determinism(torch)
    _enter_m3ae_dir(str(M3AE_DIR))
    from m3ae.modules import M3AETransformerSS
    from m3ae.transforms import keys_to_transforms
    ckpt = str(weights) if weights else str(M3AE_PRETRAINED)
    key = M3AE_KEY[bench]
    label_size = 500 if (weights and bench == "KVASIR") else None   # amendment: clean split head
    config = build_m3ae_config(ckpt, dataset_key=key, vqa_label_size=label_size)
    config = dict(config)
    config["image_size"] = M3AE_IMAGE_SIZE
    model = M3AETransformerSS(config).eval().to(dtype).to(DEV)
    if dtype == torch.float16:
        # M3AE's CLIP LayerNorm casts its input to fp32 on purpose (clip_model.py:32),
        # so its weights must stay fp32 too -- exactly what CLIP's own
        # convert_weights() does when it halves a model.
        for m in model.modules():
            if isinstance(m, torch.nn.LayerNorm):
                m.float()
    tf = keys_to_transforms(["clip"], size=M3AE_IMAGE_SIZE)[0]

    def embed(imgs):
        T, Pj = [], []
        for i in range(0, len(imgs), M3AE_BS):
            x = torch.stack([tf(im) for im in imgs[i:i + M3AE_BS]]).to(DEV, dtype)
            with torch.no_grad():
                f = model.vision_encoder(x)                 # (B, 1+577?, 768) with CLS at 0
                p = model.multi_modal_vision_proj(f)
            T.append(f[:, 1:].float().mean(1).cpu().numpy())
            Pj.append(p[:, 1:].float().mean(1).cpu().numpy())
        return {"tower": np.concatenate(T).astype(np.float32),
                "proj": np.concatenate(Pj).astype(np.float32)}
    meta = {"tower": "CLIP ViT-B/16 tokens after ln_post, CLS dropped",
            "proj": "multi_modal_vision_proj output, CLS dropped",
            "batch_size": M3AE_BS, "image_size": M3AE_IMAGE_SIZE,
            "transform": "clip_transform (Resize+CenterCrop, CLIP mean/std)",
            "pooling": "mean over visual tokens", "checkpoint": ckpt,
            "vqa_label_size": label_size, "env": "lapa"}
    return embed, meta


# -------------------------------------------------------------------- run
def run_line(bench, ruler, family, weights, ids, imgs, sha_ids, passes=("fp32", "fp16")):
    import torch
    dtype_of = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    todo = [(v, ps) for ps in passes for v in VARIANTS
            if not arr_path(bench, ruler, v, PASSES[ps]).exists()]
    if not todo:
        print("  %-7s %-22s ja completo (%s)" % (bench, ruler, ",".join(passes)), flush=True)
        return
    for ps in passes:
        if not any(p == ps for _, p in todo):
            continue
        tag, dtype = PASSES[ps], dtype_of[ps]
        t0 = time.time()
        try:
            embed, meta = (m3ae_embedder(bench, weights, dtype) if family == "M3AE"
                           else vlm_embedder(family, weights, dtype))
        except Exception as e:
            # a twin is evidence for analysis D only: its failure is logged and
            # must never take the fp32 ruler down with it
            print("     FALHA ao construir %s %s (%s): %s" % (ruler, ps, type(e).__name__,
                                                              str(e)[:160]), flush=True)
            if ps == "fp32":
                raise
            continue
        ctrl = control(embed, imgs, "%s %s" % (ruler, ps))
        key = "%s|%s" % (ruler, bench) + ("" if ps == "fp32" else "|" + ps)
        update_manifest(lambda man: man.setdefault("position_control", {}).__setitem__(
            key, {v: {"max_abs_diff": ctrl[v], "n": CONTROL_N, "passed": ctrl[v] == 0.0}
                  for v in VARIANTS}))
        if any(d != 0.0 for d in ctrl.values()):
            print("     reprovado no controle: nada gravado para %s %s %s" % (bench, ruler, ps))
            return
        try:
            a = embed(imgs)
        except Exception as e:
            print("     FALHA ao embutir %s %s (%s): %s" % (ruler, ps, type(e).__name__,
                                                            str(e)[:160]), flush=True)
            if ps == "fp32":
                raise
            continue
        role = {"fp32": "model-derived ruler",
                "fp16": "evidence only (analysis D)",
                "bf16": "model-derived ruler at the evaluation dtype (bf16)"}[ps]
        dt_name = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}[ps]
        for v in VARIANTS:
            if arr_path(bench, ruler, v, tag).exists():
                continue
            record(bench, ruler, v, tag, a[v], sha_ids, {"dtype": dt_name, "role": role})

        def rulers(man):
            man.setdefault("rulers", {})
            for v in VARIANTS:
                man["rulers"].setdefault("%s_%s%s" % (ruler, v, tag), {
                    "family": family, "variant": v, "what": meta[v],
                    "role": "%s (Gate 0: %s)" % (role,
                        "tower frozen, base encoder" if family == "QWEN2VL" or weights is None
                        else "tower trained, per line"),
                    **{k: meta[k] for k in meta if k not in VARIANTS},
                    "dtype": dt_name, "dim": int(a[v].shape[1])})
        update_manifest(rulers)
        print("  %-7s %-22s %s tower %s proj %s  %.0fs"
              % (bench, ruler, ps, a["tower"].shape, a["proj"].shape, time.time() - t0), flush=True)
        del embed
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="list lines, run the order contract, no GPU")
    ap.add_argument("--m3ae-line", nargs=3, metavar=("BENCH", "RULER", "WEIGHTS"),
                    help="internal: one M3AE line, run inside lapa ('-' = pretrained)")
    ap.add_argument("--only", choices=["vlm", "m3ae"], default=None)
    ap.add_argument("--bench", nargs="*", default=None, help="restrict to these benchmarks")
    ap.add_argument("--dtypes", nargs="*", default=["fp32", "fp16"], choices=list(PASSES),
                    help="passes to run; bf16 writes *_bf16 twins at the evaluation dtype")
    ap.add_argument("--rulers", nargs="*", default=None, help="restrict to these ruler names")
    ap.add_argument("--smoke", choices=["vlm", "m3ae"], default=None,
                    help="two images through each embedder of the family; writes nothing")
    a = ap.parse_args()

    if a.smoke:
        import torch
        b = "KVASIR"
        ids, imgs = load(ARROWS[b])
        imgs = imgs[:2]
        if a.smoke == "vlm":
            cases = [("qwen2vl_base", "QWEN2VL", None),
                     ("smolvlm_lora_s0", "SMOLVLM", W.resolve(b, "SMOLVLM_LORA", "seed0"))]
        else:
            cases = [("m3ae_base", "M3AE", None), ("m3ae_s0", "M3AE", W.resolve(b, "M3AE", "seed0"))]
        for r, fam, w in cases:
            t = time.time()
            embed, meta = (m3ae_embedder(b, w, torch.float32) if fam == "M3AE"
                           else vlm_embedder(fam, w, torch.float32))
            o = embed(imgs)
            print("  smoke %-16s tower %s proj %s  %.0fs  %s" % (
                r, o["tower"].shape, o["proj"].shape, time.time() - t,
                {k: meta[k] for k in ("batch_size", "pooling")}), flush=True)
        return

    if a.m3ae_line:
        bench, ruler, w = a.m3ae_line
        ids, imgs = load(ARROWS[bench])
        sha_ids = order_contract(bench, ids)
        run_line(bench, ruler, "M3AE", None if w == "-" else Path(w), ids, imgs, sha_ids)
        return

    L, skipped = lines()
    print("linhas: %d  (arrays: %d fp32 + %d fp16)" % (len(L), 2 * len(L), 2 * len(L)))
    for b, r, fam, w in L:
        print("  %-7s %-22s %-8s %s" % (b, r, fam, str(w).replace(str(ROOT) + "/", "") if w else "(base)"))
    if skipped:
        print("\nnao embutidas (resolvedor recusou):")
        for b, cfg, s, why in skipped:
            print("  %-7s %-13s %-6s %s" % (b, cfg, s, why))
    print("\ncontrato de ordem (ids do arrow == image_ids.json[BENCH__C0]):")
    shas = {}
    for b in P.BENCHMARKS:
        ids, _ = load(ARROWS[b])
        shas[b] = order_contract(b, ids)
        print("  %-7s %5d ids  OK  sha %s" % (b, len(ids), shas[b][:16]))
    if a.plan:
        return

    t0 = time.time()
    for b in P.BENCHMARKS:
        if a.bench and b not in a.bench:
            continue
        if a.rulers:
            L = [x for x in L if x[1] in a.rulers or x[0] != b]
        ids, imgs = load(ARROWS[b])
        sha_ids = order_contract(b, ids)
        for bb, r, fam, w in L:
            if bb != b:
                continue
            if fam == "M3AE":
                if a.only == "vlm":
                    continue
                cmd = [LAPA, str(REPO / "src/featurespace/embed_model_rulers.py"),
                       "--m3ae-line", b, r, str(w) if w else "-"]
                rc = subprocess.run(cmd, cwd=str(REPO)).returncode
                if rc != 0:
                    print("  %-7s %-22s subprocesso lapa saiu com %d -- linha pulada, ver acima"
                          % (b, r, rc), flush=True)
            else:
                if a.only == "m3ae":
                    continue
                run_line(b, r, fam, w, ids, imgs, sha_ids, passes=tuple(a.dtypes))
        del imgs
    print("\n  concluido em %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
