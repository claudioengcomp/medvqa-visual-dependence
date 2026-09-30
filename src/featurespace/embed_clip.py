"""CLIP ViT-L/14 for E2's hard-negative search.

E2 reproduces CORAL's hard negative, and a hard negative is defined by whichever
encoder did the searching. Using only a medical encoder would answer a different
question from theirs, so the search runs under two: BiomedCLIP, already in the
manifest, and CLIP ViT-L/14, the general-purpose one, added here.

Same rules as every other array in this cache (amendment 7.7): fp32, TF32 off,
deterministic algorithms, batch size fixed at 32 and recorded, and the position
control at exactly zero before anything is written.

C0 of Kvasir, SLAKE and PathVQA is embedded: E2 searches the intact test pool.
VQA-RAD stays out for the reason the pre-registration gives — one informative
question — while PathVQA's stated reason ("no question-level gold structure in
this repository") stopped holding when E1 built its arrow, and lifting that
exclusion is declared in the pre-registration rather than done quietly.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
M3 = Path(os.environ.get("M3AE_ARROWS", str(REPO / "data/finetune_arrows")))
P0A = REPO / "data/finetune_arrows"
CACHE = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "embeddings"
MODEL = "openai/clip-vit-large-patch14"
NAME = "clipL14"
BS, CONTROL_N = 32, 64
SETS = {"KVASIR": P0A / "vqa_kvasir_test_clean.arrow",
        "SLAKE": M3 / "vqa_slack_test.arrow",
        # PathVQA joins E2 once its arrow exists; see the amendment in
        # preregistration_e.md on lifting the declared exclusion
        "PATHVQA": P0A / "vqa_path_vqa_test.arrow"}


def determinism():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print("  aviso: %s" % e)


_model = {}


def embed(images, bs=BS):
    from transformers import AutoImageProcessor, CLIPVisionModelWithProjection
    if not _model:
        _model["p"] = AutoImageProcessor.from_pretrained(MODEL)
        _model["m"] = CLIPVisionModelWithProjection.from_pretrained(
            MODEL, dtype=torch.float32).to("cuda").eval()
    out = []
    for i in range(0, len(images), bs):
        px = _model["p"](images=images[i:i + bs], return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            f = _model["m"](pixel_values=px.to("cuda", torch.float32)).image_embeds
        out.append(f.float().cpu().numpy())
    return np.concatenate(out)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    determinism()
    mpath = CACHE / "manifest.json"
    man = json.loads(mpath.read_text())
    man.setdefault("position_control", {})
    man.setdefault("rulers", {})[NAME] = {
        "repo": MODEL, "role": "E2 hard-negative search, general purpose",
        "precision": "fp32", "batch_size": BS}
    t0 = time.time()
    controlled = False
    for bench, arrow in SETS.items():
        f = CACHE / ("%s__%s.npy" % (bench, NAME))
        if f.exists():
            print("  %s ja tem %s" % (bench, NAME))
            continue
        with pa.memory_map(str(arrow), "rb") as s:
            t = pa.ipc.open_file(s).read_all()
        imgs = [Image.open(io.BytesIO(b)).convert("RGB")
                for b in t.column("image").to_pylist()]
        print("  carregado %s: %d imagens" % (bench, len(imgs)), flush=True)
        if not controlled:
            s_ = imgs[:CONTROL_N]
            a, b_ = embed(s_), embed(s_[::-1])[::-1]
            d = float(np.abs(a.astype(np.float64) - b_.astype(np.float64)).max())
            man["position_control"]["%s" % NAME] = {"max_abs_diff": d,
                                                    "n": len(s_),
                                                    "passed": d == 0.0}
            mpath.write_text(json.dumps(man, indent=1))
            print("     controle de posicao %-9s max|dif| %.3e  %s"
                  % (NAME, d, "OK" if d == 0.0 else "FALHA"), flush=True)
            if d != 0.0:
                print("     reprovado: nada sera gravado")
                return
            controlled = True
        tt = time.time()
        a = embed(imgs)
        np.save(f, a)
        man["arrays"][f.stem] = {"shape": list(a.shape), "sha256": sha_file(f)}
        mpath.write_text(json.dumps(man, indent=1))
        print("  %-8s %-9s %s  %.0fs" % (bench, NAME, a.shape, time.time() - tt),
              flush=True)
        del imgs
    man.setdefault("clipL14", {})["host"] = os.environ.get("RUN_HOST", "unspecified")
    mpath.write_text(json.dumps(man, indent=1))
    print("\n  concluido em %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
