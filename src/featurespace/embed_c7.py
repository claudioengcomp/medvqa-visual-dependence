"""Embed C7 in the five rulers, under amendment 7.7.

Same machinery as the PathVQA and HEAL pass: TF32 off, deterministic algorithms,
cuDNN benchmark off, batch size fixed at 32 and recorded, fp32 throughout, and
the position control - the same image embedded at two different places in the
stream must give bit-identical vectors - required to pass at exactly zero before
any array is written.

`image_ids.json` is merged, never overwritten: it carries the row order of every
condition and a run covering a subset would otherwise erase the rest.
"""
from __future__ import annotations

import hashlib
import inspect
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
BS, CONTROL_N = 32, 64

SETS = {"KVASIR": P0A / "vqa_kvasir_test_clean_c7.arrow",
        "SLAKE": M3 / "vqa_slack_test_c7.arrow",
        "VQARAD": M3 / "vqa_vqa_rad_test_c7.arrow"}


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


def call(fn, images):
    return (fn(images, bs=BS) if "bs" in inspect.signature(fn).parameters
            else fn(images))


def position_control(fn, imgs, name):
    s = imgs[:CONTROL_N]
    a, b = call(fn, s), call(fn, s[::-1])[::-1]
    d = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
    ok = d == 0.0
    print("     controle de posicao %-11s max|dif| %.3e  %s"
          % (name, d, "OK" if ok else "FALHA"), flush=True)
    return {"max_abs_diff": d, "n": len(s), "passed": ok}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    determinism()
    from src.featurespace.g0_embed import RULERS, CACHE      # noqa: E402

    ids_path = CACHE / "image_ids.json"
    ids_all = json.loads(ids_path.read_text())
    mpath = CACHE / "manifest.json"
    man = json.loads(mpath.read_text())
    man.setdefault("position_control", {})

    t0 = time.time()
    controlled = set()
    for bench, arrow in SETS.items():
        if not arrow.exists():
            print("  AUSENTE %s: %s" % (bench, arrow.name))
            continue
        need = [r for r in RULERS
                if not (CACHE / ("%s__C7__%s.npy" % (bench, r))).exists()]
        if not need:
            print("  %s C7 ja completo" % bench)
            continue
        with pa.memory_map(str(arrow), "rb") as s:
            t = pa.ipc.open_file(s).read_all()
        ids = [str(x) for x in t.column("image_id").to_pylist()]
        imgs = [Image.open(io.BytesIO(b)).convert("RGB")
                for b in t.column("image").to_pylist()]
        ids_all["%s__C7" % bench] = ids
        print("  carregado %s C7: %d imagens" % (bench, len(imgs)), flush=True)
        for r in need:
            if r not in controlled:
                c = position_control(RULERS[r], imgs, r)
                man["position_control"]["%s|C7" % r] = c
                mpath.write_text(json.dumps(man, indent=1))
                if not c["passed"]:
                    print("     %s reprovado: nada gravado" % r)
                    continue
                controlled.add(r)
            if r not in controlled:
                continue
            tt = time.time()
            a = call(RULERS[r], imgs)
            f = CACHE / ("%s__C7__%s.npy" % (bench, r))
            np.save(f, a)
            man["arrays"][f.stem] = {"shape": list(a.shape), "sha256": sha_file(f)}
            mpath.write_text(json.dumps(man, indent=1))
            print("  %-8s C7  %-11s %s  %.0fs"
                  % (bench, r, a.shape, time.time() - tt), flush=True)
        del imgs
        ids_path.write_text(json.dumps(ids_all))

    man.setdefault("c7", {})["host"] = os.environ.get("RUN_HOST", "unspecified")
    man["c7"]["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    mpath.write_text(json.dumps(man, indent=1))
    print("\n  concluido em %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
