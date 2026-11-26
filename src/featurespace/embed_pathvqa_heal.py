"""Embed PathVQA and HEAL, under amendment 7.7.

Three things this does that `embed_conditions.py` does not.

**It merges `image_ids.json` instead of overwriting it.** That file is written
there with `write_text(json.dumps(ids_out))`, where `ids_out` holds only what
the run computed - so any run covering a subset erases the rest, including the
C0 row orders that canonical ordering depends on.

**It fixes the sources of position dependence before writing anything.** TF32
off, deterministic algorithms on, cuDNN benchmark off, batch size fixed and
recorded. Analysis G1's layer 1b found the same image embedding differently when
it moved rows - 22 of 1,291 on Kvasir C2, always the same images across three
rulers - and amendment 7.7 requires that to be zero for what is computed from
now on.

**It proves that, per ruler, before any array is written.** The control embeds a
sample twice, the second time with the order reversed, and undoes the reversal:
a position-independent encoder returns bit-identical vectors. A ruler that fails
is reported and nothing of it is written.

Rulers, cache and precision are the ones already declared; fp32 admits no
exception (amendment 7.5).
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

BS = 32                      # fixed and recorded, amendment 7.7
CONTROL_N = 64               # two batches, so the control spans a batch boundary

ROOT = REPO.parent
M3 = ROOT / "M3AE/data/finetune_arrows"
P0A = REPO / "data/finetune_arrows"

SETS = {
    "PATHVQA": {"C0": P0A / "vqa_path_vqa_test.arrow"},
    "HEAL": {"TRAIN": M3 / "vqa_heal_train.arrow",
             "C0": M3 / "vqa_heal_test.arrow",
             "C1": M3 / "vqa_heal_test_c1.arrow",
             "C2": M3 / "vqa_heal_test_c2.arrow",
             "C4_OCC_25": M3 / "vqa_heal_test_c4_occ_25.arrow",
             "C4_OCC_50": M3 / "vqa_heal_test_c4_occ_50.arrow",
             "C4_OCC_75": M3 / "vqa_heal_test_c4_occ_75.arrow",
             "C4_NOISE_25": M3 / "vqa_heal_test_c4_noise_25.arrow",
             "C4_NOISE_50": M3 / "vqa_heal_test_c4_noise_50.arrow",
             "C4_NOISE_75": M3 / "vqa_heal_test_c4_noise_75.arrow"},
}


def determinism():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:                       # older torch
        print("  aviso: use_deterministic_algorithms indisponivel (%s)" % e)


def load(path):
    with pa.memory_map(str(path), "rb") as src:
        t = pa.ipc.open_file(src).read_all()
    ids = [str(x) for x in t.column("image_id").to_pylist()]
    imgs = [Image.open(io.BytesIO(b)).convert("RGB")
            for b in t.column("image").to_pylist()]
    return ids, imgs


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def call(fn, images):
    """The colour histogram takes no batch size; every other ruler does."""
    import inspect
    return fn(images, bs=BS) if "bs" in inspect.signature(fn).parameters else fn(images)


def position_control(fn, imgs, name):
    """Same images, two different positions: the vectors must be identical."""
    s = imgs[:CONTROL_N]
    a = call(fn, s)
    b = call(fn, s[::-1])[::-1]
    d = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
    rows = int((np.abs(a - b) > 0).any(1).sum())
    ok = d == 0.0
    print("     controle de posicao %-11s max|dif| %.3e em %d de %d linhas  %s"
          % (name, d, rows, len(s), "OK" if ok else "FALHA"), flush=True)
    return {"max_abs_diff": d, "rows_differing": rows, "n": len(s), "passed": ok}


def main(which):
    determinism()
    from src.featurespace.g0_embed import RULERS, CACHE   # noqa: E402

    ids_path = CACHE / "image_ids.json"
    ids_all = json.loads(ids_path.read_text()) if ids_path.exists() else {}
    mpath = CACHE / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {"arrays": {}}
    man.setdefault("position_control", {})
    man.setdefault("determinism", {
        "tf32": False, "cudnn_benchmark": False, "cudnn_deterministic": True,
        "batch_size": BS, "amendment": "7.7",
        "host": os.environ.get("RUN_HOST", "unspecified"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"})

    t0 = time.time()
    controlled = set()
    # "HEAL=C0,C2" embeds only those conditions, so what analysis B needs can go
    # first and the rest of the ladder follow without holding it up
    for spec in which:
        bench, _, only = spec.partition("=")
        wanted = set(only.split(",")) if only else None
        for cond, arrow in SETS[bench].items():
            if wanted is not None and cond not in wanted:
                continue
            if not arrow.exists():
                print("  AUSENTE %s %s: %s" % (bench, cond, arrow.name))
                continue
            need = [r for r in RULERS
                    if not (CACHE / ("%s__%s__%s.npy" % (bench, cond, r))).exists()]
            if not need:
                print("  %s %s ja completo" % (bench, cond))
                ids_all.setdefault("%s__%s" % (bench, cond), None)
                continue
            print("  carregando %s %s (%s)" % (bench, cond, arrow.name), flush=True)
            ids, imgs = load(arrow)
            ids_all["%s__%s" % (bench, cond)] = ids
            for r in need:
                if r not in controlled:
                    c = position_control(RULERS[r], imgs, r)
                    man["position_control"][r] = c
                    mpath.write_text(json.dumps(man, indent=1))
                    if not c["passed"]:
                        print("     %s reprovado no controle: nada sera gravado dele" % r)
                        continue
                    controlled.add(r)
                if r not in controlled:
                    continue
                t = time.time()
                a = call(RULERS[r], imgs)
                f = CACHE / ("%s__%s__%s.npy" % (bench, cond, r))
                np.save(f, a)
                man["arrays"][f.stem] = {"shape": list(a.shape), "sha256": sha_file(f)}
                mpath.write_text(json.dumps(man, indent=1))
                print("  %-8s %-12s %-11s %s  %.0fs"
                      % (bench, cond, r, a.shape, time.time() - t), flush=True)
            del imgs
            ids_path.write_text(json.dumps({k: v for k, v in ids_all.items()
                                            if v is not None}))

    bad = [r for r, c in man["position_control"].items() if not c["passed"]]
    if bad:
        print("\n  REPROVADAS no controle de posicao: %s" % ", ".join(bad))
    print("\n  concluido em %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main(sys.argv[1:] or ["PATHVQA"])
