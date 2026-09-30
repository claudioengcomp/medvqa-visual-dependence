"""Embed the training bank and every condition, for analysis A.

The G0 cache holds C0 only. Analysis A needs knn(x) to the training bank for
every condition, so the bank and the ten conditions have to be embedded too.
Same rulers, same cache, same manifest, all fp32.

Naming: {BENCH}__{COND}__{ruler}.npy, with COND = TRAIN for the reference bank.
Resumable: an array already on disk is not recomputed.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.featurespace.g0_embed import RULERS, CACHE  # noqa: E402

ROOT = REPO.parent
M3 = ROOT / "M3AE/data/finetune_arrows"
P0A = REPO / "data/finetune_arrows"

# paper C3 = occlusion = c4_occ ; paper C4 = noise = c4_noise
CONDS = ["c1", "c2", "c4_occ_25", "c4_occ_50", "c4_occ_75",
         "c4_noise_25", "c4_noise_50", "c4_noise_75", "c5", "c6"]

SETS = {
    "KVASIR": {"TRAIN": P0A / "vqa_kvasir_train_clean.arrow",
               **{c.upper(): P0A / ("vqa_kvasir_test_clean_%s.arrow" % c) for c in CONDS}},
    "SLAKE": {"TRAIN": M3 / "vqa_slack_train.arrow",
              **{c.upper(): M3 / ("vqa_slack_test_%s.arrow" % c) for c in CONDS}},
    "VQARAD": {"TRAIN": M3 / "vqa_vqa_rad_train.arrow",
               **{c.upper(): M3 / ("vqa_vqa_rad_test_%s.arrow" % c) for c in CONDS}},
}


def load(path):
    with pa.memory_map(str(path), "rb") as src:
        t = pa.ipc.open_file(src).read_all()
    ids = [str(x) for x in t.column("image_id").to_pylist()]
    imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in t.column("image").to_pylist()]
    return ids, imgs


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    t0 = time.time()
    mpath = CACHE / "manifest.json"
    man = json.loads(mpath.read_text())
    ids_out = {}
    missing = []

    for bench, conds in SETS.items():
        for cond, arrow in conds.items():
            if not arrow.exists():
                missing.append("%s %s: %s" % (bench, cond, arrow.name))
                continue
            need = [r for r in RULERS
                    if not (CACHE / ("%s__%s__%s.npy" % (bench, cond, r))).exists()]
            if not need:
                continue
            ids, imgs = load(arrow)
            ids_out["%s__%s" % (bench, cond)] = ids
            for r in need:
                t = time.time()
                a = RULERS[r](imgs)
                f = CACHE / ("%s__%s__%s.npy" % (bench, cond, r))
                np.save(f, a)
                man["arrays"][f.stem] = {"shape": list(a.shape), "sha256": sha(f)}
                mpath.write_text(json.dumps(man, indent=1))
                print("  %-7s %-14s %-11s %s  %.0fs"
                      % (bench, cond, r, a.shape, time.time() - t), flush=True)
            del imgs

    (CACHE / "image_ids.json").write_text(json.dumps(ids_out))
    if missing:
        print("\n  AUSENTES (esperado: VQARAD C5 nao existe):")
        for m in missing:
            print("    " + m)
    print("\n  concluido em %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
