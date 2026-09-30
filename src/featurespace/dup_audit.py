"""Duplicate audit, second pass: keep the hashes and count distinct images.

The first pass reported pairs. A pair count does not say how much of the test
split is affected: 1,293 near-duplicate pairs could be three test images each
matching four hundred training frames, or a thousand test images each matching
one. The distinction decides whether "leakage-free" describes the split or only
its identifiers.

Hashes are written to disk so any later question costs a read, not a re-decode.
"""
from __future__ import annotations

import hashlib
import io
import json
from collections import defaultdict
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image
from scipy.fftpack import dct

REPO = Path(__file__).resolve().parents[2]
M3 = Path(os.environ.get("M3AE_ARROWS", str(REPO / "data/finetune_arrows")))
P0A = REPO / "data/finetune_arrows"
HASHES = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "image_hashes.json"
NEAR = 4

SPLITS = {
    "KVASIR": {"train": P0A / "vqa_kvasir_train_clean.arrow",
               "val": P0A / "vqa_kvasir_val_clean.arrow",
               "test": P0A / "vqa_kvasir_test_clean.arrow"},
    "SLAKE": {"train": M3 / "vqa_slack_train.arrow", "val": M3 / "vqa_slack_val.arrow",
              "test": M3 / "vqa_slack_test.arrow"},
    "VQARAD": {"train": M3 / "vqa_vqa_rad_train.arrow", "val": M3 / "vqa_vqa_rad_val.arrow",
               "test": M3 / "vqa_vqa_rad_test.arrow"},
}
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def phash64(im):
    g = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    d = dct(dct(g, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    v = d.flatten()[1:]
    return np.packbits((v > np.median(v)).astype(np.uint8))


def scan(path):
    with pa.memory_map(str(path), "rb") as s:
        t = pa.ipc.open_file(s).read_all()
    ids = [str(x) for x in t.column("image_id").to_pylist()]
    sha, ph = [], []
    for b in t.column("image").to_pylist():
        im = Image.open(io.BytesIO(b)).convert("RGB")
        sha.append(hashlib.sha256(im.tobytes()).hexdigest())
        ph.append(phash64(im).tolist())
    return ids, sha, ph


def main():
    if HASHES.exists():
        H = json.loads(HASHES.read_text())
        print("  hashes lidos do cache")
    else:
        H = {}
        for bench, splits in SPLITS.items():
            H[bench] = {}
            for split, p in splits.items():
                if not p.exists():
                    continue
                ids, sha, ph = scan(p)
                H[bench][split] = {"ids": ids, "sha": sha, "phash": ph}
                print("  %-7s %-5s %5d" % (bench, split, len(ids)), flush=True)
        HASHES.write_text(json.dumps(H))
        print("  hashes gravados em %s" % HASHES)

    report = {"near_threshold": NEAR, "benchmarks": {}}
    for bench, d in H.items():
        if "test" not in d:
            continue
        te = d["test"]
        n_test = len(te["ids"])
        bank = [(sp, i, s, np.array(p, dtype=np.uint8))
                for sp in ("train", "val") if sp in d
                for i, s, p in zip(d[sp]["ids"], d[sp]["sha"], d[sp]["phash"])]

        # exact inside test
        g = defaultdict(list)
        for i, s in zip(te["ids"], te["sha"]):
            g[s].append(i)
        inside = [v for v in g.values() if len(v) > 1]

        # exact test x bank
        bank_sha = defaultdict(list)
        for sp, i, s, _ in bank:
            bank_sha[s].append("%s:%s" % (sp, i))
        exact = [{"test": i, "also_in": bank_sha[s]}
                 for i, s in zip(te["ids"], te["sha"]) if s in bank_sha]

        # near test x bank
        B = np.array([b[3] for b in bank])
        near_pairs, near_imgs = 0, []
        for i, p in zip(te["ids"], te["phash"]):
            dist = POP[np.bitwise_xor(B, np.array(p, dtype=np.uint8))].sum(1)
            hit = int((dist <= NEAR).sum())
            if hit:
                near_pairs += hit
                j = int(dist.argmin())
                near_imgs.append({"test": i, "matches": hit,
                                  "closest": "%s:%s" % (bank[j][0], bank[j][1]),
                                  "distance": int(dist[j])})

        report["benchmarks"][bench] = {
            "counts": {k: len(d[k]["ids"]) for k in d},
            "exact_inside_test": {"groups": len(inside), "images": sum(len(x) for x in inside),
                                  "examples": inside[:10]},
            "exact_test_vs_bank": {
                "test_images_affected": len(exact),
                "fraction_of_test": round(len(exact) / n_test, 4),
                "examples": exact[:10]},
            "near_test_vs_bank": {
                "pairs": near_pairs,
                "test_images_affected": len(near_imgs),
                "fraction_of_test": round(len(near_imgs) / n_test, 4),
                "examples": sorted(near_imgs, key=lambda x: -x["matches"])[:10]},
        }
        r = report["benchmarks"][bench]
        print("\n  === %s (teste = %d imagens)" % (bench, n_test))
        print("    identicas dentro do teste     : %d grupos, %d imagens"
              % (r["exact_inside_test"]["groups"], r["exact_inside_test"]["images"]))
        print("    identicas teste x treino+val  : %d imagens de teste (%.1f%%)"
              % (r["exact_test_vs_bank"]["test_images_affected"],
                 100 * r["exact_test_vs_bank"]["fraction_of_test"]))
        print("    pHash <= %d entre splits       : %d imagens de teste (%.1f%%), %d pares"
              % (NEAR, r["near_test_vs_bank"]["test_images_affected"],
                 100 * r["near_test_vs_bank"]["fraction_of_test"],
                 r["near_test_vs_bank"]["pairs"]))

    p = REPO / "docs/v4/duplicate_audit.json"
    p.write_text(json.dumps(report, indent=1))
    print("\n  gravado %s" % p)


if __name__ == "__main__":
    main()
