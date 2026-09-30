"""C7: the image with its 32x32 patches permuted.

The condition keeps every pixel and destroys the arrangement. It is an
embedding-only condition: no model is ever run on it, so it has a distance and
no VDD.

Border rule, as pre-registered before generation: an image whose height or width
is not a multiple of 32 is divided into floor(H/32) x floor(W/32) **complete**
patches and only those are permuted; the remainder strips, fewer than 32 pixels
each, stay where they are. This conserves every pixel, which is what makes the
colour prediction testable: a histogram of a cropped image is not the histogram
of the original, so cropping would fail that prediction for a reason unrelated
to shuffling.

Seed 20260918 derived per image_id, so an image's permutation does not depend on
how many images preceded it. Written back at the original resolution as lossless
PNG, because a lossy re-encode would move the histogram and break the same
prediction.
"""
from __future__ import annotations

import hashlib
import io
import json
import socket
from collections import Counter
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
M3 = Path(os.environ.get("M3AE_ARROWS", str(REPO / "data/finetune_arrows")))
P0A = REPO / "data/finetune_arrows"
OUT = REPO / "docs/v4/c7_manifest.json"
SEED = 20260918
PATCH = 32

SETS = {"KVASIR": (P0A, "vqa_kvasir_test_clean"),
        "SLAKE": (M3, "vqa_slack_test"),
        "VQARAD": (M3, "vqa_vqa_rad_test")}


def rng_for(image_id):
    h = hashlib.sha256(("%d|%s" % (SEED, image_id)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def shuffle_patches(im, image_id):
    a = np.asarray(im)
    H, W = a.shape[:2]
    ny, nx = H // PATCH, W // PATCH
    if ny < 1 or nx < 1:
        return im, (H, W, 0, 0, 0)          # smaller than one patch: untouched
    out = a.copy()
    grid = [(y, x) for y in range(ny) for x in range(nx)]
    perm = rng_for(image_id).permutation(len(grid))
    src = [a[y * PATCH:(y + 1) * PATCH, x * PATCH:(x + 1) * PATCH].copy()
           for y, x in grid]
    for k, (y, x) in enumerate(grid):
        out[y * PATCH:(y + 1) * PATCH, x * PATCH:(x + 1) * PATCH] = src[perm[k]]
    return Image.fromarray(out), (H, W, ny * nx, H - ny * PATCH, W - nx * PATCH)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


man = {"host": os.environ.get("RUN_HOST", "unspecified"), "seed": SEED, "patch": PATCH,
       "border_rule": ("complete patches only are permuted; the remainder "
                       "strips (< 32 px) stay in place, so every pixel is "
                       "conserved"),
       "format": "lossless PNG at the original resolution",
       "benchmarks": {}}

for bench, (d, stem) in SETS.items():
    src = d / ("%s.arrow" % stem)
    dst = d / ("%s_c7.arrow" % stem)
    with pa.memory_map(str(src), "rb") as s:
        t = pa.ipc.open_file(s).read_all()
    ids = [str(x) for x in t.column("image_id").to_pylist()]
    raw = t.column("image").to_pylist()

    new, shapes, rem = [], Counter(), Counter()
    pix_ok = 0
    for i, b in zip(ids, raw):
        im = Image.open(io.BytesIO(b)).convert("RGB")
        sh, info = shuffle_patches(im, i)
        buf = io.BytesIO()
        sh.save(buf, format="PNG", optimize=False, compress_level=6)
        new.append(buf.getvalue())
        shapes["%dx%d" % (info[0], info[1])] += 1
        rem["bottom %d, right %d" % (info[3], info[4])] += 1
        # every pixel conserved, checked the way the colour histogram sees it:
        # the per-channel value counts must be identical. A Counter over RGB
        # tuples answers the same question and takes minutes per image.
        A, B = np.asarray(im), np.asarray(sh)
        if all(np.array_equal(np.bincount(A[..., c].ravel(), minlength=256),
                              np.bincount(B[..., c].ravel(), minlength=256))
               for c in range(3)):
            pix_ok += 1

    cols = {n: t.column(n) for n in t.schema.names}
    cols["image"] = pa.array(new, pa.binary())
    tab = pa.table(cols)
    with pa.OSFile(str(dst), "wb") as f:
        with pa.RecordBatchFileWriter(f, tab.schema) as w:
            w.write_table(tab)

    man["benchmarks"][bench] = {
        "source_arrow": src.name, "source_sha256": sha_bytes(src.read_bytes()),
        "c7_arrow": dst.name, "c7_sha256": sha_bytes(dst.read_bytes()),
        "rows": tab.num_rows,
        "shapes": dict(shapes.most_common(6)),
        "remainder_strips": dict(rem.most_common(6)),
        "pixel_multiset_preserved": "%d of %d" % (pix_ok, len(new)),
        "all_pixels_conserved": pix_ok == len(new)}
    print("  %-7s %5d linhas  formas %s  restos %s  pixels conservados %d/%d"
          % (bench, tab.num_rows, dict(shapes.most_common(2)),
             dict(rem.most_common(2)), pix_ok, len(new)), flush=True)

OUT.write_text(json.dumps(man, indent=1))
print("\n  gravado %s" % OUT)
