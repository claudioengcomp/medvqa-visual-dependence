"""What is left after the patch, in pixels.

The collapse claim is mechanical: the images become identical because everything
outside the patch is the same. That is checkable without a model. For every
SLAKE test image, measure the standard deviation of the ring left uncovered at
each severity, and the fraction of that ring that is pure black.

If the collapsed images have a ring that is black end to end, the reading
"radiology carries a black border" is measured rather than asserted. If their
rings differ in content, the collapse has another cause and the sentence is
wrong.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]

import numpy as np
import pyarrow as pa
from PIL import Image

M3 = Path(os.environ.get("M3AE_ARROWS", str(REPO / "data/finetune_arrows")))
COLLAPSED = set(json.loads((REPO / "docs/v4/collapse_vs_phash.json").read_text()
                           )["severities"]["C3-75"]["groups"][0])


def label(i):
    parts = [p for p in str(i).replace("//", "/").split("/") if p]
    if parts and parts[-1].lower() in ("source.jpg", "source.png"):
        return parts[-2] if len(parts) > 1 else parts[-1]
    return parts[-1] if parts else str(i)


def ring_stats(arr, frac):
    """Pixels outside a centred square patch covering `frac` of the area."""
    h, w = arr.shape[:2]
    ph, pw = int(round(h * frac ** 0.5)), int(round(w * frac ** 0.5))
    t, l = (h - ph) // 2, (w - pw) // 2
    m = np.ones((h, w), bool)
    m[t:t + ph, l:l + pw] = False
    ring = arr[m]
    return float(ring.std()), float((ring == 0).mean()), int(ring.size)


with pa.memory_map(str(M3 / "vqa_slack_test.arrow"), "rb") as s:
    t = pa.ipc.open_file(s).read_all()
ids = [label(x) for x in t.column("image_id").to_pylist()]

seen, rows = set(), []
for i, b in zip(ids, t.column("image").to_pylist()):
    if i in seen:
        continue
    seen.add(i)
    a = np.asarray(Image.open(io.BytesIO(b)).convert("L"))
    sd75, blk75, n75 = ring_stats(a, 0.75)
    sd50, blk50, _ = ring_stats(a, 0.50)
    rows.append({"image_id": i, "collapsed_at_75": i in COLLAPSED, "shape": list(a.shape),
                 "ring75_std": sd75, "ring75_black_fraction": blk75, "ring75_pixels": n75,
                 "ring50_std": sd50, "ring50_black_fraction": blk50})

col = [r for r in rows if r["collapsed_at_75"]]
oth = [r for r in rows if not r["collapsed_at_75"]]


def show(tag, rs):
    if not rs:
        return
    sd = np.array([r["ring75_std"] for r in rs])
    bk = np.array([r["ring75_black_fraction"] for r in rs])
    print("  %-14s n=%3d  ring75 std  med %6.3f  max %7.3f   preto  med %.4f  min %.4f"
          % (tag, len(rs), np.median(sd), sd.max(), np.median(bk), bk.min()))


print("=== o anel que sobra apos a oclusao de 75%% (SLAKE, %d imagens unicas)" % len(rows))
show("colapsadas", col)
show("as demais", oth)
print("\n  todas as colapsadas com anel 100%% preto? %s"
      % all(r["ring75_black_fraction"] == 1.0 for r in col))
print("  alguma NAO colapsada com anel 100%% preto? %d de %d"
      % (sum(1 for r in oth if r["ring75_black_fraction"] == 1.0), len(oth)))

out = {"severity": 0.75, "n_unique_images": len(rows), "rows": rows,
       "collapsed_all_black_ring": all(r["ring75_black_fraction"] == 1.0 for r in col),
       "non_collapsed_with_all_black_ring":
           [r["image_id"] for r in oth if r["ring75_black_fraction"] == 1.0]}
(REPO / "docs/v4/occlusion_ring.json").write_text(json.dumps(out, indent=1))
print("\n  gravado docs/v4/occlusion_ring.json")
