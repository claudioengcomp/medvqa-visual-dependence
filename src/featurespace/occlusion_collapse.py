"""How much of the image does centred occlusion destroy?

Measured, not asserted: at each severity, how many distinct images remain
distinct after the patch is applied. Two images that become byte-identical have
had everything that distinguished them covered; what is left is the border.

This is the most concrete statement available for "centred occlusion destroys
the image, not the evidence", and it needs no model.

Reported per benchmark and severity: rows, distinct images by sha256 of the
decoded bytes, and the groups that collapsed together.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
import os
from pathlib import Path

import pyarrow as pa

REPO = Path(__file__).resolve().parents[2]
M3 = Path(os.environ.get("M3AE_ARROWS", str(REPO / "data/finetune_arrows")))
P0A = REPO / "data/finetune_arrows"

SETS = {"KVASIR": (P0A, "vqa_kvasir_test_clean"),
        "SLAKE": (M3, "vqa_slack_test"),
        "VQARAD": (M3, "vqa_vqa_rad_test")}
SEV = [("C0", ""), ("C3-25", "_c4_occ_25"), ("C3-50", "_c4_occ_50"),
       ("C3-75", "_c4_occ_75"), ("C4-25", "_c4_noise_25"),
       ("C4-50", "_c4_noise_50"), ("C4-75", "_c4_noise_75")]


def label(i):
    parts = [p for p in str(i).replace("//", "/").split("/") if p]
    if parts and parts[-1].lower() in ("source.jpg", "source.png"):
        return parts[-2] if len(parts) > 1 else parts[-1]
    return parts[-1] if parts else str(i)


out = {}
for bench, (d, stem) in SETS.items():
    out[bench] = {}
    print("\n=== %s" % bench)
    for name, sfx in SEV:
        p = d / ("%s%s.arrow" % (stem, sfx))
        if not p.exists():
            continue
        with pa.memory_map(str(p), "rb") as s:
            t = pa.ipc.open_file(s).read_all()
        ids = [label(x) for x in t.column("image_id").to_pylist()]
        by = defaultdict(list)
        for i, b in zip(ids, t.column("image").to_pylist()):
            by[hashlib.sha256(b).hexdigest()].append(i)
        groups = [v for v in by.values() if len(v) > 1]
        collapsed = sum(len(v) for v in groups)
        out[bench][name] = {"rows": t.num_rows, "distinct_images": len(by),
                            "collapsed_groups": len(groups),
                            "images_in_collapsed_groups": collapsed,
                            "examples": [g[:4] for g in groups[:8]]}
        extra = ""
        if name != "C0" and len(by) < out[bench]["C0"]["distinct_images"]:
            lost = out[bench]["C0"]["distinct_images"] - len(by)
            extra = "   <-- %d imagens deixaram de ser distinguiveis" % lost
        print("  %-7s linhas %5d  imagens distintas %5d  grupos colapsados %3d%s"
              % (name, t.num_rows, len(by), len(groups), extra))
        for g in groups[:4]:
            print("        %s" % " = ".join(g[:4]))

(REPO / "docs/v4/occlusion_collapse.json").write_text(json.dumps(out, indent=1))
print("\n  gravado docs/v4/occlusion_collapse.json")
