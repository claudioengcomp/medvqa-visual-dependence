"""Did a swap change any image, or only which row it sits in?

Under canonical order by image_id, the swap embeddings are not bit-identical to
C0: MedSigLIP, DINOv3 and BiomedCLIP disagree by up to 2e-5 on Kvasir and
VQA-RAD, while DINOv2 and the colour histogram agree exactly, and SLAKE agrees
exactly everywhere. G1 was going to read that as a re-encode. Before it does,
the question has to be asked of the images rather than of their embeddings,
where it has an exact answer.

For each swap: the multiset of sha256 over the encoded image bytes, and the map
from image_id to sha256. If both match C0, no image was touched and the
disagreement lives entirely in the embedding step.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
import os
from pathlib import Path

import pyarrow as pa

REPO = Path(__file__).resolve().parents[2]
M3 = Path(os.environ.get("M3AE_ARROWS", str(REPO / "data/finetune_arrows")))
P0A = REPO / "data/finetune_arrows"

SETS = {"KVASIR": (P0A, "vqa_kvasir_test_clean"), "SLAKE": (M3, "vqa_slack_test"),
        "VQARAD": (M3, "vqa_vqa_rad_test")}
SWAPS = {"C2": "_c2", "C5": "_c5", "C6": "_c6"}


def read(p):
    with pa.memory_map(str(p), "rb") as s:
        t = pa.ipc.open_file(s).read_all()
    ids = [str(x) for x in t.column("image_id").to_pylist()]
    sha = [hashlib.sha256(b).hexdigest() for b in t.column("image").to_pylist()]
    return ids, sha


out = {}
for bench, (d, stem) in SETS.items():
    i0, s0 = read(d / (stem + ".arrow"))
    m0, c0 = dict(zip(i0, s0)), Counter(s0)
    out[bench] = {}
    print("\n=== %s   %d linhas, %d sha distintos" % (bench, len(s0), len(c0)))
    for name, sfx in SWAPS.items():
        p = d / ("%s%s.arrow" % (stem, sfx))
        if not p.exists():
            continue
        i, s = read(p)
        m = dict(zip(i, s))
        same_multiset = Counter(s) == c0
        same_map = m == m0
        moved = sum(1 for k in range(len(i)) if i[k] != i0[k])
        out[bench][name] = {"same_sha_multiset": same_multiset,
                            "same_id_to_sha_map": same_map,
                            "rows_whose_image_id_changed": moved,
                            "n_rows": len(i)}
        print("  %-3s multiconjunto de sha igual ao C0: %-4s   mapa id->sha igual: %-4s   linhas com id diferente: %d de %d"
              % (name, "sim" if same_multiset else "NAO", "sim" if same_map else "NAO",
                 moved, len(i)))
        if not same_map:
            diff = [k for k in m0 if m0.get(k) != m.get(k)]
            print("      %d image_id com sha diferente, ex.: %s" % (len(diff), diff[:3]))

p = REPO / "docs/v4/swap_bytes_identity.json"
p.write_text(json.dumps(out, indent=1))
print("\n  gravado %s" % p)
