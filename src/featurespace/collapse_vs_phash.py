"""Are the images that occlusion collapses the same images the pHash audit flagged?

Two instruments looked at SLAKE and each returned a set of images. The audit
found 24 test images with a perceptual neighbour in train or validation, and
read them as slices of volumes that cross the partition. Centred occlusion at
75% collapses 22 test images into a single indistinguishable group, and the
reading offered for that was "what is left is the border, and radiological
borders are black".

If the two sets coincide, they are one phenomenon seen twice: slices of one exam
share a border, so they both leak across partitions and collapse under
occlusion. If they do not, they are two findings and the paper says two things.

Also computed, because it is the more direct test of "same exam": whether the
collapsed images are perceptual neighbours OF EACH OTHER inside the test split,
which the audit never asked (it only compared test against the bank).
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
import os
from pathlib import Path

import numpy as np
import pyarrow as pa

REPO = Path(__file__).resolve().parents[2]
M3 = Path(os.environ.get("M3AE_ARROWS", str(REPO / "data/finetune_arrows")))
HASHES = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "image_hashes.json"
NEAR = 4
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def label(i):
    parts = [p for p in str(i).replace("//", "/").split("/") if p]
    if parts and parts[-1].lower() in ("source.jpg", "source.png"):
        return parts[-2] if len(parts) > 1 else parts[-1]
    return parts[-1] if parts else str(i)


def collapsed(stem, sfx):
    p = M3 / ("%s%s.arrow" % (stem, sfx))
    with pa.memory_map(str(p), "rb") as s:
        t = pa.ipc.open_file(s).read_all()
    ids = [label(x) for x in t.column("image_id").to_pylist()]
    by = defaultdict(list)
    for i, b in zip(ids, t.column("image").to_pylist()):
        by[hashlib.sha256(b).hexdigest()].append(i)
    return [sorted(set(v)) for v in by.values() if len(set(v)) > 1]


H = json.loads(HASHES.read_text())["SLAKE"]
test_ids = [label(i) for i in H["test"]["ids"]]
test_ph = np.array(H["test"]["phash"], dtype=np.uint8)
bank_ids = [label(i) for i in H["train"]["ids"]] + [label(i) for i in H["val"]["ids"]]
bank_ph = np.array(H["train"]["phash"] + H["val"]["phash"], dtype=np.uint8)

# test against the bank: the audit's 24
d = POP[np.bitwise_xor(test_ph[:, None, :], bank_ph[None, :, :])].sum(2)
near_bank = {}
for i in range(len(test_ids)):
    j = np.where(d[i] <= NEAR)[0]
    if len(j):
        near_bank[test_ids[i]] = sorted({bank_ids[k] for k in j})

# test against itself: never asked before
dt = POP[np.bitwise_xor(test_ph[:, None, :], test_ph[None, :, :])].sum(2)
np.fill_diagonal(dt, 99)
near_test = {}
for i in range(len(test_ids)):
    j = np.where(dt[i] <= NEAR)[0]
    if len(j):
        near_test[test_ids[i]] = sorted({test_ids[k] for k in j})

out = {"benchmark": "SLAKE", "near_threshold": NEAR,
       "audit_near_bank": {"n": len(near_bank), "ids": sorted(near_bank)},
       "near_inside_test": {"n": len(near_test), "ids": sorted(near_test)},
       "severities": {}}

print("auditoria: %d imagens de teste com vizinho pHash<=%d no banco" % (len(near_bank), NEAR))
print("novo:      %d imagens de teste com vizinho pHash<=%d DENTRO do teste" % (len(near_test), NEAR))

for name, sfx in (("C3-25", "_c4_occ_25"), ("C3-50", "_c4_occ_50"), ("C3-75", "_c4_occ_75")):
    groups = collapsed("vqa_slack_test", sfx)
    ids = sorted({i for g in groups for i in g})
    inter_bank = sorted(set(ids) & set(near_bank))
    inter_test = sorted(set(ids) & set(near_test))
    out["severities"][name] = {
        "collapsed_groups": len(groups), "images_collapsed": len(ids),
        "groups": groups,
        "overlap_with_audit_near_bank": {"n": len(inter_bank), "ids": inter_bank},
        "overlap_with_near_inside_test": {"n": len(inter_test), "ids": inter_test},
    }
    print("\n=== %s: %d grupo(s), %d imagens" % (name, len(groups), len(ids)))
    if ids:
        print("  ids: %s" % ", ".join(ids))
        print("  destas, com vizinho no banco (auditoria): %d de %d  -> %s"
              % (len(inter_bank), len(ids), ", ".join(inter_bank) or "nenhuma"))
        print("  destas, vizinhas entre si dentro do teste: %d de %d"
              % (len(inter_test), len(ids)))

# the direct question, in one line
c75 = set(i for g in out["severities"]["C3-75"]["groups"] for i in g)
nb = set(near_bank)
out["verdict"] = {
    "collapsed_75": len(c75), "audit_24": len(nb),
    "in_both": len(c75 & nb), "collapsed_only": sorted(c75 - nb), "audit_only": sorted(nb - c75),
}
print("\n=== VEREDITO ===")
print("  colapsadas em C3-75: %d   sinalizadas pela auditoria: %d   nas duas: %d"
      % (len(c75), len(nb), len(c75 & nb)))
print("  so colapsadas: %d   so auditoria: %d" % (len(c75 - nb), len(nb - c75)))

(REPO / "docs/v4/collapse_vs_phash.json").write_text(json.dumps(out, indent=1))
print("\n  gravado docs/v4/collapse_vs_phash.json")
