"""Per-row manifests of every image swap (Section 4.3): for each question whose
image was replaced under C2, C5 or C6, the original image id, the donor image id
it received, and the SHA-256 of the encoded bytes of both images.

The hashes are those of the intact test set (C0): a swap moves the bytes of an
existing test image into another row and changes nothing else, so the bytes a
row received under the swap are the bytes of the donor's C0 image. That the
shipped intervention arrows satisfy this was checked on the arrows themselves
(the multiset of SHA-256 over the encoded bytes and the id-to-hash map of every
swap arrow equal those of C0; docs/v4/swap_bytes_identity.json). Only
identifiers and hashes are written; no image.

The hash map (`{dataset: {"test": {"ids": [...], "sha": [...]}}}`) is derived
from the arrows and is not part of this repository; pass its path.

Usage:
    python -m src.analysis.build_row_manifests --hashes path/to/image_hashes.json \
        --swapmap data/swapmap.json --output-dir results/manifests
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASETS = ("KVASIR", "SLAKE", "VQARAD")
CONDITIONS = ("C2", "C5", "C6")


def build(hashes: dict, swapmap: dict, dataset: str, condition: str) -> dict | None:
    swap = swapmap[dataset]["swap"].get(condition)
    if swap is None:
        return None
    sha = dict(zip(hashes[dataset]["test"]["ids"], hashes[dataset]["test"]["sha"]))
    rows = []
    for qid, meta in sorted(swapmap[dataset]["qid"].items(), key=lambda kv: int(kv[0])):
        img = meta["img"]
        if img not in swap:
            continue
        donor = swap[img]
        rows.append({"qid": int(qid), "image_id_original": img, "image_id_received": donor,
                     "sha256_original": sha[img], "sha256_received": sha[donor]})
    same = sum(1 for r in rows if r["sha256_original"] == r["sha256_received"])
    return {"dataset": dataset, "condition": condition, "n_rows": len(rows),
            "n_images_swapped": len(swap),
            "hash": "sha256 of the encoded image bytes as stored in the intact test arrow (C0)",
            "rows_with_identical_bytes_before_and_after": same,
            "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hashes", required=True)
    ap.add_argument("--swapmap", default="data/swapmap.json")
    ap.add_argument("--output-dir", default="results/manifests")
    a = ap.parse_args(argv)
    hashes = json.loads(Path(a.hashes).read_text())
    swapmap = json.loads(Path(a.swapmap).read_text())
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        for condition in CONDITIONS:
            m = build(hashes, swapmap, dataset, condition)
            if m is None:
                continue
            path = out / f"{dataset}_{condition}_rows.json"
            path.write_text(json.dumps(m, indent=1) + "\n")
            print(f"{path}: {m['n_rows']} rows, {m['n_images_swapped']} images, "
                  f"{m['rows_with_identical_bytes_before_and_after']} rows with identical bytes")


if __name__ == "__main__":
    main()
