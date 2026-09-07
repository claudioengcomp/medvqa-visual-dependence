"""Extract the VQA-RAD domain field (`image_organ`) for C6 -- unlike
Kvasir, VQA-RAD already ships this field in the raw data (`data_RAD`/MEVF:
`image_organ`, e.g. "CHEST", "ABD", "HEAD"), so it does not have to be
derived from any hidden question. All 203 unique test-split images carry
the field (100%), with 0 inconsistencies (the same image always has the
same organ across all of its associated questions) -- CHEST=73, ABD=70,
HEAD=60.

C5 is not viable for VQA-RAD (confirmed twice, independently) -- the
pathology vocabulary (`question_type=='ABN'`) is almost entirely unique
free text (34/203 images, most with a distinct answer), without enough
repetition for a "different category" derangement to be meaningful. C6,
which does not depend on category repetition, covers 203/203 (100%) --
the best coverage of the 3 datasets used here.

image_id in the arrow follows "data/finetune_data/vqa_rad//images/<image_name>"
(confirmed by reading the image_id column of the real arrow -- the same
double-slash convention as SLAKE, inherited from M3AE).

Usage:
    python -m src.prepro.build_vqa_rad_organ_membership \
        --testset data_RAD/testset.json \
        --output-dir results/
"""
from __future__ import annotations

import argparse
import json
import os

VQA_RAD_IMAGE_ID_PREFIX = "data/finetune_data/vqa_rad//images/"


def build_organ_membership(records: list[dict]) -> dict[str, list[str]]:
    """records: list of raw testset.json rows (`image_name`, `image_organ`
    per record -- several rows per image, one per question). Returns
    {organ: [image_id, ...]}, without duplicates, preserving
    first-occurrence order."""
    by_organ: dict[str, list[str]] = {}
    seen: set[str] = set()
    for r in records:
        img_id = f"{VQA_RAD_IMAGE_ID_PREFIX}{r['image_name']}"
        if img_id in seen:
            continue
        seen.add(img_id)
        by_organ.setdefault(r["image_organ"], []).append(img_id)
    return by_organ


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.testset, "r", encoding="utf-8") as fp:
        records = json.load(fp)

    membership = build_organ_membership(records)
    for organ, ids in membership.items():
        out_path = os.path.join(args.output_dir, f"vqarad_bucket_{organ.lower()}.json")
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(ids, fp, indent=2, ensure_ascii=False)
        print(f"{len(ids)} {organ} images -> {out_path}")
