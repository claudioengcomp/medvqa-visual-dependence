"""Builds the swap map: for every intervention that substitutes images
(C2, C5, C6), records which donor image replaced each original image, plus
the question and reference answer of every question id.

This map is what makes the Counterfactual Consistency Rate (CCR) computable:
to know whether an altered answer is correct *for the image the model
actually received*, we need the donor's reference answer for the same
question.

Donors are recovered by matching image bytes (MD5) against the intact test
set, which is why the intervened arrows must be generated from the same
base file as the intact condition.

A precomputed swap map for the three datasets ships with this repository as
`data/swapmap.json`, so CCR can be verified without regenerating the
intervention data.

Usage:
    python -m src.analysis.build_swapmap \
        --arrow-dir path/to/finetune_arrows \
        --output data/swapmap.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

import pyarrow as pa

# Base (intact) test arrow per dataset. The intervened files are derived from
# these by appending the condition suffix, e.g. vqa_slack_test_c5.arrow.
BASES = {
    "SLAKE": "vqa_slack_test.arrow",
    "VQARAD": "vqa_vqa_rad_test.arrow",
    "KVASIR": "vqa_kvasir_test.arrow",
}
# Only the conditions that substitute one image for another. C1 (blank frame)
# and the parametric corruptions C4_OCC_* / C4_NOISE_* (the paper's C3 and C4)
# alter the original image instead of replacing it,
# so they have no donor and no CCR.
CONDITIONS = ["c2", "c5", "c6"]


def read_arrow(path: str) -> list[dict]:
    with pa.OSFile(path, "rb") as source:
        return pa.ipc.open_file(source).read_all().to_pylist()


def build_swapmap(arrow_dirs: list[str]) -> dict:
    files: dict[str, str] = {}
    for directory in arrow_dirs:
        for path in glob.glob(os.path.join(directory, "*.arrow")):
            files.setdefault(os.path.basename(path), path)

    out: dict = {}
    for dataset, base_name in BASES.items():
        if base_name not in files:
            print(f"  skipping {dataset}: {base_name} not found")
            continue
        base = read_arrow(files[base_name])
        out[dataset] = {"n_images": len(base), "qid": {}, "swap": {}}

        hash_to_image: dict[str, str] = {}
        question_to_image: dict[int, str] = {}
        for row in base:
            hash_to_image.setdefault(hashlib.md5(row["image"]).hexdigest(), row["image_id"])
            for i, qid in enumerate(row["question_id"]):
                question_to_image[int(qid)] = row["image_id"]
                out[dataset]["qid"][int(qid)] = {
                    "img": row["image_id"],
                    "q": row["questions"][i],
                    "a": row["answers"][i],
                    "t": int(row["answer_type"][i]),
                }

        for cond in CONDITIONS:
            cond_name = base_name.replace(".arrow", f"_{cond}.arrow")
            if cond_name not in files:
                print(f"  missing condition {dataset} {cond}")
                continue
            mapping: dict[str, str] = {}
            external = 0
            for row in read_arrow(files[cond_name]):
                original = question_to_image[int(row["question_id"][0])]
                donor = hash_to_image.get(hashlib.md5(row["image"]).hexdigest())
                if donor is None:
                    # Image not present in the intact test set: keep whatever
                    # identifier the intervened file carries.
                    donor = row["image_id"]
                    external += 1
                if donor != original:
                    mapping[original] = donor
            out[dataset]["swap"][cond.upper()] = mapping
            print(f"  {dataset} {cond}: {len(mapping)}/{len(base)} images swapped (external={external})")

    return out


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow-dir", nargs="+", required=True,
                        help="One or more directories containing the .arrow files")
    parser.add_argument("--output", default="data/swapmap.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    swapmap = build_swapmap(args.arrow_dir)
    with open(args.output, "w") as fp:
        json.dump(swapmap, fp)
    print("wrote", args.output, os.path.getsize(args.output), "bytes")
