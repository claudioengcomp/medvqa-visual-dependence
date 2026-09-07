"""C6: the same domain restriction as C5 (same modality+location), but with
a FREE finding category -- it may repeat, it may differ, it does not matter.
The next rung of the restriction ladder after C5:

    C1/C2 (free swap, any image in the dataset)
    > C5 (same domain -- modality+location -- DIFFERENT category)
    > C6 (same domain, FREE category)

C6 reuses the C5 PAIRS exactly as they are (same swap, same seed -- see
`--c5-manifest`) and complements them with new pairs formed freely
(`make_c2_shuffled_pairs.derangement`, with no label restriction at all) for
the images of the same bucket that were left out of C5 -- either because
they have no eligible pathology label (no Abnormality question, or only a
trivial answer) or because they belong to a whole bucket that C5 rejected
(majority category >50%, e.g. CT/Abdomen). The C5 pairs are kept intact on
purpose, so that VDD(C5) and VDD(C6) can be compared over the SAME subset of
images on both sides for the portion C5 already covers.

The finding that motivated this condition: of the 96 images in the SLAKE
test split, 58 fall outside C5 -- 55 of them have at least one other image
in the same domain available to pair freely (including the entire CT/Abdomen
bucket, 22 images, rejected by C5 only because of category imbalance, not
for lack of images). The remaining 3 are the only exemplar of their domain
(modality+location) in the whole test split -- not pairable by ANY
domain-restricted condition, C5 or C6.

`--bucket-membership`: the list of ALL image_ids in the bucket (any
content_type, not only Abnormality) -- built by
`build_slake_pathology_labels.build_slake_bucket_membership`, because the C5
label `.json` only covers images with an eligible pathology answer, a
strictly smaller subset of the whole bucket.

Usage (SLAKE, X-Ray/Lung bucket):
    python -m src.interventions.make_c6_organ_modality_swap \
        --input data/finetune_arrows/vqa_slack_test.arrow \
        --bucket-membership results/SLAKE_bucket_membership_XRAY_LUNG.json \
        --c5-manifest results/C5_manifest_SLAKE.json \
        --output data/finetune_arrows/vqa_slack_test_c6.arrow \
        --manifest results/C6_manifest_SLAKE.json \
        --examples-dir results/c6_examples_slake --n-examples 10 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json

from src.interventions.arrow_io import read_arrow, write_arrow
from src.interventions.make_c2_shuffled_pairs import derangement
from src.interventions.make_c5_matched_pathology_swap import save_example_pairs


def leftover_images(bucket_image_ids: list[str], c5_manifest: list[dict]) -> list[str]:
    """Images of the bucket (same organ+modality domain) that are NOT covered
    by C5 -- the candidates for C6's free pairing. Preserves the input order
    of `bucket_image_ids` (deterministic)."""
    covered = {e["qid_image"] for e in c5_manifest}
    return [img for img in bucket_image_ids if img not in covered]


def apply_organ_modality_swap(input_path: str, bucket_membership_path: str, c5_manifest_path: str,
                               output_path: str, manifest_path: str, seed: int = 42,
                               examples_dir: str | None = None, n_examples: int = 10) -> None:
    df = read_arrow(input_path)
    with open(bucket_membership_path, "r", encoding="utf-8") as fp:
        bucket_ids = set(json.load(fp))
    with open(c5_manifest_path, "r", encoding="utf-8") as fp:
        c5_manifest = json.load(fp)

    all_image_ids = df["image_id"].tolist()
    # restrict to what is actually in the evaluated arrow (test split) --
    # bucket_membership may also include train/validate images.
    bucket_present = [img for img in all_image_ids if img in bucket_ids]
    leftover = leftover_images(bucket_present, c5_manifest)

    df = df.copy()
    original_image = df["image"].tolist()
    original_image_id = df["image_id"].tolist()
    id_to_global = {img_id: i for i, img_id in enumerate(all_image_ids)}

    manifest = list(c5_manifest)

    # re-apply the C5 swaps (the input arrow here is the ORIGINAL one, with no
    # swap at all -- they must be re-applied, not assumed already in place).
    for entry in c5_manifest:
        global_i = id_to_global[entry["qid_image"]]
        donor_global = id_to_global[entry["donor_image"]]
        df.at[global_i, "image"] = original_image[donor_global]
        df.at[global_i, "image_id"] = original_image_id[donor_global]

    skipped: list[str] = []
    if len(leftover) >= 2:
        perm = derangement(len(leftover), seed=seed)
        for local_i, img in enumerate(leftover):
            donor_img = leftover[perm[local_i]]
            global_i = id_to_global[img]
            donor_global = id_to_global[donor_img]
            df.at[global_i, "image"] = original_image[donor_global]
            df.at[global_i, "image_id"] = original_image_id[donor_global]
            manifest.append({
                "qid_image": img, "original_label": None,
                "donor_image": donor_img, "donor_label": None,
            })
    else:
        skipped = leftover  # 0 or 1 left over -- no pair possible, stays out of C6

    write_arrow(df, output_path)
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)

    n_new = len(manifest) - len(c5_manifest)
    print(f"{len(manifest)} images covered by C6 (out of {len(bucket_present)} in the bucket): "
          f"{len(c5_manifest)} inherited from C5 + {n_new} new ones (free pairing) -> {output_path}")
    if skipped:
        print(f"{len(skipped)} bucket image(s) with no possible pair (only exemplar of the domain "
              f"in the test split): {skipped}")

    if examples_dir is not None and manifest:
        save_example_pairs(manifest, original_image, original_image_id, examples_dir, n_examples, seed)
        print(f"{min(n_examples, len(manifest))} example pairs (PNG + index.json) -> {examples_dir}")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--bucket-membership", required=True,
                         help="JSON with the list of ALL image_id values in the bucket (modality+location), "
                              "not only those with an eligible pathology label -- see "
                              "build_slake_pathology_labels.build_slake_bucket_membership.")
    parser.add_argument("--c5-manifest", required=True,
                         help="C5 manifest already generated for the same bucket -- the pairs are "
                              "reused intact, not recomputed.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--examples-dir", default=None)
    parser.add_argument("--n-examples", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    apply_organ_modality_swap(args.input, args.bucket_membership, args.c5_manifest,
                               args.output, args.manifest, seed=args.seed,
                               examples_dir=args.examples_dir, n_examples=args.n_examples)
