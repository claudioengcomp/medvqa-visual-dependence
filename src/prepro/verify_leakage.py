"""Verifies the image leakage in the official Kvasir-VQA-x1 split.

Reproduces the claim made in the paper (Abstract, Section 1 and Section 4.1):
3,821 of the 4,058 test images (94%) also appear in the training set.

The check is done on `img_id`, the image identifier carried by the dataset
itself -- not on an approximate image hash -- so it is exact.

The cause is a mismatch between the assignment criterion and the
documentation: the release assigns each image-QUESTION PAIR to a partition,
whereas the dataset description and the dataset card state that no test image
appears in training. A visible symptom is that the test partition holds about
10% of the questions but 62% of the images, although each image carries
roughly 25 questions -- a proportion no split by image could produce.

Requires network access to the HuggingFace Hub. Nothing else in this
repository does.

Usage:
    python -m src.prepro.verify_leakage
"""
from __future__ import annotations

import argparse

from datasets import load_dataset

DATASET = "SimulaMet/Kvasir-VQA-x1"


def verify(dataset_name: str = DATASET) -> dict:
    train = load_dataset(dataset_name, split="train")
    test = load_dataset(dataset_name, split="test")

    train_images = set(train["img_id"])
    test_images = set(test["img_id"])
    overlap = train_images & test_images

    total_images = len(train_images | test_images)
    result = {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_images": len(train_images),
        "test_images": len(test_images),
        "overlapping_images": len(overlap),
        "overlap_fraction_of_test": len(overlap) / len(test_images),
        "unique_images_total": total_images,
        "test_share_of_questions": len(test) / (len(train) + len(test)),
        "test_share_of_images": len(test_images) / total_images,
        "questions_per_image": (len(train) + len(test)) / total_images,
    }
    return result


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    r = verify(args.dataset)
    print(f"train rows                : {r['train_rows']:,}")
    print(f"test rows                 : {r['test_rows']:,}")
    print(f"unique train images       : {r['train_images']:,}")
    print(f"unique test images        : {r['test_images']:,}")
    print(f"test images also in train : {r['overlapping_images']:,} "
          f"({r['overlap_fraction_of_test']:.1%} of the test images)")
    print()
    print(f"unique images overall     : {r['unique_images_total']:,}")
    print(f"questions per image (avg) : {r['questions_per_image']:.1f}")
    print(f"test share of questions   : {r['test_share_of_questions']:.1%}")
    print(f"test share of images      : {r['test_share_of_images']:.1%}")
    print()
    print("Expected (reported in the paper): 3,821 of 4,058 test images (94%).")
