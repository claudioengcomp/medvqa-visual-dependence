"""Deterministic, image-stratified split of Kvasir-VQA-x1 -- pure logic,
no network/GPU (kept separate from the CLI script in
build_kvasir_clean_split.py, which performs the actual download).

Motivation: the official HuggingFace train/test split leaks images across
the two sides. 3,821 of its 4,058 test images (94%) also appear in
training, verified by image identifier, because the dataset assigns each
image-question PAIR to a partition rather than each image. This module
pools ALL unique images (from train+test together) and draws a new split
that is disjoint by img_id, stratified by pathology_label so that a rare
clinical category does not end up entirely on one side. There is no
patient identifier in the dataset, so the image is the finest granularity
available for the disjointness guarantee.
"""
from __future__ import annotations

import random
from collections import defaultdict


def split_records_by_image(
    records: list[dict],
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Split `records` (the output of
    src.prepro.prepare_kvasir_arrow.build_image_records -- one dict per
    unique image, carrying `pathology_label`) into train/val/test,
    disjoint by `img_id`, stratified by `pathology_label` (images with no
    pathology_label fall into the `None` stratum, treated like any other).

    `ratios` = (train, val, test), must sum to 1.0 (tolerance 1e-9).
    Deterministic: the same `seed` always produces the same assignment.
    """
    total = sum(ratios)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, they summed to {total}")

    rng = random.Random(seed)

    by_label: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_label[rec["pathology_label"]].append(rec)

    result: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for label, group in by_label.items():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        result["train"].extend(shuffled[:n_train])
        result["val"].extend(shuffled[n_train:n_train + n_val])
        result["test"].extend(shuffled[n_train + n_val:])

    return result
