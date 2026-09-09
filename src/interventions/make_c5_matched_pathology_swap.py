"""C5: like C2 (make_c2_shuffled_pairs.py), but the swap partner is
restricted — images are only swapped between the SAME modality and a
DIFFERENT FINDING CATEGORY, instead of drawing freely over the whole
dataset.

Rationale: C2 as it stands can pair, by bad luck of the random draw, two
images so different on the surface that a model may "notice the swap"
without ever having evaluated the specific clinical finding — contaminating
VDD(C2) with a generic anomaly-detection confounder rather than real visual
dependency. Restricting the swap to the same modality removes that
surface-level cue, while requiring a different finding category keeps the
correct answer genuinely changed by the swap.

Dataset-agnostic by design: it takes the arrow plus an already-built .json
of pathology labels (src/prepro/prepare_kvasir_arrow.py for Kvasir;
src/prepro/build_slake_pathology_labels.py for SLAKE) — it does not know,
and does not need to know, which dataset the labels came from.

Scoring convention identical to C2: the ground truth for each question
remains the ORIGINAL answer — only the displayed image changes. Rows whose
image has no eligible label keep their own original image (they do not take
part in the swap) and must be filtered out in the analysis (restrict to the
eligible qid subset when comparing VDD(C5) against VDD(C0)).

PERSISTENCE — read before running in production: `--output` (the .arrow
with the swapped images) may live in `data/finetune_arrows/` (gitignored)
because it can be regenerated at any time from `--manifest` plus the same
seed. BUT `--manifest` and `--examples-dir` are the ONLY evidence of which
swaps actually happened — save them under `results/` (not gitignored),
never under `data/`, so that they outlive the machine that produced them.

Usage (Kvasir) — 3 categories, not 2: measured on the REAL Kvasir-VQA-x1
test split (not a sample), Polyp=523 and Ulcerative Colitis=450 on their
own already give Polyp a 53.75% majority — so the originally considered
2-category pair FAILS (matched_derangement raises ValueError on the
majority category). Adding Esophagitis (403) as a third category gives
1,376 eligible images and drops the largest fraction to 38% — feasible, and
verified:
    python -m src.interventions.make_c5_matched_pathology_swap \
        --input data/finetune_arrows/vqa_kvasir_test.arrow \
        --pathology-labels data/finetune_arrows/vqa_kvasir_test_pathology_labels.json \
        --output data/finetune_arrows/vqa_kvasir_test_c5.arrow \
        --manifest results/C5_manifest_KVASIR.json \
        --examples-dir results/c5_examples_kvasir --n-examples 10 \
        --allowed-categories polyp "ulcerative colitis" oesophagitis \
        --seed 42

Usage (SLAKE, without --allowed-categories = every category in the
X-Ray/Lung bucket):
    python -m src.interventions.make_c5_matched_pathology_swap \
        --input data/finetune_arrows/vqa_slack_test.arrow \
        --pathology-labels src/prepro/precomputed/vqa_slack_test_pathology_labels.json \
        --output data/finetune_arrows/vqa_slack_test_c5.arrow \
        --manifest results/C5_manifest_SLAKE.json \
        --examples-dir results/c5_examples_slake --n-examples 10 \
        --seed 42
"""
from __future__ import annotations

import argparse
import heapq
import io
import json
import os

import numpy as np
from PIL import Image

from src.interventions.arrow_io import read_arrow, write_arrow


def build_eligible_pairs(image_ids: list[str], labels_by_image: dict[str, str],
                          allowed_categories: list[str] | None) -> list[str]:
    """Return the sublist of `image_ids` eligible for swapping. If
    `allowed_categories` is None, include any image that has SOME label in
    `labels_by_image` (the SLAKE case: many categories — listing them all on
    the CLI would be impractical). If it is a list, restrict to those
    categories (the Kvasir case)."""
    result = []
    for img_id in image_ids:
        label = labels_by_image.get(img_id)
        if label is None:
            continue
        if allowed_categories is not None:
            allowed = {c.strip().lower() for c in allowed_categories}
            if label.strip().lower() not in allowed:
                continue
        result.append(img_id)
    return result


def matched_derangement(labels: list[str], seed: int, max_attempts: int = 200) -> list[int]:
    """Return a permutation of range(len(labels)) such that, for every i,
    labels[perm[i]] != labels[i]. Raises ValueError if the majority category
    exceeds 50% of the total — in that case it is impossible to avoid
    self-pairing for everyone, no matter the algorithm.

    This does NOT use draw-and-reject (random permutation + check whether it
    works): with few, balanced categories — exactly the Kvasir production
    case, ~500/500 — the fraction of valid permutations over the total is
    astronomically small (about 1 in 10^299 for n=1000 split 500/500), so
    rejection would never terminate in practice. Instead the solution is
    constructed directly: group the indices by category, shuffle each group
    (using the seed), interleave the groups always taking from the largest
    remaining one ('reorganize string' / 'task scheduler with cooldown 1' —
    this guarantees that no category ends up adjacent to itself in the
    linear sequence whenever the largest category does not exceed half the
    total), and then use a circular shift by one position over that sequence
    as the final permutation — each item receives the next item of the
    interleaved sequence as its donor. The single case that linear
    interleaving does not cover on its own (the "wrap-around": the first and
    last item of the sequence falling in the same category) is fixed with a
    deterministic local swap.

    `max_attempts` is kept in the signature for backward compatibility; it
    is unused in this deterministic construction."""
    n = len(labels)
    if n < 2:
        raise ValueError("matched_derangement requires n >= 2")

    buckets: dict[str, list[int]] = {}
    for i, lbl in enumerate(labels):
        buckets.setdefault(lbl, []).append(i)

    max_size = max(len(v) for v in buckets.values())
    if max_size > n - max_size:
        raise ValueError(
            f"Cannot generate a valid pairing -- the majority category holds "
            f"{max_size} of {n} eligible items (more than 50%), so self-pairing "
            f"cannot be avoided for every item of that category."
        )

    rng = np.random.RandomState(seed)
    remaining = {lbl: list(rng.permutation(idxs)) for lbl, idxs in buckets.items()}

    heap = [(-len(v), lbl) for lbl, v in remaining.items()]
    heapq.heapify(heap)

    order: list[int] = []
    delayed: tuple[int, str] | None = None  # bucket used in the previous step, not reusable yet
    while heap:
        neg_count, lbl = heapq.heappop(heap)
        order.append(remaining[lbl].pop())
        new_count = -neg_count - 1

        if delayed is not None:
            heapq.heappush(heap, delayed)
        delayed = (-new_count, lbl) if new_count > 0 else None

    # circular wrap-around: the first and last elements of the sequence must
    # also have different categories (perm links order[-1] -> order[0]).
    if labels[order[0]] == labels[order[-1]]:
        for m in range(1, n - 1):
            if (labels[order[m]] != labels[order[-1]]
                    and labels[order[m - 1]] != labels[order[-1]]
                    and labels[order[0]] != labels[order[m]]
                    and (m + 1 == n - 1 or labels[order[m + 1]] != labels[order[m]])):
                order[m], order[-1] = order[-1], order[m]
                break
        else:
            raise ValueError(
                "Could not fix the circular wrap-around of the pairing -- "
                "uncovered edge case, review the category distribution."
            )

    perm = [0] * n
    for i in range(n):
        perm[order[i]] = order[(i + 1) % n]

    assert all(labels[perm[i]] != labels[i] for i in range(n)), (
        "invariant violated: matched_derangement produced a pairing with a repeated category"
    )
    return perm


def save_example_pairs(manifest: list[dict], original_image: list[bytes], original_image_id: list[str],
                        examples_dir: str, n_examples: int, seed: int) -> None:
    """Save `n_examples` pairs (original image, donor image) from `manifest`
    as real PNG files in `examples_dir`, plus an `index.json` describing each
    pair (labels, image_id, file name) — meant to become an example figure in
    the paper without having to reopen the original dataset later (Kvasir is
    hosted by URL, SLAKE is a local file — either may stop being accessible;
    the PNG saved here will not).

    Deterministic sampling (same seed = same examples, always)."""
    os.makedirs(examples_dir, exist_ok=True)
    id_to_bytes = dict(zip(original_image_id, original_image))

    rng = np.random.RandomState(seed)
    n = min(n_examples, len(manifest))
    sample_idx = rng.choice(len(manifest), size=n, replace=False)
    sample_idx.sort()  # stable read order; does not change which items were sampled

    index = []
    for rank, i in enumerate(sample_idx):
        entry = manifest[i]
        original_file = f"{rank:02d}_original_{entry['original_label']}.png".replace(" ", "_").replace(";", "-")
        donor_file = f"{rank:02d}_donor_{entry['donor_label']}.png".replace(" ", "_").replace(";", "-")

        Image.open(io.BytesIO(id_to_bytes[entry["qid_image"]])).convert("RGB").save(
            os.path.join(examples_dir, original_file))
        Image.open(io.BytesIO(id_to_bytes[entry["donor_image"]])).convert("RGB").save(
            os.path.join(examples_dir, donor_file))

        index.append({
            "original_image_id": entry["qid_image"], "original_label": entry["original_label"],
            "original_file": original_file,
            "donor_image_id": entry["donor_image"], "donor_label": entry["donor_label"],
            "donor_file": donor_file,
        })

    with open(os.path.join(examples_dir, "index.json"), "w", encoding="utf-8") as fp:
        json.dump(index, fp, indent=2, ensure_ascii=False)


def apply_matched_swap(input_path: str, pathology_labels_path: str, output_path: str,
                        manifest_path: str, allowed_categories: list[str] | None,
                        seed: int = 42, examples_dir: str | None = None, n_examples: int = 10) -> None:
    df = read_arrow(input_path)
    with open(pathology_labels_path, "r", encoding="utf-8") as fp:
        labels_by_image = json.load(fp)

    all_image_ids = df["image_id"].tolist()
    eligible = build_eligible_pairs(all_image_ids, labels_by_image, allowed_categories)
    eligible_set = set(eligible)

    eligible_positions = [i for i, img_id in enumerate(all_image_ids) if img_id in eligible_set]
    eligible_labels = [labels_by_image[all_image_ids[i]].strip().lower() for i in eligible_positions]

    perm_within_eligible = matched_derangement(eligible_labels, seed=seed)

    df = df.copy()
    original_image = df["image"].tolist()
    original_image_id = df["image_id"].tolist()

    manifest = []
    for local_i, global_i in enumerate(eligible_positions):
        donor_local = perm_within_eligible[local_i]
        donor_global = eligible_positions[donor_local]
        df.at[global_i, "image"] = original_image[donor_global]
        df.at[global_i, "image_id"] = original_image_id[donor_global]
        manifest.append({
            "qid_image": original_image_id[global_i],
            "original_label": eligible_labels[local_i],
            "donor_image": original_image_id[donor_global],
            "donor_label": eligible_labels[donor_local],
        })

    write_arrow(df, output_path)
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)

    print(f"{len(manifest)} images swapped (out of {len(df)} in the arrow) -> {output_path}")
    print(f"audit manifest -> {manifest_path}")

    if examples_dir is not None and manifest:
        save_example_pairs(manifest, original_image, original_image_id, examples_dir, n_examples, seed)
        print(f"{min(n_examples, len(manifest))} example pairs (PNG + index.json) -> {examples_dir}")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--pathology-labels", required=True)
    parser.add_argument("--output", required=True,
                         help="Arrow with the swapped images -- it can live in data/finetune_arrows/ "
                              "(gitignored, it is large derived data, regenerable at any time "
                              "from --manifest + --seed).")
    parser.add_argument("--manifest", required=True,
                         help="Swap audit JSON (which image became the donor of which) -- "
                              "save it under results/, NOT under data/ (gitignored). It is the "
                              "evidence for condition C5 and must persist beyond the run, since "
                              "it is the only record of which pairs were swapped.")
    parser.add_argument("--allowed-categories", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--examples-dir", default=None,
                         help="If given, saves --n-examples pairs (original, donor) as real PNG "
                              "files + index.json -- for the paper figure, without depending on "
                              "the original dataset remaining accessible later. Save it under "
                              "results/, not under data/.")
    parser.add_argument("--n-examples", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    apply_matched_swap(args.input, args.pathology_labels, args.output, args.manifest,
                        args.allowed_categories, seed=args.seed,
                        examples_dir=args.examples_dir, n_examples=args.n_examples)
