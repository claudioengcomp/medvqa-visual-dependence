"""Counterfactual Consistency Rate (CCR) -- Section 3.5 of the paper.

A drop in accuracy under an image swap admits two readings: the model
failed, or the model correctly described the image it was actually given.
VDD cannot tell them apart, because it always scores against the reference
answer of the *original* image. CCR separates them by scoring against the
reference answer of the *donor*.

For a swapped question (x, q, y) with donor x', let y' be the reference
answer of the same question q asked about x', when that question exists in
the test set. The question is INFORMATIVE only when y' differs from y: if
both images have the same reference answer, the swap reveals nothing about
whether the model read the image.

Over the informative set, every altered answer falls into exactly one of
three buckets:

    followed_donor  -- matches the donor's reference  -> this is the CCR
    kept_original   -- matches the original reference -> a textual or
                       domain prior prevailed over the image
    neither         -- some other string

Answers are compared with `answer_matches`, the same function that decides
the `correct` field behind accuracy and VDD (see inference/process_text.py).
Reference answers may be lists, so the comparison is set-based: a question is
uninformative when the two reference sets match at all.

That comparison never merges distinct reference answers of the same question:
within a question template, no two reference answers of the three benchmarks
compare as equal (0 of 260 reference strings in Kvasir-VQA-x1, 0 of 130 in
SLAKE, 0 of 120 in VQA-RAD), so no prediction can match both the original
reference and the donor's.

Section 5.3 of the paper reports the same zero numerator over a LARGER
denominator -- 0 of 353, 0 of 268 and 0 of 13 -- because it counts over the
whole test set, the set phi is defined on, while the counts above cover only
the templates that enter this module's informative computation. Both are
correct for what they measure; src/analysis/coverage.py computes the second and
docs/COVERAGE.md records the difference. The informative sets are 5,218 = 3,936 + 1,282 in
Kvasir under C5, 92 in SLAKE under C6, and 1 in VQA-RAD.

The donor mapping and the per-question references come from the swap map
(see build_swapmap.py); a precomputed one ships as data/swapmap.json.

Usage:
    python -m src.analysis.ccr \
        --results-dir results --swapmap data/swapmap.json \
        --model SMOLVLM_LORA_KVASIR --dataset KVASIR --condition C5
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import re

from src.inference.process_text import answer_matches, normalize_text

# Conditions that substitute one image for another. C1 (blank frame) and the
# parametric corruptions C4_OCC_* / C4_NOISE_* (the paper's C3 and C4)
# (occlusion/noise) alter the original image rather than replacing it, so
# there is no donor and CCR is undefined for them.
SWAP_CONDITIONS = ("C2", "C5", "C6")


def question_key(q) -> str:
    """Index key for a question's text -- NOT an answer comparison.

    Used to ask "what is the reference for this same question on the donor
    image?", so it must stay a stable dictionary key. Deliberately does not go
    through normalize_text(): stripping punctuation would change these keys and
    could merge or split templates, which would change the size of the
    informative set. That size is a published quantity (5,218 in Kvasir under
    C5) and must not move as a side effect of the answer-scoring rule.
    """
    return re.sub(r"\s+", " ", str(q).strip().lower())


def answers_match(a, refs) -> bool:
    """True when answer `a` matches any reference in `refs`.

    The single ruler of this module, and the same one behind accuracy and VDD.
    `answer_matches` is an equivalence relation on normalized strings (its
    whitespace-squashed fallback subsumes plain equality), so the set-based
    comparisons below are well defined regardless of argument order.
    """
    return answer_matches(normalize_text(str(a)), [normalize_text(str(r)) for r in refs])


def load_swapmap(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fp:
        return json.load(fp)


def load_predictions(results_dir: str, prefix: str, condition: str) -> dict | None:
    path = os.path.join(results_dir, f"{prefix}_{condition}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fp:
        record = json.load(fp)
    return {p["qid"]: p for p in record["predictions"]}


def compute_ccr(
    c0_predictions: dict,
    swapped_predictions: dict,
    qid_meta: dict,
    donor_map: dict,
) -> dict:
    """Returns the CCR breakdown over the informative swapped questions.

    `qid_meta` maps qid -> {"img": image_id, "q": question, "a": reference}
    `donor_map` maps original image_id -> donor image_id
    """
    # question text -> reference answer, indexed by image, so we can ask
    # "what is the answer to this same question on the donor image?"
    by_image = collections.defaultdict(dict)
    for qid, meta in qid_meta.items():
        by_image[meta["img"]][question_key(meta["q"])] = meta["a"]

    counts = collections.Counter()
    eligible = [q for q in c0_predictions if q in qid_meta and qid_meta[q]["img"] in donor_map]

    for qid in eligible:
        meta = qid_meta[qid]
        donor = donor_map[meta["img"]]
        donor_reference = by_image[donor].get(question_key(meta["q"]))
        if donor_reference is None:
            # The donor image does not carry this question: not verifiable.
            continue

        original_reference = meta["a"]
        if any(answers_match(x, donor_reference) for x in original_reference):
            # Same reference on both images: the swap reveals nothing.
            continue

        counts["informative"] += 1
        predicted = swapped_predictions[qid]["predicted"]
        kept = answers_match(predicted, original_reference)
        followed = answers_match(predicted, donor_reference)
        # The two references do not match each other on an informative
        # question, so at most one of these can be true.
        if kept:
            counts["kept_original"] += 1
        elif followed:
            counts["followed_donor"] += 1
        else:
            counts["neither"] += 1

        # Restricted to questions the model got right on the intact image,
        # which isolates grounding from questions it never answered well.
        if c0_predictions[qid]["correct"]:
            counts["inf_correct_c0"] += 1
            if followed:
                counts["c0correct_followed"] += 1
            elif kept:
                counts["c0correct_kept"] += 1
            else:
                counts["c0correct_neither"] += 1

    n = counts["informative"]
    result = dict(counts)
    result["ccr"] = counts["followed_donor"] / n if n else None
    result["kept_original_rate"] = counts["kept_original"] / n if n else None
    result["neither_rate"] = counts["neither"] / n if n else None
    return result


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--swapmap", default="data/swapmap.json")
    parser.add_argument("--model", required=True,
                        help="Results file prefix, e.g. SMOLVLM_LORA_KVASIR_seed0")
    parser.add_argument("--dataset", required=True, choices=["SLAKE", "VQARAD", "KVASIR"])
    parser.add_argument("--condition", required=True, choices=SWAP_CONDITIONS)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    swapmap = load_swapmap(args.swapmap)
    entry = swapmap[args.dataset]
    qid_meta = {int(k): v for k, v in entry["qid"].items()}
    donor_map = entry["swap"][args.condition]

    c0 = load_predictions(args.results_dir, args.model, "C0")
    cx = load_predictions(args.results_dir, args.model, args.condition)
    if c0 is None or cx is None:
        raise SystemExit(f"missing predictions for {args.model} (C0 and/or {args.condition})")

    result = compute_ccr(c0, cx, qid_meta, donor_map)
    print(f"{args.model} {args.dataset} {args.condition}")
    print(f"  informative questions : {result['informative']}")
    print(f"  CCR (followed donor)  : {result['ccr']:.1%}" if result["ccr"] is not None else "  CCR: n/a")
    print(f"  kept original answer  : {result['kept_original_rate']:.1%}" if result["ccr"] is not None else "")
    print(f"  neither               : {result['neither_rate']:.1%}" if result["ccr"] is not None else "")
