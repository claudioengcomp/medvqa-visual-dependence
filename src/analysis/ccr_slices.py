"""CCR by answer slice -- the numbers reported in Section 5.4.

The CCR is reported three ways: over all informative questions of a swap, and
over the two slices that partition them.

    single-valued -- neither the original nor the donor reference is a list
    list          -- either reference is a list of locations or colours,
                     written with ';' separators

The split matters because a list answer is a harder target: the model must
produce the same set of items, and the paper reports the slices separately for
that reason. Both slices, and the aggregate, are scored with `answers_match`
from ccr.py -- one ruler, the same one behind accuracy and VDD.

Aggregation follows Section 4.3: mean and sample standard deviation over
training seeds; seeds are never pooled.

Usage:
    python -m src.analysis.ccr_slices --dataset KVASIR --condition C5
    python -m src.analysis.ccr_slices --dataset SLAKE --condition C6 --json out.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics

from src.analysis.ccr import (SWAP_CONDITIONS, answers_match, load_predictions,
                              load_swapmap, question_key)

# Fine-tuned configurations per dataset, in the order of Table 1, plus the
# zero-shot model. Seed suffixes are discovered: Kvasir writes seed0 explicitly,
# SLAKE and VQA-RAD leave the first seed unsuffixed.
BASES = {
    "KVASIR": ["M3AE_KVASIR", "QWEN2VL_LORA_KVASIR", "QWEN2VL_DORA_KVASIR",
               "SMOLVLM_LORA_KVASIR", "SMOLVLM_DORA_KVASIR",
               "MEDVLTHINKER_ZEROSHOT_KVASIR"],
    "SLAKE": ["M3AE_SCRATCH_SLAKE", "QWEN2VL_LORA_SLAKE", "QWEN2VL_DORA_SLAKE",
              "SMOLVLM_LORA_SLAKE", "SMOLVLM_DORA_SLAKE",
              "MEDVLTHINKER_ZEROSHOT_SLAKE"],
    "VQARAD": ["M3AE_SCRATCH_VQARAD", "QWEN2VL_LORA_VQARAD", "QWEN2VL_DORA_VQARAD",
               "SMOLVLM_LORA_VQARAD", "SMOLVLM_DORA_VQARAD",
               "MEDVLTHINKER_ZEROSHOT_VQARAD"],
}


def is_list_answer(*references) -> bool:
    """A question falls in the list slice when either reference is a list."""
    return any(";" in str(r) for refs in references for r in refs)


def seed_prefixes(results_dir: str, base: str, condition: str) -> list[str]:
    """Prefixes of the runs of `base` that have both C0 and `condition`."""
    candidates = [base] + [f"{base}_seed{s}" for s in (0, 1, 2)]
    return [p for p in candidates
            if os.path.exists(os.path.join(results_dir, f"{p}_C0.json"))
            and os.path.exists(os.path.join(results_dir, f"{p}_{condition}.json"))]


def slice_counts(c0_predictions, swapped_predictions, qid_meta, donor_map) -> dict:
    """Bucket counts for the aggregate and for each slice, in one pass."""
    by_image = collections.defaultdict(dict)
    for meta in qid_meta.values():
        by_image[meta["img"]][question_key(meta["q"])] = meta["a"]

    counts = collections.Counter()
    for qid, prediction in c0_predictions.items():
        meta = qid_meta.get(qid)
        if meta is None or meta["img"] not in donor_map or qid not in swapped_predictions:
            continue
        donor_reference = by_image[donor_map[meta["img"]]].get(question_key(meta["q"]))
        if donor_reference is None:
            continue
        original_reference = meta["a"]
        if any(answers_match(x, donor_reference) for x in original_reference):
            continue                              # uninformative: same reference

        predicted = swapped_predictions[qid]["predicted"]
        followed = answers_match(predicted, donor_reference)
        kept = answers_match(predicted, original_reference)
        bucket = "followed" if followed else ("kept" if kept else "neither")
        for key in ("all", "list" if is_list_answer(original_reference, donor_reference)
                    else "single"):
            counts[f"{key}_n"] += 1
            counts[f"{key}_{bucket}"] += 1
            if prediction["correct"]:
                counts[f"{key}_c0"] += 1
                counts[f"{key}_c0_followed"] += followed
    return dict(counts)


def rate(counts: dict, slice_name: str, bucket: str = "followed"):
    n = counts.get(f"{slice_name}_n", 0)
    return counts.get(f"{slice_name}_{bucket}", 0) / n * 100 if n else None


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return (None, None)
    return (statistics.mean(values),
            statistics.stdev(values) if len(values) > 1 else 0.0)


def run(dataset: str, condition: str, results_dir: str, swapmap_path: str) -> dict:
    entry = load_swapmap(swapmap_path)[dataset]
    qid_meta = {int(k): v for k, v in entry["qid"].items()}
    donor_map = entry["swap"][condition]

    out = {}
    for base in BASES[dataset]:
        prefixes = seed_prefixes(results_dir, base, condition)
        if not prefixes:
            continue
        per_seed = []
        for prefix in prefixes:
            per_seed.append(slice_counts(
                load_predictions(results_dir, prefix, "C0"),
                load_predictions(results_dir, prefix, condition),
                qid_meta, donor_map))
        record = {"seeds": prefixes, "n_seeds": len(prefixes)}
        for slice_name in ("all", "single", "list"):
            record[slice_name] = {
                "n": per_seed[0].get(f"{slice_name}_n", 0),
                "ccr": mean_sd([rate(c, slice_name) for c in per_seed]),
                "kept": mean_sd([rate(c, slice_name, "kept") for c in per_seed]),
                "neither": mean_sd([rate(c, slice_name, "neither") for c in per_seed]),
                "c0_followed": mean_sd([
                    c.get(f"{slice_name}_c0_followed", 0) / c[f"{slice_name}_c0"] * 100
                    if c.get(f"{slice_name}_c0") else None for c in per_seed]),
            }
        out[base] = record
    return out


def _fmt(pair):
    return "-" if pair[0] is None else f"{pair[0]:5.1f} ± {pair[1]:.1f}"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--swapmap", default="data/swapmap.json")
    parser.add_argument("--dataset", required=True, choices=sorted(BASES))
    parser.add_argument("--condition", required=True, choices=SWAP_CONDITIONS)
    parser.add_argument("--json", help="also write the full record here")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run(args.dataset, args.condition, args.results_dir, args.swapmap)
    first = next(iter(result.values()))
    print(f"{args.dataset} {args.condition} - CCR, mean ± sd over seeds "
          f"(n per seed: all={first['all']['n']}, "
          f"single={first['single']['n']}, list={first['list']['n']})")
    print(f"{'configuration':<30}{'seeds':>6}{'all':>14}{'single':>14}{'list':>14}"
          f"{'kept (all)':>14}")
    for base, record in result.items():
        print(f"{base:<30}{record['n_seeds']:>6}"
              f"{_fmt(record['all']['ccr']):>14}{_fmt(record['single']['ccr']):>14}"
              f"{_fmt(record['list']['ccr']):>14}{_fmt(record['all']['kept']):>14}")
    if args.json:
        with open(args.json, "w") as fp:
            json.dump(result, fp, indent=1)
