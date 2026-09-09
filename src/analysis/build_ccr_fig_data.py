"""Rebuild the `ccr` block of data/fig_data.json -- the input of Figure 5.

Figure 5 shows, for every swapped question whose original and donor references
differ, the fraction of answers that match the donor's reference (the CCR), keep
the original one, or are neither. Those three fractions are recomputed here with
`answers_match` from ccr.py, so the figure and the table cannot drift apart.

Only the `ccr` block is touched; src/analysis/build_fig_data.py rebuilds the
whole file, including this block.

Counts are pooled over seeds, which is how the figure's own data has always been
laid out (its subtitle divides by the seed count to display n per seed). Pooling
is safe for these ratios because every seed carries the same denominator; the
per-seed dispersion that Section 4.3 requires is reported by ccr_slices.py,
which never pools.

Usage:
    python -m src.analysis.build_ccr_fig_data              # rewrites in place
    python -m src.analysis.build_ccr_fig_data --dry-run    # prints a diff
"""
from __future__ import annotations

import argparse
import collections
import json

from src.analysis.ccr import load_predictions, load_swapmap
from src.analysis.ccr_slices import (mean_sd, rate, seed_prefixes, slice_counts)
from src.analysis.ccr_sameproc import procedure_by_image, split_donor_map

DATASETS = ("KVASIR", "SLAKE", "VQARAD")
CONDITIONS = ("C2", "C5", "C6")


def dataset_of(base: str) -> str | None:
    return next((d for d in DATASETS if base.endswith(d)), None)


def pooled(base: str, dataset: str, condition: str, results_dir: str,
           qid_meta: dict, donor_map: dict) -> dict | None:
    """Bucket fractions over the seeds pooled together, plus the pooled n."""
    prefixes = seed_prefixes(results_dir, base, condition)
    if not prefixes:
        return None
    total = collections.Counter()
    for prefix in prefixes:
        total.update(slice_counts(load_predictions(results_dir, prefix, "C0"),
                                  load_predictions(results_dir, prefix, condition),
                                  qid_meta, donor_map))
    n = total["all_n"]
    if not n:
        return None
    return {"followed_donor": total["all_followed"] / n,
            "kept_original": total["all_kept"] / n,
            "neither": total["all_neither"] / n,
            "n": n}


def subset_entry(base: str, results_dir: str, qid_meta: dict, donor_map: dict):
    """[mean rate, sample sd over seeds, n per seed] for a C5 subset."""
    prefixes = seed_prefixes(results_dir, base, "C5")
    if not prefixes:
        return None
    per_seed = [slice_counts(load_predictions(results_dir, p, "C0"),
                             load_predictions(results_dir, p, "C5"),
                             qid_meta, donor_map) for p in prefixes]
    mean, sd = mean_sd([rate(c, "all") for c in per_seed])
    if mean is None:
        return None
    return [mean / 100, (sd or 0.0) / 100, per_seed[0]["all_n"]]


def build(existing_keys, results_dir: str, swapmap_path: str) -> dict:
    swapmap = load_swapmap(swapmap_path)
    meta_cache, out = {}, {}
    for base in existing_keys:
        dataset = dataset_of(base)
        if dataset is None:
            continue
        if dataset not in meta_cache:
            entry = swapmap[dataset]
            meta_cache[dataset] = ({int(k): v for k, v in entry["qid"].items()},
                                   entry["swap"])
        qid_meta, swap = meta_cache[dataset]

        record = {}
        if dataset == "KVASIR" and "C5" in swap:
            parts = split_donor_map(swap["C5"], procedure_by_image(qid_meta))
            for name in ("same", "cross"):
                value = subset_entry(base, results_dir, qid_meta, parts[name])
                if value:
                    record[f"C5_{name}"] = value
        for condition in CONDITIONS:
            if condition not in swap:
                continue
            value = pooled(base, dataset, condition, results_dir, qid_meta,
                           swap[condition])
            if value:
                record[condition] = value
        if record:
            out[base] = record
    return out


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--swapmap", default="data/swapmap.json")
    parser.add_argument("--fig-data", default="data/fig_data.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.fig_data) as fp:
        fig_data = json.load(fp)
    old = fig_data["ccr"]
    new = build(list(old), args.results_dir, args.swapmap)

    for base in sorted(set(old) | set(new)):
        for condition in CONDITIONS:
            a = old.get(base, {}).get(condition)
            b = new.get(base, {}).get(condition)
            if a and b and abs(a["followed_donor"] - b["followed_donor"]) > 5e-5:
                print(f"{base:<30} {condition}: CCR "
                      f"{a['followed_donor']:.4f} -> {b['followed_donor']:.4f}"
                      f"  (n {a['n']} -> {b['n']})")
    missing = set(old) - set(new)
    if missing:
        print(f"NOT REBUILT (kept as they were): {sorted(missing)}")
        for base in missing:
            new[base] = old[base]

    if args.dry_run:
        raise SystemExit(0)
    fig_data["ccr"] = new
    with open(args.fig_data, "w") as fp:
        json.dump(fig_data, fp, indent=1)
    print(f"wrote {args.fig_data}")
