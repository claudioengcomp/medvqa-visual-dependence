"""CCR on the same-procedure subset of Kvasir-VQA-x1 -- Section 5.3/5.4.

C5 pairs each image with a donor that keeps the imaging domain and changes the
clinical finding, but "endoscopy" spans two procedures. Splitting the C5 pairs
by whether the donor stays within the same procedure gives a graded distance:
the same-procedure pairs are the nearest donors available.

The procedure label is not a separate artifact. Kvasir-VQA-x1 carries it as an
ordinary question -- "What type of procedure is the image taken from?" -- whose
answer is colonoscopy or gastroscopy, so it is read straight out of the swap
map. It covers 776 of the 1,291 test images; the 8 C5 pairs where either side
lacks the label are excluded from both subsets rather than guessed.

Answers are scored with `answers_match` from ccr.py: one ruler, the same one
behind accuracy and VDD.

Usage:
    python -m src.analysis.ccr_sameproc --subset same
    python -m src.analysis.ccr_sameproc --subset cross --json out.json
"""
from __future__ import annotations

import argparse
import json
import re

from src.analysis.ccr import load_predictions, load_swapmap
from src.analysis.ccr_slices import BASES, mean_sd, rate, seed_prefixes, slice_counts

PROCEDURE_QUESTION = "what type of procedure is the image taken from?"


def procedure_by_image(qid_meta: dict) -> dict[str, str]:
    """image id -> colonoscopy | gastroscopy, for the images that carry it."""
    out = {}
    for meta in qid_meta.values():
        if re.sub(r"\s+", " ", str(meta["q"]).strip().lower()) == PROCEDURE_QUESTION:
            out[meta["img"]] = str(meta["a"][0]).strip().lower()
    return out


def split_donor_map(donor_map: dict, procedure: dict) -> dict[str, dict]:
    """Partition the C5 donor map into same-procedure and cross-procedure.

    Pairs where either side has no procedure label go to neither subset.
    """
    same, cross, unlabelled = {}, {}, {}
    for image, donor in donor_map.items():
        if image not in procedure or donor not in procedure:
            unlabelled[image] = donor
        elif procedure[image] == procedure[donor]:
            same[image] = donor
        else:
            cross[image] = donor
    return {"same": same, "cross": cross, "unlabelled": unlabelled}


def run(subset: str, results_dir: str, swapmap_path: str, condition: str = "C5") -> dict:
    entry = load_swapmap(swapmap_path)["KVASIR"]
    qid_meta = {int(k): v for k, v in entry["qid"].items()}
    parts = split_donor_map(entry["swap"][condition], procedure_by_image(qid_meta))
    donor_map = parts[subset]

    out = {"_pairs": {k: len(v) for k, v in parts.items()}}
    for base in BASES["KVASIR"]:
        prefixes = seed_prefixes(results_dir, base, condition)
        if not prefixes:
            continue
        per_seed = [slice_counts(load_predictions(results_dir, p, "C0"),
                                 load_predictions(results_dir, p, condition),
                                 qid_meta, donor_map) for p in prefixes]
        out[base] = {
            "seeds": prefixes,
            "n_seeds": len(prefixes),
            "n": per_seed[0].get("all_n", 0),
            "ccr": mean_sd([rate(c, "all") for c in per_seed]),
            "kept": mean_sd([rate(c, "all", "kept") for c in per_seed]),
            "neither": mean_sd([rate(c, "all", "neither") for c in per_seed]),
        }
    return out


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--swapmap", default="data/swapmap.json")
    parser.add_argument("--subset", default="same", choices=["same", "cross"])
    parser.add_argument("--json")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run(args.subset, args.results_dir, args.swapmap)
    pairs = result.pop("_pairs")
    print(f"Kvasir-VQA-x1 C5, {args.subset}-procedure subset "
          f"({pairs[args.subset]} of {sum(pairs.values())} pairs; "
          f"{pairs['unlabelled']} pairs unlabelled and excluded)")
    for base, record in result.items():
        ccr, sd = record["ccr"]
        print(f"  {base:<30} n={record['n']:>5}  CCR={ccr:5.1f} ± {sd:.1f}"
              f"  kept={record['kept'][0]:4.1f}")
    if args.json:
        with open(args.json, "w") as fp:
            json.dump({"pairs": pairs, **result}, fp, indent=1)
