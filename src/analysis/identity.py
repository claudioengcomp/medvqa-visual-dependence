"""The identity of Section 3.4, verified on the saved predictions (Section 5.1).

Under a swap the model receives an existing test item, so the CCR -- the
altered answer scored against the DONOR's reference -- is the accuracy the same
model already obtains under C0 on that donor item. This holds by construction
of the swap. It is verified rather than asserted because inference is not
bit-exact across environments (Section 5.6) and list answers admit reordering.

The module partitions the swapped questions into

    informative   the donor carries the question and the reference differs
    same_ref      the donor carries the question and the reference coincides
    novel         the donor does not carry the question

and, on the informative set, compares the prediction under the swap with the
prediction under C0 on the donor item, cell by cell (configuration x seed).

A single scoring rule: question_key and answers_match come from
src/analysis/ccr.py, the pair behind accuracy, VDD and the CCR.

Usage:
    python -m src.analysis.identity --dataset KVASIR --condition C5
    python -m src.analysis.identity --all --json identity.json
"""
from __future__ import annotations

import argparse
import json

from src.analysis.ccr import (answers_match, load_predictions, load_swapmap,
                              question_key)
from src.analysis.ccr_slices import seed_prefixes

# The cells the paper reports: the swap with the largest informative set of
# each dataset that has one.
REPORTED = (("KVASIR", "C5"), ("SLAKE", "C6"))

CONFIGURATIONS = {
    "KVASIR": ["M3AE_KVASIR", "QWEN2VL_LORA_KVASIR", "QWEN2VL_DORA_KVASIR",
               "SMOLVLM_LORA_KVASIR", "SMOLVLM_DORA_KVASIR"],
    "SLAKE": ["M3AE_SCRATCH_SLAKE", "QWEN2VL_LORA_SLAKE", "QWEN2VL_DORA_SLAKE",
              "SMOLVLM_LORA_SLAKE", "SMOLVLM_DORA_SLAKE"],
    "VQARAD": ["M3AE_SCRATCH_VQARAD", "QWEN2VL_LORA_VQARAD",
               "QWEN2VL_DORA_VQARAD", "SMOLVLM_LORA_VQARAD",
               "SMOLVLM_DORA_VQARAD"],
}


def strata(qid_meta: dict, donor_map: dict):
    """Partition the questions the swap touches, and name the donor item.

    Returns the partition plus, for every question whose donor carries it, the
    qid of that donor item -- which is what makes the identity checkable: the
    donor item is a test item with a prediction of its own under C0.
    """
    by_image, index = {}, {}
    for qid, meta in qid_meta.items():
        by_image.setdefault(meta["img"], {})[question_key(meta["q"])] = meta["a"]
        index[(meta["img"], question_key(meta["q"]))] = qid

    parts = {"informative": [], "same_ref": [], "novel": []}
    donor_qid, donor_ref = {}, {}
    for qid, meta in qid_meta.items():
        donor = donor_map.get(meta["img"])
        if donor is None:
            continue
        key = question_key(meta["q"])
        reference = by_image.get(donor, {}).get(key)
        if reference is None:
            parts["novel"].append(qid)
            continue
        donor_qid[qid] = index[(donor, key)]
        donor_ref[qid] = reference
        bucket = ("same_ref" if any(answers_match(a, reference) for a in meta["a"])
                  else "informative")
        parts[bucket].append(qid)
    return parts, donor_qid, donor_ref


def check_cell(c0: dict, swapped: dict, informative: list[int],
               donor_qid: dict, donor_ref: dict) -> dict:
    """The three quantities the identity relates, over the same items.

    agreement  how often the swapped prediction equals the C0 prediction on the
               donor item -- the identity itself, at the level of the answer
    ccr        the swapped answer scored against the DONOR's reference
    donor_acc  the accuracy the model already has under C0 on those donor items

    The identity says ccr and donor_acc are the same number.
    """
    usable = [q for q in informative if q in swapped and donor_qid[q] in c0]
    if not usable:
        return {"n": 0}
    agree = sum(1 for q in usable
                if answers_match(swapped[q]["predicted"],
                                 [c0[donor_qid[q]]["predicted"]]))
    ccr = sum(1 for q in usable
              if answers_match(swapped[q]["predicted"], donor_ref[q]))
    donor_acc = sum(1 for q in usable if c0[donor_qid[q]]["correct"])
    n = len(usable)
    return {"n": n, "agreement": agree / n, "ccr": ccr / n,
            "donor_accuracy": donor_acc / n,
            "difference_pp": (ccr - donor_acc) / n * 100}


def run(dataset: str, condition: str, results_dir: str, swapmap_path: str) -> dict:
    entry = load_swapmap(swapmap_path)[dataset]
    qid_meta = {int(k): v for k, v in entry["qid"].items()}
    if condition not in entry["swap"]:
        raise SystemExit(f"{dataset} has no donor map for {condition}")
    parts, donor_qid, donor_ref = strata(qid_meta, entry["swap"][condition])

    out = {"dataset": dataset, "condition": condition,
           "informative": len(parts["informative"]),
           "same_ref": len(parts["same_ref"]), "novel": len(parts["novel"]),
           "cells": {}}
    for base in CONFIGURATIONS[dataset]:
        for prefix in seed_prefixes(results_dir, base, condition):
            c0 = load_predictions(results_dir, prefix, "C0")
            swapped = load_predictions(results_dir, prefix, condition)
            if c0 is None or swapped is None:
                continue
            out["cells"][prefix] = check_cell(
                c0, swapped, parts["informative"], donor_qid, donor_ref)
    return out


def summarize(records: list[dict]) -> dict:
    cells = [(name, c) for r in records for name, c in r["cells"].items() if c["n"]]
    exact = [n for n, c in cells
             if c["agreement"] == 1.0 and abs(c["difference_pp"]) < 1e-9]
    return {"cells": len(cells), "exact": len(exact),
            "exceptions": sorted(
                ({"cell": n, "agreement": c["agreement"],
                  "difference_pp": c["difference_pp"]}
                 for n, c in cells if n not in set(exact)),
                key=lambda d: d["agreement"])}


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--swapmap", default="data/swapmap.json")
    p.add_argument("--dataset")
    p.add_argument("--condition")
    p.add_argument("--all", action="store_true", help="the cells the paper reports")
    p.add_argument("--json")
    return p.parse_args()


def main():
    args = _parse_args()
    pairs = REPORTED if args.all or not args.dataset else [(args.dataset,
                                                            args.condition)]
    records = []
    for dataset, condition in pairs:
        r = run(dataset, condition, args.results_dir, args.swapmap)
        records.append(r)
        print(f"\n=== {dataset} {condition}  "
              f"informative {r['informative']}, same reference {r['same_ref']}, "
              f"novel pairing {r['novel']}")
        print(f"{'cell':<30}{'n':>7}{'agreement':>12}{'CCR':>9}"
              f"{'donor Acc(C0)':>15}{'diff (pp)':>11}")
        for name, c in r["cells"].items():
            if not c["n"]:
                continue
            print(f"{name:<30}{c['n']:>7}{c['agreement'] * 100:>11.2f}%"
                  f"{c['ccr'] * 100:>8.1f}%{c['donor_accuracy'] * 100:>14.1f}%"
                  f"{c['difference_pp']:>11.3f}")

    s = summarize(records)
    print(f"\nthe identity holds exactly in {s['exact']} of {s['cells']} "
          f"configuration-seed cells")
    for e in s["exceptions"]:
        print(f"  exception: {e['cell']}  agreement {e['agreement'] * 100:.2f}%"
              f"  difference {e['difference_pp']:.2f} pp")

    if args.json:
        with open(args.json, "w") as fp:
            json.dump({"records": records, "summary": s}, fp, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
