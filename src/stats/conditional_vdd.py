"""VDD restricted to the questions an intervention actually touches (Section 3.3).

This is the quantity reported in Table 1 for the restricted swaps, and it is
NOT the same as the plain accuracy difference over the whole test set.

    VDD(T) = Acc_{S_T}(C0) - Acc_{S_T}(T)

where S_T is the subset of questions whose image the intervention actually
replaced. C1, C2 and the parametric corruptions (C4_OCC_*, C4_NOISE_* -- the
paper's C3 and C4) act on every test item, so S_T = D and the restriction
is a no-op. C5 and C6 are partial by construction: images without an eligible
finding label keep their original image, and a question whose image did not
change produces the same prediction under C0 and under T.

Including those untouched questions does not add information -- it simply
multiplies the estimate by the coverage |S_T|/N. On Kvasir-VQA-x1, C5 touches
8,940 of the 10,779 questions, so the unrestricted number is 0.829 times the
restricted one: 38.5 instead of 46.4 percentage points for the classifier.
Both describe the same predictions; only the restricted one answers "how much
does this model lose on the questions this intervention is about".

The eligible subset is derived from the swap map: a question belongs to S_T
when its image received a donor.

Usage:
    python -m src.stats.conditional_vdd \
        --model M3AE_KVASIR --dataset KVASIR --condition C5 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

from src.analysis.ccr import load_swapmap
from src.stats.compute_vdd import mcnemar_test

# Interventions that act on every test item: for these the restriction is a
# no-op and the result equals the plain accuracy difference.
FULL_COVERAGE_CONDITIONS = ("C1", "C2", "C4_OCC_25", "C4_OCC_50", "C4_OCC_75",
                            "C4_NOISE_25", "C4_NOISE_50", "C4_NOISE_75")


def _load(results_dir: str, prefix: str, condition: str) -> dict | None:
    path = os.path.join(results_dir, f"{prefix}_{condition}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fp:
        record = json.load(fp)
    return {p["qid"]: p for p in record["predictions"]}


def eligible_qids(qid_meta: dict, donor_map: dict) -> set[int]:
    """Questions whose image actually received a donor."""
    return {qid for qid, meta in qid_meta.items() if meta["img"] in donor_map}


def conditional_vdd(c0: dict, cx: dict, restrict_to: set[int] | None = None) -> dict:
    """VDD, McNemar and Cohen's g over `restrict_to` (all shared qids if None)."""
    shared = [q for q in c0 if q in cx]
    if restrict_to is not None:
        shared = [q for q in shared if q in restrict_to]
    if not shared:
        raise ValueError("no shared questions after restriction")

    c0_correct = [c0[q]["correct"] for q in shared]
    cx_correct = [cx[q]["correct"] for q in shared]
    acc_c0 = sum(c0_correct) / len(shared)
    acc_cx = sum(cx_correct) / len(shared)

    result = {
        "n": len(shared),
        "accuracy_c0": acc_c0,
        "accuracy_other": acc_cx,
        "vdd": acc_c0 - acc_cx,
        "mcnemar": mcnemar_test(c0_correct, cx_correct),
    }
    return result


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--swapmap", default="data/swapmap.json")
    parser.add_argument("--model", required=True,
                        help="Results prefix without the seed, e.g. M3AE_KVASIR")
    parser.add_argument("--dataset", required=True, choices=["SLAKE", "VQARAD", "KVASIR"])
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seeds", nargs="*", default=["0", "1", "2"],
                        help="Seed suffixes to aggregate; pass none for a model without seeds")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    restrict = None
    if args.condition not in FULL_COVERAGE_CONDITIONS:
        entry = load_swapmap(args.swapmap)[args.dataset]
        qid_meta = {int(k): v for k, v in entry["qid"].items()}
        donor_map = entry["swap"][args.condition]
        restrict = eligible_qids(qid_meta, donor_map)

    vdds, accs, significant, ns = [], [], 0, set()
    for seed in (args.seeds or [None]):
        prefix = args.model if seed is None else f"{args.model}_seed{seed}"
        c0 = _load(args.results_dir, prefix, "C0")
        cx = _load(args.results_dir, prefix, args.condition)
        if c0 is None or cx is None:
            print(f"  skipping {prefix}: missing C0 and/or {args.condition}")
            continue
        r = conditional_vdd(c0, cx, restrict)
        vdds.append(r["vdd"]); accs.append(r["accuracy_c0"]); ns.add(r["n"])
        significant += 1 if r["mcnemar"]["p_value"] < 0.05 else 0

    if not vdds:
        raise SystemExit("no seeds could be evaluated")

    def ms(v):
        return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)

    acc_m, acc_s = ms(accs)
    vdd_m, vdd_s = ms(vdds)
    coverage = f"{sorted(ns)[0]}" if len(ns) == 1 else f"{sorted(ns)}"
    print(f"{args.model} {args.dataset} {args.condition}")
    print(f"  questions in S_T : {coverage}")
    print(f"  seeds            : {len(vdds)}")
    print(f"  Acc(C0)          : {acc_m*100:.1f} +/- {acc_s*100:.1f}")
    print(f"  VDD              : {vdd_m*100:.1f} +/- {vdd_s*100:.1f}")
    print(f"  McNemar p<0.05   : {significant}/{len(vdds)} seeds")
