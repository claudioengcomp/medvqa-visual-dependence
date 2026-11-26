"""VDD on the subsets of the restricted swaps -- the numbers of Section 5.3.

Section 5.3 reads the cost of a swap against how far the donor is and how often
the answer actually changes. Three subsets carry that argument:

    same        the 240 C5 pairs whose donor stays within the procedure
    cross       the 369 C5 pairs that cross gastroscopy and colonoscopy
    c6_only     the images that enter C6 but not C5, because they carry no
                eligible finding label -- the swap that changes the least

For each, three quantities: the VDD under the swap, the VDD under the blank
frame *on the same questions* (so the comparison is paired, not against a
different denominator), and the fraction of verifiable answers the swap
actually changes. The ratio of the first to the third is the cost per changed
answer, which is what makes the subsets comparable to each other.

"Verifiable" means the donor image carries the same question, so its reference
answer is known; "changed" means that reference differs from the original's,
scored with `answers_match` from ccr.py -- the same ruler as everywhere else.

Usage:
    python -m src.analysis.subset_vdd
    python -m src.analysis.subset_vdd --json subsets.json
"""
from __future__ import annotations

import argparse
import json

from src.analysis.ccr import answers_match, load_swapmap, question_key
from src.analysis.ccr_sameproc import procedure_by_image, split_donor_map
from src.analysis.ccr_slices import BASES, mean_sd, seed_prefixes
from src.stats.conditional_vdd import _load, conditional_vdd

DATASET = "KVASIR"


def changed_fraction(qid_meta: dict, donor_map: dict) -> tuple[int, int]:
    """(questions whose reference changes, questions whose donor carries them)."""
    by_image = {}
    for meta in qid_meta.values():
        by_image.setdefault(meta["img"], {})[question_key(meta["q"])] = meta["a"]
    verifiable = changed = 0
    for meta in qid_meta.values():
        if meta["img"] not in donor_map:
            continue
        donor_reference = by_image[donor_map[meta["img"]]].get(question_key(meta["q"]))
        if donor_reference is None:
            continue
        verifiable += 1
        if not any(answers_match(x, donor_reference) for x in meta["a"]):
            changed += 1
    return changed, verifiable


def subsets(qid_meta: dict, swap: dict) -> dict[str, dict]:
    """The three donor maps of Section 5.3, plus the questions each touches."""
    parts = split_donor_map(swap["C5"], procedure_by_image(qid_meta))
    c6_only = {image: donor for image, donor in swap["C6"].items()
               if image not in swap["C5"]}
    return {"same": parts["same"], "cross": parts["cross"], "c6_only": c6_only}


def run(results_dir: str, swapmap_path: str) -> dict:
    entry = load_swapmap(swapmap_path)[DATASET]
    qid_meta = {int(k): v for k, v in entry["qid"].items()}
    out = {}
    for name, donor_map in subsets(qid_meta, entry["swap"]).items():
        condition = "C6" if name == "c6_only" else "C5"
        qids = {qid for qid, meta in qid_meta.items() if meta["img"] in donor_map}
        changed, verifiable = changed_fraction(qid_meta, donor_map)
        record = {"pairs": len(donor_map), "questions": len(qids),
                  "verifiable": verifiable, "changed": changed,
                  "changed_fraction": changed / verifiable if verifiable else None,
                  "condition": condition, "models": {}}
        for base in BASES[DATASET]:
            prefixes = seed_prefixes(results_dir, base, condition)
            if not prefixes:
                continue
            swap_vdds, blank_vdds = [], []
            for prefix in prefixes:
                c0 = _load(results_dir, prefix, "C0")
                cx = _load(results_dir, prefix, condition)
                c1 = _load(results_dir, prefix, "C1")
                if c0 is None or cx is None:
                    continue
                swap_vdds.append(conditional_vdd(c0, cx, qids)["vdd"] * 100)
                if c1 is not None:
                    blank_vdds.append(conditional_vdd(c0, c1, qids)["vdd"] * 100)
            if swap_vdds:
                swap_mean = mean_sd(swap_vdds)
                record["models"][base] = {
                    "vdd_swap": swap_mean,
                    "vdd_blank_same_questions": mean_sd(blank_vdds),
                    "cost_per_changed_answer": (
                        swap_mean[0] / 100 / record["changed_fraction"]
                        if record["changed_fraction"] else None),
                }
        out[name] = record
    return out


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--swapmap", default="data/swapmap.json")
    parser.add_argument("--json")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run(args.results_dir, args.swapmap)
    for name, record in result.items():
        print(f"\n{name}: {record['pairs']} pairs, {record['questions']} questions, "
              f"{record['changed']} of {record['verifiable']} verifiable answers "
              f"change ({record['changed_fraction']:.1%}) - {record['condition']}")
        print(f"  {'configuration':<30}{'VDD swap':>10}{'VDD blank':>12}{'ratio':>8}")
        for base, m in record["models"].items():
            blank = ("--" if m["vdd_blank_same_questions"][0] is None
                     else f"{m['vdd_blank_same_questions'][0]:.1f}")
            print(f"  {base:<30}{m['vdd_swap'][0]:>10.1f}{blank:>12}"
                  f"{m['cost_per_changed_answer']:>8.2f}")
    if args.json:
        with open(args.json, "w") as fp:
            json.dump(result, fp, indent=1)
