"""Accuracy and VDD split by question type -- the numbers of Section 5.2.

Closed questions have a small, fixed answer set (yes/no and the like); open
questions do not. The split matters twice in the paper: the question-only
baseline is trained and evaluated on closed questions alone, so any comparison
against that floor has to be made on closed questions, and open questions lose
more than closed ones under image removal.

The label is the benchmarks' own, not a heuristic. It travels in the swap map
as the `t` field of each question (0 closed, 1 open) and reproduces the counts
the paper reports: 416 closed and 645 open on SLAKE, 272 and 179 on VQA-RAD.
Kvasir-VQA-x1 has categorical answers throughout and is treated as closed.

This is *not* the same thing as src/analysis/answer_category.py, which buckets
predictions by the shape of the gold answer to report per-category accuracy.
That module's proxy answers a different question and must not be substituted
for the label here when reproducing Section 5.2.

Usage:
    python -m src.analysis.question_type --dataset SLAKE --condition C1
"""
from __future__ import annotations

import argparse
import json

from src.analysis.ccr import load_swapmap
from src.analysis.ccr_slices import BASES, mean_sd, seed_prefixes
from src.stats.conditional_vdd import _load, conditional_vdd

CLOSED, OPEN = 0, 1


def qids_by_type(qid_meta: dict) -> dict[int, set[int]]:
    """{0: closed qids, 1: open qids}, from the benchmarks' own label."""
    out = {CLOSED: set(), OPEN: set()}
    for qid, meta in qid_meta.items():
        out[int(meta["t"])].add(qid)
    return out


def run(dataset: str, condition: str, results_dir: str, swapmap_path: str) -> dict:
    entry = load_swapmap(swapmap_path)[dataset]
    qid_meta = {int(k): v for k, v in entry["qid"].items()}
    subsets = qids_by_type(qid_meta)

    out = {}
    for base in BASES[dataset] + [f"M0_question_only_{dataset}"]:
        prefixes = seed_prefixes(results_dir, base, condition)
        if not prefixes:
            continue
        record = {"seeds": len(prefixes)}
        for name, qids in (("closed", subsets[CLOSED]), ("open", subsets[OPEN])):
            if not qids:
                continue
            accuracies_c0, accuracies_cx, vdds = [], [], []
            for prefix in prefixes:
                c0 = _load(results_dir, prefix, "C0")
                cx = _load(results_dir, prefix, condition)
                if c0 is None or cx is None:
                    continue
                if not qids & set(c0) & set(cx):
                    # The question-only baseline answers closed questions only,
                    # so it has no predictions at all on the open subset.
                    continue
                r = conditional_vdd(c0, cx, qids)
                accuracies_c0.append(r["accuracy_c0"] * 100)
                accuracies_cx.append(r["accuracy_other"] * 100)
                vdds.append(r["vdd"] * 100)
            if vdds:
                record[name] = {"n": len(qids),
                                "accuracy_c0": mean_sd(accuracies_c0),
                                "accuracy_condition": mean_sd(accuracies_cx),
                                "vdd": mean_sd(vdds)}
        if len(record) > 1:
            out[base] = record
    return out


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--swapmap", default="data/swapmap.json")
    parser.add_argument("--dataset", required=True, choices=sorted(BASES))
    parser.add_argument("--condition", default="C1")
    parser.add_argument("--json")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run(args.dataset, args.condition, args.results_dir, args.swapmap)
    print(f"{args.dataset} {args.condition} — accuracy and VDD by question type")
    print(f"{'configuration':<30}{'closed C0':>12}{'closed cond':>13}{'VDD':>8}"
          f"{'open C0':>12}{'open cond':>12}{'VDD':>8}")
    for base, record in result.items():
        cells = []
        for name in ("closed", "open"):
            r = record.get(name)
            cells += (["--"] * 3 if r is None else
                      [f"{r['accuracy_c0'][0]:.1f}", f"{r['accuracy_condition'][0]:.1f}",
                       f"{r['vdd'][0]:.1f}"])
        print(f"{base:<30}{cells[0]:>12}{cells[1]:>13}{cells[2]:>8}"
              f"{cells[3]:>12}{cells[4]:>12}{cells[5]:>8}")
    if args.json:
        with open(args.json, "w") as fp:
            json.dump(result, fp, indent=1)
