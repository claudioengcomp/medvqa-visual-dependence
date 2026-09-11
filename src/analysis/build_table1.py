"""Builds Table 1 from the per-item predictions -- every cell, in one pass.

Table 1 carries, for each model and dataset: the intact accuracy, the VDD under
C1, C2, C5 and C6, the relative drop under C2, and the CCR.

Every cell comes from the module that owns the quantity:

    accuracy, VDD, McNemar   src/stats/conditional_vdd.py
    CCR                      src/analysis/ccr_slices.py

and nothing is recomputed independently, so the table cannot drift from the
quantities the rest of the paper reports.

Two rules of Section 4.3 are enforced structurally rather than by convention.
Fine-tuned configurations are summarized as mean and sample standard deviation
over their three seeds, and McNemar is computed *within* each seed and reported
as the count of seeds in which it is significant -- never pooled. C5 and C6 are
partial interventions, so their VDD is restricted to the questions whose image
actually received a donor; C1 and C2 touch every item, and there the restriction
is a no-op.

Usage:
    python -m src.analysis.build_table1                    # printed table
    python -m src.analysis.build_table1 --json table1.json # machine-readable
    python -m src.analysis.build_table1 --latex            # the table body
"""
from __future__ import annotations

import argparse
import json
import statistics

from src.analysis.ccr import load_swapmap
from src.analysis.ccr_slices import seed_prefixes
from src.stats.conditional_vdd import (FULL_COVERAGE_CONDITIONS, _load,
                                       conditional_vdd, eligible_qids)
from src.analysis import ccr_slices

CONDITIONS = ("C1", "C2", "C5", "C6")

# Rows of Table 1, in order, per dataset. The CCR is reported for the swap with
# the largest informative set of each dataset, and is not reportable on VQA-RAD.
ROWS = {
    "KVASIR": (["M3AE_KVASIR", "QWEN2VL_LORA_KVASIR", "QWEN2VL_DORA_KVASIR",
                "SMOLVLM_LORA_KVASIR", "SMOLVLM_DORA_KVASIR",
                "MEDVLTHINKER_ZEROSHOT_KVASIR", "M0_question_only_KVASIR"], "C5"),
    "SLAKE": (["M3AE_SCRATCH_SLAKE", "QWEN2VL_LORA_SLAKE", "QWEN2VL_DORA_SLAKE",
               "SMOLVLM_LORA_SLAKE", "SMOLVLM_DORA_SLAKE",
               "MEDVLTHINKER_ZEROSHOT_SLAKE", "M0_question_only_SLAKE"], "C6"),
    "VQARAD": (["M3AE_SCRATCH_VQARAD", "QWEN2VL_LORA_VQARAD", "QWEN2VL_DORA_VQARAD",
                "SMOLVLM_LORA_VQARAD", "SMOLVLM_DORA_VQARAD",
                "MEDVLTHINKER_ZEROSHOT_VQARAD", "M0_question_only_VQARAD"], None),
}


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return (None, None)
    return (statistics.mean(values),
            statistics.stdev(values) if len(values) > 1 else 0.0)


def row(base: str, dataset: str, ccr_condition: str | None,
        results_dir: str, swapmap_path: str) -> dict | None:
    entry = load_swapmap(swapmap_path)[dataset]
    qid_meta = {int(k): v for k, v in entry["qid"].items()}

    prefixes = seed_prefixes(results_dir, base, "C0") or (
        [base] if _load(results_dir, base, "C0") else [])
    if not prefixes:
        return None

    out = {"seeds": len(prefixes), "conditions": {}}
    accuracies = []
    for condition in CONDITIONS:
        if condition not in entry["swap"] and condition not in FULL_COVERAGE_CONDITIONS:
            continue
        restrict = (None if condition in FULL_COVERAGE_CONDITIONS
                    else eligible_qids(qid_meta, entry["swap"][condition]))
        vdds, significant, n = [], 0, None
        for prefix in prefixes:
            c0 = _load(results_dir, prefix, "C0")
            cx = _load(results_dir, prefix, condition)
            if c0 is None or cx is None:
                continue
            r = conditional_vdd(c0, cx, restrict)
            vdds.append(r["vdd"] * 100)
            significant += 1 if r["mcnemar"]["p_value"] < 0.05 else 0
            n = r["n"]
            if condition == CONDITIONS[0]:
                accuracies.append(r["accuracy_c0"] * 100)
        if vdds:
            out["conditions"][condition] = {
                "vdd": mean_sd(vdds), "n": n,
                "significant_seeds": significant, "seeds": len(vdds)}
    if not accuracies:                                   # C1 missing: fall back
        for prefix in prefixes:
            c0 = _load(results_dir, prefix, "C0")
            if c0:
                accuracies.append(sum(p["correct"] for p in c0.values())
                                  / len(c0) * 100)
    out["accuracy_c0"] = mean_sd(accuracies)

    c2 = out["conditions"].get("C2")
    out["relative_drop_c2"] = (c2["vdd"][0] / out["accuracy_c0"][0]
                               if c2 and out["accuracy_c0"][0] else None)

    out["ccr"] = (None, None)
    if ccr_condition:
        record = ccr_slices.run(dataset, ccr_condition, results_dir,
                                swapmap_path).get(base)
        if record:
            out["ccr"] = record["all"]["ccr"]
            out["ccr_n"] = record["all"]["n"]
    return out


def build(results_dir: str, swapmap_path: str) -> dict:
    table = {}
    for dataset, (bases, ccr_condition) in ROWS.items():
        for base in bases:
            record = row(base, dataset, ccr_condition, results_dir, swapmap_path)
            if record:
                record["dataset"] = dataset
                table[base] = record
    return table


def _cell(pair, seeds, marker=""):
    if pair[0] is None:
        return "--"
    if seeds < 2:
        return f"{pair[0]:.1f}{marker}"
    return f"{pair[0]:.1f} ± {pair[1]:.1f}{marker}"


def _marker(condition_record) -> str:
    if condition_record is None:
        return ""
    k, n = condition_record["significant_seeds"], condition_record["seeds"]
    return "‡" if k == n else ("†" if n > 1 and k == n - 1 else "")


def render(table: dict, latex: bool = False) -> str:
    lines = []
    for base, r in table.items():
        cells = [_cell(r["accuracy_c0"], r["seeds"])]
        for condition in CONDITIONS:
            c = r["conditions"].get(condition)
            cells.append("--" if c is None
                         else _cell(c["vdd"], r["seeds"], _marker(c)))
        cells.append("--" if r["relative_drop_c2"] is None
                     else f"{r['relative_drop_c2'] * 100:.0f}")
        cells.append(_cell(r["ccr"], r["seeds"]))
        if latex:
            body = " & ".join(c.replace("±", "$\\pm$").replace("‡", "$^{\\ddagger}$")
                              .replace("†", "$^{\\dagger}$") for c in cells)
            lines.append(f"{base} & {body} \\\\")
        else:
            lines.append(f"{base:<30}" + "".join(f"{c:>16}" for c in cells))
    if not latex:
        header = ("model", "Acc(C0)", "VDD(C1)", "VDD(C2)", "VDD(C5*)",
                  "VDD(C6*)", "VDD(C2)/Acc", "CCR")
        lines.insert(0, f"{header[0]:<30}" + "".join(f"{h:>16}" for h in header[1:]))
    return "\n".join(lines)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--swapmap", default="data/swapmap.json")
    parser.add_argument("--json", help="write the full record here")
    parser.add_argument("--latex", action="store_true", help="render table rows")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    table = build(args.results_dir, args.swapmap)
    print(render(table, latex=args.latex))
    if args.json:
        with open(args.json, "w") as fp:
            json.dump(table, fp, indent=1)
