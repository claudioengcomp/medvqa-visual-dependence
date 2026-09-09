"""Rebuilds data/fig_data.json in full -- the input of every figure.

`make_figs.py` draws Figures 2 to 5 from an aggregate that, until now, had no
generator: the figures shipped re-plottable but not recomputable. This closes
that, so a reader can go from the per-item predictions to the figures without
trusting an intermediate file.

Four blocks, each delegated to the module that owns the quantity:

    combos    accuracy and VDD per configuration, whole set, eligible subsets,
              and closed questions  -- stats/conditional_vdd, analysis/question_type
    fvsvdd    cost of a swap against the fraction of answers it changes,
              per subset            -- analysis/subset_vdd
    table1    every cell of Table 1 -- analysis/build_table1
    ccr       the three answer buckets -- analysis/build_ccr_fig_data

`floor` is kept as the empty object it already is: Figure 3 reads the
question-only floor out of `combos`, not from a block of its own.

Values are fractions, not percentages, matching what make_figs.py expects.

Usage:
    python -m src.analysis.build_fig_data                  # rewrites in place
    python -m src.analysis.build_fig_data --compare        # diff against the file
"""
from __future__ import annotations

import argparse
import json

from src.analysis import build_ccr_fig_data, build_table1, question_type, subset_vdd
from src.analysis.ccr import load_swapmap
from src.analysis.ccr_slices import BASES, mean_sd, seed_prefixes
from src.stats.conditional_vdd import _load, conditional_vdd, eligible_qids

DATASETS = ("SLAKE", "VQARAD", "KVASIR")
MIN_VERIFIABLE = 4          # below this a "fraction of answers changed" means nothing
# (key in `combos`, condition to score, subset to restrict to)
COMBO_CELLS = (("C1", "C1", None), ("C2", "C2", None),
               ("C5e", "C5", "C5"), ("C6e", "C6", "C6"),
               ("C1atC5", "C1", "C5"), ("C1atC6", "C1", "C6"))


def _bases(dataset):
    return BASES[dataset] + [f"M0_question_only_{dataset}"]


def combos_block(results_dir: str, swapmap_path: str) -> dict:
    swapmap = load_swapmap(swapmap_path)
    out = {}
    for dataset in DATASETS:
        entry = swapmap[dataset]
        qid_meta = {int(k): v for k, v in entry["qid"].items()}
        closed = question_type.qids_by_type(qid_meta)[question_type.CLOSED]
        for base in _bases(dataset):
            prefixes = seed_prefixes(results_dir, base, "C0")
            if not prefixes:
                continue
            record = {"seeds": len(prefixes)}
            accuracies, accuracies_closed = [], []
            for prefix in prefixes:
                c0 = _load(results_dir, prefix, "C0")
                accuracies.append(sum(p["correct"] for p in c0.values()) / len(c0))
                shared = closed & set(c0)
                if shared:
                    accuracies_closed.append(
                        sum(c0[q]["correct"] for q in shared) / len(shared))
            record["acc0"] = mean_sd(accuracies)
            record["acc0_closed"] = mean_sd(accuracies_closed)

            for key, condition, restrict_to in COMBO_CELLS:
                if restrict_to and restrict_to not in entry["swap"]:
                    record[key] = None       # the intervention does not exist here
                    continue
                restrict = (eligible_qids(qid_meta, entry["swap"][restrict_to])
                            if restrict_to else None)
                values = []
                for prefix in prefixes:
                    c0 = _load(results_dir, prefix, "C0")
                    cx = _load(results_dir, prefix, condition)
                    if c0 is None or cx is None:
                        continue
                    values.append(conditional_vdd(c0, cx, restrict)["vdd"])
                record[key] = mean_sd(values) if values else None

            values = []
            for prefix in prefixes:
                c0 = _load(results_dir, prefix, "C0")
                c1 = _load(results_dir, prefix, "C1")
                if c0 is None or c1 is None or not (closed & set(c0) & set(c1)):
                    continue
                values.append(conditional_vdd(c0, c1, closed)["vdd"])
            record["C1closed"] = mean_sd(values) if values else None
            out[base] = record
    return out


def fvsvdd_block(results_dir: str, swapmap_path: str) -> list:
    """Cost of a swap against the fraction of verifiable answers it changes."""
    swapmap = load_swapmap(swapmap_path)
    rows = []
    for dataset in DATASETS:
        entry = swapmap[dataset]
        qid_meta = {int(k): v for k, v in entry["qid"].items()}
        maps = {("all", condition): donor_map
                for condition, donor_map in entry["swap"].items()}
        if dataset == "KVASIR":
            for name, donor_map in subset_vdd.subsets(qid_meta, entry["swap"]).items():
                if name in ("same", "cross"):
                    maps[(name, "C5")] = donor_map

        for (subset, condition), donor_map in sorted(maps.items()):
            changed, verifiable = subset_vdd.changed_fraction(qid_meta, donor_map)
            if verifiable < MIN_VERIFIABLE:
                # Same threshold the paper applies to the CCR: below four
                # verifiable answers the fraction is not a quantity, it is a
                # coin flip. This is what keeps VQA-RAD out of the panel.
                continue
            qids = eligible_qids(qid_meta, donor_map)
            for base in BASES[dataset]:
                prefixes = seed_prefixes(results_dir, base, condition)
                if not prefixes or "MEDVLTHINKER" in base:
                    continue
                values = []
                for prefix in prefixes:
                    c0 = _load(results_dir, prefix, "C0")
                    cx = _load(results_dir, prefix, condition)
                    if c0 is None or cx is None:
                        continue
                    values.append(conditional_vdd(c0, cx, qids)["vdd"])
                if not values:
                    continue
                mean, sd = mean_sd(values)
                rows.append({"ds": dataset, "cond": condition, "subset": subset,
                             "f": changed / verifiable, "model": base,
                             "vdd": mean, "sd": sd, "nverif": verifiable})
    return rows


def table1_block(results_dir: str, swapmap_path: str) -> dict:
    """Table 1 in the nested shape the figure data has always used."""
    out = {}
    for base, r in build_table1.build(results_dir, swapmap_path).items():
        record = {"acc0": [v / 100 if v is not None else None
                           for v in r["accuracy_c0"]],
                  "seeds": r["seeds"], "ds": r["dataset"]}
        for condition in build_table1.CONDITIONS:
            c = r["conditions"].get(condition)
            record[condition] = (None if c is None else
                                 {"vdd": [v / 100 for v in c["vdd"]],
                                  "sig": c["significant_seeds"],
                                  "k": c["seeds"], "n": c["n"]})
        record["rvdd_c2"] = r["relative_drop_c2"]
        record["ccr"] = (None if r["ccr"][0] is None else
                         [r["ccr"][0] / 100, r["ccr"][1] / 100, r.get("ccr_n")])
        out[base] = record
    return out


def build(results_dir: str, swapmap_path: str, existing: dict) -> dict:
    return {"combos": combos_block(results_dir, swapmap_path),
            "floor": existing.get("floor", {}),
            "fvsvdd": fvsvdd_block(results_dir, swapmap_path),
            "ccr": build_ccr_fig_data.build(list(existing.get("ccr", {})),
                                            results_dir, swapmap_path),
            "table1": table1_block(results_dir, swapmap_path)}


def compare(old: dict, new: dict, tolerance: float = 5e-4) -> list[str]:
    """Numeric differences between two fig_data blocks, as readable lines."""
    def walk(path, a, b, out):
        if isinstance(a, tuple):
            a = list(a)
        if isinstance(b, tuple):
            b = list(b)
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    out.append(f"{path}.{key}: {'only old' if key in a else 'only new'}")
                else:
                    walk(f"{path}.{key}", a[key], b[key], out)
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)):
                walk(f"{path}[{i}]", x, y, out)
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if abs(a - b) > tolerance:
                out.append(f"{path}: {a:.4f} -> {b:.4f}")
        elif a != b:
            out.append(f"{path}: {a!r} -> {b!r}")
    lines = []
    walk("", old, new, lines)
    return lines


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--swapmap", default="data/swapmap.json")
    parser.add_argument("--fig-data", default="data/fig_data.json")
    parser.add_argument("--compare", action="store_true",
                        help="report differences instead of writing")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.fig_data) as fp:
        existing = json.load(fp)
    rebuilt = build(args.results_dir, args.swapmap, existing)

    if args.compare:
        differences = compare(existing, rebuilt)
        print("\n".join(differences) if differences
              else "identical within tolerance")
        raise SystemExit(0)
    with open(args.fig_data, "w") as fp:
        json.dump(rebuilt, fp, indent=1)
    print(f"wrote {args.fig_data}")
