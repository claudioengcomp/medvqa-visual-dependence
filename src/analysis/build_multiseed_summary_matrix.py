"""Consolidates the multi-seed aggregates (results/multiseed/*.json, produced
by src/analysis/build_multiseed_matrix.py) into a final per-model_base
matrix, in the same spirit as src/analysis/build_summary_matrix.py but using
the ALREADY aggregated fields (accuracy_c0_mean/std, vdd_mean/std,
cohens_g_mean/std/n, mcnemar_significant_count/total_seeds) instead of the
result of a single seed.

VDA is recomputed here, ONCE, over the aggregated C1 means -- never as an
average of per-seed VDA values. The harmonic mean is non-linear, so
averaging per-seed VDAs would not equal the VDA of the averaged components
and would not correspond to the quantity reported in the paper.

Usage:
    python -m src.analysis.build_multiseed_summary_matrix \
        --multiseed-dir results/multiseed --output results/summary_matrix_multiseed.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from src.analysis.vda import compute_vda


def discover_model_bases(multiseed_dir: str) -> list[str]:
    bases = set()
    for path in glob.glob(os.path.join(multiseed_dir, "*_multiseed.json")):
        with open(path) as fp:
            d = json.load(fp)
        bases.add(d["model_base"])
    return sorted(bases)


def build_model_base_entry(multiseed_dir: str, model_base: str) -> dict:
    conditions = {}
    for path in sorted(glob.glob(os.path.join(multiseed_dir, f"{model_base}_*_multiseed.json"))):
        with open(path) as fp:
            d = json.load(fp)
        if d["model_base"] != model_base:
            continue  # avoids prefix collisions (the glob is only a pre-filter; the inner field is the truth)
        conditions[d["condition"]] = d

    entry = {"conditions": conditions}
    if "C1" in conditions:
        c1 = conditions["C1"]
        entry["vda_c1"] = compute_vda(c1["accuracy_c0_mean"], c1["vdd_mean"])
    return entry


def build_multiseed_summary_matrix(multiseed_dir: str) -> dict:
    return {mb: build_model_base_entry(multiseed_dir, mb) for mb in discover_model_bases(multiseed_dir)}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiseed-dir", default="results/multiseed")
    parser.add_argument("--output", default="results/summary_matrix_multiseed.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    matrix = build_multiseed_summary_matrix(args.multiseed_dir)
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(matrix, fp, indent=2, ensure_ascii=False)
    print(f"{len(matrix)} model_bases consolidados em {args.output}")
