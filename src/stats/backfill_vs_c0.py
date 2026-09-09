"""Batch backfill of VDD/McNemar/Cohen's g/bootstrap 95% CI (stored under the
JSON key 'ldv', which holds the VDD value, see src/stats/compute_vdd.py) for every model x condition combination in
results/ that already has both C0 and the condition executed but does not yet
have its _vs_C0.json computed.

Idempotent -- it skips whatever already exists, so it is safe to run as many
times as needed (including incrementally, every time a new condition
finishes).

Usage:
    python -m src.stats.backfill_vs_c0 --results-dir results
"""
from __future__ import annotations

import argparse
import json
import os

from src.analysis.build_summary_matrix import _COND_ORDER, discover_models
from src.stats.compute_vdd import compute_summary_vs_baseline

# Combinations whose source data comes from a discarded version of the
# dataset -- they never enter the backfill and, by extension, never enter the
# multi-seed aggregation, even if the raw files are still on disk.
# M3AE_KVASIR (without a _seed suffix) was trained on a leaked
# Kvasir-VQA-x1 split, before the split was rebuilt to remove the
# train/test image overlap that inflated its accuracy (Acc(C0)=91.01%).
# Those runs are not comparable with the clean-split results and are
# excluded from every reported number.
DENYLIST_PREFIXES = {"M3AE_KVASIR"}


def _load(results_dir: str, prefix: str, suffix: str) -> dict | None:
    path = os.path.join(results_dir, f"{prefix}_{suffix}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fp:
        return json.load(fp)


def backfill_model(results_dir: str, prefix: str) -> dict:
    """Computes the _vs_C0.json for every condition of `prefix` that does not
    have one yet. Returns a report (it never raises because of a SINGLE
    problematic combination/condition -- it records the issue and moves on to
    the next one):
        {"computed": [cond, ...], "skipped_existing": [cond, ...],
         "missing_condition": [cond, ...], "n_mismatch": [(cond, n_c0, n_cond), ...]}
    """
    report = {"computed": [], "skipped_existing": [], "missing_condition": [], "n_mismatch": []}
    c0 = _load(results_dir, prefix, "C0")
    if c0 is None:
        return report  # discover_models already requires C0 -- should not happen in practice

    for cond in _COND_ORDER[1:]:
        out_path = os.path.join(results_dir, f"{prefix}_{cond}_vs_C0.json")
        if os.path.exists(out_path):
            report["skipped_existing"].append(cond)
            continue
        other = _load(results_dir, prefix, cond)
        if other is None:
            report["missing_condition"].append(cond)
            continue
        if c0.get("n") != other.get("n"):
            report["n_mismatch"].append((cond, c0.get("n"), other.get("n")))
            continue
        summary = compute_summary_vs_baseline(c0, other)
        with open(out_path, "w") as fp:
            json.dump(summary, fp, indent=2)
        report["computed"].append(cond)

    return report


def backfill_all(results_dir: str) -> dict:
    """Runs backfill_model for every model prefix discovered in results_dir,
    except those in DENYLIST_PREFIXES. Returns {prefix: report}."""
    return {
        prefix: backfill_model(results_dir, prefix)
        for prefix in discover_models(results_dir)
        if prefix not in DENYLIST_PREFIXES
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    reports = backfill_all(args.results_dir)
    total_computed = sum(len(r["computed"]) for r in reports.values())
    total_skipped = sum(len(r["skipped_existing"]) for r in reports.values())
    print(f"{len(reports)} combos scanned | {total_computed} new vs_C0 computed | {total_skipped} already existed")
    for prefix, r in reports.items():
        if r["computed"]:
            print(f"  {prefix}: +{len(r['computed'])} ({', '.join(r['computed'])})")
        if r["n_mismatch"]:
            print(f"  {prefix}: AVISO n divergente -- {r['n_mismatch']}")
