"""Noise floor of the protocol -- the measurements of Section 5.6.

Inference is not bit-exact across execution environments, so part of any
observed drop could be reproduction noise rather than signal. The intact
condition was re-run to measure how much: twice inside one environment, and
once against the environment where the original battery ran.

What is compared is the per-item prediction, not the aggregate accuracy. Two
runs can agree on accuracy to the decimal and still disagree on which items
they got right, and it is the item-level disagreement that propagates into a
paired test. The report therefore gives, for each pair of runs, the fraction of
predictions that differ, McNemar's b and c over the correctness flags, and both
accuracies, so a reader can see that the disagreement has no direction.

    same environment      C0_REP1 against C0_REP2
    across environments   the original C0 against C0_REP1

This bounds the reading of the convergence result: an effect smaller than the
across-environment band is not interpreted unless it replicates across seeds.

Usage:
    python -m src.analysis.noise_floor
"""
from __future__ import annotations

import argparse
import json
import os

from src.stats.compute_vdd import mcnemar_test

# (label, original prefix, dataset size in the paper's tables)
CONFIGURATIONS = (
    ("SmolVLM2-2.2B + LoRA, SLAKE", "SMOLVLM_LORA_SLAKE"),
    ("Qwen2-VL-2B + LoRA, SLAKE", "QWEN2VL_LORA_SLAKE"),
    ("Qwen2-VL-2B + LoRA, VQA-RAD", "QWEN2VL_LORA_VQARAD"),
)


def _load(results_dir: str, name: str) -> dict | None:
    path = os.path.join(results_dir, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fp:
        return {p["qid"]: p for p in json.load(fp)["predictions"]}


def compare_runs(a: dict, b: dict) -> dict:
    """Item-level disagreement between two runs of the same condition."""
    shared = sorted(set(a) & set(b))
    differing = [q for q in shared if str(a[q]["predicted"]) != str(b[q]["predicted"])]
    mcnemar = mcnemar_test([a[q]["correct"] for q in shared],
                           [b[q]["correct"] for q in shared])
    return {"n": len(shared),
            "differing": len(differing),
            "fraction": len(differing) / len(shared) if shared else None,
            "accuracy_a": sum(a[q]["correct"] for q in shared) / len(shared),
            "accuracy_b": sum(b[q]["correct"] for q in shared) / len(shared),
            "b": mcnemar["b"], "c": mcnemar["c"], "p_value": mcnemar["p_value"]}


def run(results_dir: str) -> dict:
    out = {}
    for label, prefix in CONFIGURATIONS:
        original = _load(results_dir, f"{prefix}_C0")
        rep1 = _load(results_dir, f"{prefix}_C0_REP1")
        rep2 = _load(results_dir, f"{prefix}_C0_REP2")
        record = {}
        if rep1 and rep2:
            record["same_environment"] = compare_runs(rep1, rep2)
        if original and rep1:
            record["across_environments"] = compare_runs(original, rep1)
        if record:
            out[label] = record
    return out


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--json")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run(args.results_dir)
    print(f"{'configuration':<32}{'same env':>22}{'across env':>24}{'McNemar':>14}")
    for label, record in result.items():
        same = record.get("same_environment")
        across = record.get("across_environments")
        same_cell = ("--" if same is None else
                     f"{same['fraction']:.2%} ({same['differing']}/{same['n']})")
        across_cell = ("--" if across is None else
                       f"{across['fraction']:.2%} ({across['differing']}/{across['n']})")
        mcnemar = "--" if across is None else f"b={across['b']}, c={across['c']}"
        print(f"{label:<32}{same_cell:>22}{across_cell:>24}{mcnemar:>14}")
    for label, record in result.items():
        across = record.get("across_environments")
        if across:
            print(f"\n{label}: accuracy {across['accuracy_a']:.2%} -> "
                  f"{across['accuracy_b']:.2%} despite {across['differing']} "
                  f"predictions changing")
    if args.json:
        with open(args.json, "w") as fp:
            json.dump(result, fp, indent=1)
