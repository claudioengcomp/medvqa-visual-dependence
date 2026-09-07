"""Consolidates all results/*.json + *_vs_C0.json files into a single
'summary matrix' (results/summary_matrix.json) -- a single source of truth
for the visual artefact and for the paper's tables, replacing the manual
copying of numbers on every update (which carried a real risk of
transcription errors).

For each model (identified by the file name results/<MODEL>_C0.json, minus
the _C0.json suffix) it computes, in addition to what already comes from
src/stats/compute_vdd.py:
- cohens_g (src.stats.compute_vdd.cohens_g), derived from the b/c counts
  already stored in each *_vs_C0.json -- there is no need to regenerate the
  already committed _vs_C0.json files;
- VDA over C1 (src.analysis.vda.compute_vda), whenever the model has a C1
  condition (the question-only baseline M0 does not);
- the accuracy breakdown by answer category in C0
  (src.analysis.answer_category.category_breakdown).

Note that *_C0.json and *_vs_C0.json must not be confused when globbing:
both end in '_C0.json', so the discovery step explicitly filters out the
'_vs_C0' files (see discover_models below).

Usage:
    python -m src.analysis.build_summary_matrix \
        --results-dir results --output results/summary_matrix.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from src.analysis.answer_category import category_breakdown
from src.analysis.vda import compute_vda
from src.stats.compute_vdd import cohens_g

_COND_ORDER = [
    "C0", "C1", "C2",
    "C4_OCC_25", "C4_OCC_50", "C4_OCC_75",
    "C4_NOISE_25", "C4_NOISE_50", "C4_NOISE_75",
    "C5", "C6",
]


def discover_models(results_dir: str) -> list[str]:
    """Returns the model names (the file prefix, without '_C0.json') that
    have a genuine C0.json in `results_dir`, sorted alphabetically. Ignores
    *_vs_C0.json -- which also ends in '_C0.json' and would be matched by a
    naive '*_C0.json' glob."""
    pattern = os.path.join(results_dir, "*_C0.json")
    names = []
    for path in sorted(glob.glob(pattern)):
        stem = os.path.basename(path)
        if "_vs_C0" in stem:
            continue
        assert stem.endswith("_C0.json")
        names.append(stem[: -len("_C0.json")])
    return sorted(set(names))


def _load(results_dir: str, model: str, suffix: str) -> dict | None:
    path = os.path.join(results_dir, f"{model}_{suffix}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fp:
        return json.load(fp)


def build_model_entry(results_dir: str, model: str) -> dict:
    c0 = _load(results_dir, model, "C0")
    if c0 is None:
        raise FileNotFoundError(f"{model}_C0.json not found in {results_dir}")

    entry = {
        "dataset": c0["dataset"],
        "n": c0["n"],
        "conditions": {
            "C0": {"accuracy": c0["accuracy"], "n": c0["n"], "n_correct": c0["n_correct"]},
        },
    }

    for cond in _COND_ORDER[1:]:
        vs_c0 = _load(results_dir, model, f"{cond}_vs_C0")
        cond_record = _load(results_dir, model, cond)
        if vs_c0 is None or cond_record is None:
            continue
        cond_entry = {
            "accuracy": cond_record["accuracy"],
            "n": cond_record["n"],
            "n_correct": cond_record["n_correct"],
            "ldv": vs_c0["ldv"],
        }
        mcnemar = vs_c0.get("mcnemar")
        if mcnemar is not None:
            cond_entry["mcnemar_p"] = mcnemar["p_value"]
            cond_entry["cohens_g"] = cohens_g(mcnemar["b"], mcnemar["c"])
        if "ci_other" in vs_c0:
            cond_entry["ci_other"] = vs_c0["ci_other"]
        if "ci_c0" in vs_c0:
            cond_entry["ci_c0"] = vs_c0["ci_c0"]
        if "n_paired" in vs_c0:
            cond_entry["n_paired"] = vs_c0["n_paired"]
        entry["conditions"][cond] = cond_entry

    if "C1" in entry["conditions"]:
        entry["vda_c1"] = compute_vda(entry["conditions"]["C0"]["accuracy"], entry["conditions"]["C1"]["ldv"])

    entry["answer_category_breakdown_c0"] = category_breakdown(c0["predictions"])

    return entry


def build_summary_matrix(results_dir: str) -> dict:
    return {model: build_model_entry(results_dir, model) for model in discover_models(results_dir)}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="results/summary_matrix.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    matrix = build_summary_matrix(args.results_dir)
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(matrix, fp, indent=2, ensure_ascii=False)
    print(f"{len(matrix)} models consolidated into {args.output}")
