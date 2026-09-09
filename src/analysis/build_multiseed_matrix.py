"""Aggregates the _vs_C0.json files of N seeds of the SAME model_base x
condition combination into a single multi-seed statistic (mean +/- std of
accuracy/VDD via src.stats.compute_multiseed.aggregate_multiseed_summary;
VDD is the metric name used in the paper for the JSON schema key 'ldv',
see src/stats/compute_vdd.py).

Only groups with at least N_SEEDS_EXPECTED available summaries are
aggregated -- combinations with fewer completed seeds are left out of the
aggregated result and are reported under "pending" (it never aggregates
partially and never errors out -- a combination still running is a normal
state, not a failure).

Combinations without a _seedN suffix in the prefix (zero-shot models; the
question-only baseline M0) do not enter the grouping -- they are not
fine-tuned, so they have no initialisation seed to vary across independent
runs, and they have no multi-seed variant by design.

Usage:
    python -m src.analysis.build_multiseed_matrix \
        --results-dir results --output-dir results/multiseed
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

from src.analysis.build_summary_matrix import _COND_ORDER
from src.stats.backfill_vs_c0 import DENYLIST_PREFIXES
from src.stats.compute_multiseed import aggregate_multiseed_summary

N_SEEDS_EXPECTED = 3
_SEED_SUFFIX_RE = re.compile(r"_seed(\d+)$")
# Longest suffix first -- there is no real collision among the current
# conditions, but this keeps the defensive check cheap in case a new
# condition ends up being a prefix of another one in the future.
_CONDITIONS_BY_LENGTH = sorted(_COND_ORDER[1:], key=len, reverse=True)


def _split_prefix(prefix: str) -> tuple[str, int | None]:
    """Returns (model_base, seed). A prefix with an explicit _seedN suffix ->
    seed = N. A prefix WITHOUT a suffix -> candidate seed = 0, following the
    naming convention of the SLAKE/VQA-RAD runs: the first (seed 0) run is
    the baseline seed and is saved with no suffix, while the seed 1
    and seed 2 runs added later use an explicit suffix. Such a candidate only
    becomes a real seed 0 in discover_multiseed_groups(), after confirming
    that the same model_base actually has a _seed1/_seed2 sibling --
    otherwise it is a genuine zero-shot/M0 combination with no multi-seed
    family (a zero-shot model, for instance, will never have a _seed1
    sibling)."""
    m = _SEED_SUFFIX_RE.search(prefix)
    if m:
        return prefix[: m.start()], int(m.group(1))
    return prefix, 0


def discover_multiseed_groups(results_dir: str) -> dict[tuple[str, str], list[tuple[int, str]]]:
    """Scans results_dir/*_vs_C0.json and groups by (model_base, condition).
    Returns {(model_base, condition): [(seed, path), ...]}.

    A prefix without a _seedN suffix only enters as seed 0 when the same
    model_base has at least one genuine _seed1 or _seed2 somewhere (which
    confirms it belongs to a multi-seed family, as in the SLAKE/VQA-RAD runs
    whose seed 0 carries no suffix). Genuinely zero-shot/M0
    combinations (with NO _seedN suffix at all, in any condition) are left
    out -- they have no training seed, hence no multi-seed by design."""
    raw: list[tuple[str, int | None, str, str]] = []  # (model_base, seed, cond, path)
    for path in sorted(glob.glob(os.path.join(results_dir, "*_vs_C0.json"))):
        stem = os.path.basename(path)[: -len("_vs_C0.json")]
        for cond in _CONDITIONS_BY_LENGTH:
            suffix = f"_{cond}"
            if not stem.endswith(suffix):
                continue
            prefix = stem[: -len(suffix)]
            if prefix in DENYLIST_PREFIXES:
                break  # discarded data (e.g. the leaked split) -- same denylist as the backfill step
            model_base, seed = _split_prefix(prefix)
            raw.append((model_base, seed, cond, path))
            break

    has_explicit_seed = {mb for mb, seed, _, _ in raw if seed and seed >= 1}

    groups: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for model_base, seed, cond, path in raw:
        if seed == 0 and model_base not in has_explicit_seed:
            continue  # genuine zero-shot/M0 -- no _seed1/_seed2 sibling anywhere
        groups.setdefault((model_base, cond), []).append((seed, path))
    return groups


# Known label inconsistency in the result files: different scripts recorded
# the SAME dataset either as "VQARAD" (no hyphen, the older convention) or as
# "VQA-RAD" (hyphenated, the literal value of the --dataset argument).
# Normalises to the hyphenated label (the more recent, CLI form) in memory
# only, before validating/aggregating -- the files on disk are never
# rewritten.
_DATASET_ALIASES = {"VQARAD": "VQA-RAD"}


def _normalize_dataset(dataset: str | None) -> str | None:
    return _DATASET_ALIASES.get(dataset, dataset)


def build_multiseed_matrix(results_dir: str, n_seeds_expected: int = N_SEEDS_EXPECTED) -> dict:
    """Returns {"aggregated": {(model_base, cond): agg}, "pending": {(model_base, cond): [seeds]}}."""
    groups = discover_multiseed_groups(results_dir)
    aggregated: dict = {}
    pending: dict = {}
    for key, seed_paths in groups.items():
        seed_paths_sorted = sorted(seed_paths)
        seeds_found = [s for s, _ in seed_paths_sorted]
        if len(seed_paths_sorted) < n_seeds_expected:
            pending[key] = seeds_found
            continue
        summaries, source_files = [], []
        for _, path in seed_paths_sorted:
            with open(path) as fp:
                summary = json.load(fp)
            summary["dataset"] = _normalize_dataset(summary.get("dataset"))
            summaries.append(summary)
            source_files.append(path)
        agg = aggregate_multiseed_summary(summaries)
        model_base, cond = key
        agg.update(model_base=model_base, condition=cond, seeds_used=seeds_found, source_files=source_files)
        aggregated[key] = agg
    return {"aggregated": aggregated, "pending": pending}


def write_multiseed_matrix(results_dir: str, output_dir: str, n_seeds_expected: int = N_SEEDS_EXPECTED) -> dict:
    result = build_multiseed_matrix(results_dir, n_seeds_expected)
    os.makedirs(output_dir, exist_ok=True)
    for (model_base, cond), agg in result["aggregated"].items():
        out_path = os.path.join(output_dir, f"{model_base}_{cond}_multiseed.json")
        with open(out_path, "w") as fp:
            json.dump(agg, fp, indent=2)
    return result


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/multiseed")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = write_multiseed_matrix(args.results_dir, args.output_dir)
    print(f"{len(result['aggregated'])} combo-conditions aggregated into {args.output_dir}")
    if result["pending"]:
        print(f"{len(result['pending'])} aguardando mais seeds:")
        for (model_base, cond), seeds in sorted(result["pending"].items()):
            print(f"  {model_base} {cond}: {len(seeds)}/{N_SEEDS_EXPECTED} seeds ({seeds})")
