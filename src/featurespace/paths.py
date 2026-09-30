"""The one place that turns (benchmark, configuration, seed, condition) into a path.

Two conventions coexist in `results/`, and nothing in the file names says so:

    Kvasir-VQA-x1     M3AE_KVASIR_seed0_C0.json
    SLAKE, VQA-RAD    M3AE_SCRATCH_SLAKE_C0.json      <- seed 0 has NO suffix
                      M3AE_SCRATCH_SLAKE_seed1_C0.json

A script that assumes one convention loses a seed of every cell of the other
two benchmarks, silently — the file is simply not found. It has cost a round
twice: once in item 4(c) of the audit, and once in the first pass of analysis C,
where ten of the fifteen G3 cells were reported as failing because a missing
seed makes the "all three seeds" rule unsatisfiable.

So no script of v4 builds a result path of its own from here on. They call
`predictions_path` or `cells`, and `tests/v4/test_paths.py` requires all three
seeds of all fifteen cells to resolve.

`*_vs_C0.json` files are comparisons already computed between two conditions,
not predictions. Matching on the trailing `_C<n>.json` swallows them; this
module never returns one.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

BENCHMARKS = ["KVASIR", "SLAKE", "VQARAD"]
SEEDS = ["seed0", "seed1", "seed2"]

# the five multi-seed configurations of G3, per benchmark. Kvasir trains the
# classifier from the authors' backbone and the other two from scratch: that is
# the design of section 4.2, not an inconsistency, and the table of G3 labels
# them "M3AE (pre-trained backbone)" and "M3AE (from scratch)".
CONFIGS = {
    "KVASIR": ["M3AE", "QWEN2VL_DORA", "QWEN2VL_LORA", "SMOLVLM_DORA", "SMOLVLM_LORA"],
    "SLAKE": ["M3AE_SCRATCH", "QWEN2VL_DORA", "QWEN2VL_LORA", "SMOLVLM_DORA",
              "SMOLVLM_LORA"],
    "VQARAD": ["M3AE_SCRATCH", "QWEN2VL_DORA", "QWEN2VL_LORA", "SMOLVLM_DORA",
               "SMOLVLM_LORA"],
}

# reference lines: one seed, drawn in the figures, never inside a gate
REFERENCE = {
    "KVASIR": ["MEDVLTHINKER_ZEROSHOT"],
    "SLAKE": ["MEDVLTHINKER_ZEROSHOT", "M0_question_only", "M3AE"],
    "VQARAD": ["MEDVLTHINKER_ZEROSHOT", "M0_question_only"],
}

# VQA-RAD has no C5: the arrow was never generated, absent by construction
# (paper section 4.1), which is why its cells are complete with C0+C2+C6.
SWAP_CONDITIONS = {"KVASIR": ["C2", "C5", "C6"], "SLAKE": ["C2", "C5", "C6"],
                   "VQARAD": ["C2", "C6"]}


def prefix(benchmark: str, config: str, seed: str, results_dir: Path = RESULTS):
    """The file prefix of one run, or None if that run does not exist.

    Tries the explicit `_seedN` spelling first and falls back to the bare name
    for seed 0, which is how SLAKE and VQA-RAD name their first run.
    """
    candidates = ["%s_%s_%s" % (config, benchmark, seed)]
    if seed == "seed0":
        candidates.append("%s_%s" % (config, benchmark))
    for c in candidates:
        if (results_dir / ("%s_C0.json" % c)).exists():
            return c
    return None


def predictions_path(benchmark, config, seed, condition, results_dir: Path = RESULTS):
    """Path to one prediction file, or None. Never returns a `_vs_C0` file."""
    p = prefix(benchmark, config, seed, results_dir)
    if p is None:
        return None
    f = results_dir / ("%s_%s.json" % (p, condition))
    return f if f.exists() and "_vs_" not in f.name else None


def cells(benchmark=None, include_reference=False, results_dir: Path = RESULTS):
    """Every (benchmark, config, seed) that resolves, in a fixed order."""
    out = []
    for b in ([benchmark] if benchmark else BENCHMARKS):
        names = list(CONFIGS[b]) + (list(REFERENCE[b]) if include_reference else [])
        for cfg in names:
            for s in SEEDS:
                p = prefix(b, cfg, s, results_dir)
                if p is not None:
                    out.append({"benchmark": b, "config": cfg, "seed": s,
                                "prefix": p, "reference": cfg not in CONFIGS[b]})
    return out


def missing(results_dir: Path = RESULTS):
    """The G3 cells that do not resolve, as (benchmark, config, seed, condition)."""
    gaps = []
    for b in BENCHMARKS:
        for cfg in CONFIGS[b]:
            for s in SEEDS:
                if prefix(b, cfg, s, results_dir) is None:
                    gaps.append((b, cfg, s, "C0"))
                    continue
                for c in ["C0"] + SWAP_CONDITIONS[b]:
                    if predictions_path(b, cfg, s, c, results_dir) is None:
                        gaps.append((b, cfg, s, c))
    return gaps
