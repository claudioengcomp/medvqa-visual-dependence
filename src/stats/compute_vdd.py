"""VDD (Visual Dependency Drop), bootstrap 95% CI and McNemar's test, following
the statistical protocol of this study.

Note on naming: the JSON schema key for the VDD value is historically named
"ldv". The key name is kept as-is for backward compatibility with the 493
already-generated result files that carry it; everywhere in prose the metric is called VDD.

Two modes are supported, depending on what the result records make available:

- Aggregate mode (available when 'n'/'n_correct' are present): VDD =
  accuracy(C0) - accuracy(C1); 95% CI from an INDEPENDENT bootstrap on each
  condition (not paired -- it does not capture item-by-item correlation).
- Paired mode (requires 'predictions' with the SAME qids in C0 and C1): adds
  McNemar's test and a paired bootstrap over the difference.

Usage:
    python -m src.stats.compute_vdd \
        --c0 results/M3AE_SCRATCH_SLAKE_C0.json --c1 results/M3AE_SCRATCH_SLAKE_C1.json \
        --output results/M3AE_SCRATCH_SLAKE_C1_vs_C0.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import stats as scipy_stats

from src.inference.results_schema import has_paired_data


def compute_vdd(acc_c0: float, acc_c1: float) -> float:
    return acc_c0 - acc_c1


def bootstrap_accuracy_ci(n_correct: int, n_total: int, n_boot: int = 10000,
                           seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    if not (0 <= n_correct <= n_total) or n_total <= 0:
        raise ValueError(f"n_correct={n_correct} is invalid for n_total={n_total}")
    rng = np.random.RandomState(seed)
    sample = np.array([1] * n_correct + [0] * (n_total - n_correct))
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        resample = rng.choice(sample, size=n_total, replace=True)
        boot_means[b] = resample.mean()
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def bootstrap_paired_diff_ci(c0_correct: list[bool], c1_correct: list[bool],
                              n_boot: int = 10000, seed: int = 42,
                              alpha: float = 0.05) -> tuple[float, float]:
    if len(c0_correct) != len(c1_correct):
        raise ValueError("c0_correct and c1_correct must have the same length (paired data)")
    n = len(c0_correct)
    if n == 0:
        raise ValueError("empty lists")
    c0 = np.asarray(c0_correct, dtype=float)
    c1 = np.asarray(c1_correct, dtype=float)
    rng = np.random.RandomState(seed)
    boot_diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_diffs[b] = c0[idx].mean() - c1[idx].mean()
    lo = float(np.percentile(boot_diffs, 100 * alpha / 2))
    hi = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))
    return lo, hi


def cohens_g(b: int, c: int) -> float | None:
    """Cohen's effect size (g) for McNemar's test -- |proportion of
    discordant pairs in the majority direction - 0.5|, ranging from 0 (b=c,
    symmetric disagreement) to 0.5 (b=0 or c=0, all disagreement in a
    single direction). Reference: Cohen, J. (1988), Statistical Power
    Analysis for the Behavioral Sciences, 2nd ed. -- the standard effect-size
    companion for McNemar, in the same spirit as reporting Cohen's d
    alongside a t-test. Undefined (None) when b+c=0 (no discordant pairs --
    the same case in which mcnemar_test already returns p=1.0)."""
    if b + c == 0:
        return None
    return abs(b / (b + c) - 0.5)


def mcnemar_test(c0_correct: list[bool], c1_correct: list[bool]) -> dict:
    """Paired McNemar test. `b` = correct in C0 and wrong in C1; `c` =
    wrong in C0 and correct in C1 (the discordant pairs). Uses the exact
    (binomial) version when b+c < 25, otherwise chi-squared with continuity
    correction. The b+c < 25 threshold is the documented statistical
    convention (e.g. Agresti, "Categorical Data Analysis") for switching
    from the exact test to the chi-squared approximation -- it is not an
    arbitrary magic number. Includes `cohens_g` (see cohens_g()) as the
    companion effect size."""
    if len(c0_correct) != len(c1_correct):
        raise ValueError("c0_correct and c1_correct must have the same length (paired data)")
    b = sum(1 for x, y in zip(c0_correct, c1_correct) if x and not y)
    c = sum(1 for x, y in zip(c0_correct, c1_correct) if not x and y)
    g = cohens_g(b, c)

    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "method": "exact", "b": b, "c": c, "cohens_g": g}

    if b + c < 25:
        result = scipy_stats.binomtest(min(b, c), b + c, 0.5)
        return {"statistic": float(min(b, c)), "p_value": float(result.pvalue), "method": "exact", "b": b, "c": c, "cohens_g": g}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(scipy_stats.chi2.sf(chi2, df=1))
    return {"statistic": float(chi2), "p_value": p_value, "method": "chi2_corrected", "b": b, "c": c, "cohens_g": g}


def _predictions_by_qid(predictions: list[dict]) -> dict:
    return {p["qid"]: p["correct"] for p in predictions}


def compute_summary(c0_record: dict, c1_record: dict, n_boot: int = 10000, seed: int = 42) -> dict:
    if c0_record["model"] != c1_record["model"] or c0_record["dataset"] != c1_record["dataset"]:
        raise ValueError("c0_record and c1_record must have the same model/dataset")
    if c0_record["condition"] != "C0":
        raise ValueError(f"c0_record must have condition='C0', got {c0_record['condition']!r}")
    if c1_record["condition"] != "C1":
        raise ValueError(f"c1_record must have condition='C1', got {c1_record['condition']!r}")

    ldv = compute_vdd(c0_record["accuracy"], c1_record["accuracy"])
    summary = {
        "model": c0_record["model"],
        "dataset": c0_record["dataset"],
        "accuracy_c0": c0_record["accuracy"],
        "accuracy_c1": c1_record["accuracy"],
        "ldv": ldv,
        "paired": False,
    }

    if "n" in c0_record and "n_correct" in c0_record:
        summary["ci_c0"] = bootstrap_accuracy_ci(c0_record["n_correct"], c0_record["n"], n_boot=n_boot, seed=seed)
    if "n" in c1_record and "n_correct" in c1_record:
        summary["ci_c1"] = bootstrap_accuracy_ci(c1_record["n_correct"], c1_record["n"], n_boot=n_boot, seed=seed + 1)

    if has_paired_data(c0_record) and has_paired_data(c1_record):
        c0_by_qid = _predictions_by_qid(c0_record["predictions"])
        c1_by_qid = _predictions_by_qid(c1_record["predictions"])
        shared_qids = sorted(set(c0_by_qid) & set(c1_by_qid))
        if shared_qids:
            c0_correct = [c0_by_qid[q] for q in shared_qids]
            c1_correct = [c1_by_qid[q] for q in shared_qids]
            summary["paired"] = True
            summary["n_paired"] = len(shared_qids)
            summary["mcnemar"] = mcnemar_test(c0_correct, c1_correct)
            summary["ldv_ci_paired"] = bootstrap_paired_diff_ci(c0_correct, c1_correct, n_boot=n_boot, seed=seed)

    return summary


def compute_summary_vs_baseline(c0_record: dict, other_record: dict, n_boot: int = 10000,
                                 seed: int = 42) -> dict:
    """Generalizes compute_summary to compare C0 against ANY other condition
    (C1, C2, C4_OCC_25, C4_NOISE_75, ...) -- used by the C4 severity curve,
    which has 6 conditions beyond C1/C2. Same VDD/bootstrap/McNemar logic as
    compute_summary; only the validation of the second condition's label is
    permissive (anything other than 'C0')."""
    if c0_record["model"] != other_record["model"] or c0_record["dataset"] != other_record["dataset"]:
        raise ValueError("c0_record and other_record must have the same model/dataset")
    if c0_record["condition"] != "C0":
        raise ValueError(f"c0_record must have condition='C0', got {c0_record['condition']!r}")
    if other_record["condition"] == "C0":
        raise ValueError("other_record cannot have condition='C0' (nothing to compare against)")

    ldv = compute_vdd(c0_record["accuracy"], other_record["accuracy"])
    summary = {
        "model": c0_record["model"],
        "dataset": c0_record["dataset"],
        "condition": other_record["condition"],
        "accuracy_c0": c0_record["accuracy"],
        "accuracy_other": other_record["accuracy"],
        "ldv": ldv,
        "paired": False,
    }

    if "n" in c0_record and "n_correct" in c0_record:
        summary["ci_c0"] = bootstrap_accuracy_ci(c0_record["n_correct"], c0_record["n"], n_boot=n_boot, seed=seed)
    if "n" in other_record and "n_correct" in other_record:
        summary["ci_other"] = bootstrap_accuracy_ci(other_record["n_correct"], other_record["n"], n_boot=n_boot, seed=seed + 1)

    if has_paired_data(c0_record) and has_paired_data(other_record):
        c0_by_qid = _predictions_by_qid(c0_record["predictions"])
        other_by_qid = _predictions_by_qid(other_record["predictions"])
        shared_qids = sorted(set(c0_by_qid) & set(other_by_qid))
        if shared_qids:
            c0_correct = [c0_by_qid[q] for q in shared_qids]
            other_correct = [other_by_qid[q] for q in shared_qids]
            summary["paired"] = True
            summary["n_paired"] = len(shared_qids)
            summary["mcnemar"] = mcnemar_test(c0_correct, other_correct)
            summary["ldv_ci_paired"] = bootstrap_paired_diff_ci(c0_correct, other_correct, n_boot=n_boot, seed=seed)

    return summary


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0", required=True)
    parser.add_argument("--c1", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.c0) as fp:
        c0_record = json.load(fp)
    with open(args.c1) as fp:
        c1_record = json.load(fp)
    summary = compute_summary(c0_record, c1_record, n_boot=args.n_boot, seed=args.seed)
    with open(args.output, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"LDV({summary['model']}, {summary['dataset']}) = {summary['ldv']:.4f} (paired={summary['paired']})")
