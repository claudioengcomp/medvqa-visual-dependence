"""Does a spread of VDD among configurations exceed seed noise? (Section 5.4)

For one condition, the 15 values of VDD (5 configurations x 3 seeds) are
decomposed into between- and within-configuration sums of squares, and the
RANGE of the five means is tested two ways:

    permutation   100,000 relabellings of the 15 values among the five groups
    Welch's ANOVA which does NOT assume equal variances

Both are reported, and when they disagree the disagreement is the result. That
is not a hedge: under C1 the seed standard deviation runs from 0.09 to 4.94 pp,
a variance ratio above three thousand, so permutation -- which assumes
exchangeability -- is anticonservative exactly where it matters, and Welch is
the test that corrects the violated assumption.

Usage:
    python -m src.stats.spread_tests --dataset KVASIR
    python -m src.stats.spread_tests --dataset KVASIR --conditions C1 C2 C5 C6
"""
from __future__ import annotations

import argparse
import json
import statistics

import numpy as np

from src.analysis.ccr import load_swapmap
from src.analysis.identity import CONFIGURATIONS
from src.stats.conditional_vdd import (FULL_COVERAGE_CONDITIONS, _load,
                                       eligible_qids)

RESAMPLES = 100_000
DEFAULT_CONDITIONS = ("C1", "C2", "C5", "C6")


def vdd_by_configuration(dataset: str, condition: str, results_dir: str,
                         swapmap_path: str) -> tuple[list[list[float]], int]:
    """[[VDD per seed] per configuration], in pp, plus the n of the condition.

    C5 and C6 are partial interventions, so they are restricted to the
    questions whose image actually received a donor; on the full-coverage
    conditions the restriction is a no-op.
    """
    entry = load_swapmap(swapmap_path)[dataset]
    qid_meta = {int(k): v for k, v in entry["qid"].items()}
    if condition in FULL_COVERAGE_CONDITIONS:
        keep = set(qid_meta)
    else:
        if condition not in entry["swap"]:
            raise SystemExit(f"{dataset} has no donor map for {condition}")
        keep = eligible_qids(qid_meta, entry["swap"][condition])

    groups = []
    for base in CONFIGURATIONS[dataset]:
        row = []
        for suffix in ("", "_seed0", "_seed1", "_seed2"):
            prefix = base + suffix
            c0 = _load(results_dir, prefix, "C0")
            cx = _load(results_dir, prefix, condition)
            if c0 is None or cx is None:
                continue
            shared = keep & set(c0) & set(cx)
            if not shared:
                continue
            row.append(sum(c0[q]["correct"] - cx[q]["correct"]
                           for q in shared) / len(shared) * 100)
        if row:
            groups.append(row)
    return groups, len(keep)


def decompose(groups: list[list[float]]) -> dict:
    flat = [v for g in groups for v in g]
    grand = statistics.mean(flat)
    between = sum(len(g) * (statistics.mean(g) - grand) ** 2 for g in groups)
    within = sum((v - statistics.mean(g)) ** 2 for g in groups for v in g)
    df_b, df_w = len(groups) - 1, len(flat) - len(groups)
    return {"ss_between": between, "ss_within": within,
            "ms_between": between / df_b, "ms_within": within / df_w,
            "df": [df_b, df_w],
            "fraction_between": between / (between + within)}


def permutation_test(groups: list[list[float]], resamples: int = RESAMPLES,
                     seed: int = 20260909) -> tuple[float, float]:
    """p for the RANGE of the group means, with the (r+1)/(N+1) correction."""
    flat = np.array([v for g in groups for v in g])
    sizes = [len(g) for g in groups]
    means = [statistics.mean(g) for g in groups]
    observed = max(means) - min(means)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(resamples):
        shuffled = rng.permutation(flat)
        m, k = [], 0
        for s in sizes:
            m.append(shuffled[k:k + s].mean())
            k += s
        if max(m) - min(m) >= observed - 1e-12:
            count += 1
    return observed, (count + 1) / (resamples + 1)


def welch_anova(groups: list[list[float]]):
    """Welch's F for k groups. Returns (F, (df1, df2), p), or Nones if undefined."""
    from scipy import stats
    n = np.array([len(g) for g in groups], float)
    m = np.array([np.mean(g) for g in groups])
    v = np.array([np.var(g, ddof=1) for g in groups])
    if np.any(v == 0):
        return None, None, None          # a group with zero variance: undefined
    w = n / v
    mw = (w * m).sum() / w.sum()
    k = len(groups)
    numerator = (w * (m - mw) ** 2).sum() / (k - 1)
    lam = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    f = numerator / (1 + 2 * (k - 2) / (k * k - 1) * lam)
    df2 = (k * k - 1) / (3 * lam)
    return f, (k - 1, df2), float(stats.f.sf(f, k - 1, df2))


def run(dataset: str, condition: str, results_dir: str, swapmap_path: str,
        resamples: int = RESAMPLES) -> dict:
    groups, n = vdd_by_configuration(dataset, condition, results_dir, swapmap_path)
    if len(groups) < 2:
        raise SystemExit(f"{dataset} {condition}: fewer than two configurations")
    d = decompose(groups)
    observed, p_perm = permutation_test(groups, resamples)
    f, df, p_welch = welch_anova(groups)
    sds = [statistics.stdev(g) if len(g) > 1 else 0.0 for g in groups]
    return {"dataset": dataset, "condition": condition, "n": n,
            "groups": groups, "range_pp": observed, **d,
            "p_permutation": p_perm, "welch_F": f,
            "welch_df": list(df) if df else None, "p_welch": p_welch,
            "seed_sd_min_pp": min(sds), "seed_sd_max_pp": max(sds),
            "variance_ratio": (max(sds) ** 2 / min(sds) ** 2
                               if min(sds) > 0 else None)}


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--swapmap", default="data/swapmap.json")
    p.add_argument("--dataset", default="KVASIR")
    p.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    p.add_argument("--resamples", type=int, default=RESAMPLES)
    p.add_argument("--json")
    return p.parse_args()


def main():
    args = _parse_args()
    out = {}
    for condition in args.conditions:
        r = run(args.dataset, condition, args.results_dir, args.swapmap,
                args.resamples)
        out[condition] = r
        print(f"\n=== {args.dataset} {condition}  (n = {r['n']:,} questions)")
        print(f"  SS between {r['ss_between']:8.3f} (df {r['df'][0]})   "
              f"SS within {r['ss_within']:8.3f} (df {r['df'][1]})   "
              f"MS ratio {r['ms_between'] / r['ms_within']:.2f}")
        print(f"  seed sd {r['seed_sd_min_pp']:.2f} to {r['seed_sd_max_pp']:.2f} pp"
              + (f"   variance ratio {r['variance_ratio']:.0f}x"
                 if r["variance_ratio"] else ""))
        print(f"  range of the five means: {r['range_pp']:.2f} pp")
        print(f"  permutation p = {r['p_permutation']:.5f}"
              f"   Welch p = "
              + ("undefined (a group has zero variance)" if r["p_welch"] is None
                 else f"{r['p_welch']:.5f}"))

    print(f"\n{'condition':<12}{'range':>10}{'p perm.':>12}{'p Welch':>12}"
          f"{'% var between':>16}")
    for condition, r in out.items():
        pw = "--" if r["p_welch"] is None else f"{r['p_welch']:.5f}"
        print(f"{condition:<12}{r['range_pp']:>9.2f}pp{r['p_permutation']:>12.5f}"
              f"{pw:>12}{r['fraction_between'] * 100:>15.1f}%")

    if args.json:
        with open(args.json, "w") as fp:
            json.dump(out, fp, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
