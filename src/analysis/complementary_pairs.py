"""Complementary-pair accuracy, Section 3.6 and Section 5.5.

The canonical measure of shortcut learning is accuracy on complementary pairs
(Goyal et al., 2017): two items (x, q, y) and (x', q, y') with the same question
and y != y', both answered under INTACT inputs. The informative set of C5
provides such pairs, with x' the donor of x.

NO INTERVENTION ENTERS THE COMPUTATION: both predictions come from C0. What
changes with respect to the CCR is the quantity measured -- the joint over the
pair rather than the marginal over the donor item.

The quantity of interest is the departure from independence,

    Delta = P(both correct) - a^2

with `a` the marginal accuracy over the 2*n positions of the pairs, together
with the rate at which the model gives the SAME answer to both images despite
their different references -- the direct signature of answering from the
question.

The reading rule was fixed before the analysis: if |Delta| stays below 0.5 pp in
every configuration, the joint quantity is determined by the marginals and adds
nothing to accuracy.

Deduplication. The donor map is a derangement, so i -> j and j -> k produce
distinct pairs and one item may sit in more than one. A pair is an UNORDERED
set {i, j}, deduplicated by frozenset; no item is dropped for appearing in more
than one pair, since dropping would change the set under analysis and break the
correspondence with the informative set.

Usage:
    python -m src.analysis.complementary_pairs --dataset KVASIR --condition C5
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics

from src.analysis.ccr import answers_match, load_predictions, load_swapmap
from src.analysis.ccr_slices import seed_prefixes
from src.analysis.identity import CONFIGURATIONS, strata

THRESHOLD_PP = 0.5          # fixed in advance; see Section 3.6


def build_pairs(qid_meta: dict, donor_map: dict):
    """Deduplicated unordered pairs, plus the image each pair is clustered on."""
    parts, donor_qid, _ = strata(qid_meta, donor_map)
    directed = [(i, donor_qid[i]) for i in parts["informative"]]
    seen, pairs = set(), []
    for i, j in directed:
        key = frozenset((i, j))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(tuple(sorted((i, j))))
    membership = collections.Counter()
    for a, b in pairs:
        membership[a] += 1
        membership[b] += 1
    clusters = [qid_meta[a]["img"] for a, _ in pairs]
    return {"directed": len(directed), "pairs": pairs,
            "two_cycles_collapsed": len(directed) - len(pairs),
            "items": len(membership), "clusters": len(set(clusters)),
            "pairs_per_item": dict(sorted(collections.Counter(
                membership.values()).items()))}


def measure(pairs: list[tuple[int, int]], c0: dict) -> dict:
    """Joint and marginal quantities of one configuration-seed cell."""
    both = one = neither = identical = hits = 0
    for a, b in pairs:
        ca, cb = bool(c0[a]["correct"]), bool(c0[b]["correct"])
        both += ca and cb
        one += ca != cb
        neither += not ca and not cb
        identical += answers_match(c0[a]["predicted"], [c0[b]["predicted"]])
        hits += int(ca) + int(cb)
    n = len(pairs)
    marginal = hits / (2 * n)
    return {"n_pairs": n, "p_both": both / n, "p_one": one / n,
            "p_neither": neither / n, "identical_answer": identical / n,
            "marginal_accuracy": marginal,
            "delta_pp": (both / n - marginal ** 2) * 100}


def run(dataset: str, condition: str, results_dir: str, swapmap_path: str) -> dict:
    entry = load_swapmap(swapmap_path)[dataset]
    qid_meta = {int(k): v for k, v in entry["qid"].items()}
    if condition not in entry["swap"]:
        raise SystemExit(f"{dataset} has no donor map for {condition}")
    built = build_pairs(qid_meta, entry["swap"][condition])

    out = {"dataset": dataset, "condition": condition,
           "construction": {k: v for k, v in built.items() if k != "pairs"},
           "configurations": {}}
    for base in CONFIGURATIONS[dataset]:
        cells = []
        for prefix in seed_prefixes(results_dir, base, "C0"):
            c0 = load_predictions(results_dir, prefix, "C0")
            if c0 is None or any(q not in c0 for p in built["pairs"] for q in p):
                continue
            cells.append(measure(built["pairs"], c0))
        if not cells:
            continue
        out["configurations"][base] = {
            "seeds": len(cells),
            **{k: statistics.mean(c[k] for c in cells)
               for k in ("p_both", "p_one", "p_neither", "identical_answer",
                         "marginal_accuracy", "delta_pp")}}
    return out


def verdict(record: dict) -> dict:
    """The rule fixed in advance: below 0.5 pp everywhere means it collapses."""
    deltas = {b: r["delta_pp"] for b, r in record["configurations"].items()}
    if not deltas:
        return {"decided": False}
    largest = max(abs(d) for d in deltas.values())
    return {"decided": True, "largest_abs_delta_pp": largest,
            "threshold_pp": THRESHOLD_PP,
            "collapses_into_accuracy": largest < THRESHOLD_PP,
            "signs": sorted({("positive" if d > 0 else "negative" if d < 0 else "zero")
                             for d in deltas.values()})}


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--swapmap", default="data/swapmap.json")
    p.add_argument("--dataset", default="KVASIR")
    p.add_argument("--condition", default="C5")
    p.add_argument("--json")
    return p.parse_args()


def main():
    args = _parse_args()
    r = run(args.dataset, args.condition, args.results_dir, args.swapmap)
    c = r["construction"]
    print(f"\n=== {args.dataset} {args.condition}")
    print(f"  directed pairs (= |I|)     : {c['directed']:,}")
    print(f"  two-cycles collapsed       : {c['two_cycles_collapsed']}")
    print(f"  distinct items involved    : {c['items']:,}")
    print(f"  clusters (distinct images) : {c['clusters']:,}")
    print(f"  pairs per item             : " +
          "  ".join(f"{k}x:{v}" for k, v in c["pairs_per_item"].items()))

    print(f"\n{'configuration':<24}{'P(both)':>10}{'P(one)':>9}{'P(neither)':>12}"
          f"{'same answer':>13}{'a':>8}{'Delta':>10}")
    for base, v in r["configurations"].items():
        print(f"{base:<24}{v['p_both'] * 100:>9.2f}%{v['p_one'] * 100:>8.2f}%"
              f"{v['p_neither'] * 100:>11.2f}%{v['identical_answer'] * 100:>12.2f}%"
              f"{v['marginal_accuracy'] * 100:>7.2f}%{v['delta_pp']:>9.2f}pp")

    d = verdict(r)
    if d["decided"]:
        print(f"\n  rule fixed in advance: |Delta| < {d['threshold_pp']} pp in every "
              f"configuration means the joint adds nothing to accuracy.")
        print(f"  largest |Delta| = {d['largest_abs_delta_pp']:.2f} pp -> "
              f"{'COLLAPSES' if d['collapses_into_accuracy'] else 'DOES NOT COLLAPSE'}")
        print(f"  signs across configurations: {', '.join(d['signs'])}")

    if args.json:
        with open(args.json, "w") as fp:
            json.dump({"record": r, "verdict": d}, fp, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
