"""Does VDD degrade monotonically with severity under C3 and C4? (Section 5.4)

The paper calls this "a check fixed before the analysis", so the definition is
here, in code, rather than settled after seeing the numbers:

1. Monotonicity is reported BOTH per seed and on the mean over seeds. The claim
   in the paper uses the PER-SEED reading, the conservative one: it must hold in
   all three seeds of a configuration rather than once on their mean.
2. Non-strict (VDD_25 <= VDD_50 <= VDD_75) is primary, since accuracy is
   discrete over a fixed denominator and an exact tie is not a reversal. The
   strict count is reported beside it.
3. The main count uses ZERO tolerance: any decrease as severity rises is a
   violation, however small. Violations smaller than the seed-noise band are
   reported as such, and still counted.
4. C3 and C4 are counted separately, and their conjunction is reported too,
   because a universal claim has to carry the conjunction.

The paper writes C3 for occlusion and C4 for noise; the code records both under
the C4_ prefix, which docs/CONDITION_LABELS.md explains.

Usage:
    python -m src.analysis.severity --dataset KVASIR --json severity.json
    python -m src.analysis.severity --all
"""
from __future__ import annotations

import argparse
import json
import statistics

from src.analysis.identity import CONFIGURATIONS
from src.stats.conditional_vdd import _load

RUNGS = {"C3": "C4_OCC", "C4": "C4_NOISE"}      # paper label -> code prefix
SEVERITIES = (25, 50, 75)
NOISE_BAND_PP = 0.8      # largest seed sd of intact accuracy; secondary reading


def coverage(results_dir: str, datasets) -> dict:
    """Which configuration x seed x rung x severity cells have predictions."""
    present, missing = 0, []
    for dataset in datasets:
        for base in CONFIGURATIONS[dataset]:
            for suffix in ("", "_seed0", "_seed1", "_seed2"):
                prefix = base + suffix
                if _load(results_dir, prefix, "C0") is None:
                    continue
                for prefix_code in RUNGS.values():
                    for sev in SEVERITIES:
                        if _load(results_dir, prefix, f"{prefix_code}_{sev}") is None:
                            missing.append(f"{prefix}_{prefix_code}_{sev}")
                        else:
                            present += 1
    return {"present": present, "missing": missing,
            "cells": present + len(missing)}


def vdd(c0: dict, cx: dict) -> tuple[float | None, int]:
    shared = set(c0) & set(cx)
    if not shared:
        return None, 0
    return (sum(c0[q]["correct"] - cx[q]["correct"]
                for q in shared) / len(shared) * 100, len(shared))


def aggregate(dataset: str, results_dir: str) -> dict:
    out = {}
    for base in CONFIGURATIONS[dataset]:
        prefixes = [base + s for s in ("", "_seed0", "_seed1", "_seed2")
                    if _load(results_dir, base + s, "C0") is not None]
        record = {"dataset": dataset, "seeds": len(prefixes), "rungs": {}}
        for rung, code in RUNGS.items():
            record["rungs"][rung] = {}
            for sev in SEVERITIES:
                per_seed = []
                for prefix in prefixes:
                    c0 = _load(results_dir, prefix, "C0")
                    cx = _load(results_dir, prefix, f"{code}_{sev}")
                    if c0 is None or cx is None:
                        continue
                    value, n = vdd(c0, cx)
                    if value is None:
                        continue
                    per_seed.append({"seed": prefix, "vdd": value, "n": n})
                if not per_seed:
                    continue
                values = [x["vdd"] for x in per_seed]
                record["rungs"][rung][str(sev)] = {
                    "per_seed": per_seed, "vdd_mean": statistics.mean(values),
                    "vdd_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "n": per_seed[0]["n"]}
        out[base] = record
    return out


def monotone(values, strict: bool = False) -> bool:
    return (all(b > a for a, b in zip(values, values[1:])) if strict
            else all(b >= a for a, b in zip(values, values[1:])))


def violations(values) -> list[dict]:
    return [{"from": SEVERITIES[i], "to": SEVERITIES[i + 1],
             "vdd_from": values[i], "vdd_to": values[i + 1],
             "drop_pp": values[i] - values[i + 1]}
            for i in range(len(values) - 1) if values[i + 1] < values[i]]


def verdict(table: dict) -> dict:
    out = {"on_mean": {}, "per_seed": {}, "ties": {}, "violations": []}
    for rung in RUNGS:
        ok_mean, ok_seed, ties = [], [], []
        for base, record in table.items():
            d = record["rungs"].get(rung, {})
            if len(d) != len(SEVERITIES):
                continue
            means = [d[str(s)]["vdd_mean"] for s in SEVERITIES]
            if monotone(means):
                ok_mean.append(base)
                if not monotone(means, strict=True):
                    ties.append(base)
            for v in violations(means):
                out["violations"].append({"scope": "mean", "rung": rung,
                                          "configuration": base, **v})
            all_seeds = True
            for i in range(record["seeds"]):
                try:
                    values = [d[str(s)]["per_seed"][i]["vdd"] for s in SEVERITIES]
                except IndexError:
                    continue
                if not monotone(values):
                    all_seeds = False
                    for v in violations(values):
                        out["violations"].append({
                            "scope": d["25"]["per_seed"][i]["seed"],
                            "rung": rung, "configuration": base, **v})
            if all_seeds:
                ok_seed.append(base)
        out["on_mean"][rung] = ok_mean
        out["per_seed"][rung] = ok_seed
        out["ties"][rung] = ties
    return out


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--dataset")
    p.add_argument("--all", action="store_true")
    p.add_argument("--json")
    return p.parse_args()


def main():
    args = _parse_args()
    datasets = list(CONFIGURATIONS) if args.all or not args.dataset else [args.dataset]

    cov = coverage(args.results_dir, datasets)
    print(f"coverage: {cov['present']}/{cov['cells']} cells")
    if cov["missing"]:
        print(f"  MISSING ({len(cov['missing'])}): {cov['missing'][:8]}")
        print("  the claim shrinks to what was actually evaluated; nothing is "
              "interpolated")
    else:
        print("  no gap")

    table = {}
    for dataset in datasets:
        table.update(aggregate(dataset, args.results_dir))

    print(f"\n{'configuration':<24}{'rung':>6}" +
          "".join(f"{f'VDD{s}':>15}" for s in SEVERITIES) + f"{'mono':>7}")
    for base, record in table.items():
        for rung in RUNGS:
            d = record["rungs"].get(rung, {})
            if len(d) != len(SEVERITIES):
                print(f"{base:<24}{rung:>6}   INCOMPLETE")
                continue
            means = [d[str(s)]["vdd_mean"] for s in SEVERITIES]
            sds = [d[str(s)]["vdd_sd"] for s in SEVERITIES]
            print(f"{base:<24}{rung:>6}"
                  + "".join(f"{m:>9.1f}+-{s:<4.1f}" for m, s in zip(means, sds))
                  + f"{'yes' if monotone(means) else 'NO':>7}")

    v = verdict(table)
    total = len(table)
    print(f"\nverdict (zero tolerance, non-strict)")
    for rung in RUNGS:
        print(f"  {rung}: on the mean {len(v['on_mean'][rung])}/{total}"
              f"   |   per seed {len(v['per_seed'][rung])}/{total}"
              f"   |   with a tie {len(v['ties'][rung])}")
    both_mean = set(v["on_mean"]["C3"]) & set(v["on_mean"]["C4"])
    both_seed = set(v["per_seed"]["C3"]) & set(v["per_seed"]["C4"])
    print(f"  C3 and C4 together: on the mean {len(both_mean)}/{total}"
          f"   |   per seed {len(both_seed)}/{total}")

    if v["violations"]:
        print(f"\n  violations ({len(v['violations'])}), named:")
        for d in sorted(v["violations"], key=lambda x: -x["drop_pp"]):
            band = ("  (below the seed-noise band)" if d["drop_pp"] < NOISE_BAND_PP
                    else "")
            print(f"    {d['configuration']:<24}{d['rung']:>4}  {d['scope']:<32}"
                  f"{d['from']}->{d['to']}  {d['vdd_from']:.1f} -> "
                  f"{d['vdd_to']:.1f}  drop {d['drop_pp']:.2f} pp{band}")
    else:
        print("\n  no violation")

    if args.json:
        with open(args.json, "w") as fp:
            json.dump({"coverage": cov, "table": table, "verdict": v,
                       "tolerance_pp": 0.0,
                       "noise_band_pp_secondary": NOISE_BAND_PP}, fp, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
