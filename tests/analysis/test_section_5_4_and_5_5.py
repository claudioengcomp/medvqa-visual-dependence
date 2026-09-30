"""Tests for the complementary pairs (Section 5.5) and the dispersion of
Section 5.4: the permutation and Welch tests, and the severity monotonicity.

Two rules of these sections were fixed BEFORE the analysis and are pinned here
so that a later change cannot quietly relax them: the 0.5 pp threshold of
Section 3.6, and the zero tolerance of the monotonicity check.
"""
from __future__ import annotations

import pytest

from src.analysis.complementary_pairs import build_pairs, run as pairs_run, verdict
from src.analysis.ccr import load_swapmap
from src.analysis.severity import (RUNGS, aggregate, coverage, monotone,
                                   verdict as severity_verdict)
from src.stats.spread_tests import run as spread_run

RESULTS = "results"
SWAPMAP = "data/swapmap.json"


# ---------------------------------------------------------------- Section 5.5
@pytest.fixture(scope="module")
def pairs():
    return pairs_run("KVASIR", "C5", RESULTS, SWAPMAP)


def test_pair_construction_matches_section_5_5(pairs):
    c = pairs["construction"]
    assert c["directed"] == 5216
    assert c["two_cycles_collapsed"] == 0     # the derangement has none here
    assert c["items"] == 7007
    assert c["clusters"] == 412


def test_deduplication_keeps_pairs_unordered_and_drops_no_item():
    entry = load_swapmap(SWAPMAP)["KVASIR"]
    qid_meta = {int(k): v for k, v in entry["qid"].items()}
    built = build_pairs(qid_meta, entry["swap"]["C5"])
    seen = {frozenset(p) for p in built["pairs"]}
    assert len(seen) == len(built["pairs"])           # no duplicate pair
    assert all(a < b for a, b in built["pairs"])      # canonical order
    involved = {q for p in built["pairs"] for q in p}
    assert len(involved) == built["items"]


def test_joint_and_marginal_match_section_5_5(pairs):
    both = [v["p_both"] for v in pairs["configurations"].values()]
    marginal = [v["marginal_accuracy"] for v in pairs["configurations"].values()]
    same = [v["identical_answer"] for v in pairs["configurations"].values()]
    assert min(both) == pytest.approx(0.642, abs=5e-4)
    assert max(both) == pytest.approx(0.669, abs=5e-4)
    assert min(marginal) == pytest.approx(0.803, abs=5e-4)
    assert max(marginal) == pytest.approx(0.817, abs=5e-4)
    assert min(same) == pytest.approx(0.058, abs=5e-4)
    assert max(same) == pytest.approx(0.087, abs=5e-4)


def test_delta_values_and_the_threshold_fixed_in_advance(pairs):
    deltas = sorted(v["delta_pp"] for v in pairs["configurations"].values())
    for got, want in zip(deltas, [-0.25, -0.03, 0.01, 0.18, 0.50]):
        assert got == pytest.approx(want, abs=5e-3)
    d = verdict(pairs)
    assert d["threshold_pp"] == 0.5           # not to be relaxed to fit a result
    assert d["collapses_into_accuracy"] is True
    assert d["signs"] == ["negative", "positive"]   # fluctuation around zero


# ---------------------------------------------------------------- Section 5.4
@pytest.mark.parametrize("condition,range_pp,p_perm,p_welch", [
    ("C1", 9.72, 0.0426, 0.0529),
    ("C2", 1.59, 0.0099, 0.0004),
    ("C5", 1.79, 0.0063, 0.0009),
    ("C6", 1.67, 0.0153, 0.0010)])
def test_spread_tests_match_section_5_4(condition, range_pp, p_perm, p_welch):
    r = spread_run("KVASIR", condition, RESULTS, SWAPMAP, resamples=100_000)
    assert r["range_pp"] == pytest.approx(range_pp, abs=5e-3)
    assert r["p_permutation"] == pytest.approx(p_perm, abs=5e-4)
    assert r["p_welch"] == pytest.approx(p_welch, abs=5e-5)


def test_c1_is_where_the_two_tests_disagree():
    """The condition with the largest spread has the weakest evidence."""
    c1 = spread_run("KVASIR", "C1", RESULTS, SWAPMAP, resamples=100_000)
    c5 = spread_run("KVASIR", "C5", RESULTS, SWAPMAP, resamples=100_000)
    assert c1["range_pp"] > c5["range_pp"]                 # larger spread
    assert c1["p_permutation"] < 0.05 < c1["p_welch"]      # and they straddle 0.05
    assert c5["p_permutation"] < 0.05 and c5["p_welch"] < 0.05
    assert c1["variance_ratio"] > 1000      # why permutation is anticonservative


def test_severity_coverage_has_no_gap():
    cov = coverage(RESULTS, ["KVASIR", "SLAKE", "VQARAD"])
    assert cov["missing"] == []
    assert cov["present"] == 270            # 15 configs x 3 seeds x 2 rungs x 3


@pytest.fixture(scope="module")
def severity_table():
    table = {}
    for dataset in ("KVASIR", "SLAKE", "VQARAD"):
        table.update(aggregate(dataset, RESULTS))
    return table


def test_monotonicity_matches_section_5_4(severity_table):
    v = severity_verdict(severity_table)
    assert len(severity_table) == 15
    assert len(v["on_mean"]["C3"]) == 15
    assert len(v["on_mean"]["C4"]) == 15     # all 15 on the mean over seeds
    assert len(v["per_seed"]["C3"]) == 15
    assert len(v["per_seed"]["C4"]) == 14    # 44 of the 45 cells


def test_the_single_violation_is_named(severity_table):
    v = severity_verdict(severity_table)
    per_seed = [d for d in v["violations"] if d["scope"] != "mean"]
    assert len(per_seed) == 1
    d = per_seed[0]
    assert d["configuration"] == "QWEN2VL_DORA_VQARAD"
    assert d["rung"] == "C4"
    assert (d["from"], d["to"]) == (50, 75)
    assert d["drop_pp"] == pytest.approx(0.44, abs=5e-3)


def test_no_tie_so_strict_and_non_strict_agree(severity_table):
    """The non-strict choice turned out to be inert; that is worth pinning."""
    v = severity_verdict(severity_table)
    assert v["ties"]["C3"] == [] and v["ties"]["C4"] == []
    for record in severity_table.values():
        for rung in RUNGS:
            d = record["rungs"][rung]
            means = [d[str(s)]["vdd_mean"] for s in (25, 50, 75)]
            assert monotone(means) == monotone(means, strict=True)
