"""Tests for the identity check of Section 5.1.

The identity is the central claim of the paper, so what is pinned here is not
only the headline count but WHICH cells are the exceptions: a change that kept
"27 of 30" while moving the exceptions elsewhere would be a different result
wearing the same number.
"""
from __future__ import annotations

import pytest

from src.analysis.identity import REPORTED, run, strata, summarize
from src.analysis.ccr import load_swapmap

RESULTS = "results"
SWAPMAP = "data/swapmap.json"


@pytest.fixture(scope="module")
def records():
    return [run(ds, cond, RESULTS, SWAPMAP) for ds, cond in REPORTED]


def test_strata_partition_every_touched_question_exactly_once():
    entry = load_swapmap(SWAPMAP)["KVASIR"]
    qid_meta = {int(k): v for k, v in entry["qid"].items()}
    parts, donor_qid, donor_ref = strata(qid_meta, entry["swap"]["C5"])
    buckets = [set(parts[k]) for k in ("informative", "same_ref", "novel")]
    assert not (buckets[0] & buckets[1]) and not (buckets[0] & buckets[2])
    assert not (buckets[1] & buckets[2])
    touched = sum(1 for m in qid_meta.values() if m["img"] in entry["swap"]["C5"])
    assert sum(len(b) for b in buckets) == touched
    # a donor item is named for exactly the questions the donor carries
    assert set(donor_qid) == buckets[0] | buckets[1]
    assert set(donor_ref) == set(donor_qid)


@pytest.mark.parametrize("dataset,condition,informative", [
    ("KVASIR", "C5", 5216), ("SLAKE", "C6", 92)])
def test_informative_set_sizes_match_the_paper(dataset, condition, informative):
    r = run(dataset, condition, RESULTS, SWAPMAP)
    assert r["informative"] == informative


def test_kvasir_c5_strata_sizes_match_section_5_2(records):
    """|I| = 5,216, |U| = 2,886 and |N| = 832 over |S_T| = 8,934 (corrected donor map, Section 5.2)."""
    r = records[0]
    assert (r["informative"], r["same_ref"], r["novel"]) == (5216, 2886, 832)
    assert r["informative"] + r["same_ref"] + r["novel"] == 8934


def test_identity_holds_in_27_of_30_cells(records):
    s = summarize(records)
    assert s["cells"] == 30
    assert s["exact"] == 27


def test_the_three_exceptions_are_the_ones_the_paper_names(records):
    """Section 5.1 names them: one SLAKE classifier seed and two SmolVLM2 seeds."""
    s = summarize(records)
    by_cell = {e["cell"]: e for e in s["exceptions"]}
    assert set(by_cell) == {"M3AE_SCRATCH_SLAKE",
                            "SMOLVLM_LORA_KVASIR_seed2",
                            "SMOLVLM_DORA_KVASIR_seed0"}
    assert by_cell["M3AE_SCRATCH_SLAKE"]["agreement"] == pytest.approx(0.8804, abs=5e-5)
    assert by_cell["SMOLVLM_LORA_KVASIR_seed2"]["agreement"] == pytest.approx(0.9898, abs=5e-5)
    assert by_cell["SMOLVLM_DORA_KVASIR_seed0"]["agreement"] == pytest.approx(0.9912, abs=5e-5)


def test_the_two_kvasir_exceptions_stay_below_the_run_to_run_band(records):
    """Section 5.6 measures 2.7-5.9% disagreement across environments."""
    s = summarize(records)
    for e in s["exceptions"]:
        if e["cell"].endswith("KVASIR_seed2") or e["cell"].endswith("KVASIR_seed0"):
            assert (1 - e["agreement"]) * 100 < 2.7


def test_ccr_equals_donor_accuracy_wherever_the_identity_holds(records):
    """What the identity asserts, cell by cell, not only in the aggregate."""
    for r in records:
        for name, c in r["cells"].items():
            if not c["n"] or c["agreement"] != 1.0:
                continue
            assert c["ccr"] == pytest.approx(c["donor_accuracy"], abs=1e-12), name
            assert c["difference_pp"] == pytest.approx(0.0, abs=1e-9), name
