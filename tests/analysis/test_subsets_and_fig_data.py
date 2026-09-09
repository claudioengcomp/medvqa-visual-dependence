"""Tests for the subset VDD report and the figure-data builder.

Both exist to close the same gap: the numbers of Section 5.3 and the input of
every figure had no generator, so they could be read but not recomputed. The
tests pin what the paper states about the subsets, and check that rebuilding the
figure data from the predictions reproduces the aggregate the figures were drawn
from.
"""
import os

import pytest

from src.analysis.build_fig_data import MIN_VERIFIABLE, combos_block, fvsvdd_block
from src.analysis.subset_vdd import run as subset_run

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SWAPMAP = os.path.join(REPO, "data", "swapmap.json")
RESULTS = os.path.join(REPO, "results")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(SWAPMAP) and os.path.isdir(RESULTS)),
    reason="swapmap or results not available")


@pytest.fixture(scope="module")
def subsets():
    return subset_run(RESULTS, SWAPMAP)


def test_subset_sizes_match_the_paper(subsets):
    """240 pairs stay within the procedure, 369 cross it, 166 enter C6 only."""
    assert subsets["same"]["pairs"] == 240
    assert subsets["cross"]["pairs"] == 369
    assert subsets["c6_only"]["pairs"] == 166


def test_the_nearer_donor_changes_more_answers_and_costs_less(subsets):
    """Section 5.3: 68% of answers change within a procedure, 62% across it,
    yet the crossing swap costs more -- that is the whole point of the ratio."""
    assert round(subsets["same"]["changed_fraction"], 3) == 0.681
    assert round(subsets["cross"]["changed_fraction"], 3) == 0.622
    # The claim is about the five fine-tuned configurations; the zero-shot
    # model barely reacts to either swap and is reported separately.
    def swap_vdds(name):
        return [m["vdd_swap"][0] for base, m in subsets[name]["models"].items()
                if "MEDVLTHINKER" not in base]
    assert max(swap_vdds("same")) < min(swap_vdds("cross"))


def test_cost_per_changed_answer_matches_the_published_ratios(subsets):
    def ratios(name):
        return [m["cost_per_changed_answer"]
                for base, m in subsets[name]["models"].items()
                if "MEDVLTHINKER" not in base]
    assert [round(r, 2) for r in sorted(ratios("same"))][0] == 0.59
    assert [round(r, 2) for r in sorted(ratios("same"))][-1] == 0.63
    assert [round(r, 2) for r in sorted(ratios("cross"))][0] == 0.80
    assert [round(r, 2) for r in sorted(ratios("cross"))][-1] == 0.83


def test_blank_frame_is_scored_on_the_same_questions(subsets):
    """The comparison of Section 5.3 is paired: 40.6 against 36.0 for the
    classifier on the same-procedure subset, not against a different set."""
    m3ae = subsets["same"]["models"]["M3AE_KVASIR"]
    assert round(m3ae["vdd_swap"][0], 1) == 40.6
    assert round(m3ae["vdd_blank_same_questions"][0], 1) == 36.0


def test_fig_data_combos_reproduce_table_1_accuracies():
    combos = combos_block(RESULTS, SWAPMAP)
    assert round(combos["M3AE_KVASIR"]["acc0"][0] * 100, 1) == 87.1
    assert round(combos["M0_question_only_SLAKE"]["acc0_closed"][0] * 100, 1) == 64.2
    # VQA-RAD has no C5, and the key is present as null rather than absent.
    assert combos["M3AE_SCRATCH_VQARAD"]["C5e"] is None


def test_fig_data_drops_subsets_too_small_to_carry_a_fraction():
    """VQA-RAD has one or two verifiable answers per swap: a 'fraction of
    answers changed' computed on those is noise, and is left out."""
    rows = fvsvdd_block(RESULTS, SWAPMAP)
    assert rows, "no rows built"
    assert all(row["nverif"] >= MIN_VERIFIABLE for row in rows)
    assert not [row for row in rows if row["ds"] == "VQARAD"]
    assert {row["subset"] for row in rows if row["ds"] == "KVASIR"} == {
        "all", "same", "cross"}
