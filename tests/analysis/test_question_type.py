"""Tests for the question-type split of Section 5.2.

The label these tests defend is the benchmarks' own, carried by the swap map.
A previous note in answer_category.py claimed it was unrecoverable, and the
numbers of Section 5.2 were produced outside the repository as a result; the
counts below are what makes the claim checkable from a clean checkout.
"""
import os

import pytest

from src.analysis.ccr import load_swapmap
from src.analysis.question_type import CLOSED, OPEN, qids_by_type, run

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SWAPMAP = os.path.join(REPO, "data", "swapmap.json")
RESULTS = os.path.join(REPO, "results")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(SWAPMAP) and os.path.isdir(RESULTS)),
    reason="swapmap or results not available")

# Counts reported in Section 4.1 and the footnote of Table 1.
PUBLISHED_COUNTS = {"SLAKE": (416, 645), "VQARAD": (272, 179), "KVASIR": (10779, 0)}


@pytest.mark.parametrize("dataset,expected", PUBLISHED_COUNTS.items())
def test_official_label_reproduces_the_published_counts(dataset, expected):
    entry = load_swapmap(SWAPMAP)[dataset]
    subsets = qids_by_type({int(k): v for k, v in entry["qid"].items()})
    assert (len(subsets[CLOSED]), len(subsets[OPEN])) == expected


def test_slake_closed_accuracy_matches_the_floor_comparison():
    """Section 5.2 compares against a baseline trained on closed questions,
    so the accuracies it quotes must be on the closed subset."""
    result = run("SLAKE", "C1", RESULTS, SWAPMAP)
    m3ae = result["M3AE_SCRATCH_SLAKE"]["closed"]
    assert round(m3ae["accuracy_c0"][0], 1) == 87.3
    assert round(m3ae["accuracy_condition"][0], 1) == 57.8
    assert round(result["M0_question_only_SLAKE"]["closed"]["accuracy_c0"][0], 1) == 64.2


def test_open_questions_lose_more_than_closed_ones():
    """SmolVLM2 with LoRA on SLAKE: 56.5 against 32.9 percentage points."""
    record = run("SLAKE", "C1", RESULTS, SWAPMAP)["SMOLVLM_LORA_SLAKE"]
    assert round(record["open"]["vdd"][0], 1) == 56.5
    assert round(record["closed"]["vdd"][0], 1) == 32.9
    assert record["open"]["vdd"][0] > record["closed"]["vdd"][0]


def test_question_only_baseline_has_no_open_questions():
    """It is trained and evaluated on closed questions alone."""
    record = run("SLAKE", "C1", RESULTS, SWAPMAP)["M0_question_only_SLAKE"]
    assert "closed" in record
    assert "open" not in record
