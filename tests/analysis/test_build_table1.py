"""Tests for the Table 1 builder.

The point of the builder is that the table is regenerable, so the tests check
that it reproduces the published cells rather than that it runs. They also pin
the two rules of Section 4.3 that a table builder is most likely to break:
seeds are summarized, never pooled, and the partial interventions are
restricted to the questions they touch.
"""
import os

import pytest

from src.analysis.build_table1 import CONDITIONS, build, row

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SWAPMAP = os.path.join(REPO, "data", "swapmap.json")
RESULTS = os.path.join(REPO, "results")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(SWAPMAP) and os.path.isdir(RESULTS)),
    reason="swapmap or results not available")

# Published Table 1, Kvasir-VQA-x1: Acc(C0), VDD(C1), VDD(C2), VDD(C5*), VDD(C6*)
PUBLISHED = {
    "M3AE_KVASIR": (87.1, 28.6, 32.0, 46.4, 42.0),
    "QWEN2VL_LORA_KVASIR": (86.5, 37.3, 31.1, 46.9, 42.3),
    "QWEN2VL_DORA_KVASIR": (86.7, 34.4, 31.4, 46.9, 42.4),
    "SMOLVLM_LORA_KVASIR": (87.6, 38.3, 32.7, 48.2, 43.6),
    "SMOLVLM_DORA_KVASIR": (86.9, 37.6, 32.5, 47.1, 42.6),
}
COVERAGE = {"C1": 10779, "C2": 10779, "C5": 8940, "C6": 10271}


@pytest.fixture(scope="module")
def table():
    return build(RESULTS, SWAPMAP)


@pytest.mark.parametrize("base,expected", PUBLISHED.items())
def test_reproduces_the_published_cells(table, base, expected):
    record = table[base]
    assert round(record["accuracy_c0"][0], 1) == expected[0]
    for condition, value in zip(CONDITIONS, expected[1:]):
        assert round(record["conditions"][condition]["vdd"][0], 1) == value, condition


@pytest.mark.parametrize("condition,n", COVERAGE.items())
def test_partial_interventions_are_restricted_to_what_they_touch(table, condition, n):
    """C5 and C6 are scored on the questions whose image received a donor."""
    assert table["M3AE_KVASIR"]["conditions"][condition]["n"] == n


def test_every_fine_tuned_row_aggregates_three_seeds(table):
    for base, record in table.items():
        expected = 1 if ("MEDVLTHINKER" in base or "M0_" in base) else 3
        assert record["seeds"] == expected, base


def test_mcnemar_is_counted_per_seed_never_pooled(table):
    """Significance is a count out of the seeds, not a single pooled test."""
    for record in table.values():
        for condition in record["conditions"].values():
            assert 0 <= condition["significant_seeds"] <= condition["seeds"]


def test_ccr_comes_from_the_swap_with_the_largest_informative_set(table):
    assert table["M3AE_KVASIR"]["ccr_n"] == 5218          # Kvasir: C5
    assert table["M3AE_SCRATCH_SLAKE"]["ccr_n"] == 92     # SLAKE: C6
    assert table["M3AE_SCRATCH_VQARAD"]["ccr"] == (None, None)


def test_single_run_models_carry_no_standard_deviation(table):
    record = table["MEDVLTHINKER_ZEROSHOT_KVASIR"]
    assert record["seeds"] == 1
    assert record["accuracy_c0"][1] == 0.0
