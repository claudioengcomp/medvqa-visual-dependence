"""Tests for the CCR slice and subset reports.

These modules exist because the numbers of Section 5.4 used to be produced by
ad-hoc commands that were never committed. The tests pin the structural
quantities those numbers rest on -- how many questions fall in each slice, how
many image pairs stay within a procedure -- so a future edit cannot move a
published denominator without a test failing.
"""
import os

import pytest

from src.analysis.ccr import load_swapmap
from src.analysis.ccr_sameproc import (procedure_by_image, split_donor_map)
from src.analysis.ccr_slices import is_list_answer, run, seed_prefixes

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SWAPMAP = os.path.join(REPO, "data", "swapmap.json")
RESULTS = os.path.join(REPO, "results")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(SWAPMAP) and os.path.isdir(RESULTS)),
    reason="swapmap or results not available")


def test_list_slice_triggers_on_either_side():
    assert is_list_answer(["flesh; red"], ["white"]) is True
    assert is_list_answer(["white"], ["flesh; red"]) is True
    assert is_list_answer(["white"], ["red"]) is False


def test_kvasir_c5_slice_sizes_match_the_paper():
    """5,218 informative per seed = 3,936 single-valued + 1,282 list."""
    record = run("KVASIR", "C5", RESULTS, SWAPMAP)["M3AE_KVASIR"]
    assert record["all"]["n"] == 5218
    assert record["single"]["n"] == 3936
    assert record["list"]["n"] == 1282
    assert record["single"]["n"] + record["list"]["n"] == record["all"]["n"]


def test_slake_c6_is_all_single_valued():
    """92 informative per seed, none of them a list (Section 5.4)."""
    record = run("SLAKE", "C6", RESULTS, SWAPMAP)["M3AE_SCRATCH_SLAKE"]
    assert (record["all"]["n"], record["single"]["n"], record["list"]["n"]) == (92, 92, 0)


def test_vqarad_ccr_stays_unreportable():
    """Fewer than four informative questions, so no CCR is reported there."""
    for condition in ("C2", "C6"):
        for record in run("VQARAD", condition, RESULTS, SWAPMAP).values():
            assert record["all"]["n"] < 4


def test_three_seeds_are_found_for_every_fine_tuned_configuration():
    for base in ("M3AE_KVASIR", "SMOLVLM_LORA_KVASIR"):
        assert len(seed_prefixes(RESULTS, base, "C5")) == 3
    for base in ("M3AE_SCRATCH_SLAKE", "QWEN2VL_LORA_SLAKE"):
        assert len(seed_prefixes(RESULTS, base, "C6")) == 3
    # The zero-shot model is evaluated once (Section 4.3).
    assert len(seed_prefixes(RESULTS, "MEDVLTHINKER_ZEROSHOT_KVASIR", "C5")) == 1


def test_same_procedure_partition_matches_the_paper():
    """240 pairs stay within a procedure, 369 cross it (Section 5.3)."""
    entry = load_swapmap(SWAPMAP)["KVASIR"]
    qid_meta = {int(k): v for k, v in entry["qid"].items()}
    parts = split_donor_map(entry["swap"]["C5"], procedure_by_image(qid_meta))
    assert len(parts["same"]) == 240
    assert len(parts["cross"]) == 369
    assert len(parts["unlabelled"]) == 8
    assert sum(len(p) for p in parts.values()) == len(entry["swap"]["C5"])


def test_procedure_labels_come_from_the_dataset_itself():
    entry = load_swapmap(SWAPMAP)["KVASIR"]
    procedure = procedure_by_image({int(k): v for k, v in entry["qid"].items()})
    assert set(procedure.values()) == {"colonoscopy", "gastroscopy"}
    assert len(procedure) == 776
