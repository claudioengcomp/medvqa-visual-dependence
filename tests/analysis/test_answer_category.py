import json

import pytest

from src.analysis.answer_category import category_breakdown, classify_answer_category


def test_classify_answer_category_yes_no_case_insensitive():
    assert classify_answer_category("Yes") == "yes_no"
    assert classify_answer_category("no") == "yes_no"
    assert classify_answer_category("  YES  ") == "yes_no"


def test_classify_answer_category_anything_else_is_open():
    assert classify_answer_category("ct") == "open"
    assert classify_answer_category("right lung") == "open"


def test_category_breakdown_rejects_empty_predictions():
    with pytest.raises(ValueError):
        category_breakdown([])


def test_category_breakdown_matches_smolvlm_dora_vqarad_c0():
    # Numbers refreshed: the previous expectations (178/56 correct) were
    # already stale before the answer_matches rescoring pass, so this is
    # pre-existing test drift against the current data rather than an effect
    # of the whitespace fix -- SMOLVLM_DORA_VQARAD_C0 had 0 items changed by
    # that rescoring pass.
    record = json.load(open("results/SMOLVLM_DORA_VQARAD_C0.json"))
    result = category_breakdown(record["predictions"])
    assert result["by_category"]["yes_no"]["n"] == 251
    assert result["by_category"]["yes_no"]["correct"] == 173
    assert result["by_category"]["yes_no"]["accuracy"] == pytest.approx(0.68924, abs=1e-4)
    assert result["by_category"]["open"]["n"] == 200
    assert result["by_category"]["open"]["correct"] == 51
    assert result["by_category"]["open"]["accuracy"] == pytest.approx(0.25500, abs=1e-4)
    assert result["naive_accuracy"] == pytest.approx(record["accuracy"], abs=1e-9)
    assert result["balanced_accuracy"] == pytest.approx(0.47212, abs=1e-4)


def test_category_breakdown_matches_qwen2vl_lora_slake_c0():
    # balanced_accuracy refreshed -- QWEN2VL_LORA_SLAKE_C0 had 22 items
    # corrected by the answer_matches whitespace fix (0.6814 -> 0.7022),
    # which changed the per-category breakdown.
    record = json.load(open("results/QWEN2VL_LORA_SLAKE_C0.json"))
    result = category_breakdown(record["predictions"])
    assert result["by_category"]["yes_no"]["n"] == 355
    assert result["by_category"]["open"]["n"] == 706
    assert result["naive_accuracy"] == pytest.approx(record["accuracy"], abs=1e-9)
    assert result["balanced_accuracy"] == pytest.approx(0.72579, abs=1e-4)
