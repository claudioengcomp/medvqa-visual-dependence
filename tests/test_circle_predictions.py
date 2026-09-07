import pytest

from src.inference.circle_predictions import build_results_record, score_prediction
from src.inference.results_schema import ResultsSchemaError


def test_score_prediction_exact_match_is_correct():
    extracted, correct = score_prediction("yes", "Is there a tumor?", ["yes"])
    assert extracted == "yes"
    assert correct is True


def test_score_prediction_mismatch_is_incorrect():
    extracted, correct = score_prediction("no", "Is there a tumor?", ["yes"])
    assert extracted == "no"
    assert correct is False


def test_score_prediction_matches_any_gold_answer():
    _, correct = score_prediction("left lung", "Where is it?", ["right lung", "left lung"])
    assert correct is True


def test_score_prediction_strips_thinking_block():
    raw = "<think>reasoning...</think>The answer is: yes"
    extracted, correct = score_prediction(raw, "Is there a tumor?", ["yes"])
    assert extracted == "yes"
    assert correct is True


def test_score_prediction_normalizes_case_and_punctuation():
    _, correct = score_prediction("Yes.", "Is there a tumor?", ["yes"])
    assert correct is True


def test_build_results_record_computes_accuracy_from_predictions():
    predictions = [
        {"qid": 1, "predicted": "yes", "gold": "yes", "correct": True},
        {"qid": 2, "predicted": "no", "gold": "yes", "correct": False},
    ]
    record = build_results_record("QWEN2VL_LORA", "SLAKE", "C0", predictions)
    assert record["n"] == 2
    assert record["n_correct"] == 1
    assert record["accuracy"] == 0.5


def test_build_results_record_rejects_invalid_condition():
    predictions = [{"qid": 1, "predicted": "yes", "gold": "yes", "correct": True}]
    with pytest.raises(ResultsSchemaError):
        build_results_record("QWEN2VL_LORA", "SLAKE", "NOT_A_CONDITION", predictions)
