import json
from pathlib import Path

import pytest

from src.baselines.question_only import (
    evaluate_question_only_model,
    load_closed_qa_pairs,
    train_question_only_model,
)

SLAKE_DIR = Path.home() / "Downloads" / "Slake1.0"
SLAKE_TRAIN_JSON = SLAKE_DIR / "train.json"
SLAKE_TEST_JSON = SLAKE_DIR / "test.json"


def _write_json(path, samples):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(samples, fp)


def test_load_closed_qa_pairs_filters_language_and_type(tmp_path):
    samples = [
        {"qid": 1, "question": "Is there a tumor?", "answer": "Yes", "q_lang": "en", "answer_type": "CLOSED"},
        {"qid": 2, "question": "Existe um tumor?", "answer": "Sim", "q_lang": "zh", "answer_type": "CLOSED"},
        {"qid": 3, "question": "What organ is this?", "answer": "Liver", "q_lang": "en", "answer_type": "OPEN"},
    ]
    path = tmp_path / "data.json"
    _write_json(path, samples)

    pairs = load_closed_qa_pairs(str(path))

    assert len(pairs) == 1
    assert pairs[0] == {"qid": 1, "question": "Is there a tumor?", "answer": "yes"}


def test_train_and_evaluate_perfectly_memorizable_case():
    train_pairs = [
        {"qid": 1, "question": "Is there a tumor?", "answer": "yes"},
        {"qid": 2, "question": "Is this a chest x-ray?", "answer": "no"},
        {"qid": 3, "question": "Is there a tumor?", "answer": "yes"},
        {"qid": 4, "question": "Is this a chest x-ray?", "answer": "no"},
    ]
    test_pairs = [
        {"qid": 5, "question": "Is there a tumor?", "answer": "yes"},
        {"qid": 6, "question": "Is this a chest x-ray?", "answer": "no"},
    ]

    model = train_question_only_model(train_pairs)
    results = evaluate_question_only_model(model, test_pairs, condition="C1")

    assert results["accuracy"] == 1.0
    assert results["n"] == 2
    assert results["n_correct"] == 2
    assert results["condition"] == "C1"
    assert len(results["predictions"]) == 2
    assert all(p["correct"] for p in results["predictions"])


def test_train_question_only_model_rejects_empty_input():
    with pytest.raises(ValueError):
        train_question_only_model([])


@pytest.mark.skipif(
    not (SLAKE_TRAIN_JSON.exists() and SLAKE_TEST_JSON.exists()),
    reason=f"real SLAKE data not found at {SLAKE_DIR}",
)
def test_m0_real_slake_result_is_locked():
    """Locks the real Step-5 M0 result so a silent drift (sklearn bump,
    accidental edit to load_closed_qa_pairs filtering, etc.) fails loudly.

    M0 is the comparison floor every later vision-using model gets measured
    against, so this number must not move without someone noticing.
    """
    train_pairs = load_closed_qa_pairs(str(SLAKE_TRAIN_JSON))
    test_pairs = load_closed_qa_pairs(str(SLAKE_TEST_JSON))

    model = train_question_only_model(train_pairs)
    results = evaluate_question_only_model(model, test_pairs, condition="C0")

    assert results["n"] == 416
    assert results["n_correct"] == 267
    assert results["accuracy"] == pytest.approx(267 / 416)
