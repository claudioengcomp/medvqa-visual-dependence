"""Pure per-item scoring logic for the generative pipeline (Qwen2-VL,
SmolVLM2 etc.) -- it reuses the exact answer-extraction and normalization
functions applied by the training-time accuracy metric
(src/inference/process_text.py: extract_answer, normalize_text,
answer_matches) rather than reimplementing them, so scoring here cannot drift
from scoring there. Deliberately free of any torch/transformers dependency,
which lets this module be unit-tested locally without the GPU stack."""
from __future__ import annotations

from src.inference.process_text import answer_matches, extract_answer, normalize_text

from src.inference.results_schema import validate_results_dict


def score_prediction(
    raw_prediction: str, question: str, gold_answers: list[str]
) -> tuple[str, bool]:
    """Extract the core answer from `raw_prediction` (strip reasoning blocks,
    honour the 'the answer is:' marker, truncate open-ended answers) and check
    whether it matches any of the `gold_answers` after normalization -- the
    exact same logic the aggregate accuracy metric applies internally
    (extract_answer first, then normalize_text), applied here per item rather
    than per batch, so that `predicted`/`correct` can be recorded for each
    individual question."""
    extracted = extract_answer(raw_prediction, question)
    extracted_norm = normalize_text(extracted)
    gold_norms = [normalize_text(a) for a in gold_answers]
    correct = answer_matches(extracted_norm, gold_norms)
    return extracted, correct


def build_results_record(
    model: str, dataset: str, condition: str, predictions: list[dict]
) -> dict:
    """Assemble the complete results record (see results_schema) from a list
    of already-scored per-item predictions ({"qid", "predicted", "gold",
    "correct"}), computing accuracy/n/n_correct directly from them -- the same
    format written by src/inference/extract_predictions_m3ae.py, so it
    inherits the same statistical apparatus (VDD, bootstrap CIs, paired
    McNemar)."""
    n = len(predictions)
    n_correct = sum(1 for p in predictions if p["correct"])
    record = {
        "model": model,
        "dataset": dataset,
        "condition": condition,
        "accuracy": (n_correct / n) if n else 0.0,
        "n": n,
        "n_correct": n_correct,
        "predictions": predictions,
    }
    validate_results_dict(record)
    return record
