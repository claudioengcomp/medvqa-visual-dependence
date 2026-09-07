"""Classifies each prediction item into an answer category derived from the
gold answer, and computes per-category accuracy plus balanced accuracy.

This is a proxy, not the official "open/closed" question-type label of
SLAKE/VQA-RAD: the official question-type label was not preserved in
results/*.json, which only stores qid/predicted/gold/correct (see
src/inference/results_schema.py). Recovering the official label would require the
original preprocessed dataset shards, which are no longer available in the
execution environment used for the runs. 'yes_no' covers the two most common
gold answers of closed-form Med-VQA; any other gold answer is classified as
'open'. The proxy was validated against the real data before being adopted."""
from __future__ import annotations

_YES_NO = {"yes", "no"}


def classify_answer_category(gold: str) -> str:
    return "yes_no" if gold.strip().lower() in _YES_NO else "open"


def category_breakdown(predictions: list[dict]) -> dict:
    """Returns {"by_category": {category: {n, correct, accuracy}},
    "balanced_accuracy": the macro average, unweighted by the size of each
    category, "naive_accuracy": the ordinary aggregate accuracy (weighted by
    category size), identical to the 'accuracy' already reported elsewhere}."""
    if not predictions:
        raise ValueError("predictions is empty")

    stats: dict[str, dict] = {}
    total_correct = 0
    for p in predictions:
        cat = classify_answer_category(p["gold"])
        bucket = stats.setdefault(cat, {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += int(p["correct"])
        total_correct += int(p["correct"])

    for bucket in stats.values():
        bucket["accuracy"] = bucket["correct"] / bucket["n"]

    balanced_accuracy = sum(b["accuracy"] for b in stats.values()) / len(stats)
    naive_accuracy = total_correct / len(predictions)

    return {
        "by_category": stats,
        "balanced_accuracy": balanced_accuracy,
        "naive_accuracy": naive_accuracy,
    }
