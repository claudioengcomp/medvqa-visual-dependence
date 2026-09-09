"""Classifies each prediction item into an answer category derived from the
gold answer, and computes per-category accuracy plus balanced accuracy.

This is a proxy, and a different quantity from the official "open/closed"
question-type label of SLAKE/VQA-RAD. It buckets by the shape of the gold
answer: 'yes_no' covers the two most common gold answers of closed-form
Med-VQA, and any other gold answer is classified as 'open'. The proxy was
validated against the real data before being adopted.

Do NOT use it to reproduce the question-type split of Section 5.2. The
official label is available: it travels in data/swapmap.json as the `t` field
of each question and reproduces the counts the paper reports (416 closed and
645 open on SLAKE, 272 and 179 on VQA-RAD). src/analysis/question_type.py
reads it. It is absent only from results/*.json, which stores just
qid/predicted/gold/correct (see src/inference/results_schema.py) -- an earlier
version of this note read that absence as the label being unrecoverable, which
was true of the results files alone and false of the repository."""
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
