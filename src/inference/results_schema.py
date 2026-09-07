"""Schema and validation for the evaluation harness' results records.

A "results record" describes the performance of ONE model under ONE condition
(C0/C1/C2) on ONE dataset. Two levels of detail are supported:

- Aggregate mode (always mandatory): model, dataset, condition, accuracy.
  'n'/'n_correct' are optional, but required for bootstrap CIs.
- Paired mode (optional): 'predictions', a list of
  {"qid": ..., "predicted": str, "gold": str, "correct": bool}, one entry per
  item. Required for the McNemar test and the paired bootstrap.
"""
from __future__ import annotations

REQUIRED_KEYS = ("model", "dataset", "condition", "accuracy")
VALID_CONDITIONS = (
    "C0", "C1", "C2", "C5", "C6",
    "C4_OCC_25", "C4_OCC_50", "C4_OCC_75",
    "C4_NOISE_25", "C4_NOISE_50", "C4_NOISE_75",
    # C7 is reserved: a positional-occlusion placebo condition. It was
    # renamed from C6 to C7 once C6 came to mean "same organ + modality,
    # free category" swap. Not implemented yet -- listed here only to
    # reserve the name so it cannot be reused for something else.
    "C7",
)


class ResultsSchemaError(ValueError):
    pass


def validate_results_dict(d: dict) -> None:
    """Raise ResultsSchemaError if `d` does not follow the schema. Returns nothing on success."""
    missing = [k for k in REQUIRED_KEYS if k not in d]
    if missing:
        raise ResultsSchemaError(f"Missing required fields: {missing}")

    if d["condition"] not in VALID_CONDITIONS:
        raise ResultsSchemaError(
            f"condition={d['condition']!r} is invalid, it must be one of {VALID_CONDITIONS}"
        )

    if not isinstance(d["accuracy"], (int, float)) or not (0.0 <= float(d["accuracy"]) <= 1.0):
        raise ResultsSchemaError(f"accuracy={d['accuracy']!r} must be a number between 0 and 1")

    has_n = "n" in d
    has_n_correct = "n_correct" in d
    if has_n != has_n_correct:
        raise ResultsSchemaError("'n' and 'n_correct' must both be present or both be absent")
    if has_n and (not isinstance(d["n"], int) or d["n"] <= 0):
        raise ResultsSchemaError(f"n={d['n']!r} must be a positive integer")
    if has_n_correct and (not isinstance(d["n_correct"], int) or not (0 <= d["n_correct"] <= d["n"])):
        raise ResultsSchemaError(f"n_correct={d['n_correct']!r} must be between 0 and n")

    if "predictions" in d:
        preds = d["predictions"]
        if not isinstance(preds, list) or not preds:
            raise ResultsSchemaError("'predictions', if present, must be a non-empty list")
        for i, p in enumerate(preds):
            for key in ("qid", "predicted", "gold", "correct"):
                if key not in p:
                    raise ResultsSchemaError(f"predictions[{i}] is missing the field '{key}'")
            if not isinstance(p["correct"], bool):
                raise ResultsSchemaError(f"predictions[{i}]['correct'] must be a bool")


def has_paired_data(d: dict) -> bool:
    return "predictions" in d and len(d["predictions"]) > 0
