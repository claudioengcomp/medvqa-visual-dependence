import pytest

from src.inference.results_schema import ResultsSchemaError, has_paired_data, validate_results_dict


def test_valid_aggregate_only_record_passes():
    validate_results_dict({"model": "M3AE", "dataset": "SLAKE", "condition": "C0", "accuracy": 0.84})


def test_valid_full_record_with_n_passes():
    validate_results_dict({
        "model": "M3AE", "dataset": "SLAKE", "condition": "C0", "accuracy": 0.84,
        "n": 100, "n_correct": 84,
    })


def test_valid_paired_record_passes():
    record = {
        "model": "M0", "dataset": "SLAKE", "condition": "C0", "accuracy": 1.0,
        "predictions": [{"qid": 1, "predicted": "yes", "gold": "yes", "correct": True}],
    }
    validate_results_dict(record)
    assert has_paired_data(record) is True


def test_missing_required_key_raises():
    with pytest.raises(ResultsSchemaError):
        validate_results_dict({"model": "M3AE", "dataset": "SLAKE", "accuracy": 0.5})


def test_invalid_condition_raises():
    with pytest.raises(ResultsSchemaError):
        validate_results_dict({"model": "M3AE", "dataset": "SLAKE", "condition": "C9", "accuracy": 0.5})


def test_accuracy_out_of_range_raises():
    with pytest.raises(ResultsSchemaError):
        validate_results_dict({"model": "M3AE", "dataset": "SLAKE", "condition": "C0", "accuracy": 1.5})


def test_n_without_n_correct_raises():
    with pytest.raises(ResultsSchemaError):
        validate_results_dict({"model": "M3AE", "dataset": "SLAKE", "condition": "C0", "accuracy": 0.5, "n": 10})


def test_predictions_missing_field_raises():
    with pytest.raises(ResultsSchemaError):
        validate_results_dict({
            "model": "M0", "dataset": "SLAKE", "condition": "C0", "accuracy": 1.0,
            "predictions": [{"qid": 1, "predicted": "yes"}],
        })


def test_has_paired_data_false_when_absent():
    assert has_paired_data({"model": "M3AE", "dataset": "SLAKE", "condition": "C0", "accuracy": 0.5}) is False


def test_c4_conditions_are_valid():
    for cond in ["C4_OCC_25", "C4_OCC_50", "C4_OCC_75", "C4_NOISE_25", "C4_NOISE_50", "C4_NOISE_75"]:
        validate_results_dict({"model": "M3AE", "dataset": "SLAKE", "condition": cond, "accuracy": 0.5})


def test_c5_condition_is_valid():
    # C5 = swap restricted to the same modality+location, but with a
    # DIFFERENT finding category. Not to be confused with the "paraphrase"
    # condition that other protocols also label C5 — the name collision is
    # deliberate and accepted here.
    validate_results_dict({"model": "QWEN2VL_LORA", "dataset": "KVASIR", "condition": "C5", "accuracy": 0.5})


def test_c6_condition_is_valid():
    # C6 = same domain restriction as C5 (modality+location), but the
    # finding category is UNCONSTRAINED -- it reuses the C5 pairs and then
    # completes the rest of the bucket with free pairing.
    validate_results_dict({"model": "QWEN2VL_LORA", "dataset": "SLAKE", "condition": "C6", "accuracy": 0.5})


def test_c7_condition_is_reserved_but_valid():
    # C7 = reserved for a positional-occlusion placebo condition. It was
    # renamed from C6 to C7 once C6 came to mean "same organ+modality,
    # unconstrained category". Not implemented yet -- the schema only
    # reserves the name.
    validate_results_dict({"model": "QWEN2VL_LORA", "dataset": "SLAKE", "condition": "C7", "accuracy": 0.5})
