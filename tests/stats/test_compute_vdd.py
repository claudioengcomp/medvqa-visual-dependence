import pytest

from src.stats.compute_vdd import (
    bootstrap_accuracy_ci,
    bootstrap_paired_diff_ci,
    cohens_g,
    compute_vdd,
    compute_summary,
    compute_summary_vs_baseline,
    mcnemar_test,
)


def test_compute_vdd_is_simple_difference():
    assert compute_vdd(0.85, 0.60) == pytest.approx(0.25)


def test_bootstrap_accuracy_ci_contains_point_estimate_and_is_reproducible():
    lo, hi = bootstrap_accuracy_ci(n_correct=80, n_total=100, n_boot=2000, seed=1)
    assert lo <= 0.80 <= hi
    lo2, hi2 = bootstrap_accuracy_ci(n_correct=80, n_total=100, n_boot=2000, seed=1)
    assert (lo, hi) == (lo2, hi2)


def test_bootstrap_accuracy_ci_rejects_invalid_counts():
    with pytest.raises(ValueError):
        bootstrap_accuracy_ci(n_correct=150, n_total=100)


def test_bootstrap_accuracy_ci_zero_correct_is_degenerate():
    lo, hi = bootstrap_accuracy_ci(n_correct=0, n_total=100, n_boot=500, seed=1)
    assert (lo, hi) == (0.0, 0.0)


def test_bootstrap_accuracy_ci_all_correct_is_degenerate():
    lo, hi = bootstrap_accuracy_ci(n_correct=100, n_total=100, n_boot=500, seed=1)
    assert (lo, hi) == (1.0, 1.0)


def test_mcnemar_test_no_difference_gives_p_value_one():
    c0 = [True, False, True, False, True, False] * 5
    c1 = [True, False, True, False, True, False] * 5
    result = mcnemar_test(c0, c1)
    assert result["b"] == 0
    assert result["c"] == 0
    assert result["p_value"] == 1.0


def test_mcnemar_test_detects_asymmetric_disagreement():
    c0 = [True] * 15 + [False] * 5
    c1 = [False] * 15 + [False] * 5
    result = mcnemar_test(c0, c1)
    assert result["b"] == 15
    assert result["c"] == 0
    assert result["method"] == "exact"
    assert result["p_value"] < 0.01


def test_bootstrap_paired_diff_ci_reproducible():
    c0 = [True, True, False, True, False, True, True, False]
    c1 = [False, True, False, False, False, True, True, False]
    lo, hi = bootstrap_paired_diff_ci(c0, c1, n_boot=2000, seed=3)
    lo2, hi2 = bootstrap_paired_diff_ci(c0, c1, n_boot=2000, seed=3)
    assert (lo, hi) == (lo2, hi2)
    assert lo <= hi


def test_compute_summary_aggregate_mode():
    c0 = {"model": "M3AE", "dataset": "SLAKE", "condition": "C0", "accuracy": 0.85, "n": 100, "n_correct": 85}
    c1 = {"model": "M3AE", "dataset": "SLAKE", "condition": "C1", "accuracy": 0.60, "n": 100, "n_correct": 60}
    summary = compute_summary(c0, c1, n_boot=500, seed=1)
    assert summary["ldv"] == pytest.approx(0.25)
    assert summary["paired"] is False
    assert "ci_c0" in summary and "ci_c1" in summary


def test_compute_summary_paired_mode():
    c0_preds = [{"qid": i, "predicted": "x", "gold": "x", "correct": i < 8} for i in range(10)]
    c1_preds = [{"qid": i, "predicted": "x", "gold": "x", "correct": i < 5} for i in range(10)]
    c0 = {"model": "M3AE", "dataset": "SLAKE", "condition": "C0", "accuracy": 0.8, "predictions": c0_preds}
    c1 = {"model": "M3AE", "dataset": "SLAKE", "condition": "C1", "accuracy": 0.5, "predictions": c1_preds}
    summary = compute_summary(c0, c1, n_boot=500, seed=1)
    assert summary["paired"] is True
    assert summary["n_paired"] == 10
    assert "mcnemar" in summary
    assert "ldv_ci_paired" in summary


def test_compute_summary_rejects_mismatched_model():
    c0 = {"model": "M3AE", "dataset": "SLAKE", "condition": "C0", "accuracy": 0.8}
    c1 = {"model": "SMOLVLM_LORA", "dataset": "SLAKE", "condition": "C1", "accuracy": 0.5}
    with pytest.raises(ValueError):
        compute_summary(c0, c1)


def test_compute_summary_rejects_swapped_conditions():
    c0 = {"model": "M3AE", "dataset": "SLAKE", "condition": "C1", "accuracy": 0.6}
    c1 = {"model": "M3AE", "dataset": "SLAKE", "condition": "C0", "accuracy": 0.8}
    with pytest.raises(ValueError):
        compute_summary(c0, c1)


def test_compute_summary_vs_baseline_accepts_arbitrary_condition_aggregate():
    c0 = {"model": "M3AE", "dataset": "SLAKE", "condition": "C0", "accuracy": 0.85, "n": 100, "n_correct": 85}
    c4 = {"model": "M3AE", "dataset": "SLAKE", "condition": "C4_OCC_50", "accuracy": 0.55, "n": 100, "n_correct": 55}
    summary = compute_summary_vs_baseline(c0, c4, n_boot=500, seed=1)
    assert summary["condition"] == "C4_OCC_50"
    assert summary["ldv"] == pytest.approx(0.30)
    assert summary["paired"] is False
    assert "ci_c0" in summary and "ci_other" in summary


def test_compute_summary_vs_baseline_paired_mode():
    c0_preds = [{"qid": i, "predicted": "x", "gold": "x", "correct": i < 8} for i in range(10)]
    c4_preds = [{"qid": i, "predicted": "x", "gold": "x", "correct": i < 4} for i in range(10)]
    c0 = {"model": "M3AE", "dataset": "SLAKE", "condition": "C0", "accuracy": 0.8, "predictions": c0_preds}
    c4 = {"model": "M3AE", "dataset": "SLAKE", "condition": "C4_NOISE_75", "accuracy": 0.4, "predictions": c4_preds}
    summary = compute_summary_vs_baseline(c0, c4, n_boot=500, seed=1)
    assert summary["paired"] is True
    assert summary["n_paired"] == 10
    assert "mcnemar" in summary
    assert "ldv_ci_paired" in summary


def test_compute_summary_vs_baseline_rejects_c0_as_other():
    c0 = {"model": "M3AE", "dataset": "SLAKE", "condition": "C0", "accuracy": 0.8}
    also_c0 = {"model": "M3AE", "dataset": "SLAKE", "condition": "C0", "accuracy": 0.8}
    with pytest.raises(ValueError):
        compute_summary_vs_baseline(c0, also_c0)


def test_compute_summary_vs_baseline_rejects_mismatched_model():
    c0 = {"model": "M3AE", "dataset": "SLAKE", "condition": "C0", "accuracy": 0.8}
    other = {"model": "SMOLVLM_LORA", "dataset": "SLAKE", "condition": "C4_OCC_25", "accuracy": 0.5}
    with pytest.raises(ValueError):
        compute_summary_vs_baseline(c0, other)


def test_cohens_g_zero_when_discordance_is_symmetric():
    assert cohens_g(b=10, c=10) == pytest.approx(0.0)


def test_cohens_g_half_when_discordance_is_one_directional():
    assert cohens_g(b=15, c=0) == pytest.approx(0.5)


def test_cohens_g_is_none_when_no_discordant_pairs():
    assert cohens_g(b=0, c=0) is None


def test_mcnemar_test_includes_cohens_g_matching_real_qwen2vl_lora_vqarad_c1():
    # reproduces the real b=74, c=39 from results/QWEN2VL_LORA_VQARAD_C1_vs_C0.json
    c0 = [True] * 74 + [False] * 39 + [True] * 100 + [False] * 100
    c1 = [False] * 74 + [True] * 39 + [True] * 100 + [False] * 100
    result = mcnemar_test(c0, c1)
    assert result["b"] == 74
    assert result["c"] == 39
    assert result["cohens_g"] == pytest.approx(abs(74 / 113 - 0.5), abs=1e-9)
