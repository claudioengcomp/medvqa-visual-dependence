import pytest

from src.stats.compute_multiseed import aggregate_multiseed_summary


def _summary(accuracy, ldv, mcnemar_p, cohens_g):
    # Mirrors the real schema of
    # src.stats.compute_vdd.compute_summary_vs_baseline: the visual
    # dependency metric lives under the key "ldv" (a historical name in the
    # code -- the paper calls it VDD, but it is the same number), and
    # "cohens_g" lives INSIDE "mcnemar" rather than as a top-level key (see
    # compute_vdd.py: mcnemar_test() always includes cohens_g in the dict it
    # returns).
    return {
        "accuracy_c0": accuracy,
        "ldv": ldv,
        "mcnemar": {"p_value": mcnemar_p, "cohens_g": cohens_g},
    }


def test_aggregate_reports_mean_and_std_of_accuracy():
    summaries = [_summary(0.70, 0.10, 0.001, 0.3),
                 _summary(0.72, 0.12, 0.002, 0.32),
                 _summary(0.68, 0.08, 0.03, 0.28)]
    result = aggregate_multiseed_summary(summaries)

    assert result["accuracy_c0_mean"] == pytest.approx(0.70, abs=1e-6)
    assert result["accuracy_c0_std"] == pytest.approx(0.02, abs=1e-6)


def test_aggregate_reports_mean_and_std_of_vdd():
    summaries = [_summary(0.70, 0.10, 0.001, 0.3),
                 _summary(0.72, 0.12, 0.002, 0.32),
                 _summary(0.68, 0.08, 0.03, 0.28)]
    result = aggregate_multiseed_summary(summaries)

    assert result["vdd_mean"] == pytest.approx(0.10, abs=1e-6)


def test_aggregate_counts_significant_mcnemar_at_default_alpha():
    summaries = [_summary(0.70, 0.10, 0.001, 0.3),
                 _summary(0.72, 0.12, 0.002, 0.32),
                 _summary(0.68, 0.08, 0.30, 0.28)]  # this one is not significant
    result = aggregate_multiseed_summary(summaries, alpha=0.05)

    assert result["mcnemar_significant_count"] == 2
    assert result["mcnemar_total_seeds"] == 3


def test_aggregate_requires_at_least_one_summary():
    with pytest.raises(ValueError, match="at least 1"):
        aggregate_multiseed_summary([])


def test_aggregate_single_seed_std_is_zero():
    summaries = [_summary(0.70, 0.10, 0.001, 0.3)]
    result = aggregate_multiseed_summary(summaries)

    assert result["accuracy_c0_std"] == 0.0
    assert result["n_seeds"] == 1


def test_aggregate_counts_significant_mcnemar_at_custom_alpha():
    # p=0.03 is not significant at alpha=0.01 (although it would be at alpha=0.05)
    summaries = [_summary(0.70, 0.10, 0.001, 0.3),
                 _summary(0.72, 0.12, 0.002, 0.32),
                 _summary(0.68, 0.08, 0.03, 0.28)]
    result = aggregate_multiseed_summary(summaries, alpha=0.01)

    assert result["mcnemar_significant_count"] == 2
    assert result["mcnemar_total_seeds"] == 3


def test_aggregate_works_with_five_seeds():
    summaries = [_summary(0.70, 0.10, 0.001, 0.3),
                 _summary(0.72, 0.12, 0.002, 0.32),
                 _summary(0.68, 0.08, 0.03, 0.28),
                 _summary(0.74, 0.14, 0.20, 0.10),
                 _summary(0.66, 0.06, 0.50, 0.05)]
    result = aggregate_multiseed_summary(summaries)

    assert result["n_seeds"] == 5
    assert result["mcnemar_total_seeds"] == 5
    assert result["accuracy_c0_mean"] == pytest.approx(0.70, abs=1e-6)
    assert result["mcnemar_significant_count"] == 3


def test_aggregate_raises_key_error_when_summary_missing_mcnemar():
    # Unpaired data (no 'predictions' shared between C0/C1) does not produce
    # the 'mcnemar' key in compute_summary_vs_baseline -- this module
    # requires paired data on EVERY seed. That is intentional: aggregating
    # without McNemar/cohens_g would be meaningless here.
    summaries = [_summary(0.70, 0.10, 0.001, 0.3),
                 {"accuracy_c0": 0.72, "ldv": 0.12}]  # no 'mcnemar'
    with pytest.raises(KeyError):
        aggregate_multiseed_summary(summaries)


def test_aggregate_excludes_none_cohens_g_from_mean_and_reports_count():
    summaries = [_summary(0.70, 0.10, 0.001, 0.3),
                 _summary(0.72, 0.12, 0.002, 0.32),
                 _summary(0.68, 0.08, 1.0, None)]  # b=c=0, cohens_g undefined
    result = aggregate_multiseed_summary(summaries)

    assert result["cohens_g_n"] == 2
    assert result["cohens_g_mean"] == pytest.approx(0.31, abs=1e-6)
    assert result["n_seeds"] == 3


def test_aggregate_cohens_g_all_none_returns_none_not_error():
    summaries = [_summary(0.70, 0.10, 1.0, None),
                 _summary(0.72, 0.12, 1.0, None)]
    result = aggregate_multiseed_summary(summaries)

    assert result["cohens_g_mean"] is None
    assert result["cohens_g_std"] is None
    assert result["cohens_g_n"] == 0


def test_aggregate_raises_when_model_mismatched_across_seeds():
    s1 = _summary(0.70, 0.10, 0.001, 0.3)
    s2 = _summary(0.72, 0.12, 0.002, 0.32)
    s1["model"] = "LaPA"
    s2["model"] = "LLaVA-Med"

    with pytest.raises(ValueError, match="model"):
        aggregate_multiseed_summary([s1, s2])


def test_aggregate_raises_when_dataset_mismatched_across_seeds():
    s1 = _summary(0.70, 0.10, 0.001, 0.3)
    s2 = _summary(0.72, 0.12, 0.002, 0.32)
    s1["dataset"] = "Kvasir-VQA"
    s2["dataset"] = "SLAKE"

    with pytest.raises(ValueError, match="dataset"):
        aggregate_multiseed_summary([s1, s2])


def test_aggregate_allows_same_combo_across_seeds():
    s1 = _summary(0.70, 0.10, 0.001, 0.3)
    s2 = _summary(0.72, 0.12, 0.002, 0.32)
    s1["model"] = s2["model"] = "LaPA"
    s1["dataset"] = s2["dataset"] = "Kvasir-VQA"
    s1["condition"] = s2["condition"] = "C1"

    result = aggregate_multiseed_summary([s1, s2])
    assert result["n_seeds"] == 2


def test_aggregate_raises_when_first_summary_lacks_key_but_later_ones_mismatch():
    # Regression guard: _validate_same_combo must not anchor on summaries[0].
    # Here summaries[0] has no 'model', while summaries[1] and summaries[2]
    # carry DIFFERENT values -- the mismatch must still be caught.
    s1 = _summary(0.70, 0.10, 0.001, 0.3)  # no 'model'
    s2 = _summary(0.72, 0.12, 0.002, 0.32)
    s3 = _summary(0.68, 0.08, 0.03, 0.28)
    s2["model"] = "LaPA"
    s3["model"] = "LLaVA-Med"

    with pytest.raises(ValueError, match="model"):
        aggregate_multiseed_summary([s1, s2, s3])
