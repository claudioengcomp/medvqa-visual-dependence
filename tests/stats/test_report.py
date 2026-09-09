from src.stats.report import render_multiseed_report, render_report


def test_render_report_includes_all_rows_and_values():
    summaries = [
        {
            "model": "M3AE", "dataset": "SLAKE", "accuracy_c0": 0.8473, "accuracy_c1": 0.61,
            "ldv": 0.2373, "ldv_ci_paired": (0.15, 0.32), "mcnemar": {"p_value": 0.0001},
        },
        {
            "model": "M0_question_only", "dataset": "SLAKE", "accuracy_c0": 0.55, "accuracy_c1": 0.55,
            "ldv": 0.0,
        },
    ]
    report = render_report(summaries)

    assert "M3AE" in report
    assert "0.8473" in report
    assert "0.2373" in report
    assert "[0.150, 0.320]" in report
    assert "M0_question_only" in report
    assert "0.0000" in report
    assert "—" in report


def test_render_report_falls_back_to_aggregate_ci_when_no_paired_ci():
    summaries = [{
        "model": "M3AE", "dataset": "SLAKE", "accuracy_c0": 0.84, "accuracy_c1": 0.60,
        "ldv": 0.24, "ci_c0": (0.80, 0.88), "ci_c1": (0.55, 0.65),
    }]
    report = render_report(summaries)
    assert "~[" in report


def _agg(condition, n_seeds=3, acc_mean=0.70, acc_std=0.02, vdd_mean=0.10, vdd_std=0.01,
         g_mean=0.3, g_std=0.02, g_n=3, sig=3, total=3):
    return {
        "condition": condition, "n_seeds": n_seeds,
        "accuracy_c0_mean": acc_mean, "accuracy_c0_std": acc_std,
        "vdd_mean": vdd_mean, "vdd_std": vdd_std,
        "cohens_g_mean": g_mean, "cohens_g_std": g_std, "cohens_g_n": g_n,
        "mcnemar_significant_count": sig, "mcnemar_total_seeds": total,
    }


def test_render_multiseed_report_includes_model_and_values():
    matrix = {
        "SMOLVLM_LORA_KVASIR": {
            "conditions": {"C1": _agg("C1")},
            "vda_c1": 0.1806,
        }
    }
    report = render_multiseed_report(matrix)

    assert "SMOLVLM_LORA_KVASIR" in report
    assert "0.7000 ± 0.0200" in report
    assert "0.1000 ± 0.0100" in report
    assert "0.1806" in report
    assert "3/3" in report
    assert "0.3000 ± 0.0200 (n=3)" in report


def test_render_multiseed_report_vda_only_shown_on_c1_row():
    matrix = {
        "SMOLVLM_LORA_KVASIR": {
            "conditions": {"C1": _agg("C1"), "C2": _agg("C2")},
            "vda_c1": 0.1806,
        }
    }
    report = render_multiseed_report(matrix)
    lines = [l for l in report.split("\n") if "| SMOLVLM_LORA_KVASIR |" in l]

    c1_line = next(l for l in lines if "| C1 |" in l)
    c2_line = next(l for l in lines if "| C2 |" in l)
    assert "0.1806" in c1_line
    assert "0.1806" not in c2_line
    assert c2_line.count("—") >= 1


def test_render_multiseed_report_handles_undefined_cohens_g():
    matrix = {
        "M0_question_only_SLAKE": {
            "conditions": {"C1": _agg("C1", g_mean=None, g_std=None, g_n=0)},
        }
    }
    report = render_multiseed_report(matrix)

    assert "M0_question_only_SLAKE" in report
    assert "| — |\n" in report or report.strip().endswith("—")
