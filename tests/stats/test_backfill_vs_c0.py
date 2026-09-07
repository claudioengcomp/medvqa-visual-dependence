import json
import os

import pytest

from src.stats.backfill_vs_c0 import DENYLIST_PREFIXES, backfill_all, backfill_model


def _write(tmp_path, name, record):
    (tmp_path / f"{name}.json").write_text(json.dumps(record))


def _record(model="FOO", dataset="KVASIR", condition="C0", accuracy=0.8, n=100, n_correct=80):
    return {
        "model": model, "dataset": dataset, "condition": condition,
        "accuracy": accuracy, "n": n, "n_correct": n_correct,
    }


def test_backfill_model_computes_vs_c0_when_missing_and_both_exist(tmp_path):
    _write(tmp_path, "FOO_C0", _record(condition="C0", accuracy=0.8, n_correct=80))
    _write(tmp_path, "FOO_C1", _record(condition="C1", accuracy=0.6, n_correct=60))

    report = backfill_model(str(tmp_path), "FOO")

    assert report["computed"] == ["C1"]
    out = json.loads((tmp_path / "FOO_C1_vs_C0.json").read_text())
    assert out["ldv"] == pytest.approx(0.2)


def test_backfill_model_skips_when_vs_c0_already_exists(tmp_path):
    _write(tmp_path, "FOO_C0", _record(condition="C0"))
    _write(tmp_path, "FOO_C1", _record(condition="C1", accuracy=0.6, n_correct=60))
    _write(tmp_path, "FOO_C1_vs_C0", {"ldv": 0.999, "sentinel": True})

    report = backfill_model(str(tmp_path), "FOO")

    assert report["skipped_existing"] == ["C1"]
    # the existing file must not have been overwritten
    out = json.loads((tmp_path / "FOO_C1_vs_C0.json").read_text())
    assert out == {"ldv": 0.999, "sentinel": True}


def test_backfill_model_reports_missing_condition_without_erroring(tmp_path):
    _write(tmp_path, "FOO_C0", _record(condition="C0"))
    # no other condition exists

    report = backfill_model(str(tmp_path), "FOO")

    assert "C1" in report["missing_condition"]
    assert report["computed"] == []


def test_backfill_model_reports_n_mismatch_and_does_not_abort_batch(tmp_path):
    _write(tmp_path, "FOO_C0", _record(condition="C0", n=100, n_correct=80))
    _write(tmp_path, "FOO_C1", _record(condition="C1", n=90, n_correct=60))  # different n
    _write(tmp_path, "FOO_C2", _record(condition="C2", n=100, n_correct=70))  # same n

    report = backfill_model(str(tmp_path), "FOO")

    assert report["n_mismatch"] == [("C1", 100, 90)]
    assert report["computed"] == ["C2"]
    assert not os.path.exists(tmp_path / "FOO_C1_vs_C0.json")
    assert os.path.exists(tmp_path / "FOO_C2_vs_C0.json")


def test_backfill_all_skips_denylisted_prefixes(tmp_path):
    assert "M3AE_KVASIR" in DENYLIST_PREFIXES
    _write(tmp_path, "M3AE_KVASIR_C0", _record(model="M3AE", condition="C0"))
    _write(tmp_path, "M3AE_KVASIR_C1", _record(model="M3AE", condition="C1", accuracy=0.5, n_correct=50))
    _write(tmp_path, "FOO_C0", _record(condition="C0"))
    _write(tmp_path, "FOO_C1", _record(condition="C1", accuracy=0.6, n_correct=60))

    reports = backfill_all(str(tmp_path))

    assert "M3AE_KVASIR" not in reports
    assert "FOO" in reports
    assert not os.path.exists(tmp_path / "M3AE_KVASIR_C1_vs_C0.json")
    assert os.path.exists(tmp_path / "FOO_C1_vs_C0.json")
