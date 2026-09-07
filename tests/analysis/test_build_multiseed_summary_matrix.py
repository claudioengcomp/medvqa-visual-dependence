import json

import pytest

from src.analysis.build_multiseed_summary_matrix import (
    build_model_base_entry,
    build_multiseed_summary_matrix,
    discover_model_bases,
)


def _multiseed_entry(model_base, condition, accuracy_c0_mean=0.70, vdd_mean=0.10):
    return {
        "model_base": model_base, "condition": condition,
        "n_seeds": 3, "accuracy_c0_mean": accuracy_c0_mean, "accuracy_c0_std": 0.02,
        "vdd_mean": vdd_mean, "vdd_std": 0.01,
        "cohens_g_mean": 0.3, "cohens_g_std": 0.02, "cohens_g_n": 3,
        "mcnemar_significant_count": 3, "mcnemar_total_seeds": 3,
        "seeds_used": [0, 1, 2], "source_files": [],
    }


def _write(tmp_path, name, record):
    (tmp_path / f"{name}.json").write_text(json.dumps(record))


def test_discover_model_bases_reads_internal_field_not_filename(tmp_path):
    _write(tmp_path, "SMOLVLM_LORA_KVASIR_C1_multiseed", _multiseed_entry("SMOLVLM_LORA_KVASIR", "C1"))
    _write(tmp_path, "SMOLVLM_DORA_KVASIR_C1_multiseed", _multiseed_entry("SMOLVLM_DORA_KVASIR", "C1"))

    bases = discover_model_bases(str(tmp_path))

    assert bases == ["SMOLVLM_DORA_KVASIR", "SMOLVLM_LORA_KVASIR"]


def test_build_model_base_entry_collects_all_conditions(tmp_path):
    _write(tmp_path, "SMOLVLM_LORA_KVASIR_C1_multiseed", _multiseed_entry("SMOLVLM_LORA_KVASIR", "C1"))
    _write(tmp_path, "SMOLVLM_LORA_KVASIR_C2_multiseed", _multiseed_entry("SMOLVLM_LORA_KVASIR", "C2"))

    entry = build_model_base_entry(str(tmp_path), "SMOLVLM_LORA_KVASIR")

    assert set(entry["conditions"]) == {"C1", "C2"}


def test_build_model_base_entry_computes_vda_once_over_aggregated_means(tmp_path):
    _write(tmp_path, "SMOLVLM_LORA_KVASIR_C1_multiseed",
           _multiseed_entry("SMOLVLM_LORA_KVASIR", "C1", accuracy_c0_mean=0.70, vdd_mean=0.10))

    entry = build_model_base_entry(str(tmp_path), "SMOLVLM_LORA_KVASIR")

    from src.analysis.vda import compute_vda
    expected = compute_vda(0.70, 0.10)
    assert entry["vda_c1"] == pytest.approx(expected)


def test_build_model_base_entry_has_no_vda_without_c1(tmp_path):
    _write(tmp_path, "SMOLVLM_LORA_KVASIR_C2_multiseed", _multiseed_entry("SMOLVLM_LORA_KVASIR", "C2"))

    entry = build_model_base_entry(str(tmp_path), "SMOLVLM_LORA_KVASIR")

    assert "vda_c1" not in entry


def test_glob_prefix_collision_is_filtered_by_internal_field(tmp_path):
    # "SMOLVLM_LORA_KVASIR" is a textual prefix of
    # "SMOLVLM_LORA_KVASIR_EXTRA" -- a glob would pick up both, so the
    # internal model_base field must filter out what does not really belong.
    _write(tmp_path, "SMOLVLM_LORA_KVASIR_C1_multiseed", _multiseed_entry("SMOLVLM_LORA_KVASIR", "C1"))
    _write(tmp_path, "SMOLVLM_LORA_KVASIR_EXTRA_C1_multiseed", _multiseed_entry("SMOLVLM_LORA_KVASIR_EXTRA", "C1"))

    entry = build_model_base_entry(str(tmp_path), "SMOLVLM_LORA_KVASIR")

    assert list(entry["conditions"]["C1"].keys()) == list(_multiseed_entry("SMOLVLM_LORA_KVASIR", "C1").keys())
    assert entry["conditions"]["C1"]["model_base"] == "SMOLVLM_LORA_KVASIR"


def test_build_multiseed_summary_matrix_covers_all_model_bases(tmp_path):
    _write(tmp_path, "SMOLVLM_LORA_KVASIR_C1_multiseed", _multiseed_entry("SMOLVLM_LORA_KVASIR", "C1"))
    _write(tmp_path, "SMOLVLM_DORA_KVASIR_C1_multiseed", _multiseed_entry("SMOLVLM_DORA_KVASIR", "C1"))

    matrix = build_multiseed_summary_matrix(str(tmp_path))

    assert set(matrix) == {"SMOLVLM_LORA_KVASIR", "SMOLVLM_DORA_KVASIR"}
