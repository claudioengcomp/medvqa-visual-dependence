import json

import pytest

from src.analysis.build_multiseed_matrix import build_multiseed_matrix, discover_multiseed_groups


def _vs_c0(accuracy_c0=0.7, ldv=0.1, p=0.01, g=0.3, dataset=None):
    d = {
        "accuracy_c0": accuracy_c0, "ldv": ldv,
        "mcnemar": {"p_value": p, "cohens_g": g},
    }
    if dataset is not None:
        d["dataset"] = dataset
    return d


def _write(tmp_path, name, record):
    (tmp_path / f"{name}.json").write_text(json.dumps(record))


def test_aggregates_when_three_seeds_present(tmp_path):
    for seed, acc in ((0, 0.70), (1, 0.72), (2, 0.68)):
        _write(tmp_path, f"SMOLVLM_LORA_KVASIR_seed{seed}_C1_vs_C0", _vs_c0(accuracy_c0=acc))

    result = build_multiseed_matrix(str(tmp_path))

    key = ("SMOLVLM_LORA_KVASIR", "C1")
    assert key in result["aggregated"]
    assert result["pending"] == {}
    agg = result["aggregated"][key]
    assert agg["n_seeds"] == 3
    assert agg["accuracy_c0_mean"] == pytest.approx(0.70, abs=1e-6)
    assert agg["seeds_used"] == [0, 1, 2]
    assert agg["model_base"] == "SMOLVLM_LORA_KVASIR"
    assert agg["condition"] == "C1"


def test_reports_pending_when_fewer_than_three_seeds(tmp_path):
    _write(tmp_path, "SMOLVLM_LORA_KVASIR_seed0_C1_vs_C0", _vs_c0())
    _write(tmp_path, "SMOLVLM_LORA_KVASIR_seed1_C1_vs_C0", _vs_c0())
    # seed2 does not exist yet

    result = build_multiseed_matrix(str(tmp_path))

    key = ("SMOLVLM_LORA_KVASIR", "C1")
    assert key not in result["aggregated"]
    assert result["pending"][key] == [0, 1]


def test_does_not_confuse_lora_and_dora_groups(tmp_path):
    for seed in (0, 1, 2):
        _write(tmp_path, f"SMOLVLM_LORA_KVASIR_seed{seed}_C1_vs_C0", _vs_c0(accuracy_c0=0.70))
    for seed in (0, 1, 2):
        _write(tmp_path, f"SMOLVLM_DORA_KVASIR_seed{seed}_C1_vs_C0", _vs_c0(accuracy_c0=0.50))

    result = build_multiseed_matrix(str(tmp_path))

    assert ("SMOLVLM_LORA_KVASIR", "C1") in result["aggregated"]
    assert ("SMOLVLM_DORA_KVASIR", "C1") in result["aggregated"]
    assert result["aggregated"][("SMOLVLM_LORA_KVASIR", "C1")]["accuracy_c0_mean"] == pytest.approx(0.70, abs=1e-6)
    assert result["aggregated"][("SMOLVLM_DORA_KVASIR", "C1")]["accuracy_c0_mean"] == pytest.approx(0.50, abs=1e-6)


def test_zero_shot_without_seed_suffix_is_excluded_from_grouping(tmp_path):
    _write(tmp_path, "MEDVLTHINKER_ZEROSHOT_KVASIR_C1_vs_C0", _vs_c0())

    groups = discover_multiseed_groups(str(tmp_path))

    assert groups == {}


def test_bare_prefix_without_seed_suffix_is_treated_as_seed0_when_siblings_exist(tmp_path):
    # Naming convention inherited from the SLAKE/VQA-RAD runs: the original
    # seed0 was saved WITHOUT a _seedN suffix; only the seed1/seed2 files
    # added later carry an explicit suffix.
    _write(tmp_path, "QWEN2VL_LORA_SLAKE_C1_vs_C0", _vs_c0(accuracy_c0=0.60))  # implicit seed0
    _write(tmp_path, "QWEN2VL_LORA_SLAKE_seed1_C1_vs_C0", _vs_c0(accuracy_c0=0.62))
    _write(tmp_path, "QWEN2VL_LORA_SLAKE_seed2_C1_vs_C0", _vs_c0(accuracy_c0=0.58))

    result = build_multiseed_matrix(str(tmp_path))

    key = ("QWEN2VL_LORA_SLAKE", "C1")
    assert key in result["aggregated"]
    assert result["aggregated"][key]["seeds_used"] == [0, 1, 2]
    assert result["aggregated"][key]["accuracy_c0_mean"] == pytest.approx(0.60, abs=1e-6)


def test_bare_prefix_without_any_seed_sibling_stays_excluded(tmp_path):
    # A genuinely zero-shot combo (MedVLThinker) -- no _seed1/_seed2 sibling
    # under ANY condition -- so it is not an implicit seed0 of anything.
    _write(tmp_path, "MEDVLTHINKER_ZEROSHOT_SLAKE_C1_vs_C0", _vs_c0())
    _write(tmp_path, "MEDVLTHINKER_ZEROSHOT_SLAKE_C2_vs_C0", _vs_c0())

    result = build_multiseed_matrix(str(tmp_path))

    assert result["aggregated"] == {}
    assert result["pending"] == {}


def test_denylisted_prefix_is_excluded_even_with_seed_siblings(tmp_path):
    # M3AE_KVASIR (no seed suffix) is the discarded pre-fix combo trained on
    # the leaked split -- it is in DENYLIST_PREFIXES and must never be
    # treated as the "implicit seed0" of M3AE_KVASIR_seed1/2, even though
    # both exist and together look like a family of 3 seeds.
    _write(tmp_path, "M3AE_KVASIR_C1_vs_C0", _vs_c0(accuracy_c0=0.91))  # discarded data
    _write(tmp_path, "M3AE_KVASIR_seed1_C1_vs_C0", _vs_c0(accuracy_c0=0.60))
    _write(tmp_path, "M3AE_KVASIR_seed2_C1_vs_C0", _vs_c0(accuracy_c0=0.61))

    result = build_multiseed_matrix(str(tmp_path))

    key = ("M3AE_KVASIR", "C1")
    assert key not in result["aggregated"]
    assert result["pending"].get(key) == [1, 2]


def test_normalizes_dataset_label_aliases_before_validating(tmp_path):
    # A real inconsistency in the stored results: "VQARAD" (no hyphen) vs
    # "VQA-RAD" (hyphenated) for the SAME dataset, written by different
    # scripts.
    _write(tmp_path, "QWEN2VL_LORA_VQARAD_C1_vs_C0", _vs_c0(dataset="VQARAD"))
    _write(tmp_path, "QWEN2VL_LORA_VQARAD_seed1_C1_vs_C0", _vs_c0(dataset="VQA-RAD"))
    _write(tmp_path, "QWEN2VL_LORA_VQARAD_seed2_C1_vs_C0", _vs_c0(dataset="VQA-RAD"))

    result = build_multiseed_matrix(str(tmp_path))

    # no "inconsistent combo" ValueError -- the 3 seeds aggregate normally
    assert ("QWEN2VL_LORA_VQARAD", "C1") in result["aggregated"]


def test_condition_with_underscore_is_parsed_correctly(tmp_path):
    for seed in (0, 1, 2):
        _write(tmp_path, f"SMOLVLM_LORA_KVASIR_seed{seed}_C4_OCC_25_vs_C0", _vs_c0())

    groups = discover_multiseed_groups(str(tmp_path))

    assert ("SMOLVLM_LORA_KVASIR", "C4_OCC_25") in groups
    assert len(groups[("SMOLVLM_LORA_KVASIR", "C4_OCC_25")]) == 3

