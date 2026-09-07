import pytest

from src.analysis.vda import compute_vda


def test_compute_vda_matches_qwen2vl_lora_slake():
    # Acc(C0)=0.6955702167766258, VDD(C1)=0.37040527803958523 (results/QWEN2VL_LORA_SLAKE_*)
    vda = compute_vda(0.6955702167766258, 0.37040527803958523)
    assert vda == pytest.approx(0.48338, abs=1e-4)


def test_compute_vda_matches_smolvlm_lora_slake():
    # Acc(C0)=0.7313854853911405, VDD(C1)=0.4439208294062206 (results/SMOLVLM_LORA_SLAKE_*)
    vda = compute_vda(0.7313854853911405, 0.4439208294062206)
    assert vda == pytest.approx(0.55249, abs=1e-4)


def test_compute_vda_matches_smolvlm_dora_vqarad():
    # Acc(C0)=0.5188470066518847, VDD(C1)=0.13968957871396892 (results/SMOLVLM_DORA_VQARAD_*)
    vda = compute_vda(0.5188470066518847, 0.13968957871396892)
    assert vda == pytest.approx(0.22012, abs=1e-4)


def test_compute_vda_zero_ldv_gives_zero():
    assert compute_vda(0.9, 0.0) == 0.0


def test_compute_vda_stays_between_min_and_max_of_inputs():
    # Genuine property of the harmonic mean of 2 positive values:
    # min(a,b) <= H <= max(a,b), NOT H <= min(a,b). The stricter bound was
    # written first by mistake; TDD caught it here before any wrong number
    # was published.
    vda = compute_vda(0.5, 0.3)
    assert 0.3 - 1e-9 <= vda <= 0.5 + 1e-9


def test_compute_vda_returns_none_when_denominator_non_positive():
    assert compute_vda(0.1, -0.2) is None
