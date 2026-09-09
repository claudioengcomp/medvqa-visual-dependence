"""Tests for the noise-floor report of Section 5.6.

The floor bounds how small an effect the protocol can claim, so these tests pin
the published measurements rather than the code's shape.
"""
import os

import pytest

from src.analysis.noise_floor import compare_runs, run

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "results")

pytestmark = pytest.mark.skipif(not os.path.isdir(RESULTS),
                                reason="results not available")


def test_within_one_environment_the_pipeline_is_deterministic():
    """Zero differing predictions, in both architectures."""
    for record in run(RESULTS).values():
        assert record["same_environment"]["differing"] == 0


@pytest.mark.parametrize("label,differing,n", [
    ("SmolVLM2-2.2B + LoRA, SLAKE", 63, 1061),
    ("Qwen2-VL-2B + LoRA, SLAKE", 35, 1061),
    ("Qwen2-VL-2B + LoRA, VQA-RAD", 12, 451),
])
def test_across_environments_matches_the_published_band(label, differing, n):
    record = run(RESULTS)[label]["across_environments"]
    assert (record["differing"], record["n"]) == (differing, n)
    assert 0.026 <= record["fraction"] <= 0.060


def test_the_disagreements_have_no_direction():
    """b and c are close in all three: noise adds variance, not bias."""
    for record in run(RESULTS).values():
        across = record["across_environments"]
        assert abs(across["b"] - across["c"]) <= 3
        assert across["p_value"] > 0.05


def test_accuracy_can_be_unchanged_while_predictions_move():
    """On VQA-RAD 12 predictions change and the accuracy does not move --
    which is why the floor is measured per item, not on the aggregate."""
    across = run(RESULTS)["Qwen2-VL-2B + LoRA, VQA-RAD"]["across_environments"]
    assert across["differing"] == 12
    assert round(across["accuracy_a"], 4) == round(across["accuracy_b"], 4)


def test_identical_runs_compare_as_identical():
    a = {1: {"predicted": "yes", "correct": True},
         2: {"predicted": "no", "correct": False}}
    assert compare_runs(a, a)["differing"] == 0
    assert compare_runs(a, a)["fraction"] == 0.0
