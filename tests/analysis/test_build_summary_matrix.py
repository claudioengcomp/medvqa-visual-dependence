import json
import os

import pytest

from src.analysis.build_summary_matrix import build_model_entry, build_summary_matrix, discover_models


def test_discover_models_ignores_vs_c0_files(tmp_path):
    (tmp_path / "FOO_C0.json").write_text("{}")
    (tmp_path / "FOO_C1_vs_C0.json").write_text("{}")
    (tmp_path / "FOO_C4_OCC_25_vs_C0.json").write_text("{}")
    (tmp_path / "BAR_C0.json").write_text("{}")
    assert discover_models(str(tmp_path)) == ["BAR", "FOO"]


def test_discover_models_on_real_results_dir_finds_51_files():
    # The earlier count of 15 became obsolete once the whole protocol was
    # expanded to multi-seed (3 seeds per combo on Kvasir/SLAKE/VQA-RAD):
    # every LoRA/DoRA combo of every dataset turned into 3 files
    # (_seed0/1/2, or a bare file plus _seed1/_seed2), raising the count to
    # 51. M3AE_KVASIR (bare, no _seed suffix) was dropped from scope -- it
    # is the pre-fix M3AE trained on the leaked Kvasir-VQA-x1 split, so it
    # was moved out of results/ and no longer counts as a valid model. The
    # M3AE_KVASIR_seed0/1/2 combos (clean split, post-fix) still count
    # normally.
    models = discover_models("results")
    assert len(models) == 51
    assert "QWEN2VL_LORA_SLAKE" in models
    assert "SMOLVLM_DORA_VQARAD" in models
    assert "M0_question_only_SLAKE" in models
    assert "M3AE_KVASIR" not in models
    assert "M3AE_KVASIR_seed0" in models
    assert "MEDVLTHINKER_ZEROSHOT_SLAKE" in models


def test_build_model_entry_qwen2vl_lora_vqarad_has_vda_and_cohens_g():
    # Numbers refreshed: the old expectations ("VQA-RAD" with a hyphen and
    # accuracy=0.5011) were already stale BEFORE the answer_matches
    # rescoring pass -- QWEN2VL_LORA_VQARAD_C0 had 0 items corrected by the
    # whitespace fix, so this was pre-existing test drift and not an effect
    # of that fix.
    entry = build_model_entry("results", "QWEN2VL_LORA_VQARAD")
    assert entry["dataset"] == "VQARAD"
    assert entry["conditions"]["C0"]["accuracy"] == pytest.approx(0.49667405764966743)
    assert entry["conditions"]["C1"]["cohens_g"] == pytest.approx(0.18103, abs=1e-4)
    assert entry["vda_c1"] == pytest.approx(0.15684, abs=1e-4)
    assert "answer_category_breakdown_c0" in entry
    assert entry["answer_category_breakdown_c0"]["naive_accuracy"] == pytest.approx(entry["conditions"]["C0"]["accuracy"])


def test_build_model_entry_m0_has_vda_zero_now_that_c1_was_measured():
    # M0 originally had only C0, and an earlier version of this test simply
    # checked that "C1" not in entry["conditions"]. C1 was then actually run
    # (src/baselines/question_only.py --condition C1, on the same SLAKE
    # train/test.json) to confirm empirically -- rather than by assumption --
    # what the M0 docstring already claimed by construction: a question-only
    # classifier never reads the image, so condition C1 (image replaced by a
    # neutral one) cannot change any prediction. The 416 C1 predictions came
    # out byte-for-byte identical to the C0 ones. Hence VDD(M0, C1)=0.0000
    # and VDA(M0)=0.0000 (a harmonic mean with VDD=0 is always 0, whatever
    # the accuracy). McNemar has b=c=0 (no discordant pairs) -- exactly the
    # case where cohens_g() returns None by definition, so "C1"."cohens_g"
    # is None here rather than absent.
    entry = build_model_entry("results", "M0_question_only_SLAKE")
    assert entry["conditions"]["C1"]["accuracy"] == pytest.approx(entry["conditions"]["C0"]["accuracy"])
    assert entry["conditions"]["C1"]["ldv"] == pytest.approx(0.0)
    assert entry["conditions"]["C1"]["mcnemar_p"] == pytest.approx(1.0)
    assert entry["conditions"]["C1"]["cohens_g"] is None
    assert entry["vda_c1"] == pytest.approx(0.0)


def test_build_summary_matrix_returns_all_51_files():
    matrix = build_summary_matrix("results")
    assert len(matrix) == 51
    assert set(matrix.keys()) == set(discover_models("results"))


def test_build_summary_matrix_cli_writes_valid_json(tmp_path):
    import subprocess
    import sys

    output = tmp_path / "summary_matrix.json"
    result = subprocess.run(
        [sys.executable, "-m", "src.analysis.build_summary_matrix",
         "--results-dir", "results", "--output", str(output)],
        capture_output=True, text=True, cwd=os.getcwd(),
    )
    assert result.returncode == 0, result.stderr
    written = json.loads(output.read_text())
    assert len(written) == 51


def test_build_model_entry_includes_c5_and_c6_when_present():
    # QWEN2VL_LORA_SLAKE has both C5 and C6 -- both had to be added to
    # _COND_ORDER, which previously stopped at C4_NOISE_75.
    entry = build_model_entry("results", "QWEN2VL_LORA_SLAKE")
    assert "C5" in entry["conditions"]
    assert "C6" in entry["conditions"]
    assert entry["conditions"]["C6"]["ldv"] == pytest.approx(0.0448, abs=1e-3)
