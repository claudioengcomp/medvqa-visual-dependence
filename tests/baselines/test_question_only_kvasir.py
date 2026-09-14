"""Tests for the Kvasir question-only floor (Table 1).

The floor is trained on an Arrow file that is not redistributed here, so the
tests are split in two. The synthetic ones pin the behaviour of the loader and
the ruler on data built in `tmp_path`. The published ones recompute 7,086 of
10,779 from the committed predictions, the committed swapmap and the ruler --
no Arrow and no training -- which is what makes the number in the paper
checkable by a reader who only has this repository.
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest

from src.baselines.question_only_kvasir import (CONDITIONS, load_test_items,
                                                load_train_pairs, rescore,
                                                score, write_conditions)

RESULTS = Path("results")
SWAPMAP = "data/swapmap.json"
PUBLISHED_N = 10779
PUBLISHED_CORRECT = 7086


def _write_arrow(path, questions, answers, splits):
    table = pa.table({
        "questions": pa.array(questions, type=pa.list_(pa.string())),
        "answers": pa.array(answers, type=pa.list_(pa.list_(pa.string()))),
        "split": pa.array(splits, type=pa.string())})
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    return str(path)


# ------------------------------------------------------------------ synthetic
def test_load_train_pairs_flattens_one_row_per_image(tmp_path):
    """The Arrow holds one row per IMAGE, with parallel question/answer lists."""
    path = _write_arrow(
        tmp_path / "t.arrow",
        questions=[["Is there a polyp?", "How many?"], ["Have all polyps been removed?"]],
        answers=[[["Yes"], ["1"]], [["not relevant"]]],
        splits=["train", "train"])

    pairs = load_train_pairs(path)

    assert len(pairs) == 3                      # 2 + 1, not 2 rows
    assert pairs[0] == {"question": "Is there a polyp?", "answer": "yes"}
    assert pairs[2]["answer"] == "not relevant"


def test_load_train_pairs_refuses_an_arrow_that_is_not_training_only(tmp_path):
    """Training on the test split would make the floor meaningless, silently."""
    path = _write_arrow(tmp_path / "t.arrow", [["q"]], [[["a"]]], ["test"])
    with pytest.raises(SystemExit, match="training-only"):
        load_train_pairs(path)


def test_load_train_pairs_refuses_an_arrow_missing_columns(tmp_path):
    table = pa.table({"questions": pa.array([["q"]], type=pa.list_(pa.string()))})
    path = tmp_path / "t.arrow"
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    with pytest.raises(SystemExit, match="missing the columns"):
        load_train_pairs(str(path))


def test_load_train_pairs_refuses_a_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        load_train_pairs(str(tmp_path / "absent.arrow"))


def test_score_uses_the_repository_ruler_not_exact_string():
    """`Polyp.` and `polyp` are the same answer; exact match would miss it."""
    items = [{"qid": 1, "question": "What?", "refs": ["polyp"]},
             {"qid": 2, "question": "What?", "refs": ["ulcer"]}]

    record = score(["Polyp.", "polyp"], items)

    assert record["n_correct"] == 1                 # first merges, second does not
    assert record["predictions"][0]["correct"] is True
    assert record["predictions"][1]["correct"] is False
    assert record["accuracy"] == 0.5
    assert record["dataset"] == "KVASIR"


def test_score_refuses_a_length_mismatch():
    items = [{"qid": 1, "question": "q", "refs": ["a"]}]
    with pytest.raises(ValueError, match="2 predictions for 1 items"):
        score(["a", "b"], items)


def test_write_conditions_writes_the_floor_under_every_condition(tmp_path):
    """VDD is zero by construction, so every condition carries the same rows."""
    items = [{"qid": 1, "question": "q", "refs": ["a"]}]
    record = score(["a"], items)

    written = write_conditions(record, str(tmp_path))

    assert len(written) == len(CONDITIONS) == 11
    loaded = [json.loads(Path(p).read_text()) for p in written]
    assert {d["condition"] for d in loaded} == set(CONDITIONS)
    assert all(d["predictions"] == loaded[0]["predictions"] for d in loaded)
    assert all(d["n_correct"] == 1 for d in loaded)


# ------------------------------------------------------------------ published
@pytest.fixture(scope="module")
def published():
    return {c: json.loads((RESULTS / ("M0_question_only_KVASIR_%s.json" % c)).read_text())
            for c in CONDITIONS}


def test_the_published_floor_is_the_number_the_paper_reports(published):
    c0 = published["C0"]
    assert (c0["n_correct"], c0["n"]) == (PUBLISHED_CORRECT, PUBLISHED_N)
    assert c0["accuracy"] == pytest.approx(0.6574, abs=5e-5)


def test_the_floor_recomputes_from_committed_data_without_the_arrow(published):
    """The assertion that makes the published number verifiable.

    No training and no Arrow: the stored `predicted` strings are scored again
    against the swapmap references with the repository ruler. A drift in the
    ruler, in the swapmap or in the predictions changes this count.
    """
    items = load_test_items(SWAPMAP)

    again = rescore(published["C0"], items)

    assert again["n_correct"] == PUBLISHED_CORRECT
    assert again["n"] == PUBLISHED_N
    assert again["predictions"] == published["C0"]["predictions"]


def test_every_condition_carries_the_same_predictions(published):
    """What makes it a floor: the image changes, the prediction does not."""
    reference = published["C0"]["predictions"]
    for condition, record in published.items():
        assert record["predictions"] == reference, condition
        assert record["n_correct"] == PUBLISHED_CORRECT, condition


def test_the_qids_are_exactly_the_clean_split_test_set(published):
    """The floor must be measured on the same items as Acc(C0), not a subset."""
    expected = {i["qid"] for i in load_test_items(SWAPMAP)}
    assert {p["qid"] for p in published["C0"]["predictions"]} == expected
    assert len(expected) == PUBLISHED_N


def test_the_floor_runs_on_every_question_unlike_slake_and_vqarad(published):
    """Kvasir has no open/closed label, so there is no scope footnote here."""
    assert published["C0"]["n"] == PUBLISHED_N
    slake = json.loads((RESULTS / "M0_question_only_SLAKE_C0.json").read_text())
    assert slake["n"] < published["C0"]["n"]      # SLAKE is closed-only: 416
