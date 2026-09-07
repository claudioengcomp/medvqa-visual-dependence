import json

import pytest

from src.interventions.c5_eligible_qids import (
    eligible_qids_from_manifest,
    filter_record_to_qids,
)
from tests.helpers import write_sample_arrow


def test_eligible_qids_from_manifest_returns_qids_of_swapped_images_only(tmp_path):
    arrow_path = tmp_path / "in.arrow"
    write_sample_arrow(str(arrow_path), n=3)
    # sample_rows(n=3): image_id="imgs/xmlab{i}/source.jpg", question_id=[i*2, i*2+1]
    manifest = [
        {"qid_image": "imgs/xmlab0/source.jpg", "original_label": "a",
         "donor_image": "imgs/xmlab1/source.jpg", "donor_label": "b"},
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    qids = eligible_qids_from_manifest(str(arrow_path), str(manifest_path))
    assert qids == {0, 1}  # only xmlab0's questions, not xmlab1/xmlab2


def test_eligible_qids_from_manifest_empty_manifest_returns_empty_set(tmp_path):
    arrow_path = tmp_path / "in.arrow"
    write_sample_arrow(str(arrow_path), n=2)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps([]))

    assert eligible_qids_from_manifest(str(arrow_path), str(manifest_path)) == set()


def test_filter_record_to_qids_recomputes_accuracy():
    record = {
        "model": "M3AE", "dataset": "SLAKE", "condition": "C0",
        "predictions": [
            {"qid": 1, "predicted": "a", "gold": "a", "correct": True},
            {"qid": 2, "predicted": "b", "gold": "c", "correct": False},
            {"qid": 3, "predicted": "a", "gold": "a", "correct": True},
        ],
        "n": 3, "n_correct": 2, "accuracy": 2 / 3,
    }
    filtered = filter_record_to_qids(record, qids={1, 2})
    assert filtered["n"] == 2
    assert filtered["n_correct"] == 1
    assert filtered["accuracy"] == 0.5
    assert {p["qid"] for p in filtered["predictions"]} == {1, 2}
    # does not mutate the original record
    assert record["n"] == 3


def test_filter_record_to_qids_raises_without_predictions():
    with pytest.raises(ValueError):
        filter_record_to_qids(
            {"model": "M", "dataset": "D", "condition": "C0", "n": 1, "n_correct": 1, "accuracy": 1.0},
            qids={1},
        )


def test_filter_record_to_qids_empty_intersection_gives_zero_accuracy():
    record = {
        "model": "M3AE", "dataset": "SLAKE", "condition": "C0",
        "predictions": [{"qid": 1, "predicted": "a", "gold": "a", "correct": True}],
        "n": 1, "n_correct": 1, "accuracy": 1.0,
    }
    filtered = filter_record_to_qids(record, qids={999})
    assert filtered["n"] == 0
    assert filtered["n_correct"] == 0
    assert filtered["accuracy"] == 0.0
