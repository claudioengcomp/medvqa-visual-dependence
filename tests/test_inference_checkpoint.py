import json

from src.inference.inference_checkpoint import append_checkpoint, filter_pending, load_checkpoint


def test_load_checkpoint_missing_file_returns_empty():
    preds, done = load_checkpoint("/tmp/does-not-exist-battery-checkpoint.jsonl")
    assert preds == []
    assert done == set()


def test_load_checkpoint_none_path_returns_empty():
    preds, done = load_checkpoint(None)
    assert preds == []
    assert done == set()


def test_append_then_load_round_trips(tmp_path):
    path = str(tmp_path / "ckpt.jsonl")
    append_checkpoint(path, [
        {"qid": 1, "predicted": "yes", "gold": "yes", "correct": True},
        {"qid": 2, "predicted": "no", "gold": "yes", "correct": False},
    ])
    preds, done = load_checkpoint(path)
    assert len(preds) == 2
    assert done == {1, 2}


def test_append_is_additive_not_overwriting(tmp_path):
    path = str(tmp_path / "ckpt.jsonl")
    append_checkpoint(path, [{"qid": 1, "predicted": "yes", "gold": "yes", "correct": True}])
    append_checkpoint(path, [{"qid": 2, "predicted": "no", "gold": "no", "correct": True}])
    preds, done = load_checkpoint(path)
    assert len(preds) == 2
    assert done == {1, 2}


def test_append_checkpoint_none_path_is_noop(tmp_path):
    append_checkpoint(None, [{"qid": 1, "predicted": "yes", "gold": "yes", "correct": True}])
    # nothing to check beyond "it does not raise" -- no path, no file


def test_append_checkpoint_empty_predictions_does_not_create_file(tmp_path):
    path = str(tmp_path / "never_created.jsonl")
    append_checkpoint(path, [])
    import os
    assert not os.path.exists(path)


def test_filter_pending_skips_done_qids():
    examples = [{"id": 1, "x": "a"}, {"id": 2, "x": "b"}, {"id": 3, "x": "c"}]
    pending = filter_pending(examples, done_qids={2})
    assert [e["id"] for e in pending] == [1, 3]


def test_filter_pending_empty_done_qids_returns_everything():
    examples = [{"id": 1}, {"id": 2}]
    assert filter_pending(examples, done_qids=set()) == examples


def test_filter_pending_coerces_qid_types():
    # the dataset may hand back an id as np.int64 or str -- always compare as int
    examples = [{"id": "5"}, {"id": 6}]
    pending = filter_pending(examples, done_qids={5})
    assert [e["id"] for e in pending] == [6]


def test_checkpoint_file_is_valid_jsonl_one_object_per_line(tmp_path):
    path = str(tmp_path / "ckpt.jsonl")
    append_checkpoint(path, [{"qid": 1, "predicted": "a", "gold": "a", "correct": True}])
    append_checkpoint(path, [{"qid": 2, "predicted": "b", "gold": "b", "correct": True}])
    with open(path) as fp:
        lines = fp.readlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # must not raise
