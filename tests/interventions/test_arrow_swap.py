import pytest

from src.interventions.arrow_swap import swap_test_arrow


def _make_arrow_dir(tmp_path, target_bytes: bytes):
    arrow_dir = tmp_path / "data" / "finetune_arrows"
    arrow_dir.mkdir(parents=True)
    target = arrow_dir / "vqa_slack_test.arrow"
    target.write_bytes(target_bytes)
    return target


def test_swap_test_arrow_restores_original_after_block(tmp_path):
    target = _make_arrow_dir(tmp_path, b"ORIGINAL")
    other = tmp_path / "vqa_slack_test_c1.arrow"
    other.write_bytes(b"C1-VERSION")

    with swap_test_arrow(str(tmp_path), str(other)) as swapped_path:
        assert swapped_path == str(target)
        assert target.read_bytes() == b"C1-VERSION"

    assert target.read_bytes() == b"ORIGINAL"


def test_swap_test_arrow_restores_even_on_exception(tmp_path):
    target = _make_arrow_dir(tmp_path, b"ORIGINAL")
    other = tmp_path / "vqa_slack_test_c1.arrow"
    other.write_bytes(b"C1-VERSION")

    with pytest.raises(RuntimeError):
        with swap_test_arrow(str(tmp_path), str(other)):
            raise RuntimeError("boom")

    assert target.read_bytes() == b"ORIGINAL"


def test_swap_test_arrow_missing_target_raises(tmp_path):
    (tmp_path / "data" / "finetune_arrows").mkdir(parents=True)
    other = tmp_path / "c1.arrow"
    other.write_bytes(b"X")
    with pytest.raises(FileNotFoundError):
        with swap_test_arrow(str(tmp_path), str(other)):
            pass


def test_swap_test_arrow_missing_source_raises(tmp_path):
    _make_arrow_dir(tmp_path, b"ORIGINAL")
    with pytest.raises(FileNotFoundError):
        with swap_test_arrow(str(tmp_path), str(tmp_path / "does_not_exist.arrow")):
            pass


def test_swap_test_arrow_accepts_custom_filename_for_other_datasets(tmp_path):
    """For VQA-RAD the test file is named 'vqa_vqa_rad_test.arrow', not
    'vqa_slack_test.arrow', so the filename must be parameterisable."""
    arrow_dir = tmp_path / "data" / "finetune_arrows"
    arrow_dir.mkdir(parents=True)
    target = arrow_dir / "vqa_vqa_rad_test.arrow"
    target.write_bytes(b"ORIGINAL-RAD")
    other = tmp_path / "vqa_vqa_rad_test_c1.arrow"
    other.write_bytes(b"C1-VERSION-RAD")

    with swap_test_arrow(str(tmp_path), str(other), test_arrow_filename="vqa_vqa_rad_test.arrow") as swapped_path:
        assert swapped_path == str(target)
        assert target.read_bytes() == b"C1-VERSION-RAD"

    assert target.read_bytes() == b"ORIGINAL-RAD"


def test_swap_test_arrow_condition_equal_to_target_is_a_noop(tmp_path):
    """The C0 case: the "condition file" IS vqa_slack_test.arrow itself, so
    no real swap happens. shutil.copy2(x, x) would raise SameFileError if it
    were not handled — this test exists to lock exactly that case down."""
    target = _make_arrow_dir(tmp_path, b"ORIGINAL")

    with swap_test_arrow(str(tmp_path), str(target)) as swapped_path:
        assert swapped_path == str(target)
        assert target.read_bytes() == b"ORIGINAL"

    assert target.read_bytes() == b"ORIGINAL"
