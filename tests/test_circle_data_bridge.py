from PIL import Image as PILImage

from src.inference.circle_data_bridge import load_condition_dataset
from tests.helpers import write_sample_arrow_with_real_images


def test_load_condition_dataset_flattens_one_row_per_question(tmp_path):
    path = tmp_path / "condition.arrow"
    write_sample_arrow_with_real_images(str(path), n=2, size=64)
    ds = load_condition_dataset(str(path))
    assert len(ds) == 4  # 2 images x 2 questions each (sample_rows fixture)


def test_load_condition_dataset_decodes_images_as_pil(tmp_path):
    path = tmp_path / "condition.arrow"
    write_sample_arrow_with_real_images(str(path), n=1, size=64)
    ds = load_condition_dataset(str(path))
    example = ds[0]
    assert isinstance(example["image"], PILImage.Image)
    assert example["image"].size == (64, 64)


def test_load_condition_dataset_preserves_real_question_id(tmp_path):
    path = tmp_path / "condition.arrow"
    write_sample_arrow_with_real_images(str(path), n=2, size=64)
    ds = load_condition_dataset(str(path))
    ids = sorted(example["id"] for example in ds)
    assert ids == [0, 1, 2, 3]  # question_id from the fixture: [[0,1],[2,3]]


def test_load_condition_dataset_matches_med_vlm_normalizer_contract(tmp_path):
    path = tmp_path / "condition.arrow"
    write_sample_arrow_with_real_images(str(path), n=1, size=64)
    ds = load_condition_dataset(str(path))
    example = ds[0]
    assert set(example.keys()) == {"image", "question", "answers", "id"}
    assert isinstance(example["question"], str)
    assert isinstance(example["answers"], list)
    assert all(isinstance(a, str) for a in example["answers"])


def test_load_condition_dataset_reuses_same_image_across_its_questions(tmp_path):
    path = tmp_path / "condition.arrow"
    write_sample_arrow_with_real_images(str(path), n=2, size=64)
    ds = load_condition_dataset(str(path))
    # The first two rows (id 0 and 1) come from the same image (row 0 of the
    # arrow file) - their pixels must be identical.
    by_id = {example["id"]: example for example in ds}
    assert list(by_id[0]["image"].getdata()) == list(by_id[1]["image"].getdata())
