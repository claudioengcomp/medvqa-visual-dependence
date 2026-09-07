import json

import pytest
from PIL import Image

from src.interventions.arrow_io import read_arrow
from src.interventions.make_c5_matched_pathology_swap import (
    apply_matched_swap,
    build_eligible_pairs,
    matched_derangement,
)
from tests.helpers import write_sample_arrow, write_sample_arrow_with_real_images


def test_matched_derangement_never_pairs_same_category():
    labels = ["polyp"] * 5 + ["ulcerative colitis"] * 5
    perm = matched_derangement(labels, seed=42)
    for i, donor_idx in enumerate(perm):
        assert labels[donor_idx] != labels[i]


def test_matched_derangement_is_a_valid_permutation():
    labels = ["polyp"] * 5 + ["ulcerative colitis"] * 5
    perm = matched_derangement(labels, seed=42)
    assert sorted(perm) == list(range(10))


def test_matched_derangement_deterministic_for_same_seed():
    labels = ["polyp"] * 5 + ["ulcerative colitis"] * 5
    assert matched_derangement(labels, seed=7) == matched_derangement(labels, seed=7)


def test_matched_derangement_raises_if_category_is_majority():
    labels = ["polyp"] * 8 + ["ulcerative colitis"] * 2
    with pytest.raises(ValueError):
        matched_derangement(labels, seed=42)


def test_matched_derangement_works_with_more_than_two_categories():
    # exercises the SLAKE case (many possible categories, not just 2 as in Kvasir)
    labels = ["cardiomegaly"] * 4 + ["atelectasis"] * 4 + ["pneumonia"] * 4 + ["nodule"] * 4
    perm = matched_derangement(labels, seed=1)
    for i, donor_idx in enumerate(perm):
        assert labels[donor_idx] != labels[i]


def test_matched_derangement_scales_to_real_kvasir_size_balanced_50_50():
    # Regression guard: draw-and-reject (the original implementation) is
    # statistically hopeless at this scale (~1 in 10^299 per attempt) -- the
    # deterministic construction must keep working here.
    labels = ["polyp"] * 500 + ["ulcerative colitis"] * 500
    for seed in range(10):
        perm = matched_derangement(labels, seed=seed)
        assert sorted(perm) == list(range(1000))
        assert all(labels[perm[i]] != labels[i] for i in range(1000))


def test_matched_derangement_handles_odd_n_with_three_categories():
    labels = ["a"] * 3 + ["b"] * 2 + ["c"] * 2  # n=7, feasible (largest <= floor(n/2))
    for seed in range(20):
        perm = matched_derangement(labels, seed=seed)
        assert sorted(perm) == list(range(7))
        assert all(labels[perm[i]] != labels[i] for i in range(7))


def test_matched_derangement_boundary_majority_exactly_half_is_feasible():
    labels = ["a"] * 50 + ["b"] * 25 + ["c"] * 25  # largest = exactly 50%, still feasible
    perm = matched_derangement(labels, seed=0)
    assert all(labels[perm[i]] != labels[i] for i in range(100))


def test_build_eligible_pairs_filters_by_allowed_categories():
    image_ids = ["a", "b", "c", "d"]
    labels_by_image = {"a": "polyp", "b": "ulcerative colitis", "c": "esophagitis", "d": "polyp"}
    eligible = build_eligible_pairs(image_ids, labels_by_image, allowed_categories=["polyp", "ulcerative colitis"])
    assert eligible == ["a", "b", "d"]


def test_build_eligible_pairs_allows_none_to_mean_all_labeled_categories():
    image_ids = ["a", "b", "c"]
    labels_by_image = {"a": "cardiomegaly", "b": "atelectasis"}  # "c" has no label
    eligible = build_eligible_pairs(image_ids, labels_by_image, allowed_categories=None)
    assert eligible == ["a", "b"]


def test_apply_matched_swap_end_to_end(tmp_path):
    arrow_path = tmp_path / "in.arrow"
    write_sample_arrow(str(arrow_path), n=6)  # image_id: imgs/xmlab0..5/source.jpg (fixture)

    labels = {
        "imgs/xmlab0/source.jpg": "polyp", "imgs/xmlab1/source.jpg": "polyp",
        "imgs/xmlab2/source.jpg": "polyp",
        "imgs/xmlab3/source.jpg": "ulcerative colitis", "imgs/xmlab4/source.jpg": "ulcerative colitis",
        "imgs/xmlab5/source.jpg": "ulcerative colitis",
    }
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels))

    output_path = tmp_path / "out.arrow"
    manifest_path = tmp_path / "manifest.json"
    apply_matched_swap(str(arrow_path), str(labels_path), str(output_path), str(manifest_path),
                        allowed_categories=["polyp", "ulcerative colitis"], seed=42)

    df_out = read_arrow(str(output_path))
    assert len(df_out) == 6
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest) == 6
    for entry in manifest:
        assert entry["original_label"] != entry["donor_label"]


def test_apply_matched_swap_saves_example_pairs_as_png_with_index(tmp_path):
    arrow_path = tmp_path / "in.arrow"
    write_sample_arrow_with_real_images(str(arrow_path), n=6)  # real image bytes, not just "IMG0"

    labels = {
        "imgs/xmlab0/source.jpg": "polyp", "imgs/xmlab1/source.jpg": "polyp",
        "imgs/xmlab2/source.jpg": "polyp",
        "imgs/xmlab3/source.jpg": "ulcerative colitis", "imgs/xmlab4/source.jpg": "ulcerative colitis",
        "imgs/xmlab5/source.jpg": "ulcerative colitis",
    }
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels))

    examples_dir = tmp_path / "examples"
    apply_matched_swap(str(arrow_path), str(labels_path), str(tmp_path / "out.arrow"),
                        str(tmp_path / "manifest.json"),
                        allowed_categories=["polyp", "ulcerative colitis"], seed=42,
                        examples_dir=str(examples_dir), n_examples=3)

    index = json.loads((examples_dir / "index.json").read_text())
    assert len(index) == 3
    for entry in index:
        assert entry["original_label"] != entry["donor_label"]
        original_png = examples_dir / entry["original_file"]
        donor_png = examples_dir / entry["donor_file"]
        assert original_png.exists()
        assert donor_png.exists()
        # confirm these are real, decodable PNGs and not just copied bytes
        Image.open(original_png).verify()
        Image.open(donor_png).verify()


def test_apply_matched_swap_examples_deterministic_for_same_seed(tmp_path):
    arrow_path = tmp_path / "in.arrow"
    write_sample_arrow_with_real_images(str(arrow_path), n=6)
    labels = {
        "imgs/xmlab0/source.jpg": "polyp", "imgs/xmlab1/source.jpg": "polyp",
        "imgs/xmlab2/source.jpg": "polyp",
        "imgs/xmlab3/source.jpg": "ulcerative colitis", "imgs/xmlab4/source.jpg": "ulcerative colitis",
        "imgs/xmlab5/source.jpg": "ulcerative colitis",
    }
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels))

    indexes = []
    for run in range(2):
        examples_dir = tmp_path / f"examples_{run}"
        apply_matched_swap(str(arrow_path), str(labels_path), str(tmp_path / f"out_{run}.arrow"),
                            str(tmp_path / f"manifest_{run}.json"),
                            allowed_categories=["polyp", "ulcerative colitis"], seed=42,
                            examples_dir=str(examples_dir), n_examples=3)
        indexes.append(json.loads((examples_dir / "index.json").read_text()))

    assert indexes[0] == indexes[1]


def test_apply_matched_swap_leaves_ineligible_rows_untouched(tmp_path):
    arrow_path = tmp_path / "in.arrow"
    write_sample_arrow(str(arrow_path), n=4)

    # only 2 of the 4 images have an eligible label; the other 2 have no label at all
    labels = {"imgs/xmlab0/source.jpg": "polyp", "imgs/xmlab1/source.jpg": "ulcerative colitis"}
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels))

    df_in = read_arrow(str(arrow_path))
    output_path = tmp_path / "out.arrow"
    manifest_path = tmp_path / "manifest.json"
    apply_matched_swap(str(arrow_path), str(labels_path), str(output_path), str(manifest_path),
                        allowed_categories=["polyp", "ulcerative colitis"], seed=42)

    df_out = read_arrow(str(output_path))
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest) == 2
    # rows 2 and 3 (unlabelled) must keep their original image
    assert df_out["image"].iloc[2] == df_in["image"].iloc[2]
    assert df_out["image"].iloc[3] == df_in["image"].iloc[3]
