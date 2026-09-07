import json

from src.interventions.arrow_io import read_arrow
from src.interventions.make_c6_organ_modality_swap import (
    apply_organ_modality_swap,
    leftover_images,
)
from tests.helpers import write_sample_arrow


def test_leftover_images_excludes_c5_covered():
    bucket = ["a", "b", "c", "d"]
    c5_manifest = [{"qid_image": "a", "donor_image": "b"}, {"qid_image": "b", "donor_image": "a"}]
    assert leftover_images(bucket, c5_manifest) == ["c", "d"]


def test_leftover_images_preserves_bucket_order():
    bucket = ["z", "y", "x"]
    assert leftover_images(bucket, c5_manifest=[]) == ["z", "y", "x"]


def test_apply_organ_modality_swap_keeps_c5_pairs_intact(tmp_path):
    arrow_path = tmp_path / "in.arrow"
    write_sample_arrow(str(arrow_path), n=6)  # imgs/xmlab0..5/source.jpg

    bucket = [f"imgs/xmlab{i}/source.jpg" for i in range(6)]
    bucket_path = tmp_path / "bucket.json"
    bucket_path.write_text(json.dumps(bucket))

    c5_manifest = [
        {"qid_image": "imgs/xmlab0/source.jpg", "original_label": "polyp",
         "donor_image": "imgs/xmlab1/source.jpg", "donor_label": "ulcerative colitis"},
        {"qid_image": "imgs/xmlab1/source.jpg", "original_label": "ulcerative colitis",
         "donor_image": "imgs/xmlab0/source.jpg", "donor_label": "polyp"},
    ]
    c5_manifest_path = tmp_path / "c5_manifest.json"
    c5_manifest_path.write_text(json.dumps(c5_manifest))

    output_path = tmp_path / "out.arrow"
    manifest_path = tmp_path / "manifest.json"
    apply_organ_modality_swap(str(arrow_path), str(bucket_path), str(c5_manifest_path),
                               str(output_path), str(manifest_path), seed=42)

    manifest = json.loads(manifest_path.read_text())
    # the 2 C5 pairs must appear unchanged in the C6 manifest
    c6_c5_entries = [e for e in manifest if e["qid_image"] in ("imgs/xmlab0/source.jpg", "imgs/xmlab1/source.jpg")]
    assert {e["qid_image"]: e["donor_image"] for e in c6_c5_entries} == {
        "imgs/xmlab0/source.jpg": "imgs/xmlab1/source.jpg",
        "imgs/xmlab1/source.jpg": "imgs/xmlab0/source.jpg",
    }


def test_apply_organ_modality_swap_pairs_leftover_freely(tmp_path):
    arrow_path = tmp_path / "in.arrow"
    write_sample_arrow(str(arrow_path), n=6)

    bucket = [f"imgs/xmlab{i}/source.jpg" for i in range(6)]
    bucket_path = tmp_path / "bucket.json"
    bucket_path.write_text(json.dumps(bucket))

    # only xmlab0/xmlab1 are paired in C5 -- xmlab2..5 are left over for C6
    c5_manifest = [
        {"qid_image": "imgs/xmlab0/source.jpg", "original_label": "polyp",
         "donor_image": "imgs/xmlab1/source.jpg", "donor_label": "ulcerative colitis"},
        {"qid_image": "imgs/xmlab1/source.jpg", "original_label": "ulcerative colitis",
         "donor_image": "imgs/xmlab0/source.jpg", "donor_label": "polyp"},
    ]
    c5_manifest_path = tmp_path / "c5_manifest.json"
    c5_manifest_path.write_text(json.dumps(c5_manifest))

    output_path = tmp_path / "out.arrow"
    manifest_path = tmp_path / "manifest.json"
    apply_organ_modality_swap(str(arrow_path), str(bucket_path), str(c5_manifest_path),
                               str(output_path), str(manifest_path), seed=42)

    manifest = json.loads(manifest_path.read_text())
    # 2 inherited from C5 + 4 new ones (xmlab2..5, all paired among themselves)
    assert len(manifest) == 6
    covered = {e["qid_image"] for e in manifest}
    assert covered == set(bucket)

    df_out = read_arrow(str(output_path))
    df_in = read_arrow(str(arrow_path))
    # xmlab2..5 must have had their image changed (they did not keep their own)
    for i in range(2, 6):
        assert df_out["image"].iloc[i] != df_in["image"].iloc[i]


def test_apply_organ_modality_swap_skips_single_leftover(tmp_path):
    arrow_path = tmp_path / "in.arrow"
    write_sample_arrow(str(arrow_path), n=3)

    bucket = [f"imgs/xmlab{i}/source.jpg" for i in range(3)]
    bucket_path = tmp_path / "bucket.json"
    bucket_path.write_text(json.dumps(bucket))

    # xmlab0/xmlab1 are paired in C5; xmlab2 is left over alone -- no pairing possible
    c5_manifest = [
        {"qid_image": "imgs/xmlab0/source.jpg", "original_label": "polyp",
         "donor_image": "imgs/xmlab1/source.jpg", "donor_label": "ulcerative colitis"},
        {"qid_image": "imgs/xmlab1/source.jpg", "original_label": "ulcerative colitis",
         "donor_image": "imgs/xmlab0/source.jpg", "donor_label": "polyp"},
    ]
    c5_manifest_path = tmp_path / "c5_manifest.json"
    c5_manifest_path.write_text(json.dumps(c5_manifest))

    output_path = tmp_path / "out.arrow"
    manifest_path = tmp_path / "manifest.json"
    apply_organ_modality_swap(str(arrow_path), str(bucket_path), str(c5_manifest_path),
                               str(output_path), str(manifest_path), seed=42)

    manifest = json.loads(manifest_path.read_text())
    assert len(manifest) == 2  # only the ones inherited from C5 -- xmlab2 is left out
    assert "imgs/xmlab2/source.jpg" not in {e["qid_image"] for e in manifest}

    df_out = read_arrow(str(output_path))
    df_in = read_arrow(str(arrow_path))
    assert df_out["image"].iloc[2] == df_in["image"].iloc[2]  # original image, untouched


def test_apply_organ_modality_swap_deterministic_for_same_seed(tmp_path):
    arrow_path = tmp_path / "in.arrow"
    write_sample_arrow(str(arrow_path), n=6)

    bucket = [f"imgs/xmlab{i}/source.jpg" for i in range(6)]
    bucket_path = tmp_path / "bucket.json"
    bucket_path.write_text(json.dumps(bucket))
    c5_manifest_path = tmp_path / "c5_manifest.json"
    c5_manifest_path.write_text(json.dumps([]))  # everything is left over for free pairing

    manifests = []
    for run in range(2):
        output_path = tmp_path / f"out_{run}.arrow"
        manifest_path = tmp_path / f"manifest_{run}.json"
        apply_organ_modality_swap(str(arrow_path), str(bucket_path), str(c5_manifest_path),
                                   str(output_path), str(manifest_path), seed=42)
        manifests.append(json.loads(manifest_path.read_text()))

    assert manifests[0] == manifests[1]
