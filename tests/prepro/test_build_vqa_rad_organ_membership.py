from src.prepro.build_vqa_rad_organ_membership import (
    VQA_RAD_IMAGE_ID_PREFIX,
    build_organ_membership,
)


def test_vqa_rad_image_id_prefix():
    # confirmed by reading the image_id column of the real arrow file
    assert VQA_RAD_IMAGE_ID_PREFIX == "data/finetune_data/vqa_rad//images/"


def test_build_organ_membership_groups_by_organ():
    records = [
        {"image_name": "synpic1.jpg", "image_organ": "CHEST"},
        {"image_name": "synpic2.jpg", "image_organ": "ABD"},
        {"image_name": "synpic3.jpg", "image_organ": "HEAD"},
    ]
    membership = build_organ_membership(records)
    assert membership == {
        "CHEST": ["data/finetune_data/vqa_rad//images/synpic1.jpg"],
        "ABD": ["data/finetune_data/vqa_rad//images/synpic2.jpg"],
        "HEAD": ["data/finetune_data/vqa_rad//images/synpic3.jpg"],
    }


def test_build_organ_membership_deduplicates_repeated_image():
    # the same image appears on several rows (one per question)
    records = [
        {"image_name": "synpic1.jpg", "image_organ": "CHEST"},
        {"image_name": "synpic1.jpg", "image_organ": "CHEST"},
        {"image_name": "synpic1.jpg", "image_organ": "CHEST"},
    ]
    membership = build_organ_membership(records)
    assert membership == {"CHEST": ["data/finetune_data/vqa_rad//images/synpic1.jpg"]}


def test_build_organ_membership_preserves_first_occurrence_order():
    records = [
        {"image_name": "synpic3.jpg", "image_organ": "CHEST"},
        {"image_name": "synpic1.jpg", "image_organ": "CHEST"},
        {"image_name": "synpic2.jpg", "image_organ": "CHEST"},
    ]
    membership = build_organ_membership(records)
    assert membership["CHEST"] == [
        "data/finetune_data/vqa_rad//images/synpic3.jpg",
        "data/finetune_data/vqa_rad//images/synpic1.jpg",
        "data/finetune_data/vqa_rad//images/synpic2.jpg",
    ]
