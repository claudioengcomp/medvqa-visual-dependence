from src.prepro.build_slake_pathology_labels import (
    SLAKE_IMAGE_ID_PREFIX,
    build_slake_bucket_membership,
    build_slake_labels,
    canonical_label_per_image,
    normalize_answer,
)


def test_slake_image_id_prefix_matches_real_arrow_and_m3ae_source():
    # Confirmed twice against real runs: by reading the image_id column of
    # the actual arrow file, and by reading prepro_vqa_slack() in the
    # zhjohnchan/M3AE source (data_root already ends in "/", which is where
    # the double slash comes from -- it is not a typo).
    assert SLAKE_IMAGE_ID_PREFIX == "data/finetune_data/slack//imgs/"


def test_canonical_label_per_image_picks_deterministic_first_when_multiple():
    # canonical_label_per_image expects an already-normalised answer
    # (lowercase, no surrounding whitespace) — normalisation happens inside
    # build_slake_labels via normalize_answer(), before reaching here.
    records = [
        {"img_name": "xmlab1/source.jpg", "answer": "pneumonia"},
        {"img_name": "xmlab1/source.jpg", "answer": "atelectasis"},
    ]
    labels = canonical_label_per_image(records)
    assert labels["xmlab1/source.jpg"] == "atelectasis"  # alphabetical order


def test_build_slake_labels_filters_by_modality_location_and_trivial_answers():
    records = [
        {"img_name": "xmlab1/source.jpg", "content_type": "Abnormality", "q_lang": "en",
         "modality": "X-Ray", "location": "Lung", "answer": "Cardiomegaly"},
        {"img_name": "xmlab2/source.jpg", "content_type": "Abnormality", "q_lang": "en",
         "modality": "CT", "location": "Abdomen", "answer": "Liver Cancer"},  # outside the default bucket
        {"img_name": "xmlab3/source.jpg", "content_type": "Abnormality", "q_lang": "en",
         "modality": "X-Ray", "location": "Lung", "answer": "Yes"},  # trivial, must be excluded
        {"img_name": "xmlab4/source.jpg", "content_type": "Organ", "q_lang": "en",
         "modality": "X-Ray", "location": "Lung", "answer": "Lung"},  # wrong content_type
    ]
    labels = build_slake_labels(records, modality="X-Ray", location="Lung")
    assert labels == {"data/finetune_data/slack//imgs/xmlab1/source.jpg": "cardiomegaly"}


def test_build_slake_labels_includes_chinese_only_pathology_answers():
    # SLAKE is not a parallel EN/ZH corpus for content_type=='Abnormality'
    # -- 61 images in the real test split only carry that question in
    # Chinese, with no English equivalent (confirmed by inspecting
    # test.json). The label is used purely internally (for the C5 pairing)
    # and is never shown to the model, so including it without an English
    # question is safe.
    records = [
        {"img_name": "xmlab1/source.jpg", "content_type": "Abnormality", "q_lang": "zh",
         "modality": "X-Ray", "location": "Lung", "answer": "心脏肥大"},
    ]
    labels = build_slake_labels(records, modality="X-Ray", location="Lung")
    assert labels == {"data/finetune_data/slack//imgs/xmlab1/source.jpg": "cardiomegaly"}


def test_build_slake_labels_excludes_chinese_trivial_and_organ_answers():
    records = [
        {"img_name": "xmlab1/source.jpg", "content_type": "Abnormality", "q_lang": "zh",
         "modality": "X-Ray", "location": "Lung", "answer": "是"},  # "yes", trivial
        {"img_name": "xmlab2/source.jpg", "content_type": "Abnormality", "q_lang": "zh",
         "modality": "X-Ray", "location": "Lung", "answer": "肺"},  # "lung", an organ name
    ]
    assert build_slake_labels(records, modality="X-Ray", location="Lung") == {}


def test_normalize_answer_fixes_known_typos():
    # Real findings from running against the actual SLAKE data:
    # "Noudle"/"cardiomegal"/"cardiomega" are typos in the original dataset,
    # not distinct findings.
    assert normalize_answer("noudle") == "nodule"
    assert normalize_answer("cardiomegal") == "cardiomegaly"
    assert normalize_answer("cardiomega") == "cardiomegaly"
    assert normalize_answer("atelectasis") == "atelectasis"  # unchanged


def test_normalize_answer_excludes_organ_names_that_are_not_pathology():
    # Real finding: "lung" (33 images!), "heart" and "both" show up as
    # answers to content_type=='Abnormality' questions in SLAKE, but they
    # are organ names / ambiguous answers rather than pathologies —
    # normalize_answer returns None to signal exclusion.
    assert normalize_answer("lung") is None
    assert normalize_answer("heart") is None
    assert normalize_answer("both") is None


def test_normalize_answer_translates_chinese_pathology_terms():
    # A small, closed vocabulary, enumerated exhaustively from test.json --
    # it is translated to the same canonical category name used on the
    # English side, keeping the pairing consistent across both languages.
    assert normalize_answer("心脏肥大") == "cardiomegaly"
    assert normalize_answer("肺炎") == "pneumonia"
    assert normalize_answer("气胸") == "pneumothorax"


def test_normalize_answer_canonicalizes_compound_term_order():
    # Real finding in the MRI/Brain_Tissue bucket: English SLAKE writes the
    # SAME clinical category in more than one term order for different
    # images. Without canonicalisation this "fragments" into distinct
    # categories by accident of ordering rather than by a different finding.
    assert (normalize_answer("Brain Enhancing Tumor, Brain Edema")
            == normalize_answer("Brain Edema, Brain Enhancing Tumor"))
    assert normalize_answer("Mass, Atelectasis") == normalize_answer("Atelectasis, Mass")


def test_normalize_answer_single_term_unaffected_by_canonicalization():
    assert normalize_answer("cardiomegaly") == "cardiomegaly"


def test_normalize_answer_excludes_chinese_trivial_and_organ_answers():
    # "不是" ("no"/"is not") escaped the first version of this list and was
    # mistakenly treated as a pathology category in the first real C5 run.
    # It was caught by inspecting the generated manifest, not by a test, so
    # it is locked down here to stop the mistake from recurring.
    assert normalize_answer("是") is None  # "yes"
    assert normalize_answer("否") is None  # "no"
    assert normalize_answer("不是") is None  # "no"/"is not"
    assert normalize_answer("肺") is None  # "lung", an organ name
    assert normalize_answer("心脏") is None  # "heart", an organ name


def test_build_slake_bucket_membership_includes_any_content_type():
    # Unlike build_slake_labels, this neither filters by content_type nor
    # excludes trivial answers -- it only needs to know that the image
    # exists in this bucket (modality+location) so that C6 can find whatever
    # C5 left over.
    records = [
        {"img_name": "xmlab1/source.jpg", "content_type": "Abnormality",
         "modality": "X-Ray", "location": "Lung", "answer": "Cardiomegaly"},
        {"img_name": "xmlab2/source.jpg", "content_type": "Organ",
         "modality": "X-Ray", "location": "Lung", "answer": "Lung"},  # no pathology label, still in the bucket
        {"img_name": "xmlab3/source.jpg", "content_type": "Modality",
         "modality": "CT", "location": "Abdomen", "answer": "CT"},  # wrong bucket
    ]
    membership = build_slake_bucket_membership(records, modality="X-Ray", location="Lung")
    assert membership == [
        "data/finetune_data/slack//imgs/xmlab1/source.jpg",
        "data/finetune_data/slack//imgs/xmlab2/source.jpg",
    ]


def test_build_slake_bucket_membership_deduplicates_repeated_image():
    records = [
        {"img_name": "xmlab1/source.jpg", "content_type": "Abnormality",
         "modality": "X-Ray", "location": "Lung", "answer": "Cardiomegaly"},
        {"img_name": "xmlab1/source.jpg", "content_type": "Organ",
         "modality": "X-Ray", "location": "Lung", "answer": "Lung"},
    ]
    membership = build_slake_bucket_membership(records, modality="X-Ray", location="Lung")
    assert membership == ["data/finetune_data/slack//imgs/xmlab1/source.jpg"]


def test_build_slake_labels_applies_normalization_and_drops_organ_answers():
    records = [
        {"img_name": "xmlab1/source.jpg", "content_type": "Abnormality", "q_lang": "en",
         "modality": "X-Ray", "location": "Lung", "answer": "Noudle"},
        {"img_name": "xmlab2/source.jpg", "content_type": "Abnormality", "q_lang": "en",
         "modality": "X-Ray", "location": "Lung", "answer": "Lung"},
    ]
    labels = build_slake_labels(records, modality="X-Ray", location="Lung")
    assert labels == {"data/finetune_data/slack//imgs/xmlab1/source.jpg": "nodule"}  # xmlab2 excluded (it was just "lung")
