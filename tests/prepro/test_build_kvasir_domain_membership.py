from src.prepro.build_kvasir_domain_membership import (
    extract_domain_membership,
    split_by_procedure,
)


def test_extract_domain_membership_finds_procedure_question():
    rows = [
        {"image_id": "kvasir/img1", "questions": ["Is there text?", "What type of procedure is the image taken from?"],
         "answers": ["no", "colonoscopy"]},
        {"image_id": "kvasir/img2", "questions": ["What type of procedure is the image taken from?"],
         "answers": ["Gastroscopy"]},  # capitalised -- must be normalised
    ]
    membership = extract_domain_membership(rows)
    assert membership == {"kvasir/img1": "colonoscopy", "kvasir/img2": "gastroscopy"}


def test_extract_domain_membership_skips_images_without_the_question():
    rows = [{"image_id": "kvasir/img1", "questions": ["Is there text?"], "answers": ["no"]}]
    assert extract_domain_membership(rows) == {}


def test_extract_domain_membership_handles_list_wrapped_answer():
    # the real arrow file stores the answer as a list (sometimes a numpy
    # array) -- see _first_answer, the same pattern already used by the
    # landmark extraction.
    rows = [{"image_id": "kvasir/img1", "questions": ["What type of procedure is the image taken from?"],
             "answers": [["colonoscopy"]]}]
    assert extract_domain_membership(rows) == {"kvasir/img1": "colonoscopy"}


def test_split_by_procedure_separates_into_two_lists():
    membership = {"a": "colonoscopy", "b": "gastroscopy", "c": "colonoscopy"}
    colonoscopy, gastroscopy = split_by_procedure(membership)
    assert sorted(colonoscopy) == ["a", "c"]
    assert gastroscopy == ["b"]


def test_split_by_procedure_ignores_unexpected_values():
    # not observed in practice (the module found the vocabulary to be closed
    # over colonoscopy/gastroscopy), but it must not break if something else
    # shows up -- such a value is simply left out of both lists.
    membership = {"a": "colonoscopy", "b": "endoscopy capsule"}
    colonoscopy, gastroscopy = split_by_procedure(membership)
    assert colonoscopy == ["a"]
    assert gastroscopy == []
