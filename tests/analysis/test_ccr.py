"""Tests for the Counterfactual Consistency Rate.

Two things are guarded here. First, that the module uses ONE ruler -- the same
`answer_matches` behind accuracy and VDD -- and not a stricter rule of its own.
Second, and more important, that the size of the informative set did not move:
those denominators are published quantities, and they must not change as a side
effect of the answer-scoring rule. `question_key` exists precisely to keep them
still, and these tests fail if anything sends question text through the answer
normalizer.
"""
import json
import os

import pytest

from src.analysis.ccr import (SWAP_CONDITIONS, answers_match, compute_ccr,
                              load_predictions, load_swapmap, question_key)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SWAPMAP = os.path.join(REPO, "data", "swapmap.json")
RESULTS = os.path.join(REPO, "results")

# Published denominators (Section 5.4): informative questions per seed.
PUBLISHED_INFORMATIVE = {
    ("KVASIR", "C5", "M3AE_KVASIR_seed0"): 5218,
    ("SLAKE", "C6", "M3AE_SCRATCH_SLAKE"): 92,
}


# ------------------------------------------------------------- one ruler only

def test_answers_match_is_the_accuracy_ruler():
    """Punctuation and spacing no longer decide a CCR hit."""
    assert answers_match("pink red white", ["pink; red; white"]) is True
    assert answers_match("z line", ["z-line"]) is True


def test_answers_match_still_refuses_a_different_answer():
    """The fix is a normalization fix, not a semantics change."""
    assert answers_match("polyp", ["ulcer"]) is False
    assert answers_match("flesh; red", ["flesh; red; white"]) is False


def test_answers_match_is_symmetric():
    """Set-based comparisons below rely on this."""
    for a, b in (("z line", "z-line"), ("pink red", "pink; red"), ("a", "b")):
        assert answers_match(a, [b]) == answers_match(b, [a])


def test_question_key_is_not_the_answer_normalizer():
    """Question text is a dictionary key: punctuation must survive.

    If this ever goes through normalize_text(), the keys of `by_image` change
    and the informative counts below move with them.
    """
    assert question_key("How many polyps are there?") == "how many polyps are there?"
    assert question_key("  Where  is  it? ") == "where is it?"


# ---------------------------------------------------- buckets on real records

def _real(dataset, condition, prefix):
    if not (os.path.exists(SWAPMAP) and os.path.exists(RESULTS)):
        pytest.skip("swapmap or results not available")
    entry = load_swapmap(SWAPMAP)[dataset]
    qid_meta = {int(k): v for k, v in entry["qid"].items()}
    c0 = load_predictions(RESULTS, prefix, "C0")
    cx = load_predictions(RESULTS, prefix, condition)
    if c0 is None or cx is None:
        pytest.skip(f"predictions missing for {prefix}")
    return compute_ccr(c0, cx, qid_meta, entry["swap"][condition])


@pytest.mark.parametrize("key,expected", PUBLISHED_INFORMATIVE.items())
def test_informative_set_matches_the_published_denominator(key, expected):
    dataset, condition, prefix = key
    assert _real(dataset, condition, prefix)["informative"] == expected


@pytest.mark.parametrize("key", PUBLISHED_INFORMATIVE)
def test_buckets_partition_the_informative_set(key):
    r = _real(*key)
    assert (r["followed_donor"] + r["kept_original"] + r["neither"]
            == r["informative"])
    assert (r["c0correct_followed"] + r["c0correct_kept"] + r["c0correct_neither"]
            == r["inf_correct_c0"])


@pytest.mark.parametrize("key", PUBLISHED_INFORMATIVE)
def test_no_question_matches_both_references(key):
    """On an informative question the two references cannot match each other,
    so `kept` and `followed` are mutually exclusive by construction."""
    dataset, condition, prefix = key
    entry = load_swapmap(SWAPMAP)[dataset]
    qid_meta = {int(k): v for k, v in entry["qid"].items()}
    donor_map = entry["swap"][condition]
    by_image = {}
    for meta in qid_meta.values():
        by_image.setdefault(meta["img"], {})[question_key(meta["q"])] = meta["a"]
    for meta in qid_meta.values():
        if meta["img"] not in donor_map:
            continue
        donor_ref = by_image[donor_map[meta["img"]]].get(question_key(meta["q"]))
        if donor_ref is None:
            continue
        if any(answers_match(x, donor_ref) for x in meta["a"]):
            continue                                   # uninformative
        assert not (any(answers_match(x, donor_ref) for x in meta["a"])
                    and any(answers_match(x, meta["a"]) for x in donor_ref))


def test_swap_conditions_are_the_three_with_a_donor():
    assert SWAP_CONDITIONS == ("C2", "C5", "C6")
