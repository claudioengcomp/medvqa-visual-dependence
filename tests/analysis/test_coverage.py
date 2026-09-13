"""Tests for the coverage analysis of Section 5.3.

Three layers, in increasing order of what they depend on:

1. A synthetic mini-benchmark whose |V|, |I| and phi are derived by hand. It
   depends on nothing but the scoring rule, so it is the only check that still
   means something if the real data are wrong.
2. The floor invariant: phi is reported only when most draws clear |V| >= 20.
   This is the rule of Section 3.4 pinned as a test, after a rendering bug
   once displayed a phi computed on 3 draws out of 1,000 as a full-scale
   result.
3. Regression on the published numbers, for the benchmarks that ship with the
   repository. The external ones (PathVQA, PMC-VQA, ProbMed) are skipped when
   their files are absent, since they are not redistributed here.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from src.analysis.ccr import answers_match
from src.analysis.coverage import (FLOOR_V, MASTER_SEED, duplication,
                                   from_swapmap, precompute,
                                   reference_collisions, scale, validate)

SWAPMAP = "data/swapmap.json"


# --------------------------------------------------------------------------
# 1. synthetic mini-benchmark
# --------------------------------------------------------------------------
#     #  img  question              reference    purpose
#     1  A    what is this?         polyp        different references
#     2  B    what is this?         ulcer          -> enters V and I
#     3  A    is it red?            yes          identical references
#     4  B    is it red?            yes            -> enters V, not I
#     5  C    how many lesions?     3            unique question -> not in V
#     6  C    colour?               Polyp.       differs only in case and stop
#     7  D    colour?               polyp          -> the rule MUST merge them
#     8  C    any instrument?       no           a real difference, but a prefix
#     9  D    any instrument?       none           -> the rule MUST NOT merge
#    10  A    unique to a?          alpha        unique question -> not in V
#
# Verifiable donors per item: 1+1+1+1+0+1+1+1+1+0 = 8
# Informative donors per item: 1+1+0+0+0+0+0+1+1+0 = 4   -> phi = 0.50
SYNTHETIC = [
    {"iid": "1", "img": "A", "q": "what is this?", "refs": ["polyp"]},
    {"iid": "2", "img": "B", "q": "what is this?", "refs": ["ulcer"]},
    {"iid": "3", "img": "A", "q": "is it red?", "refs": ["yes"]},
    {"iid": "4", "img": "B", "q": "is it red?", "refs": ["yes"]},
    {"iid": "5", "img": "C", "q": "how many lesions?", "refs": ["3"]},
    {"iid": "6", "img": "C", "q": "colour?", "refs": ["Polyp."]},
    {"iid": "7", "img": "D", "q": "colour?", "refs": ["polyp"]},
    {"iid": "8", "img": "C", "q": "any instrument?", "refs": ["no"]},
    {"iid": "9", "img": "D", "q": "any instrument?", "refs": ["none"]},
    {"iid": "10", "img": "A", "q": "unique to a?", "refs": ["alpha"]},
]
for _it in SYNTHETIC:
    _it.setdefault("stratum", None)
    _it.setdefault("substratum", None)

IMAGES = ["A", "B", "C", "D"]
EXPECTED_VERIFIABLE = {"1": {"B"}, "2": {"A"}, "3": {"B"}, "4": {"A"}, "5": set(),
                       "6": {"D"}, "7": {"C"}, "8": {"D"}, "9": {"C"}, "10": set()}
EXPECTED_INFORMATIVE = {"1": {"B"}, "2": {"A"}, "3": set(), "4": set(), "5": set(),
                        "6": set(), "7": set(), "8": {"D"}, "9": {"C"}, "10": set()}


def test_the_rule_merges_case_and_punctuation_but_not_a_prefix():
    assert answers_match("Polyp.", ["polyp"])
    assert not answers_match("no", ["none"])


def test_verifiable_and_informative_sets_donor_by_donor():
    pre = precompute(SYNTHETIC)
    n = pre["n_img"]
    image_id = {img: k for k, img in enumerate(sorted({i["img"] for i in SYNTHETIC}))}
    V, I = set(pre["V"].tolist()), set(pre["I"].tolist())

    for pos, it in enumerate(SYNTHETIC):
        for donor in IMAGES:
            if donor == it["img"]:
                continue                  # a derangement never returns the own image
            verifiable = int(pre["q_of_item"][pos]) * n + image_id[donor] in V
            informative = int(pre["g_of_item"][pos]) * n + image_id[donor] in I
            assert verifiable == (donor in EXPECTED_VERIFIABLE[it["iid"]]), \
                f"item {it['iid']} donor {donor}: verifiable"
            assert informative == (donor in EXPECTED_INFORMATIVE[it["iid"]]), \
                f"item {it['iid']} donor {donor}: informative"


def test_synthetic_totals_and_phi():
    pre = precompute(SYNTHETIC)
    n = pre["n_img"]
    image_id = {img: k for k, img in enumerate(sorted({i["img"] for i in SYNTHETIC}))}
    V, I = set(pre["V"].tolist()), set(pre["I"].tolist())
    verifiable = informative = 0
    for pos, it in enumerate(SYNTHETIC):
        for donor in IMAGES:
            if donor == it["img"]:
                continue
            verifiable += int(pre["q_of_item"][pos]) * n + image_id[donor] in V
            informative += int(pre["g_of_item"][pos]) * n + image_id[donor] in I
    assert verifiable == 8
    assert informative == 4
    assert informative / verifiable == pytest.approx(0.50)


def test_floor_suppresses_phi_on_the_synthetic_set():
    """Ten items: |V| stays far below 20, so phi must come out suppressed."""
    pre = precompute(SYNTHETIC)
    r = scale(pre, np.random.default_rng(MASTER_SEED), None, None, 200)
    assert r["V_mean"] < FLOOR_V
    assert r["fraction_below_floor"] == 1.0
    assert r["phi_reportable"] is False


# --------------------------------------------------------------------------
# 2. the floor invariant
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dataset", ["KVASIR", "SLAKE", "VQARAD"])
def test_phi_is_reportable_only_with_a_majority_of_draws(dataset):
    items = from_swapmap(dataset, SWAPMAP)
    pre = precompute(items)
    r = scale(pre, np.random.default_rng(MASTER_SEED), None, None, 60)
    majority = r["reportable_draws"] > r["valid_draws"] / 2
    assert r["phi_reportable"] == majority
    if r["phi_reportable"]:
        assert r["V_mean"] >= FLOOR_V
        assert r["phi_mean"] is not None and r["phi_ci95"] is not None
    else:
        # the value may still be stored; it must simply not be reported
        assert r["reportable_draws"] <= r["valid_draws"] / 2


# --------------------------------------------------------------------------
# 3. regression on the published numbers
# --------------------------------------------------------------------------
def test_step2_reproduces_the_published_slake_and_vqarad_numbers():
    """The blocking validation: SLAKE C2 is phi = 56.0% over 125 verifiable."""
    out = validate(SWAPMAP)
    assert out["SLAKE C2"]["V"] == 125
    assert out["SLAKE C2"]["I"] == 70
    assert out["SLAKE C2"]["phi"] == pytest.approx(0.560, abs=5e-4)
    assert out["VQA-RAD C6"]["I"] == 1
    assert out["VQA-RAD C6"]["I"] < 4


@pytest.mark.parametrize("dataset,collisions,tested", [
    ("KVASIR", 0, 353), ("SLAKE", 0, 268), ("VQARAD", 0, 13)])
def test_reference_collisions_match_section_5_3(dataset, collisions, tested):
    out = reference_collisions(from_swapmap(dataset, SWAPMAP))
    assert out["collisions"] == collisions
    assert out["references_tested"] == tested


@pytest.mark.parametrize("dataset,questions,images,distinct,max_per_q", [
    ("KVASIR", 10779, 1291, 18, 777),
    ("SLAKE", 1061, 96, 319, 49),
    ("VQARAD", 451, 203, 430, 4)])
def test_duplication_matches_section_5_3(dataset, questions, images, distinct, max_per_q):
    r = duplication(dataset, from_swapmap(dataset, SWAPMAP))
    assert r["test_questions"] == questions
    assert r["images"] == images
    assert r["distinct_questions"] == distinct
    assert r["max_images_per_question"] == max_per_q


@pytest.mark.parametrize("dataset,phi,lo,hi", [
    ("KVASIR", 0.487, 0.479, 0.496),
    ("SLAKE", 0.595, 0.515, 0.678)])
def test_phi_matches_figure_4(dataset, phi, lo, hi):
    """The full-scale phi and its interval, as annotated in Figure 4.

    1,000 draws with the master seed, the same as the published run.
    """
    pre = precompute(from_swapmap(dataset, SWAPMAP))
    r = scale(pre, np.random.default_rng(MASTER_SEED), None, None, 1000)
    assert r["phi_reportable"]
    assert r["phi_mean"] == pytest.approx(phi, abs=5e-4)
    assert r["phi_ci95"][0] == pytest.approx(lo, abs=5e-4)
    assert r["phi_ci95"][1] == pytest.approx(hi, abs=5e-4)
    assert r["fraction_below_floor"] == 0.0


def test_vqarad_stays_collapsed():
    """VQA-RAD must not quietly become estimable under a future change."""
    pre = precompute(from_swapmap("VQARAD", SWAPMAP))
    r = scale(pre, np.random.default_rng(MASTER_SEED), None, None, 200)
    assert r["fraction_below_floor"] == 1.0
    assert r["reportable_draws"] == 0
    assert r["phi_reportable"] is False


@pytest.mark.skipif(not os.path.exists("data/coverage/ProbMed__test.json"),
                    reason="ProbMed is a gated release and is not redistributed")
def test_probmed_collisions_match_section_5_3():
    from src.analysis.coverage import probmed
    out = reference_collisions(probmed("data/coverage"))
    assert out["collisions"] == 0
    assert out["references_tested"] == 7858
