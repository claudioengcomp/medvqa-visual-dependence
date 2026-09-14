"""The counts the README publishes, asserted against the directory.

Three times a number in the README's opening sentence went stale while the
directory moved under it: a superseded model prefix was deleted and the
paired-comparison count kept the old total; the prediction count was corrected
by arithmetic on a figure that was already wrong; and a floor was added without
recounting. Each time the prose was edited by hand and nothing checked it.

So the numbers are not repeated here as literals to be kept in step by hand.
They are read out of the README and compared with what `results/` actually
holds. Adding, removing or regenerating a file now fails this test until the
sentence is updated, which is the only way a count in prose stays true.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

RESULTS = Path("results")
README = Path("README.md")


def _basenames(pattern):
    return [p.name for p in RESULTS.glob(pattern)]


def _is_comparison(name):
    return name.endswith("_vs_C0.json")


def _is_manifest(name):
    return "manifest" in name


def _is_baseline(name):
    return name.startswith("M0_question_only_")


# ------------------------------------------------------------------ the disk
def prediction_files():
    """One per model x seed x condition. Not comparisons, not manifests, and
    not the multi-seed aggregates, which live in results/multiseed."""
    return sorted(n for n in _basenames("*.json")
                  if not _is_comparison(n) and not _is_manifest(n))


def comparison_files():
    return sorted(_basenames("*_vs_C0.json"))


def baseline_comparisons():
    return sorted(n for n in comparison_files() if _is_baseline(n))


def reported_comparisons():
    """The paired tests the paper counts: every comparison that is a test.

    The question-only baseline is excluded because its predictions do not
    change, so b = c = 0 and there is no test to report -- not because its
    files happen to be absent. `test_the_exclusion_is_exactly_the_degenerate_set`
    is what holds those two statements together.
    """
    return sorted(n for n in comparison_files() if not _is_baseline(n))


def _mcnemar(name):
    return json.loads((RESULTS / name).read_text())["mcnemar"]


# ---------------------------------------------------------------- the README
def readme_counts():
    text = " ".join(README.read_text().split())
    pattern = (r"(\d+) prediction files, (\d+) paired comparisons against the "
               r"intact condition \((\d+) comparison files on disk, (\d+) of "
               r"them the question-only baseline")
    match = re.search(pattern, text)
    assert match, "the README no longer states its counts in the expected form"
    aggregates = re.search(r"and (\d+) multi-seed aggregates", text)
    assert aggregates, "the README no longer states the multi-seed count"
    return {"predictions": int(match.group(1)),
            "reported": int(match.group(2)),
            "comparisons": int(match.group(3)),
            "baseline": int(match.group(4)),
            "aggregates": int(aggregates.group(1))}


# ----------------------------------------------------------------- the checks
def test_the_readme_counts_match_the_directory():
    """The assertion whose absence let three numbers go stale."""
    stated = readme_counts()
    assert stated["predictions"] == len(prediction_files())
    assert stated["comparisons"] == len(comparison_files())
    assert stated["baseline"] == len(baseline_comparisons())
    assert stated["reported"] == len(reported_comparisons())
    assert stated["aggregates"] == len(list((RESULTS / "multiseed").iterdir()))


def test_the_four_counts_are_arithmetically_consistent():
    """Stating both a total and a part invites them to drift apart."""
    stated = readme_counts()
    assert stated["reported"] + stated["baseline"] == stated["comparisons"]


def test_the_exclusion_is_exactly_the_degenerate_set():
    """Why the baseline may be left out: b = c = 0 is not a null test, it is
    no test. If a counted comparison were ever degenerate, or a baseline one
    ever informative, the sentence in the README would be a different claim."""
    for name in baseline_comparisons():
        m = _mcnemar(name)
        assert (m["b"], m["c"]) == (0, 0), name
        assert m["cohens_g"] is None, name
    for name in reported_comparisons():
        m = _mcnemar(name)
        assert m["b"] + m["c"] > 0, name


@pytest.mark.parametrize("prefix,expected", [("MEDVLTHINKER_", 29), ("M0_", 29)])
def test_the_zero_shot_and_baseline_comparisons_are_complete(prefix, expected):
    """Kvasir C6 was the last missing cell; these pin the matrix as closed."""
    assert len([n for n in comparison_files() if n.startswith(prefix)]) == expected


def test_the_fine_tuned_comparisons_are_the_435_the_paper_reports():
    fine_tuned = [n for n in reported_comparisons()
                  if not n.startswith("MEDVLTHINKER_")]
    assert len(fine_tuned) == 435
