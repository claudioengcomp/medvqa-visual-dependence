"""Four anchor numbers of docs/v4/numbers_v4.md, read back from the JSONs they cite.

The sheet is generated (src/featurespace/make_numbers.py) and every line names
its JSON; this test keeps the shipped copy of the sheet and the shipped JSONs
from drifting apart: G1 (C1 highest in 30 of 30 cells), G3 (14 of 15 cells),
analysis B (near fraction 0.617 over 478 pairs) and analysis D (W1 0.3244).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SHEET = Path("docs/v4/numbers_v4.md")
V4 = Path("docs/v4")


def _line(label):
    text = SHEET.read_text()
    m = re.search(r"\*\*%s\*\* — `([^`]*)`" % re.escape(label), text)
    assert m, label
    return m.group(1)


def test_g1_c1_highest_in_30_of_30_cells():
    rows = json.loads((V4 / "g1_results.json").read_text())["checks"]["c1_highest"]["rows"]
    n = sum(1 for r in rows if r["c1_highest"])
    assert (n, len(rows)) == (30, 30)
    assert _line("C1 highest").startswith("30 of 30 cells")


def test_g3_passes_in_14_of_15_cells():
    g3 = json.loads((V4 / "g3_results.json").read_text())
    assert (int(g3["cells_passing"]), len(g3["cells"])) == (14, 15)
    assert "14 of 15 cells" in _line("G3 verdict")


def test_analysis_b_near_fraction_over_478_pairs():
    cell = json.loads((V4 / "analysis_b.json").read_text())["cells"]["KVASIR|C2|medsiglip"]
    assert int(cell["classifiable"]) == 478
    assert round(float(cell["near_fraction"]), 4) == 0.6172
    assert _line("C2 near fraction, KVASIR") == "0.6172 over 478 classifiable pairs"


def test_analysis_d_w1_between_kvasir_and_slake():
    w1 = json.loads((V4 / "analysis_d.json").read_text())["w1"]
    value = w1 if isinstance(w1, (int, float)) else next(v for v in _numbers(w1))
    assert round(float(value), 4) == 0.3244
    assert _line("W1 between the d distributions, Kvasir and SLAKE") == "0.3244"


def _numbers(obj):
    if isinstance(obj, (int, float)):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _numbers(v)
