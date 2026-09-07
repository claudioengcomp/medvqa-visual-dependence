"""Derive the subset of `question_id` eligible for the VDD(C5) vs. VDD(C0)
comparison — only the questions whose images took part in the restricted
swap (`src/interventions/make_c5_matched_pathology_swap.py`).

Important detail: the `--manifest` produced by that script stores
`image_id` under the key `qid_image` (a misleading name — it is an IMAGE
id, not a question id), whereas a results record from
`extract_predictions_m3ae.py` has one `predictions` row per QUESTION
(`qid`). Without this filter, `compute_summary_vs_baseline`
(src/stats/compute_vdd.py) would compare C0 and C5 over the WHOLE test set
— including questions whose image was never swapped (a category not
eligible for C5), diluting the measured effect with noise from questions
that are not part of the condition.

`compute_summary_vs_baseline` has no built-in support for restricting the
comparison to a list of qids (confirmed by reading its source); this module
provides that restriction.

Usage (in the notebook, after generating C5 and running inference):
    from src.interventions.c5_eligible_qids import (
        eligible_qids_from_manifest, filter_record_to_qids,
    )
    qids = eligible_qids_from_manifest(arrow_original, manifest_path)
    c0_restrito = filter_record_to_qids(c0_record, qids)
    c5_restrito = filter_record_to_qids(c5_record, qids)
    summary = compute_summary_vs_baseline(c0_restrito, c5_restrito)
"""
from __future__ import annotations

import json

from src.interventions.arrow_io import read_arrow


def eligible_qids_from_manifest(arrow_path: str, manifest_path: str) -> set[int]:
    """`arrow_path`: the ORIGINAL arrow (the same `--input` used by
    make_c5_matched_pathology_swap.py — it has `image_id` + `question_id`
    per row, one row per image). `manifest_path`: the `--manifest` produced
    by that script. Returns the set of `question_id` (int) of ALL questions
    whose image was swapped (i.e. eligible for C5)."""
    with open(manifest_path, "r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    swapped_image_ids = {entry["qid_image"] for entry in manifest}

    df = read_arrow(arrow_path)
    qids: set[int] = set()
    for _, row in df.iterrows():
        if row["image_id"] in swapped_image_ids:
            qids.update(int(q) for q in row["question_id"])
    return qids


def filter_record_to_qids(record: dict, qids: set[int]) -> dict:
    """Return a COPY of `record` (schema from src/inference/results_schema.py) with
    `predictions` restricted to `qids`, and `n`/`n_correct`/`accuracy`
    recomputed over that subset — the original file is left untouched.
    Raises ValueError if `record` has no `predictions` (filtering by qid
    only makes sense with item-level paired data)."""
    if "predictions" not in record:
        raise ValueError("record has no 'predictions' -- filtering by qid requires per-item paired data")
    filtered_predictions = [p for p in record["predictions"] if p["qid"] in qids]
    n = len(filtered_predictions)
    n_correct = sum(1 for p in filtered_predictions if p["correct"])
    filtered = dict(record)
    filtered["predictions"] = filtered_predictions
    filtered["n"] = n
    filtered["n_correct"] = n_correct
    filtered["accuracy"] = (n_correct / n) if n else 0.0
    return filtered
