"""The identity of Section 3.4, verified on the saved predictions (Section 5.1).

Under a swap the model receives an existing test item, so the CCR -- the
altered answer scored against the DONOR's reference -- is the accuracy the same
model already obtains under C0 on that donor item. This holds by construction
of the swap. It is verified rather than asserted because inference is not
bit-exact across environments (Section 5.6) and list answers admit reordering.

The module partitions the swapped questions into

    informative   the donor carries the question and the reference differs
    same_ref      the donor carries the question and the reference coincides
    novel         the donor does not carry the question

and, on the informative set, compares the prediction under the swap with the
prediction under C0 on the donor item, cell by cell (configuration x seed).

A single scoring rule: question_key and answers_match come from
src/analysis/ccr.py, the pair behind accuracy, VDD and the CCR.

Two ways of naming the donor item (`--pair-by`):

    sha   (default, what the paper reports) the donor item is every C0 item
          whose image has the same SHA-256 as the bytes the swapped row received
          and whose question text is the same; read from the per-row manifests
          in results/manifests/. Byte-identical copies of one picture under
          distinct image identifiers (Kvasir-VQA-x1 has 34 test images with an
          identical copy in the training set and 7 within the test set) are the
          same item, so the identity is checked against all of them.
    id    the donor item is the C0 item with the donor's image identifier. A
          copy under another identifier is then a different item, and the
          identity can fail where the model answered the two copies differently.

Three cells have their C0 and swap predictions from different environments or
batch compositions (Section 5.6); `--reruns` replaces them by the re-runs in
results/reruns/ (same checkpoints, one session or batch size 1), the files the
paper's 30 of 30 is computed on. Without it, on the published predictions, the
identity holds in 27 of 30 cells.

Usage:
    python -m src.analysis.identity --dataset KVASIR --condition C5
    python -m src.analysis.identity --all --json identity.json
    python -m src.analysis.identity --all --pair-by id
    python -m src.analysis.identity --all --reruns
"""
from __future__ import annotations

import argparse
import json
import os

from src.analysis.ccr import (answers_match, load_predictions, load_swapmap,
                              question_key)
from src.analysis.ccr_slices import seed_prefixes

# The cells the paper reports: the swap with the largest informative set of
# each dataset that has one.
REPORTED = (("KVASIR", "C5"), ("SLAKE", "C6"))

CONFIGURATIONS = {
    "KVASIR": ["M3AE_KVASIR", "QWEN2VL_LORA_KVASIR", "QWEN2VL_DORA_KVASIR",
               "SMOLVLM_LORA_KVASIR", "SMOLVLM_DORA_KVASIR"],
    "SLAKE": ["M3AE_SCRATCH_SLAKE", "QWEN2VL_LORA_SLAKE", "QWEN2VL_DORA_SLAKE",
              "SMOLVLM_LORA_SLAKE", "SMOLVLM_DORA_SLAKE"],
    "VQARAD": ["M3AE_SCRATCH_VQARAD", "QWEN2VL_LORA_VQARAD",
               "QWEN2VL_DORA_VQARAD", "SMOLVLM_LORA_VQARAD",
               "SMOLVLM_DORA_VQARAD"],
}


# prefix -> {condition: file in results/reruns/} for the three cells of Section 5.6
RERUNS = {
    "M3AE_SCRATCH_SLAKE": {"C0": "M3AE_SCRATCH_SLAKE_C0_sameenv.json",
                           "C6": "M3AE_SCRATCH_SLAKE_C6_sameenv.json"},
    "SMOLVLM_LORA_KVASIR_seed2": {"C0": "SMOLVLM_LORA_KVASIR_seed2_C0_bs1.json",
                                  "C5": "SMOLVLM_LORA_KVASIR_seed2_C5_bs1.json"},
    "SMOLVLM_DORA_KVASIR_seed0": {"C0": "SMOLVLM_DORA_KVASIR_seed0_C0_bs1.json",
                                  "C5": "SMOLVLM_DORA_KVASIR_seed0_C5_bs1.json"},
}


def load_with_reruns(results_dir: str, prefix: str, condition: str, reruns_dir: str | None):
    """The prediction file of one run, replaced by its re-run when asked and available."""
    if reruns_dir and condition in RERUNS.get(prefix, {}):
        path = os.path.join(reruns_dir, RERUNS[prefix][condition])
        if os.path.exists(path):
            with open(path) as fp:
                return {p["qid"]: p for p in json.load(fp)["predictions"]}
    return load_predictions(results_dir, prefix, condition)


def strata(qid_meta: dict, donor_map: dict):
    """Partition the questions the swap touches, and name the donor item.

    Returns the partition plus, for every question whose donor carries it, the
    qid of that donor item -- which is what makes the identity checkable: the
    donor item is a test item with a prediction of its own under C0.
    """
    by_image, index = {}, {}
    for qid, meta in qid_meta.items():
        by_image.setdefault(meta["img"], {})[question_key(meta["q"])] = meta["a"]
        index[(meta["img"], question_key(meta["q"]))] = qid

    parts = {"informative": [], "same_ref": [], "novel": []}
    donor_qid, donor_ref = {}, {}
    for qid, meta in qid_meta.items():
        donor = donor_map.get(meta["img"])
        if donor is None:
            continue
        key = question_key(meta["q"])
        reference = by_image.get(donor, {}).get(key)
        if reference is None:
            parts["novel"].append(qid)
            continue
        donor_qid[qid] = index[(donor, key)]
        donor_ref[qid] = reference
        bucket = ("same_ref" if any(answers_match(a, reference) for a in meta["a"])
                  else "informative")
        parts[bucket].append(qid)
    return parts, donor_qid, donor_ref


def sha_twins(qid_meta: dict, donor_map: dict, manifest_path: str):
    """For every swapped question, the C0 items that carry the SAME PICTURE (by
    SHA-256 of the image bytes) and the same question as the donor item.

    The manifest gives the SHA-256 of the original and of the received bytes of
    every swapped row; every test image appears in it either as an original or
    as a donor, so the image -> sha map covers the whole test set.
    """
    with open(manifest_path) as fp:
        rows = json.load(fp)["rows"]
    sha = {}
    for r in rows:
        sha[r["image_id_original"]] = r["sha256_original"]
        sha[r["image_id_received"]] = r["sha256_received"]
    by_sha_q = {}
    for qid, meta in qid_meta.items():
        s = sha.get(meta["img"])
        if s is not None:
            by_sha_q.setdefault((s, question_key(meta["q"])), []).append(qid)
    twins = {}
    for qid, meta in qid_meta.items():
        donor = donor_map.get(meta["img"])
        if donor is None or donor not in sha:
            continue
        twins[qid] = by_sha_q.get((sha[donor], question_key(meta["q"])), [])
    return twins


def check_cell(c0: dict, swapped: dict, informative: list[int],
               donor_qid: dict, donor_ref: dict, twins: dict | None = None) -> dict:
    """The three quantities the identity relates, over the same items.

    agreement  how often the swapped prediction equals the C0 prediction on the
               donor item -- the identity itself, at the level of the answer
    ccr        the swapped answer scored against the DONOR's reference
    donor_acc  the accuracy the model already has under C0 on those donor items

    The identity says ccr and donor_acc are the same number.

    With `twins` (pairing by image bytes) the donor item is the set of C0 items
    that carry the same picture and question; the swapped prediction agrees when
    it equals the C0 prediction of any of them, and donor_acc reads the
    correctness of the twin it agrees with (of the first twin when it agrees
    with none). Twins whose C0 predictions differ among themselves are counted
    in `ambiguous_twins`.
    """
    if twins is None:
        usable = [q for q in informative if q in swapped and donor_qid[q] in c0]
        if not usable:
            return {"n": 0}
        agree = sum(1 for q in usable
                    if answers_match(swapped[q]["predicted"],
                                     [c0[donor_qid[q]]["predicted"]]))
        ccr = sum(1 for q in usable
                  if answers_match(swapped[q]["predicted"], donor_ref[q]))
        donor_acc = sum(1 for q in usable if c0[donor_qid[q]]["correct"])
        n = len(usable)
        return {"n": n, "agreement": agree / n, "ccr": ccr / n,
                "donor_accuracy": donor_acc / n,
                "difference_pp": (ccr - donor_acc) / n * 100,
                "exact": agree == n and ccr == donor_acc}
    usable = [q for q in informative if q in swapped
              and any(t in c0 for t in twins.get(q, []))]
    if not usable:
        return {"n": 0}
    agree = ccr = donor_acc = ambiguous = 0
    for q in usable:
        ts = [t for t in twins[q] if t in c0]
        preds = {str(c0[t]["predicted"]) for t in ts}
        ambiguous += len(preds) > 1
        hit = [t for t in ts if answers_match(swapped[q]["predicted"],
                                              [c0[t]["predicted"]])]
        agree += bool(hit)
        ccr += answers_match(swapped[q]["predicted"], donor_ref[q])
        # the twin's C0 prediction scored against the reference of the donor
        # item the swap named (a copy under another identifier may carry its
        # own reference; the identity is about the same picture and question)
        donor_acc += answers_match(c0[(hit or ts)[0]]["predicted"], donor_ref[q])
    n = len(usable)
    return {"n": n, "agreement": agree / n, "ccr": ccr / n,
            "donor_accuracy": donor_acc / n,
            "difference_pp": (ccr - donor_acc) / n * 100,
            "ambiguous_twins": ambiguous, "exact": agree == n and ccr == donor_acc}


def run(dataset: str, condition: str, results_dir: str, swapmap_path: str,
        pair_by: str = "sha", manifests_dir: str | None = None,
        reruns_dir: str | None = None) -> dict:
    entry = load_swapmap(swapmap_path)[dataset]
    qid_meta = {int(k): v for k, v in entry["qid"].items()}
    if condition not in entry["swap"]:
        raise SystemExit(f"{dataset} has no donor map for {condition}")
    parts, donor_qid, donor_ref = strata(qid_meta, entry["swap"][condition])
    twins = None
    if pair_by == "sha":
        manifest = os.path.join(manifests_dir or os.path.join(results_dir, "manifests"),
                                f"{dataset}_{condition}_rows.json")
        twins = sha_twins(qid_meta, entry["swap"][condition], manifest)

    out = {"dataset": dataset, "condition": condition, "pair_by": pair_by,
           "reruns": bool(reruns_dir),
           "informative": len(parts["informative"]),
           "same_ref": len(parts["same_ref"]), "novel": len(parts["novel"]),
           "cells": {}}
    for base in CONFIGURATIONS[dataset]:
        for prefix in seed_prefixes(results_dir, base, condition):
            c0 = load_with_reruns(results_dir, prefix, "C0", reruns_dir)
            swapped = load_with_reruns(results_dir, prefix, condition, reruns_dir)
            if c0 is None or swapped is None:
                continue
            out["cells"][prefix] = check_cell(
                c0, swapped, parts["informative"], donor_qid, donor_ref, twins)
    return out


def summarize(records: list[dict]) -> dict:
    cells = [(name, c) for r in records for name, c in r["cells"].items() if c["n"]]
    exact = [n for n, c in cells if c["exact"]]
    return {"cells": len(cells), "exact": len(exact),
            "exceptions": sorted(
                ({"cell": n, "agreement": c["agreement"],
                  "difference_pp": c["difference_pp"]}
                 for n, c in cells if n not in set(exact)),
                key=lambda d: d["agreement"])}


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--swapmap", default="data/swapmap.json")
    p.add_argument("--dataset")
    p.add_argument("--condition")
    p.add_argument("--all", action="store_true", help="the cells the paper reports")
    p.add_argument("--pair-by", choices=("sha", "id"), default="sha",
                   help="name the donor item by image bytes (sha, the paper) or by image identifier (id)")
    p.add_argument("--manifests-dir", default=None,
                   help="per-row manifests with the SHA-256 (default: <results-dir>/manifests)")
    p.add_argument("--reruns", nargs="?", const="results/reruns", default=None,
                   help="use the re-runs of the three cells of Section 5.6 (default dir: results/reruns)")
    p.add_argument("--json")
    return p.parse_args()


def main():
    args = _parse_args()
    pairs = REPORTED if args.all or not args.dataset else [(args.dataset,
                                                            args.condition)]
    records = []
    for dataset, condition in pairs:
        r = run(dataset, condition, args.results_dir, args.swapmap,
                args.pair_by, args.manifests_dir, args.reruns)
        records.append(r)
        print(f"\n=== {dataset} {condition}  paired by {args.pair_by}  "
              f"informative {r['informative']}, same reference {r['same_ref']}, "
              f"novel pairing {r['novel']}")
        print(f"{'cell':<30}{'n':>7}{'agreement':>12}{'CCR':>9}"
              f"{'donor Acc(C0)':>15}{'diff (pp)':>11}")
        for name, c in r["cells"].items():
            if not c["n"]:
                continue
            print(f"{name:<30}{c['n']:>7}{c['agreement'] * 100:>11.2f}%"
                  f"{c['ccr'] * 100:>8.1f}%{c['donor_accuracy'] * 100:>14.1f}%"
                  f"{c['difference_pp']:>11.3f}")

    s = summarize(records)
    print(f"\nthe identity holds exactly in {s['exact']} of {s['cells']} "
          f"configuration-seed cells")
    for e in s["exceptions"]:
        print(f"  exception: {e['cell']}  agreement {e['agreement'] * 100:.2f}%"
              f"  difference {e['difference_pp']:.2f} pp")

    if args.json:
        with open(args.json, "w") as fp:
            json.dump({"records": records, "summary": s}, fp, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
