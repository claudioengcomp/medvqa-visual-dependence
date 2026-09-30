"""Coverage analysis of Section 5.3: where phi is estimable, and at what scale.

phi and |V| are properties of the TEST SET and of the SWAP, not of a model, so
this module runs no model and needs no GPU: it reads questions and reference
answers only.

    V_T   swapped questions whose donor image carries the SAME question
    I_T   the subset of V_T whose reference answer differs
    phi   |I_T| / |V_T|

For each benchmark the module draws `--draws` derangements of the test items
with a fixed master seed, and reports phi only when |V| clears the floor in
more than half of the draws (Section 3.4). It repeats the whole computation on
random subsets of 100 items, the scale at which such probes are commonly run.

A single scoring rule. `question_key` and `answers_match` are imported from
src/analysis/ccr.py, the same pair that decides accuracy, VDD and the CCR. No
normalization is defined here, and no semantic similarity is used anywhere: on
a small answer vocabulary it degrades the matching.

Benchmarks. SLAKE, VQA-RAD and Kvasir-VQA-x1 are read from the swap map that
ships with this repository. PathVQA, PMC-VQA and ProbMed are read from files
the user downloads first (see `--data-dir` and docs/COVERAGE.md); they are not
redistributed here. ProbMed is a gated release and needs credentialing from its
maintainers.

Usage:
    python -m src.analysis.coverage --step1          # question duplication only
    python -m src.analysis.coverage --all --json coverage.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os

import numpy as np

from src.analysis.ccr import answers_match, question_key

MASTER_SEED = 20260910
DRAWS = 1000
FLOOR_V = 20
SUBSAMPLE = 100

# Benchmarks that ship with this repository, read from the swap map.
IN_REPO = {"Kvasir-VQA-x1": "KVASIR", "SLAKE": "SLAKE", "VQA-RAD": "VQARAD"}

PATHVQA_PARQUETS = (
    "path-vqa__test-00000-of-00003-e9adadb4799f44d3.parquet",
    "path-vqa__test-00001-of-00003-7ea98873fc919813.parquet",
    "path-vqa__test-00002-of-00003-1628308435019820.parquet",
)
PROBMED_CATEGORY = {"modality": "Modality", "body_part": "Organ",
                    "abnormality": "Abnormality", "entity": "Condition",
                    "grounding": "Position"}


class MissingData(SystemExit):
    """Raised, not worked around, when an input the analysis needs is absent."""


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def from_swapmap(dataset: str, swapmap_path: str) -> list[dict]:
    """The test split of a benchmark that ships with the repository."""
    if not os.path.exists(swapmap_path):
        raise MissingData(f"swap map not found: {swapmap_path}")
    with open(swapmap_path) as fp:
        entry = json.load(fp)[dataset]
    items = []
    for qid, meta in entry["qid"].items():
        t = str(meta.get("t", "")).upper()
        kind = {"CLOSED": "closed", "OPEN": "open"}.get(t)
        items.append({"iid": str(qid), "img": str(meta["img"]),
                      "q": str(meta["q"]),
                      "refs": [str(a) for a in meta["a"]],
                      "stratum": kind, "substratum": None})
    return items


def path_vqa(data_dir: str) -> list[dict]:
    """PathVQA test split, from the parquet files of the released test set.

    The images carry no usable file name in the parquet, so image identity is
    the sha1 of the image bytes -- the same hash-matching criterion that
    build_swapmap.py uses. Two byte-identical images under different names
    would count as one, which can only RAISE |V| and so never strengthens a
    verdict that coverage collapses.

    PathVQA has no type field. The benchmark's own convention is that a
    question is closed when its gold answer is yes/no, and that is the rule
    applied here.
    """
    import pyarrow.parquet as pq

    missing = [f for f in PATHVQA_PARQUETS
               if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        raise MissingData(f"PathVQA parquet files not found in {data_dir}: {missing}")

    items, n = [], 0
    for name in PATHVQA_PARQUETS:
        table = pq.read_table(os.path.join(data_dir, name),
                              columns=["image", "question", "answer"])
        image = table.column("image").combine_chunks()
        for raw, q, a in zip(image.field("bytes").to_pylist(),
                             table.column("question").to_pylist(),
                             table.column("answer").to_pylist()):
            digest = hashlib.sha1(raw).hexdigest() if raw else f"NOBYTES_{n}"
            kind = "closed" if str(a).strip().lower() in ("yes", "no") else "open"
            items.append({"iid": f"pvqa_{n}", "img": digest, "q": str(q),
                          "refs": [str(a)], "stratum": kind, "substratum": None})
            n += 1
    return items


def pmc_vqa(data_dir: str, filename: str) -> list[dict]:
    """PMC-VQA, multiple choice.

    The reference is NOT the option letter: two items with the same question
    may carry different option sets, so comparing labels would be wrong. The
    `Answer` column holds the CONTENT of the correct option and is what is
    used; consistency against `Choice <Answer_label>` is checked and any
    divergence is counted and reported rather than silently resolved.
    """
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise MissingData(f"PMC-VQA file not found: {path}")

    items, empty, divergent = [], 0, 0
    with open(path, newline="", encoding="utf-8") as fp:
        for n, row in enumerate(csv.DictReader(fp)):
            content = str(row.get("Answer") or "").strip()
            if not content:
                empty += 1
                continue
            label = str(row.get("Answer_label") or "").strip()
            letter = label[0].upper() if label else ""
            raw = row.get(f"Choice {letter}")
            if raw is not None:
                option = str(raw).split(":", 1)[-1].strip()
                if option and not answers_match(content, [option]):
                    divergent += 1
            items.append({"iid": f"pmc_{n}", "img": str(row["Figure_path"]),
                          "q": str(row["Question"]), "refs": [content],
                          "stratum": None, "substratum": None})
    if empty or divergent:
        print(f"  note PMC-VQA ({filename}): {empty} rows without an answer "
              f"(dropped); {divergent} where Answer disagrees with "
              f"Choice[Answer_label] (kept, content prevails)")
    return items


def probmed(data_dir: str, filename: str = "ProbMed__test.json") -> list[dict]:
    """ProbMed test split. Gated release: the file must be obtained first.

    `qa_type` carries both strata at once: the diagnostic category and whether
    the question is a ground-truth or an adversarial (hallucinated entity) one.
    The numeric suffixes (entity_gt_3) index the n-th entity of the same image
    and are not a stratum; they collapse into the category.
    """
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise MissingData(
            f"ProbMed file not found: {path}\n"
            "ProbMed is a gated release: request access from its maintainers "
            "and place test/test.json here. It is not redistributed with this "
            "repository.")
    with open(path) as fp:
        raw = json.load(fp)

    items, unmapped = [], 0
    for n, row in enumerate(raw):
        qa = str(row.get("qa_type", ""))
        root = qa.split("_gt")[0].split("_hallu")[0]
        category = PROBMED_CATEGORY.get(root)
        if category is None:
            unmapped += 1
            continue
        items.append({"iid": f"probmed_{n}", "img": str(row["image"]),
                      "q": str(row["question"]), "refs": [str(row["answer"])],
                      "stratum": category,
                      "substratum": "adversarial" if "_hallu" in qa else "ground truth"})
    if unmapped:
        print(f"  note ProbMed: {unmapped} rows with an unmapped qa_type")
    return items


def benchmarks(swapmap_path: str, data_dir: str) -> dict:
    """Name -> loader. External benchmarks fail loudly when their file is absent."""
    out = {name: (lambda d=ds: from_swapmap(d, swapmap_path))
           for name, ds in IN_REPO.items()}
    out["PathVQA"] = lambda: path_vqa(data_dir)
    out["PMC-VQA (test clean)"] = lambda: pmc_vqa(data_dir, "PMC-VQA__test_clean.csv")
    out["PMC-VQA (test)"] = lambda: pmc_vqa(data_dir, "PMC-VQA__test.csv")
    out["ProbMed"] = lambda: probmed(data_dir)
    return out


# --------------------------------------------------------------------------
# Step 1: question duplication -- deterministic, and it bounds E[|V|]
# --------------------------------------------------------------------------
def duplication(name: str, items: list[dict]) -> dict:
    by_question = collections.defaultdict(set)
    for it in items:
        by_question[question_key(it["q"])].add(it["img"])
    images_per_question = {q: len(v) for q, v in by_question.items()}
    duplicated = sum(1 for it in items
                     if images_per_question[question_key(it["q"])] >= 2)
    return {"benchmark": name, "test_questions": len(items),
            "images": len({it["img"] for it in items}),
            "distinct_questions": len(by_question),
            "questions_on_2plus_images": sum(1 for v in images_per_question.values()
                                             if v >= 2),
            "items_with_duplicated_question": duplicated,
            "fraction_duplicated": duplicated / len(items) if items else 0.0,
            "max_images_per_question": max(images_per_question.values(), default=0),
            "by_stratum": dict(collections.Counter(
                it["stratum"] for it in items if it["stratum"])),
            "by_substratum": dict(collections.Counter(
                it["substratum"] for it in items if it["substratum"]))}


# --------------------------------------------------------------------------
# Step 4: does the scoring rule merge two distinct references?
# --------------------------------------------------------------------------
def reference_collisions(items: list[dict]) -> dict:
    """Distinct reference strings, within one question template, merged by the rule.

    Denominator: the distinct reference strings that live in templates carrying
    more than one, counted over the WHOLE test set. Note that the docstring of
    ccr.py reports a different denominator for the same zero numerator, because
    it counts only the templates that enter the informative computation.
    """
    by_template = collections.defaultdict(set)
    for it in items:
        for r in it["refs"]:
            by_template[question_key(it["q"])].add(str(r))
    tested = collided = 0
    for refs in by_template.values():
        refs = sorted(refs)
        if len(refs) < 2:
            continue
        tested += len(refs)
        for i, a in enumerate(refs):
            if any(answers_match(a, [b]) for j, b in enumerate(refs) if j != i):
                collided += 1
    return {"references_tested": tested, "collisions": collided}


# --------------------------------------------------------------------------
# Step 3: phi over random derangements
# --------------------------------------------------------------------------
def precompute(items: list[dict]) -> dict:
    """Resolve the scoring rule ONCE, before any draw.

    Whether a donor makes item i verifiable does not depend on i, only on its
    QUESTION: i is verifiable iff the pair (donor, q_i) exists in the test set.
    Whether it makes i informative depends on i only through its REFERENCE
    CLASS within that question.

    Keying by (question) and by (question, reference class) instead of by item
    keeps both sets the size of the (image, question) index -- tens of
    thousands -- instead of the item x candidate-donor product, which on a
    benchmark with dense questions exceeds 10^8.
    """
    images = sorted({it["img"] for it in items})
    image_id = {v: k for k, v in enumerate(images)}
    n = len(images)

    reference = {}
    for it in items:
        reference.setdefault((it["img"], question_key(it["q"])), it["refs"])

    q_id, class_id, q_of_item, g_of_item = {}, {}, [], []
    for it in items:
        q = question_key(it["q"])
        cls = tuple(sorted(str(r) for r in it["refs"]))
        q_of_item.append(q_id.setdefault(q, len(q_id)))
        g_of_item.append(class_id.setdefault((q, cls), len(class_id)))

    by_question = collections.defaultdict(set)
    for (img, q) in reference:
        by_question[q].add(image_id[img])

    keys_v = [q_id[q] * n + d for q, ds in by_question.items() for d in ds]
    keys_i = []
    for (q, cls), gid in class_id.items():
        base = gid * n
        for d in by_question[q]:
            if not any(answers_match(x, reference[(images[d], q)]) for x in cls):
                keys_i.append(base + d)

    return {"n_img": n,
            "image_of_item": np.array([image_id[it["img"]] for it in items], np.int64),
            "q_of_item": np.array(q_of_item, np.int64),
            "g_of_item": np.array(g_of_item, np.int64),
            "stratum": np.array([it["stratum"] or "" for it in items]),
            "V": np.array(sorted(keys_v), np.int64),
            "I": np.array(sorted(keys_i), np.int64)}


def derangement_by_image(images: np.ndarray, rng, attempts: int = 200):
    """A permutation sigma with image[sigma(i)] != image[i] for every i.

    Random permutation followed by repair: conflicting positions are swapped
    with randomly drawn ones until none is left. Returns None when it does not
    converge, which happens when almost every item shares one image; such a
    draw is discarded rather than patched.
    """
    n = len(images)
    if np.unique(images).size < 2:
        return None
    sigma = rng.permutation(n)
    for _ in range(attempts):
        bad = np.flatnonzero(images[sigma] == images)
        if bad.size == 0:
            return sigma
        target = rng.integers(0, n, bad.size)
        a, b = sigma[bad].copy(), sigma[target].copy()
        sigma[bad], sigma[target] = b, a
    return None


def scale(pre: dict, rng, subsample: int | None = None,
          stratum: str | None = None, draws: int = DRAWS,
          floor: int = FLOOR_V) -> dict:
    """`draws` derangements; subsample=None uses the full test set.

    A stratum restricts WHO is counted, never who may be a donor: the swap
    happens over the whole set, as the real intervention does.
    """
    all_images, n_img = pre["image_of_item"], pre["n_img"]
    n_items = all_images.size
    every = np.arange(n_items)
    counted = (np.ones(n_items, bool) if stratum is None
               else pre["stratum"] == stratum)

    rows, discarded = [], 0
    for _ in range(draws):
        sel = (every if subsample is None
               else rng.choice(n_items, size=min(subsample, n_items), replace=False))
        images = all_images[sel]
        sigma = derangement_by_image(images, rng)
        if sigma is None:
            discarded += 1
            continue
        target = counted[sel]
        if not target.any():
            continue
        donors = images[sigma][target]
        v = int(np.isin(pre["q_of_item"][sel[target]] * n_img + donors, pre["V"]).sum())
        i = int(np.isin(pre["g_of_item"][sel[target]] * n_img + donors, pre["I"]).sum())
        rows.append((int(target.sum()), v, i))

    if not rows:
        return {"valid_draws": 0, "discarded": discarded}
    vs = np.array([r[1] for r in rows], float)
    is_ = np.array([r[2] for r in rows], float)
    above = vs >= floor
    out = {"valid_draws": len(rows), "discarded": discarded,
           "S_mean": float(np.mean([r[0] for r in rows])),
           "V_mean": float(vs.mean()), "V_median": float(np.median(vs)),
           "V_min": float(vs.min()), "V_max": float(vs.max()),
           "I_mean": float(is_.mean()),
           "fraction_below_floor": float((~above).mean()),
           "reportable_draws": int(above.sum())}
    if above.any():
        phi = is_[above] / vs[above]
        out.update({"phi_mean": float(phi.mean()),
                    "phi_ci95": [float(np.percentile(phi, 2.5)),
                                 float(np.percentile(phi, 97.5))]})
    else:
        out.update({"phi_mean": None, "phi_ci95": None})
    # phi is REPORTED only when most draws clear the floor. A cell where 3 of
    # 1,000 draws clear it has a phi computed on those 3, and displaying it
    # would contradict the floor rule itself. The value stays in the JSON and
    # the counter says how many draws it rests on: nothing is hidden, it just
    # stops being reported.
    out["phi_reportable"] = bool(above.mean() > 0.5)
    return out


# --------------------------------------------------------------------------
# Step 2: the blocking validation against published numbers
# --------------------------------------------------------------------------
def validate(swapmap_path: str) -> dict:
    """Reproduce SLAKE C2 (phi = 56.0% over 125 verifiable), VQA-RAD (|I| < 4) and
    Kvasir-VQA-x1 C5 (|V| = 8,102, |I| = 5,216, phi = 0.6438; Table 4 and Eq. 2).

    Uses the RECORDED swap map, not a draw: those numbers came from that map.
    Draws enter only in Step 3.
    """
    with open(swapmap_path) as fp:
        swapmap = json.load(fp)
    out = {}
    for name, ds, cond in (("SLAKE", "SLAKE", "C2"), ("VQA-RAD", "VQARAD", "C6"),
                           ("VQA-RAD", "VQARAD", "C2"), ("Kvasir-VQA-x1", "KVASIR", "C5"),
                           ("Kvasir-VQA-x1", "KVASIR", "C2")):
        entry = swapmap[ds]
        qid_meta = {int(k): v for k, v in entry["qid"].items()}
        donors = entry["swap"][cond]
        reference = {}
        for meta in qid_meta.values():
            reference.setdefault((meta["img"], question_key(meta["q"])), meta["a"])
        s = v = i = 0
        for meta in qid_meta.values():
            d = donors.get(meta["img"])
            if d is None:
                continue
            s += 1
            target = reference.get((d, question_key(meta["q"])))
            if target is None:
                continue
            v += 1
            if not any(answers_match(x, target) for x in meta["a"]):
                i += 1
        out[f"{name} {cond}"] = {"S": s, "V": v, "I": i,
                                 "phi": i / v if v else None}
    return out


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--swapmap", default="data/swapmap.json")
    p.add_argument("--data-dir", default="data/coverage",
                   help="where the downloaded PathVQA / PMC-VQA / ProbMed files are")
    p.add_argument("--draws", type=int, default=DRAWS)
    p.add_argument("--floor", type=int, default=FLOOR_V)
    p.add_argument("--step1", action="store_true", help="duplication table only")
    p.add_argument("--all", action="store_true")
    p.add_argument("--json", help="write the full record here")
    p.add_argument("--skip-missing", action="store_true",
                   help="skip benchmarks whose files are absent instead of stopping")
    return p.parse_args()


def main():
    args = _parse_args()
    print(f"master seed {MASTER_SEED} | {args.draws} draws | floor |V| >= "
          f"{args.floor} | subsample n = {SUBSAMPLE}\n")

    data, skipped = {}, []
    for name, load in benchmarks(args.swapmap, args.data_dir).items():
        try:
            data[name] = load()
        except MissingData as exc:
            if not args.skip_missing:
                raise
            skipped.append(name)
            print(f"  SKIPPED {name}: {str(exc).splitlines()[0]}")
    if skipped:
        print()

    print("=" * 108)
    print("STEP 1 -- question duplication")
    print("=" * 108)
    head = ("benchmark", "test Qs", "images", "distinct Qs", "Qs on 2+ imgs",
            "dup. items", "% dup.", "max imgs/Q")
    print(f"{head[0]:<22}" + "".join(f"{h:>14}" for h in head[1:]))
    step1 = {}
    for name, items in data.items():
        r = duplication(name, items)
        step1[name] = r
        print(f"{name:<22}{r['test_questions']:>14,}{r['images']:>14,}"
              f"{r['distinct_questions']:>14,}{r['questions_on_2plus_images']:>14,}"
              f"{r['items_with_duplicated_question']:>14,}"
              f"{r['fraction_duplicated'] * 100:>13.1f}%"
              f"{r['max_images_per_question']:>14,}")
        if r["by_substratum"]:
            print(f"{'':<22}  strata: " +
                  ", ".join(f"{k} {v:,}" for k, v in sorted(r["by_stratum"].items()))
                  + "  |  " + ", ".join(f"{k} {v:,}"
                                        for k, v in sorted(r["by_substratum"].items())))

    print("\n" + "=" * 108)
    print("STEP 2 -- validation against published numbers (blocking)")
    print("=" * 108)
    step2 = validate(args.swapmap)
    for k, v in step2.items():
        phi = "n/a" if v["phi"] is None else f"{v['phi'] * 100:.1f}%"
        print(f"  {k:<14} |S|={v['S']:>6}  |V|={v['V']:>5}  |I|={v['I']:>5}  phi={phi}")
    ok_slake = (step2["SLAKE C2"]["V"] == 125
                and abs((step2["SLAKE C2"]["phi"] or 0) - 0.56) < 0.005)
    ok_rad = step2["VQA-RAD C6"]["I"] < 4
    print(f"\n  target SLAKE C2: |V| = 125 and phi ~ 56%   -> "
          f"{'REPRODUCED' if ok_slake else 'NOT REPRODUCED'}")
    print(f"  target VQA-RAD: |I| < 4, phi not estimable -> "
          f"{'REPRODUCED' if ok_rad else 'NOT REPRODUCED'}")
    if not (ok_slake and ok_rad):
        raise SystemExit("STOP: Step 2 did not reproduce. The fault is in the "
                         "pipeline, not in the data.")

    print("\n" + "=" * 108)
    print("STEP 4 -- reference collisions within a question template")
    print("=" * 108)
    step4 = {}
    for name, items in data.items():
        c = reference_collisions(items)
        step4[name] = c
        print(f"  {name:<22} {c['collisions']} collisions in "
              f"{c['references_tested']} references tested")

    if args.step1 and not args.all:
        if args.json:
            with open(args.json, "w") as fp:
                json.dump({"step1": step1, "step2": step2, "step4": step4}, fp, indent=1)
        return

    print("\n" + "=" * 108)
    print("STEP 3 -- phi per draw (derangement by image)")
    print("=" * 108)
    head = ("benchmark", "scale", "|D|", "|V| mean", "|I| mean", "phi",
            "95% CI", "< floor", "report./n")
    print(f"{head[0]:<22}{head[1]:<22}{head[2]:>10}{head[3]:>11}{head[4]:>11}"
          f"{head[5]:>9}{head[6]:>18}{head[7]:>9}{head[8]:>12}")
    step3 = {}
    for name, items in data.items():
        pre = precompute(items)
        strata = sorted({it["stratum"] for it in items if it["stratum"]})
        for stratum, suffix in [(None, "")] + [(s, " " + s.lower()) for s in strata]:
            for label, sub in (("(a) full test set", None),
                               (f"(b) n={SUBSAMPLE}", SUBSAMPLE)):
                rng = np.random.default_rng(MASTER_SEED)
                r = scale(pre, rng, sub, stratum, args.draws, args.floor)
                step3[f"{name}{suffix} | {label}"] = r
                if not r.get("valid_draws"):
                    print(f"{name + suffix:<22}{label:<22}  NO VALID DRAW")
                    continue
                ok = r["phi_reportable"]
                phi = f"{r['phi_mean'] * 100:.1f}%" if ok else "--"
                ci = (f"[{r['phi_ci95'][0] * 100:.1f}, {r['phi_ci95'][1] * 100:.1f}]"
                      if ok else "--")
                print(f"{name + suffix:<22}{label:<22}{r['S_mean']:>10,.0f}"
                      f"{r['V_mean']:>11.1f}{r['I_mean']:>11.1f}{phi:>9}{ci:>18}"
                      f"{r['fraction_below_floor'] * 100:>8.0f}%"
                      f"{r['reportable_draws']:>12,}")

    if args.json:
        with open(args.json, "w") as fp:
            json.dump({"master_seed": MASTER_SEED, "draws": args.draws,
                       "floor_V": args.floor, "subsample": SUBSAMPLE,
                       "skipped": skipped, "step1": step1, "step2": step2,
                       "step3": step3, "step4": step4}, fp, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
