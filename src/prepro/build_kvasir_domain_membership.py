"""Extract the Kvasir domain field (procedure) from the already built
evaluation arrow -- analogous to SLAKE's `location`, except that Kvasir
does not ship that field in the raw data. The question "What type of
procedure is the image taken from?" answers colonoscopy/gastroscopy for
776 of the 1,291 images of the image-disjoint test split (60%; 1,756 of 4,058, 43%, in the original split) -- cross-validated against the
existing pathology_label (polyp/ulcerative colitis should be colonoscopy;
oesophagitis/barretts should be gastroscopy): of the 682 images carrying
both signals, only 2 were inconsistent (99.7% agreement).

Used by C6 (src/interventions/make_c6_organ_modality_swap.py) as
`--bucket-membership`, one bucket per procedure -- the same role that
build_slake_pathology_labels.build_slake_bucket_membership plays for SLAKE.

Usage:
    python -m src.prepro.build_kvasir_domain_membership \
        --input data/finetune_arrows/vqa_kvasir_test.arrow \
        --output-colonoscopy results/kvasir_clean_bucket_colonoscopy.json \
        --output-gastroscopy results/kvasir_clean_bucket_gastroscopy.json
"""
from __future__ import annotations

import argparse
import json

_PROCEDURE_QUESTION = "what type of procedure is the image taken from?"


def _first_answer(answer) -> str:
    """Normalize an answer value (which may arrive as a string, a list, or
    a pandas/arrow numpy array) down to its first item, as a string."""
    if hasattr(answer, "__len__") and not isinstance(answer, str):
        answer = answer[0]
    return str(answer).strip().lower()


def extract_domain_membership(rows: list[dict]) -> dict[str, str]:
    """rows: list of {"image_id":..., "questions": [...], "answers": [...]}
    (one row per image, in the evaluation arrow format). Returns
    {image_id: procedure} only for the images that actually have this
    question answered -- most do not (56.7% of the test split lack it, see
    the module docstring)."""
    membership: dict[str, str] = {}
    for row in rows:
        for q, a in zip(row["questions"], row["answers"]):
            if q.strip().lower() == _PROCEDURE_QUESTION:
                membership[row["image_id"]] = _first_answer(a)
                break
    return membership


def split_by_procedure(membership: dict[str, str]) -> tuple[list[str], list[str]]:
    """Split {image_id: procedure} into two lists (colonoscopy,
    gastroscopy) -- values outside that pair (never observed in practice,
    see the module docstring) are ignored rather than raising an error."""
    colonoscopy = [img for img, proc in membership.items() if proc == "colonoscopy"]
    gastroscopy = [img for img, proc in membership.items() if proc == "gastroscopy"]
    return colonoscopy, gastroscopy


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-colonoscopy", required=True)
    parser.add_argument("--output-gastroscopy", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    from src.interventions.arrow_io import read_arrow

    args = _parse_args()
    df = read_arrow(args.input)
    rows = [{"image_id": r["image_id"], "questions": r["questions"], "answers": r["answers"]}
            for _, r in df.iterrows()]

    membership = extract_domain_membership(rows)
    colonoscopy, gastroscopy = split_by_procedure(membership)

    with open(args.output_colonoscopy, "w", encoding="utf-8") as fp:
        json.dump(colonoscopy, fp, indent=2, ensure_ascii=False)
    with open(args.output_gastroscopy, "w", encoding="utf-8") as fp:
        json.dump(gastroscopy, fp, indent=2, ensure_ascii=False)

    print(f"{len(colonoscopy)} colonoscopy images -> {args.output_colonoscopy}")
    print(f"{len(gastroscopy)} gastroscopy images -> {args.output_gastroscopy}")
    print(f"{len(membership)} images with an identified procedure, out of {len(rows)} in the arrow "
          f"({100 * len(membership) / len(rows):.1f}%)")
