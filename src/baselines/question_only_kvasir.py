"""M0: the question-only floor for Kvasir-VQA-x1 (Table 1).

The question-only baseline of `question_only.py` reads the JSON format of
SLAKE and VQA-RAD and accepts only those two datasets. Kvasir-VQA-x1 is
distributed as an Arrow file and carries no open/closed label, so the floor for
its five configurations lived outside the repository until this module. Three
sentences of the paper -- that every fine-tuned model falls to or below a
question-only baseline -- depend on the number it produces, so it belongs here.

Same learner as the other two datasets: TF-IDF over the question text plus
logistic regression, reusing `train_question_only_model`. The model never sees
the image, so its accuracy is identical under C0/C1/C2/C4 (x6)/C5/C6 and its
VDD is zero by construction. That invariance is the point of a floor, and
`write_conditions` writes the same predictions under all eleven condition names
so that every downstream reader finds the row where it expects it.

THREE DECISIONS THAT CHANGE THE NUMBER, fixed here rather than left implicit:

1. SOURCE. Training reads `vqa_kvasir_train_clean.arrow`, the same file the
   five fine-tuned models were trained on (38,211 questions over 4,514 images).
   The validation split does not enter. The test items come from the swapmap,
   the same source `build_table1.py` reads, so the qids line up with every
   other prediction file.

2. SCOPE. On SLAKE and VQA-RAD the baseline runs on closed questions only (416
   and 272), because those datasets mix open and closed and exact match is
   meaningful only on the closed ones. Kvasir-VQA-x1 carries no such label and
   is closed-vocabulary by construction, so the floor runs on all 10,779 test
   questions -- the same denominator as Acc(C0) in Table 1, which makes this
   row MORE comparable than the other two, not less. Footnote c does not apply
   to it.

3. RULER. The SLAKE and VQA-RAD baseline compares `pred == answer`, exact
   string. This one uses `answer_matches(normalize_text(...))`, the ruler
   behind every other result file, because its `correct` field is read beside
   the other models'. Checked before deciding: on the recorded SLAKE and
   VQA-RAD predictions the two comparisons return the same counts (267/416 and
   169/272), so adopting the repository ruler does not make the three floors
   incomparable. On Kvasir the choice matters, because some answers are lists.

Usage:
    python -m src.baselines.question_only_kvasir \
        --arrow data/finetune_arrows/vqa_kvasir_train_clean.arrow

The Arrow file is not redistributed here; it is built by the Kvasir clean-split
code. Without it the floor cannot be retrained, but the published number stays
verifiable: `rescore` recomputes 7,086/10,779 from the committed predictions,
the committed swapmap and the ruler alone, and the tests do exactly that.

Retraining reproduces the published files byte for byte, and does so under
scikit-learn 1.4.2 and 1.9.1 alike -- checked, because a floor that moved with
the solver version would not be a floor anyone else could land on.
"""
from __future__ import annotations

import argparse
import json
import os

from src.baselines.question_only import train_question_only_model
from src.inference.process_text import answer_matches, normalize_text
from src.inference.results_schema import validate_results_dict

# Every condition that leaves the question text untouched: C1/C2/C4 perturb the
# image and C5/C6 swap it between donor pairs, so the floor is the same in all.
CONDITIONS = ("C0", "C1", "C2", "C5", "C6",
              "C4_OCC_25", "C4_OCC_50", "C4_OCC_75",
              "C4_NOISE_25", "C4_NOISE_50", "C4_NOISE_75")

DEFAULT_ARROW = "data/finetune_arrows/vqa_kvasir_train_clean.arrow"
DEFAULT_SWAPMAP = "data/swapmap.json"


def load_train_pairs(arrow_path: str) -> list[dict]:
    """(question, answer) of the clean training split, from the training Arrow.

    The file is an IPC *file* with one row per IMAGE: `questions` and `answers`
    are parallel lists, so flattening them yields exactly the 38,211 questions
    that qid_map.csv marks as training. Refusing an Arrow that carries any
    other split is what keeps the floor from being trained on its own test set.
    """
    import pyarrow as pa

    if not os.path.exists(arrow_path):
        raise SystemExit("STOP: training Arrow not found at %s" % arrow_path)
    with pa.memory_map(arrow_path, "rb") as src:
        table = pa.ipc.open_file(src).read_all()
    missing = {"questions", "answers", "split"} - set(table.schema.names)
    if missing:
        raise SystemExit("STOP: Arrow is missing the columns %s" % sorted(missing))
    splits = set(table.column("split").to_pylist())
    if splits != {"train"}:
        raise SystemExit("STOP: Arrow is not training-only, it holds %s" % sorted(splits))

    questions = table.column("questions").to_pylist()
    answers = table.column("answers").to_pylist()
    pairs = []
    for row_questions, row_answers in zip(questions, answers):
        for question, answer in zip(row_questions, row_answers):
            # Each answer is a list; on Kvasir it holds a single element, which
            # may itself be a ';'-separated list of locations.
            target = answer[0] if isinstance(answer, list) else answer
            pairs.append({"question": str(question),
                          "answer": str(target).strip().lower()})
    return pairs


def load_test_items(swapmap_path: str = DEFAULT_SWAPMAP) -> list[dict]:
    """The 10,779 clean-split test items, keyed by the qids of the prediction files."""
    with open(swapmap_path, encoding="utf-8") as fp:
        qid = json.load(fp)["KVASIR"]["qid"]
    return [{"qid": int(k), "question": str(v["q"]),
             "refs": [str(r) for r in v["a"]]} for k, v in qid.items()]


def score(predicted, items: list[dict], condition: str = "C0") -> dict:
    """Build a results record from predictions already aligned with `items`.

    Kept separate from the model so that the published number can be checked
    without retraining: `rescore` feeds it the committed predictions.
    """
    if len(predicted) != len(items):
        raise ValueError("got %d predictions for %d items" % (len(predicted), len(items)))
    predictions, n_correct = [], 0
    for item, pred in zip(items, predicted):
        correct = bool(answer_matches(normalize_text(str(pred)),
                                      [normalize_text(r) for r in item["refs"]]))
        n_correct += int(correct)
        predictions.append({
            "qid": item["qid"], "predicted": str(pred),
            "gold": item["refs"][0] if len(item["refs"]) == 1 else item["refs"],
            "correct": correct})
    n = len(items)
    record = {"model": "M0_question_only", "dataset": "KVASIR",
              "condition": condition, "accuracy": n_correct / n,
              "n": n, "n_correct": n_correct, "predictions": predictions}
    validate_results_dict(record)
    return record


def rescore(record: dict, items: list[dict]) -> dict:
    """Recompute the floor from a stored record's `predicted` strings.

    This is the path that makes the published 65.74% verifiable from committed
    data alone: no Arrow, no training, only the predictions, the swapmap and
    the ruler. A drift in any of the three shows up as a different count.
    """
    by_qid = {p["qid"]: p["predicted"] for p in record["predictions"]}
    missing = [i["qid"] for i in items if i["qid"] not in by_qid]
    if missing:
        raise ValueError("record has no prediction for %d qids, e.g. %s"
                         % (len(missing), missing[:5]))
    return score([by_qid[i["qid"]] for i in items], items,
                 condition=record.get("condition", "C0"))


def write_conditions(record: dict, out_dir: str) -> list[str]:
    """Write the same predictions under all eleven condition names.

    Not duplication for its own sake: every downstream reader globs by
    condition, and a floor whose VDD is zero must be present in each one for
    the comparison against the models to be possible at all.
    """
    written = []
    for condition in CONDITIONS:
        entry = dict(record, condition=condition)
        validate_results_dict(entry)
        path = os.path.join(out_dir, "M0_question_only_KVASIR_%s.json" % condition)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(entry, fp, ensure_ascii=False)
        written.append(path)
    return written


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow", default=DEFAULT_ARROW)
    parser.add_argument("--swapmap", default=DEFAULT_SWAPMAP)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--dry-run", action="store_true",
                        help="train and report the floor without writing files")
    return parser.parse_args()


def main():
    args = _parse_args()
    train_pairs = load_train_pairs(args.arrow)
    items = load_test_items(args.swapmap)
    print("train: %d questions | test: %d questions" % (len(train_pairs), len(items)))
    print("distinct answers in training: %d" % len({p["answer"] for p in train_pairs}))

    model = train_question_only_model(train_pairs)
    record = score(model.predict([i["question"] for i in items]), items)
    print("\nKVASIR QUESTION-ONLY FLOOR: %d/%d = %.2f%%"
          % (record["n_correct"], record["n"], record["accuracy"] * 100))

    if args.dry_run:
        return
    written = write_conditions(record, args.out_dir)
    print("wrote %d files to %s" % (len(written), args.out_dir))


if __name__ == "__main__":
    main()
