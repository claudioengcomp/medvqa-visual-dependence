"""M0: degenerate question-only baseline for SLAKE and VQA-RAD.

Trains a simple classifier (TF-IDF over the question text plus logistic
regression) that predicts the answer from the question text ALONE, never
seeing the image. It serves as a control: by construction its accuracy is
identical under C0/C1/C2/C4 (x6)/C5/C6, because it never looks at the image
and none of those conditions alters the question text -- C1/C2/C4 perturb the
image, C5/C6 swap the image between donor pairs, but the wording of the
question is always the same. Its VDD is therefore zero by construction, which
is the point: it establishes the floor against which models that genuinely
use vision are compared.

Scope: closed questions only (answer_type == "CLOSED"), which have a fixed
answer vocabulary and admit exact-match scoring. Open-ended questions are out
of scope for this baseline, since exact match over free-form text would not
yield a meaningful floor.

Generalized to VQA-RAD: the q_lang == 'en' filter was hardcoded for the SLAKE
format (which is bilingual and carries that field). VQA-RAD has no such field
(it is monolingual English by construction), and applying the filter blindly
discarded the entire dataset -- s.get("q_lang") returns None, None != "en", so
every sample was dropped. The filter is now applied only when the field is
actually present in the record.

Usage:
    python -m src.baselines.question_only \
        --train_json Slake1.0/train.json \
        --test_json Slake1.0/test.json \
        --dataset SLAKE \
        --condition C0 \
        --output results/M0_question_only_SLAKE_C0.json

    python -m src.baselines.question_only \
        --train_json vqa_rad_prepared/trainset.json \
        --test_json vqa_rad_prepared/testset.json \
        --dataset VQA-RAD \
        --condition C0 \
        --output results/M0_question_only_VQARAD_C0.json
"""
from __future__ import annotations

import argparse
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.inference.results_schema import VALID_CONDITIONS, validate_results_dict


def load_closed_qa_pairs(json_path: str) -> list[dict]:
    """Read a train/validate/test.json file (SLAKE or VQA-RAD) and keep only
    the closed-type questions (answer_type == 'CLOSED'). The language filter
    (q_lang == 'en') is applied only if the field exists in the record --
    SLAKE is bilingual and carries that field, whereas VQA-RAD does not,
    being monolingual by nature of the original dataset."""
    with open(json_path, "r", encoding="utf-8") as fp:
        samples = json.load(fp)

    pairs = []
    for s in samples:
        if "q_lang" in s and s.get("q_lang") != "en":
            continue
        if str(s.get("answer_type", "")).strip().upper() != "CLOSED":
            continue
        pairs.append({
            "qid": s["qid"],
            "question": s["question"],
            "answer": str(s["answer"]).strip().lower(),
        })
    return pairs


def train_question_only_model(train_pairs: list[dict]) -> Pipeline:
    if not train_pairs:
        raise ValueError("train_pairs is empty -- nothing to train on")
    questions = [p["question"] for p in train_pairs]
    answers = [p["answer"] for p in train_pairs]
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(questions, answers)
    return model


def evaluate_question_only_model(model: Pipeline, test_pairs: list[dict],
                                  condition: str = "C0", dataset: str = "SLAKE") -> dict:
    if not test_pairs:
        raise ValueError("test_pairs is empty -- nothing to evaluate")
    questions = [p["question"] for p in test_pairs]
    predicted = model.predict(questions)

    predictions = []
    n_correct = 0
    for p, pred in zip(test_pairs, predicted):
        correct = bool(pred == p["answer"])
        n_correct += int(correct)
        predictions.append({
            "qid": p["qid"], "predicted": str(pred), "gold": p["answer"], "correct": correct,
        })

    n = len(test_pairs)
    record = {
        "model": "M0_question_only",
        "dataset": dataset,
        "condition": condition,
        "accuracy": n_correct / n,
        "n": n,
        "n_correct": n_correct,
        "predictions": predictions,
    }
    validate_results_dict(record)
    return record


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--test_json", required=True)
    parser.add_argument("--dataset", default="SLAKE", choices=["SLAKE", "VQA-RAD"])
    parser.add_argument("--condition", default="C0", choices=list(VALID_CONDITIONS))
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_pairs = load_closed_qa_pairs(args.train_json)
    test_pairs = load_closed_qa_pairs(args.test_json)
    model = train_question_only_model(train_pairs)
    results = evaluate_question_only_model(model, test_pairs, condition=args.condition, dataset=args.dataset)
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    print(f"M0: acc={results['accuracy']:.4f} ({results['n_correct']}/{results['n']}) -> {args.output}")
