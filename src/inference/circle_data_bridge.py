"""Translate a condition `.arrow` file (M3AE schema -- one row per image,
carrying per-image lists of questions/answers, see
src/interventions/arrow_io.py) into a HuggingFace `datasets.Dataset` in the
per-example format the generative pipeline expects:
{"image": PIL.Image, "question": str, "answers": list[str], "id": int}.

Each image is flattened into the N questions it carries, reusing the same
image bytes (decoded automatically by HuggingFace's `datasets.Image()`
feature when a row is accessed, never decoded by hand here). `id` is the real
`question_id` from the arrow rather than a positional index: preserving the
true qid is what keeps items paired across conditions, which is exactly what
the paired McNemar test
(src.stats.compute_vdd.compute_summary_vs_baseline) relies on."""
from __future__ import annotations

from datasets import Dataset, Features, Image as HFImage, Sequence, Value

from src.interventions.arrow_io import read_arrow

CONDITION_DATASET_FEATURES = Features({
    "image": HFImage(),
    "question": Value("string"),
    "answers": Sequence(Value("string")),
    "id": Value("int64"),
})


def load_condition_dataset(condition_arrow_path: str) -> Dataset:
    """Read `condition_arrow_path` (M3AE schema) and return a
    `datasets.Dataset` with one row per question, ready to be consumed by the
    chat-template collators and the generation loop of the generative
    pipeline without requiring any change to that package."""
    df = read_arrow(condition_arrow_path)

    images: list[bytes] = []
    questions: list[str] = []
    answers: list[list[str]] = []
    ids: list[int] = []

    for _, row in df.iterrows():
        image_bytes = row["image"]
        for question, answer_list, qid in zip(
            row["questions"], row["answers"], row["question_id"]
        ):
            images.append(image_bytes)
            questions.append(question)
            answers.append(list(answer_list))
            ids.append(int(qid))

    return Dataset.from_dict(
        {"image": images, "question": questions, "answers": answers, "id": ids},
        features=CONDITION_DATASET_FEATURES,
    )
