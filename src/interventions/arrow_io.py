"""Shared utilities to read/write .arrow files in the same format used by
prepro/make_arrow.py::make_arrow_vqa in the upstream M3AE/LaPA codebase, whose data format this repository reuses.

Schema (column order, confirmed by reading the make_arrow_vqa source):
    image          bytes      - binary contents of the image file (jpg)
    questions      list[str]
    answers        list[list[str]]
    answer_labels  list[list[int]]
    answer_scores  list[list[float]]
    image_id       str        - image file path (used as the key)
    question_id    list[int]
    answer_type    list[int]  - 0=closed, 1=open
    split          str        - "train" | "val" | "test"
"""
from __future__ import annotations

import os

import pyarrow as pa
import pandas as pd

ARROW_COLUMNS = [
    "image",
    "questions",
    "answers",
    "answer_labels",
    "answer_scores",
    "image_id",
    "question_id",
    "answer_type",
    "split",
]


def read_arrow(path: str) -> pd.DataFrame:
    """Read an .arrow file written in the make_arrow_vqa format and return a DataFrame."""
    with pa.OSFile(path, "rb") as source:
        table = pa.ipc.open_file(source).read_all()
    df = table.to_pandas()
    missing = [c for c in ARROW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"File {path} is missing the expected columns: {missing}")
    return df


def write_arrow(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to .arrow in the same format used by make_arrow_vqa
    (same column order, same RecordBatchFileWriter).

    Creates the parent directory of `path` if it does not exist. This was
    found the hard way at run time: `data/finetune_arrows/` is gitignored
    and therefore never exists in a fresh clone - pa.OSFile then raises a
    low-level `FileNotFoundError` with no clear message, after minutes of
    work have already been done (e.g. downloading ~4,000 Kvasir images)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df = df[ARROW_COLUMNS]
    table = pa.Table.from_pandas(df, preserve_index=False)
    with pa.OSFile(path, "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)


def count_questions(df: pd.DataFrame) -> int:
    """Count the actual number of evaluated questions in an arrow DataFrame.

    Each ROW of the arrow file is an IMAGE, which carries a LIST of questions
    in the 'questions' column - counting rows (len(df)) counts images, not
    questions. This exact bug produced a wrong bootstrap 95% CI in the first
    M3AE result (n=96 images instead of n=1061 questions)."""
    return int(sum(len(qs) for qs in df["questions"]))
