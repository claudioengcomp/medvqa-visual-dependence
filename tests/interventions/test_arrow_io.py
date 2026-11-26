import os

import pandas as pd
import pyarrow as pa
import pytest

from src.interventions.arrow_io import ARROW_COLUMNS, count_questions, read_arrow, write_arrow
from tests.helpers import sample_rows, write_sample_arrow


def _norm(x):
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    return x


def test_round_trip_preserves_schema_and_values(tmp_path):
    path = tmp_path / "test.arrow"
    write_sample_arrow(str(path), n=2)
    df = read_arrow(str(path))
    expected = sample_rows(n=2)

    assert list(df.columns) == ARROW_COLUMNS
    assert len(df) == 2
    for col in ARROW_COLUMNS:
        for i in range(2):
            assert _norm(df[col].iloc[i]) == _norm(expected[col][i]), f"col={col} row={i}"


def test_read_arrow_missing_columns_raises(tmp_path):
    bad_path = tmp_path / "bad.arrow"
    df = pd.DataFrame({"image": [b"x"]})
    table = pa.Table.from_pandas(df, preserve_index=False)
    with pa.OSFile(str(bad_path), "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)

    with pytest.raises(ValueError):
        read_arrow(str(bad_path))


def test_count_questions_sums_questions_not_rows(tmp_path):
    path = tmp_path / "sample.arrow"
    write_sample_arrow(str(path), n=3)  # 3 rows (images), 2 questions each
    df = read_arrow(str(path))

    assert len(df) == 3  # image count - the WRONG value the bug used to report
    assert count_questions(df) == 6  # the real question count


def test_write_arrow_creates_missing_parent_directory(tmp_path):
    # Real finding from an actual run in an ephemeral execution environment:
    # data/finetune_arrows/ is gitignored and does not exist in a fresh
    # clone - pa.OSFile raised a low-level FileNotFoundError only after
    # minutes of work had already been done (e.g. downloading ~4,000 Kvasir
    # images).
    nested_path = tmp_path / "a" / "b" / "c" / "out.arrow"
    assert not os.path.isdir(nested_path.parent)

    df = pd.DataFrame(sample_rows(n=1))
    write_arrow(df, str(nested_path))

    assert os.path.isfile(nested_path)
    assert len(read_arrow(str(nested_path))) == 1
