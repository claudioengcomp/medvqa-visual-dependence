import re

import pytest

from src.interventions.arrow_io import read_arrow
from src.interventions.make_c2_shuffled_pairs import derangement, shuffle_image_pairs
from tests.helpers import write_sample_arrow


def test_derangement_has_no_fixed_points_and_is_a_permutation():
    for seed in [0, 1, 42, 12345]:
        perm = derangement(20, seed)
        assert sorted(perm.tolist()) == list(range(20))
        assert all(perm[i] != i for i in range(20))


def test_derangement_handles_odd_n():
    # n=7 exercises the len(fixed) == 1 wraparound branch (j = (i + 1) % n)
    # for an odd-sized permutation, which the n=20/10/6 cases above never hit.
    for seed in [0, 1, 42, 12345]:
        for n in [7, 15]:
            perm = derangement(n, seed)
            assert sorted(perm.tolist()) == list(range(n))
            assert all(perm[i] != i for i in range(n))


def test_derangement_is_deterministic():
    assert derangement(10, seed=7).tolist() == derangement(10, seed=7).tolist()


def test_derangement_rejects_n_below_2():
    with pytest.raises(ValueError):
        derangement(1, seed=0)


def test_shuffle_image_pairs_preserves_non_image_columns_and_deranges_images(tmp_path):
    input_path = tmp_path / "in.arrow"
    output_path = tmp_path / "out.arrow"
    write_sample_arrow(str(input_path), n=6)

    shuffle_image_pairs(str(input_path), str(output_path), seed=42)

    df_in = read_arrow(str(input_path))
    df_out = read_arrow(str(output_path))

    assert list(df_out["split"]) == list(df_in["split"])
    assert [list(q) for q in df_out["questions"]] == [list(q) for q in df_in["questions"]]

    for i in range(len(df_in)):
        assert df_out["image"].iloc[i] != df_in["image"].iloc[i]
        assert df_out["image_id"].iloc[i] != df_in["image_id"].iloc[i]

    assert sorted(df_out["image"].tolist()) == sorted(df_in["image"].tolist())
    assert sorted(df_out["image_id"].tolist()) == sorted(df_in["image_id"].tolist())

    # image and image_id must still be paired to the same original row after
    # shuffling (both were deranged by the same permutation).
    def _index_from_image(b: bytes) -> int:
        return int(re.match(rb"IMG(\d+)", b).group(1))

    def _index_from_image_id(s: str) -> int:
        return int(re.match(r"imgs/xmlab(\d+)/source\.jpg", s).group(1))

    for i in range(len(df_out)):
        assert _index_from_image(df_out["image"].iloc[i]) == _index_from_image_id(df_out["image_id"].iloc[i])
