import io

import numpy as np
import pytest
from PIL import Image

from src.interventions.arrow_io import read_arrow
from src.interventions.make_c4_severity import (
    add_noise_to_image_bytes,
    apply_severity_condition,
    condition_label,
    occlude_image_bytes,
)
from tests.helpers import write_sample_arrow_with_real_images


def _solid_jpeg_bytes(size=64, color=(200, 30, 30)) -> bytes:
    img = Image.new("RGB", (size, size), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_occlude_image_bytes_blackens_center_keeps_corner_and_size():
    original = _solid_jpeg_bytes(size=64, color=(200, 30, 30))
    occluded = occlude_image_bytes(original, severity=0.25)

    img = Image.open(io.BytesIO(occluded))
    assert img.size == (64, 64)

    arr = np.asarray(img)
    corner = arr[0, 0]
    center = arr[32, 32]
    assert corner[0] > 150  # the corner keeps the original colour (high red)
    assert center.max() < 50  # the centre turned black (occlusion patch)


def test_occlude_image_bytes_rejects_invalid_severity():
    original = _solid_jpeg_bytes()
    with pytest.raises(ValueError):
        occlude_image_bytes(original, severity=0.0)
    with pytest.raises(ValueError):
        occlude_image_bytes(original, severity=1.0)


def test_add_noise_to_image_bytes_is_reproducible_and_changes_pixels():
    original = _solid_jpeg_bytes(size=64, color=(128, 128, 128))
    noisy_a = add_noise_to_image_bytes(original, severity=0.5, seed=7)
    noisy_b = add_noise_to_image_bytes(original, severity=0.5, seed=7)
    noisy_c = add_noise_to_image_bytes(original, severity=0.5, seed=8)

    assert noisy_a == noisy_b  # same seed -> byte-for-byte identical
    assert noisy_a != noisy_c  # different seed -> different result

    arr_original = np.asarray(Image.open(io.BytesIO(original))).astype(np.float32)
    arr_noisy = np.asarray(Image.open(io.BytesIO(noisy_a))).astype(np.float32)
    assert arr_noisy.std() > arr_original.std() + 10  # the original is flat (low std)


def test_add_noise_to_image_bytes_rejects_invalid_severity():
    original = _solid_jpeg_bytes()
    with pytest.raises(ValueError):
        add_noise_to_image_bytes(original, severity=0.0, seed=1)


def test_apply_severity_condition_occlusion_preserves_columns_except_image(tmp_path):
    input_path = tmp_path / "in.arrow"
    output_path = tmp_path / "out.arrow"
    write_sample_arrow_with_real_images(str(input_path), n=2)

    apply_severity_condition(str(input_path), str(output_path), kind="occlusion", severity=0.5)

    df_in = read_arrow(str(input_path))
    df_out = read_arrow(str(output_path))
    assert len(df_out) == len(df_in)
    assert list(df_out["image_id"]) == list(df_in["image_id"])
    for qs_out, qs_in in zip(df_out["questions"], df_in["questions"]):
        assert list(qs_out) == list(qs_in)
    assert list(df_out["image"]) != list(df_in["image"])


def test_apply_severity_condition_noise_preserves_columns_except_image(tmp_path):
    input_path = tmp_path / "in.arrow"
    output_path = tmp_path / "out.arrow"
    write_sample_arrow_with_real_images(str(input_path), n=2)

    apply_severity_condition(str(input_path), str(output_path), kind="noise", severity=0.25, seed=42)

    df_in = read_arrow(str(input_path))
    df_out = read_arrow(str(output_path))
    assert len(df_out) == len(df_in)
    assert list(df_out["image"]) != list(df_in["image"])


def test_apply_severity_condition_rejects_invalid_kind(tmp_path):
    input_path = tmp_path / "in.arrow"
    write_sample_arrow_with_real_images(str(input_path), n=1)
    with pytest.raises(ValueError):
        apply_severity_condition(str(input_path), str(tmp_path / "out.arrow"), kind="blur", severity=0.5)


def test_condition_label_formats_correctly():
    assert condition_label("occlusion", 0.25) == "C4_OCC_25"
    assert condition_label("occlusion", 0.75) == "C4_OCC_75"
    assert condition_label("noise", 0.5) == "C4_NOISE_50"
