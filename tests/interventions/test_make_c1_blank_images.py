import io

from PIL import Image

from src.interventions.arrow_io import read_arrow
from src.interventions.make_c1_blank_images import blank_out_images, make_blank_jpeg_bytes
from tests.helpers import write_sample_arrow


def test_blank_out_images_replaces_bytes_and_keeps_rest(tmp_path):
    input_path = tmp_path / "in.arrow"
    output_path = tmp_path / "out.arrow"
    write_sample_arrow(str(input_path), n=2)

    blank_out_images(str(input_path), str(output_path), size=32)

    df_in = read_arrow(str(input_path))
    df_out = read_arrow(str(output_path))

    assert len(df_out) == len(df_in)
    assert list(df_out["image_id"]) == list(df_in["image_id"])
    assert list(df_out["split"]) == list(df_in["split"])

    expected_blank = make_blank_jpeg_bytes(size=32)
    for b in df_out["image"]:
        assert b == expected_blank
        img = Image.open(io.BytesIO(b))
        assert img.size == (32, 32)

    assert list(df_out["image"]) != list(df_in["image"])
