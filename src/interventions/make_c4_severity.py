"""Parametric corruption: generate copies of the test .arrow file with a
controlled severity of image damage, in two independent families:

- occlusion: black square patch at the center covering X% of the image AREA
  (side = image_size × sqrt(X), centered).  ->  labelled C4_OCC_<X>
- noise: additive Gaussian noise with standard deviation = X% of the 8-bit
  dynamic range (0-255), pixels clipped back to [0, 255].  ->  labelled C4_NOISE_<X>

LABEL MAPPING -- the paper treats these as two separate rungs of the ladder,
C3 (occlusion) and C4 (noise), while this script emits both under the shared
`C4_` prefix. The code labels are frozen because they are embedded in the
names and in the "condition" field of the result files already generated; see
the correspondence table in README.md. In short: C4_OCC_* is the paper's C3,
C4_NOISE_* is the paper's C4.

Unlike C1 (a single point, a fully neutral image), these conditions yield a
degradation curve over X ∈ {0.25, 0.50, 0.75}, so visual dependency can be
measured as a dose-response relationship rather than a single on/off
contrast.

Usage:
    python -m src.interventions.make_c4_severity \
        --input data/finetune_arrows/vqa_slack_test.arrow \
        --output data/finetune_arrows/vqa_slack_test_c4_occ_25.arrow \
        --kind occlusion --severity 0.25
"""
from __future__ import annotations

import argparse
import io

import numpy as np
from PIL import Image

from src.interventions.arrow_io import read_arrow, write_arrow

VALID_KINDS = ("occlusion", "noise")
VALID_SEVERITIES = (0.25, 0.50, 0.75)


def occlude_image_bytes(image_bytes: bytes, severity: float, fill_color=(0, 0, 0)) -> bytes:
    """Replace a centered square patch covering `severity` of the image AREA
    with a constant color, leaving the rest untouched."""
    if not (0.0 < severity < 1.0):
        raise ValueError(f"severity={severity} must be in (0, 1)")
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    side_w = int(round(w * (severity ** 0.5)))
    side_h = int(round(h * (severity ** 0.5)))
    left = (w - side_w) // 2
    top = (h - side_h) // 2
    patch = Image.new("RGB", (side_w, side_h), fill_color)
    img.paste(patch, (left, top))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def add_noise_to_image_bytes(image_bytes: bytes, severity: float, seed: int) -> bytes:
    """Add additive Gaussian noise to the image; standard deviation =
    severity * 255 (a fraction of the 8-bit dynamic range). Deterministic for
    a fixed (bytes, severity, seed)."""
    if not (0.0 < severity < 1.0):
        raise ValueError(f"severity={severity} must be in (0, 1)")
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    rng = np.random.RandomState(seed)
    noise = rng.normal(loc=0.0, scale=severity * 255.0, size=arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(noisy).save(buf, format="JPEG")
    return buf.getvalue()


def apply_severity_condition(input_path: str, output_path: str, kind: str, severity: float,
                              seed: int = 42) -> None:
    if kind not in VALID_KINDS:
        raise ValueError(f"kind={kind!r} must be one of {VALID_KINDS}")
    df = read_arrow(input_path)
    df = df.copy()
    if kind == "occlusion":
        df["image"] = [occlude_image_bytes(b, severity) for b in df["image"]]
    else:
        rng = np.random.RandomState(seed)
        per_row_seeds = rng.randint(0, 2**31 - 1, size=len(df))
        df["image"] = [
            add_noise_to_image_bytes(b, severity, seed=int(s))
            for b, s in zip(df["image"], per_row_seeds)
        ]
    write_arrow(df, output_path)


def condition_label(kind: str, severity: float) -> str:
    """'occlusion', 0.25 -> 'C4_OCC_25'; 'noise', 0.5 -> 'C4_NOISE_50'."""
    prefix = "OCC" if kind == "occlusion" else "NOISE"
    pct = int(round(severity * 100))
    return f"C4_{prefix}_{pct}"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kind", required=True, choices=list(VALID_KINDS))
    parser.add_argument("--severity", type=float, required=True, choices=list(VALID_SEVERITIES))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    apply_severity_condition(args.input, args.output, args.kind, args.severity, seed=args.seed)
    print(f"C4 ({args.kind}, severity={args.severity}) escrito em {args.output}")
