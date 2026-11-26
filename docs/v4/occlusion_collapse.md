# Occlusion collapses images; noise does not

## What was measured

At each severity, how many of the test images are still distinguishable from one
another - counted by sha256 of the encoded bytes, so two images are "the same"
only when every pixel agrees.

| benchmark | C0 | C3-25 | C3-50 | C3-75 | C4-25 | C4-50 | C4-75 |
|---|---|---|---|---|---|---|---|
| Kvasir | 1,284 | 1,284 | 1,284 | 1,284 | 1,291 | 1,291 | 1,291 |
| SLAKE | 96 | 96 | **89** | **75** | 96 | 96 | 96 |
| VQA-RAD | 203 | 203 | 203 | 203 | 203 | 203 | 203 |

Kvasir starts at 1,284 rather than 1,291 because of the seven pairs the content
audit already found inside the test split. That figure never moves: centred
occlusion merges nothing on Kvasir that was not already merged.

SLAKE loses 7 images at 50% and 21 at 75%, and they are not scattered pairs -
they are **one group of 22 images that become a single image**. Just under a
quarter of the SLAKE test split stops being distinguishable at the strongest
occlusion.

Noise never merges anything, at any severity, on any benchmark. The distinction
is not that noise preserves information: at 75% of the intensity range it
destroys a great deal. It is that **occlusion collapses distinct inputs into one
and noise degrades without collapsing**, because each noise realisation differs.

## Why it happens, in pixels

`harness/v4/occlusion_ring.py` measures the ring left uncovered by the patch.

| group | n | ring std, median | ring std, max | fraction pure black, median |
|---|---|---|---|---|
| the 22 collapsed | 22 | **0.000** | **0.000** | **1.000** |
| the other 74 | 74 | 36.1 | 88.5 | 0.789 |

Every collapsed image has a ring that is black end to end, and all 22 are
240x240. One further image, `xmlab18`, also has an all-black ring and does not
collapse: it is 256x256, so its occluded bytes differ. Collapse requires a black
border **and** a shared resolution; nothing else is involved.

This is the sentence for the paper, and it is about the interaction rather than
about the intervention:

> Centred occlusion is not a uniform intervention across imaging domains. On
> SLAKE it merges 22 of 96 test images into a single indistinguishable input at
> 75% area, because what the patch leaves behind is an all-black radiological
> border; on Kvasir it merges none, because the endoscope's frame differs
> between images. Noise at matched severity merges nothing anywhere.

## It is not the duplication the audit found

The obvious guess was that the 22 are the slices of shared volumes that the
pHash audit flagged - the same phenomenon through two instruments. Checked in
`harness/v4/collapse_vs_phash.py`, it is not.

| set | n |
|---|---|
| collapsed at C3-75 | 22 |
| flagged by the audit (pHash <= 4 against train/val) | 24 |
| **in both** | **3** |
| collapsed only | 19 |
| flagged only | 21 |

Only `xmlab442`, `xmlab474` and `xmlab483` are in both, which is about what
independence over 96 images would give. The collapsed images are not even
perceptual neighbours of each other while intact: 2 of the 22 have a pHash
neighbour inside the test split, against 8 in the split overall.

So these are **two findings, not one**. The audit's 24 are slices of volumes
that cross the partition - a property of how SLAKE was built. The collapsed 22
are images that happen to share a border and a resolution - a property of what
the intervention destroys. Occlusion does not reveal pre-existing duplication;
it manufactures duplication that was not there.

## Consequence for the ladder

C3-75 on SLAKE is measuring 75 distinct inputs presented 96 times, with 22 rows
carrying the same picture. Any accuracy at that cell is partly a measurement of
how the model answers one image under 22 different questions. The severity is
still interpretable as an upper bound on how far occlusion can go, but it is not
a clean 96-image condition and the paper should not treat it as one.

Evidence: `docs/v4/occlusion_collapse.json`,
`docs/v4/occlusion_ring.json`, `docs/v4/collapse_vs_phash.json`.
