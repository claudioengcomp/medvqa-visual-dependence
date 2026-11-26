# Verifying the image leakage in the official Kvasir-VQA-x1 split

The paper reports that **3,821 of the 4,058 test images (94%) of the official
Kvasir-VQA-x1 release also appear in its training set**. This document explains
how to verify that claim independently.

## Run the check

```bash
pip install datasets
python -m src.prepro.verify_leakage
```

This downloads the official `train` and `test` splits from the HuggingFace Hub
and compares the sets of `img_id` values. Expected output:

```
train rows                : 143,594
test rows                 : 15,955
unique train images       : 6,212
unique test images        : 4,058
test images also in train : 3,821 (94.2% of the test images)

unique images overall     : 6,449
questions per image (avg) : 24.7
test share of questions   : 10.0%
test share of images      : 62.9%
```

## Why the check is exact

The comparison uses `img_id`, the image identifier the dataset itself carries.
It is not an approximate image hash or a filename heuristic, so there is no
false-positive matching.

## Why the leak exists

The release assigns each image–**question pair** to a partition, whereas the
dataset description and the dataset card state that no test image appears in
training. The two criteria are incompatible, and the release follows the first.

The last two lines of the output make the consequence visible: the test
partition holds about 10% of the questions but 62% of the images, although each
image carries roughly 25 questions. No split by image could produce that
combination - the images are spread across both partitions, and only the
question–answer pairs are separated.

## Consequence

Any result computed on the official split overestimates generalization, because
a model can memorize an image during training and be asked a different question
about the same image at test time.

The identifier-disjoint split released here is described in
[`../data/kvasir_clean_split/README.md`](../data/kvasir_clean_split/README.md),
and the absence of overlap is asserted by an automated test:

```bash
pytest tests/prepro/test_build_kvasir_clean_split.py
```
