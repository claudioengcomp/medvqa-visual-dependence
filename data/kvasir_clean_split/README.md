# Image-disjoint split of Kvasir-VQA-x1

This directory contains the exact image-to-split and question-to-split mapping
of the image-disjoint split of Kvasir-VQA-x1 used in the paper. It is the
artifact referred to in Section 4.1 ("the split map is released with the code").

## 1. The problem this solves

The official HuggingFace release of `SimulaMet/Kvasir-VQA-x1` leaks images
between training and test. **3,821 of its 4,058 test images (94%) also appear
in the training set**, verified by image identifier.

The cause is a mismatch between the assignment criterion and the documentation:
the release assigns each image–question **pair** to a partition, whereas the
dataset description and dataset card state that no test image appears in
training. A further symptom is that the test partition holds 10% of the
questions but 62% of the images, although each image carries about 25
questions — a proportion no split by image could produce.

Any result computed on that split therefore overestimates generalization.

## 2. How the clean split is built

- **Source**: `SimulaMet/Kvasir-VQA-x1`, splits `train` (143,594 rows, 6,212
  unique `img_id`) and `test` (15,955 rows, 4,058 unique `img_id`) from the
  HuggingFace Hub, **merged and deduplicated by `img_id`** → **6,449 unique
  images** that carry at least one question (51 of the 6,500 source images
  carry none).
- **Granularity**: by `img_id`. The dataset schema carries no patient or
  examination identifier, so image level is the finest granularity available
  (see Limitations).
- **Answer decomposition**: the composite natural-language answers are
  decomposed into atomic `(question, answer)` facts, and duplicates per image
  and question are removed. This is the same pipeline used for evaluation, so
  training and evaluation share one data format.
- **Proportions**: 70/10/20 (train/val/test).
- **Stratification**: by `pathology_label` — the answer to the template
  question *"Are there any abnormalities in the image? Check all that are
  present."*, lowercased — so that a rare clinical category does not end up
  entirely on one side of the split. Images without a label fall into a `None`
  stratum, handled like any other.
- **Seed**: `42`, fixed and deterministic (`random.Random(42)`).
- **Algorithm**: `src/prepro/kvasir_split.py::split_records_by_image` groups
  records by `pathology_label`, shuffles each group with the seeded RNG, and
  cuts each group at `round(n*0.7)` / `round(n*0.1)` / remainder, concatenating
  the three splits across all groups.

### Rebuilding from scratch

```bash
python -m src.prepro.build_kvasir_clean_split \
  --output-dir data/finetune_arrows \
  --seed 42
```

Produces in `data/finetune_arrows/`:

- `vqa_kvasir_train_clean.arrow`, `vqa_kvasir_val_clean.arrow`,
  `vqa_kvasir_test_clean.arrow`
- `kvasir_clean_split_map.csv` (`img_id` → split)
- `vqa_kvasir_clean_pathology_labels.json` (`pathology_label` per image)

## 3. Exact counts

### By image (`split_map.csv`)

| split | images |
|---|---|
| train | 4,514 |
| val | 644 |
| test | 1,291 |
| **total** | **6,449** |

### By atomic question (`qid_map.csv`, read from the generated arrow files)

| split | questions | min `question_id` | max `question_id` |
|---|---|---|---|
| train | 38,211 | 0 | 38,210 |
| val | 5,494 | 1,000,000 | 1,005,493 |
| test | 10,779 | 2,000,000 | 2,010,778 |
| **total** | **54,484** | | |

No `question_id` repeats across splits, and no `img_id` appears in more than
one split. This is asserted by an automated test
(`tests/prepro/test_build_kvasir_clean_split.py`), not only checked by hand.

## 4. Reproducibility check

Beyond the theoretical guarantee of a fixed seed, reproducibility was verified
empirically: the HuggingFace dataset (`train` + `test`) was reloaded from
scratch, the record construction and the split were re-run with
`ratios=(0.7, 0.1, 0.2)` and `seed=42`, and the result was compared `img_id` by
`img_id` against the existing split map.

```
train HF rows=143594  test HF rows=15955
records (unique images after dedup): 6449
recomputed counts: {'train': 4514, 'val': 644, 'test': 1291}
total original: 6449   total recomputed: 6449
mismatches (same key, different split): 0
only in original: 0
only in recomputed: 0
```

Zero divergences over 6,449 images: the same seed over the same raw HuggingFace
data reproduces exactly the same split.

## 5. Files in this directory

| File | Contents |
|---|---|
| `split_map.csv` | `img_id,split` — one row per unique image (6,449) |
| `qid_map.csv` | `question_id,img_id,split` — one row per atomic question (54,484) |
| `README.md` | this file |

Only identifiers are distributed here. No image data from Kvasir-VQA-x1 is
redistributed; the images must be obtained from the original dataset.

## 6. Limitations

The split is drawn by image because the dataset provides no patient or
examination identifier. Frames from the same procedure that carry different
`img_id` values are not detectable and may therefore land on different sides of
the split. This is a limitation inherited from the dataset schema, not
introduced by this split.
