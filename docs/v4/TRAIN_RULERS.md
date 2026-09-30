# Rulers on the training bank

Table 1 validates the five neutral rulers by leave-one-out kNN (k = 10) on the
intact test images only, while analysis A measures the distance of each test
image to the training bank. Two questions follow: do the rulers organise the
training bank as well, and how does the distance to the bank behave for
training images (leave-one-out) against intact test images (C0)? Both are
answered from cached embeddings, without a GPU, by
`src/featurespace/train_rulers.py`. Outputs: `train_rulers_table1.json`
(kNN on the bank, neutral rulers and model-derived rulers), `train_rulers_distances.json`
(distance distributions, anchors and the near-duplicate detector) and
`train_rulers_inputs.json` (every input array and label file with its SHA-256).
Two regressions gate the computation: Table 1 (`g0_results.json`) is reproduced
to the digit in all 30 cells, and the C0 distance to the bank of analysis A
(`analysis_a.json`) in all 15 cells.

## Labels of the bank

The bank is labelled with the same question-to-label rules as the test set:
Kvasir-VQA-x1 procedure from the procedure question and the landmark map
(2,756 of 4,514 images; 4 images with a question/landmark conflict dropped),
finding from the clean-split pathology labels with composite labels dropped
(2,138), organ of VQA-RAD from `image_organ` (314 of 314), modality of SLAKE
from the `modality` field of its annotation files (450 of 450; the C6
manifests of the test set carry the same field, 93 of 93 agree).

## kNN on the bank, neutral rulers (test / bank / bank − test)

| task | n test / bank | MedSigLIP | DINOv3 | BiomedCLIP | DINOv2 | colour |
|---|---|---|---|---|---|---|
| kvasir_procedure | 776 / 2,756 | 0.996 / 0.996 / −0.000 | 0.995 / 0.995 / +0.000 | 0.987 / 0.995 / +0.008 | 0.990 / 0.991 / +0.001 | 0.960 / 0.976 / +0.016 |
| kvasir_finding | 612 / 2,138 | 0.982 / 0.989 / +0.007 | 0.971 / 0.983 / +0.013 | 0.953 / 0.969 / +0.017 | 0.925 / 0.943 / +0.018 | 0.918 / 0.942 / +0.024 |
| kvasir_finding_in_colonoscopy | 424 / 1,479 | 0.986 / 0.992 / +0.006 | 0.967 / 0.982 / +0.015 | 0.972 / 0.969 / −0.003 | 0.918 / 0.930 / +0.013 | 0.910 / 0.930 / +0.020 |
| vqarad_organ | 203 / 314 | 1.000 / 1.000 / 0 | 0.966 / 0.984 / +0.019 | 1.000 / 1.000 / 0 | 0.985 / 0.990 / +0.005 | 0.808 / 0.854 / +0.046 |
| slake_modality | 93 / 450 | 0.957 / 1.000 / +0.043 | 0.968 / 0.987 / +0.019 | 0.968 / 1.000 / +0.032 | 0.968 / 0.984 / +0.017 | 0.850 / 0.962 / +0.113 |

Pre-fixed reading (|Δ| ≤ 0.02): met in 25 of 30 cells; the exceptions are the
colour histogram on kvasir_finding and vqarad_organ, and MedSigLIP, BiomedCLIP
and colour on slake_modality, where the test set has 93 labelled images. The
gastroscopy branch (one class in 185 of 187 test images) is excluded, as in the
paper. Model-derived rulers (18 on Kvasir-VQA-x1, 14 on SLAKE) are in the JSON.

## Distance to the bank: training (leave-one-out) against test (C0)

Per benchmark and ruler, `train_rulers_distances.json` records mean, median,
q10 and q90 of both distributions, the Wasserstein-1 distance between them, the
share of test images above the bank median, and where C1, C2, C7 and
occlusion-75 fall relative to that median. For the four content rulers: on
Kvasir-VQA-x1 W1 is 0.0011–0.0023 and 48.0–50.1% of the test images lie above
the bank median (MedSigLIP: bank median 0.0673, test median 0.0660); on SLAKE
W1 0.0038–0.0094 and 42.7–52.1%; on VQA-RAD W1 0.0052–0.0190 and 33.0–41.4%.
C1, C7 and occlusion-75 lie above the bank median for every test image in all
12 content-ruler cells (median minus bank median +0.407 to +0.938, +0.191 to
+0.848 and +0.168 to +0.881); C2 coincides with C0 (−0.018 to +0.002).

## Near duplicates (1-NN of the test set into the bank)

Kvasir-VQA-x1: under MedSigLIP 35 test images at distance below 1e-4, 318
below 0.01 and 630 below 0.05, of 1,291; the other rulers give 174–527 below
0.01. Byte-level check: 34 test images have a byte-identical copy in the bank
under another identifier (7 further duplicate pairs inside the test set, 63
inside the bank). SLAKE: 0 below 0.01 under MedSigLIP, DINOv3 and DINOv2 (2
under BiomedCLIP, 63 under colour), no byte-identical copy. VQA-RAD: 203 of 203
test images are byte-identical to a training image with the same identifier,
the known construction of that benchmark. The identifiers of the Kvasir cases
below 0.01 are listed in the JSON.
