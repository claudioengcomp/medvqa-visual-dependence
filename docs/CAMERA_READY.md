# Analyses added for the final version

The reviewers asked for uncertainty that respects questions sharing an image,
for a policy on multiple comparisons, and for the comparison with the
question-only baseline to be made on the same questions. The three analyses
below answer that. They read the stored per-item predictions; none runs a
model, and none changes a number of the submitted version.

## Paired comparisons: clustered intervals and Holm's correction

    python -m src.stats.paired_cluster

Output: `docs/camera_ready/paired_cluster.json`. For each of the 464 paired
comparisons against C0 (435 for the fine-tuned models), the file gives the
discordant counts, McNemar's test, Cohen's g, a percentile interval from 2,000
bootstrap resamples of the original images as clusters, and the p-value under
Holm's correction over the family.

| rule | fine-tuned (435) | all (464) |
|---|---|---|
| McNemar p < 0.05, uncorrected | 383 | 409 |
| Holm over the family | 327 | 346 |
| image-clustered interval excludes zero | 383 | 409 |

The 56 comparisons lost under Holm are on VQA-RAD (48) and SLAKE (8). The
clustered interval and McNemar's test disagree in two cells, in opposite
directions. The same file recomputes VDD(C5) on Kvasir-VQA-x1 over the 8,934
rows of the per-row manifest; Table 3 does not change.

## Question-only baseline on the same closed questions

    python -m src.stats.baseline_same_subset

Output: `docs/camera_ready/baseline_same_subset.json`. Accuracy under C1 of
every fine-tuned checkpoint, restricted to the questions the baseline is
evaluated on.

| Benchmark | Baseline | Fine-tuned under C1, mean over seeds |
|---|---|---|
| Kvasir-VQA-x1 (10,779 questions) | 65.74% | 49.19 to 58.52% |
| SLAKE (416 closed) | 64.18% | 48.96 to 57.77% |
| VQA-RAD (272 closed) | 62.13% | classifier 55.27%; generative 61.27 to 61.76% |

On the mean over its seeds every configuration is below the baseline of its
benchmark. Per checkpoint, 43 of 45 are below; the two exceptions are
Qwen2-VL + LoRA seed 0 and Qwen2-VL + DoRA seed 1 on VQA-RAD, one question
above it (62.50%).

## Logistic fit of analysis C with clustered intervals

    V4_CACHE=<embedding cache> python -m src.featurespace.logistic_cluster

Output: `docs/camera_ready/logistic_cluster.json`. The fit of flip against donor
distance per checkpoint and seed, with the Wald interval and a percentile
interval from 2,000 bootstrap resamples of the original images. The Wald fit
reproduces `docs/v4/analysis_c.json` in all 45 cells. The criterion fixed in
advance (a positive coefficient whose interval excludes zero, in all three
seeds) holds in 14 of 15 cells under both intervals for MedSigLIP, with the
same exception, one seed of the classifier on VQA-RAD. Under the colour
histogram it holds in 11 of 15 cells (Wald) and 7 of 15 (clustered).

This script needs the C0 test embeddings of the three benchmarks, like
`analysis_c.py`; the other two need nothing beyond this repository.
