# Correct Answers to the Wrong Image
### On What Image-Substitution Measures Count as Error in Medical VQA

Companion repository for the paper of the same name — code, the leakage-free
split and the per-item predictions.

It contains everything needed to verify the paper's claims: the leakage-free
split of Kvasir-VQA-x1, the intervention code that defines the C1-C6 ladder,
the statistics (VDD, McNemar, Cohen's *g*, bootstrap, multi-seed aggregation),
the Counterfactual Consistency Rate, the coverage analysis that computes phi
without running any model, and **the per-item predictions of every model, seed
and condition** — 557 prediction files, 464 paired comparisons against the
intact condition (493 comparison files on disk, 29 of them the question-only
baseline, whose predictions do not change and which has b = c = 0 in every
condition), and 145 multi-seed aggregates. Every count here is asserted by
`tests/test_published_counts.py`, against the directory rather than prose.

No dataset images are redistributed. Only identifiers, model predictions and
code.

---

## Quick start

```bash
pip install -r requirements.txt
pytest                      # the protocol's unit tests, including the zero-overlap test
```

Nothing above needs a GPU or a dataset download.

---

## Where each claim of the paper lives

| Paper | Claim | Where |
|---|---|---|
| Abstract, §1, §4.1 | 3,821 of 4,058 official Kvasir-VQA-x1 test images appear in its training set | `docs/LEAKAGE.md` |
| Abstract, §1, §4.1 | Release of a stratified, leakage-free split | `data/kvasir_clean_split/` |
| §4.1 | *"a test asserts that no image appears in two partitions"* | `tests/prepro/test_build_kvasir_clean_split.py` |
| §3.2 | The intervention ladder C1-C6 (Figure 2) | `src/interventions/` |
| §3.3 | VDD over the whole test set | `src/stats/compute_vdd.py` |
| §3.3 | VDD **restricted to the questions an intervention touches** (what Table 3 reports for C5/C6) | `src/stats/conditional_vdd.py` |
| §3.7 | McNemar, Cohen's *g*, paired bootstrap CI | `src/stats/compute_vdd.py` |
| §3.4 | Counterfactual Consistency Rate and the fraction phi | `src/analysis/ccr.py` |
| §3.4, §5.1 | **The identity**: the CCR is the accuracy under C0 on the donor items, verified cell by cell | `src/analysis/identity.py` |
| §3.5, §5.2 | The swap cost as a function of phi and accuracy | `src/analysis/ccr_slices.py`, `src/analysis/ccr_sameproc.py` |
| §3.6, §5.5 | Complementary-pair accuracy and the departure from independence | `src/analysis/complementary_pairs.py` |
| **§5.3, Figure 4** | **Coverage analysis on six benchmarks: where phi is estimable and at what scale, computed without any model** | `src/analysis/coverage.py`, `docs/COVERAGE.md` |
| §5.3 | Downloading the external benchmarks of the coverage analysis | `src/analysis/fetch_coverage_data.py` |
| §5.4 | Permutation test and Welch's ANOVA on the spread among configurations | `src/stats/spread_tests.py` |
| §5.4 | Monotonicity of VDD with severity under occlusion and noise | `src/analysis/severity.py` |
| §5.4 | The question-only floor (M0) on SLAKE and VQA-RAD, closed questions only | `src/baselines/question_only.py` |
| §5.4 | The question-only floor (M0) on Kvasir-VQA-x1 — 65.74% (7,086/10,779), every question | `src/baselines/question_only_kvasir.py` |
| §5.6 | The noise floor: same environment against across environments | `src/analysis/noise_floor.py` |
| §5.1 | CCR by answer slice — aggregate, single-valued, list | `src/analysis/ccr_slices.py` |
| §5.2 | The same-procedure and cross-procedure C5 subsets | `src/analysis/ccr_sameproc.py`, `src/analysis/subset_vdd.py` |
| §4.1 | Accuracy and VDD split by question type (official open/closed label) | `src/analysis/question_type.py` |
| §3.8 | Multi-seed aggregation (mean ± sd; *k*/3 significance) | `src/stats/compute_multiseed.py` |
| §4.2 | Training hyperparameters of every fine-tuned configuration | `configs/` |
| **Table 3** | The whole table, rebuilt from the predictions | `src/analysis/build_table1.py` |
| Table 3 | All numbers | `results/` (see below) |
| Figures 2-4 | Plots | `src/analysis/make_figs.py` + `data/fig_data.json` |
| Figures 2-4 | The data behind them, rebuilt from the predictions | `src/analysis/build_fig_data.py` |

The module is called `build_table1.py` because the main results table was
Table 1 in an earlier draft; it is Table 3 in the paper. The name is kept
because it appears in the provenance of generated files, and renaming it would
break the link between them and the code that produced them.

---

## One ruler for accuracy, VDD and the CCR

Every quantity in the paper is scored with the same rule: `answer_matches` from
`src/inference/process_text.py`, which compares answers after lowercasing and
normalizing whitespace and punctuation. It decides the `correct` field behind
accuracy and VDD, and it decides all three answer buckets of the CCR, including
the test of whether a swapped question is informative.

| Quantity, Kvasir-VQA-x1 under C5 | Value |
|---|---|
| CCR over all 5,216 informative questions per seed | **79.3 - 80.8%** |
| ... on the 3,935 with single-valued answers | **89.6 - 90.9%** |
| ... on the 1,281 whose answer is a list | **46.5 - 49.6%** |
| answers keeping the original reference | **2.6 - 4.4%** |
| same-procedure subset (2,026 questions per seed) | **82.4 - 84.2%** |
| standard deviation across seeds | up to **1.0 pp** |
| SLAKE under C6 (92 informative questions per seed) | 56.9 - 69.6% |
| VQA-RAD | fewer than four informative questions; not reportable |

### Amendment E7: the zero-shot row

The `MEDVLTHINKER_ZEROSHOT_*` files are re-scored under amendment E7 (dated
19 September 2026): on closed questions, those whose reference is `yes` or
`no`, the tokens `negative` and `positive` count as `no` and `yes`. Nothing
else is accepted. The rule changes only the `correct` field of the items it
touches (each one carries `"closed_synonym": "negative->no"` or
`"positive->yes"`) and the file-level `n_correct` and `accuracy`; the predicted
text is never edited, `answer_matches` is untouched, and every file that was
re-scored records it under `closed_synonyms_e7` with the number of items
flipped. The fine-tuned models and the question-only baseline never emit those
two tokens, so no other file moves. The `_vs_C0.json` comparisons of this row
are regenerated from the re-scored files; the Kvasir C6 file of this row is the
only one of the row without a `closed_synonyms_e7` field.

The rule never merges distinct reference answers of the same question: within a
question template, no two reference answers of the three benchmarks compare as
equal (0 of 260 reference strings in Kvasir-VQA-x1, 0 of 130 in SLAKE, 0 of 120
in VQA-RAD), so no prediction can match both the original reference and the
donor's, and the informative sets are exactly the denominators the tables
report. `tests/analysis/` pins those denominators.

---

## Condition labels: paper vs. code

**Read this before matching any number to the paper.** The paper's ladder has
seven rungs; the code uses six condition prefixes, because the two parametric
corruptions share one generator and one naming scheme.

| Paper | Code label | What it does |
|---|---|---|
| C0 | `C0` | intact — the original image |
| C1 | `C1` | image replaced by a constant mid-gray frame |
| C2 | `C2` | free swap — fixed-seed derangement over the whole test set |
| **C3** | **`C4_OCC_25`, `C4_OCC_50`, `C4_OCC_75`** | occlusion — centered black square covering 25/50/75% of the image **area** |
| **C4** | **`C4_NOISE_25`, `C4_NOISE_50`, `C4_NOISE_75`** | noise — zero-mean Gaussian, σ = 25/50/75% of the 8-bit dynamic range |
| C5 | `C5` | restricted swap — donor of the same imaging domain, **different** finding |
| C6 | `C6` | restricted swap — donor of the same imaging domain, finding **free** |

The two rows in bold are the only place where the labels diverge. The paper
separates occlusion (C3) from noise (C4) as distinct rungs; the code generates
both from a single script (`src/interventions/make_c4_severity.py --kind
occlusion|noise`) and labels both under the `C4_` prefix, distinguished by
`_OCC_` and `_NOISE_`.

**The code labels are frozen on purpose.** They appear in the names of 612
result files, inside the `"condition"` field of 306 of them, in the keys of
`data/swapmap.json`, and in the whitelist that validates every result record.
Renaming them would break the correspondence with the data already generated,
so this table — not a rename — is the mapping of record.

Consequently, **every filename and every `"condition"` field in `results/`
follows the code labels above**, never the paper's C3/C4 numbering. A file
named `SMOLVLM_LORA_KVASIR_seed0_C4_OCC_50.json` holds the paper's **C3 at
50% severity**.

Everywhere else the two agree: C0, C1, C2, C5 and C6 are the same string in
both.

### One quantity is named differently too

| Paper | Code label | What it is |
|---|---|---|
| VDD | `"ldv"` — a JSON key in `results/*_vs_C0.json`, and the label printed by `src/stats/compute_vdd.py` | The **Visual Dependency Drop**. Same quantity, same values, under the key name the generated files carry. |

As with the condition labels, the correspondence is this table rather than a
rename, so the shipped result files are exactly what the runs produced.

---

## Reproducing the numbers

Three levels, in increasing cost.

### 1. Read the aggregated results (no computation)

```
results/multiseed/                     mean ± sd of accuracy and VDD, k/3 McNemar,
                                       Cohen's g, per configuration and condition
results/summary_matrix_multiseed.json  the same, consolidated
results/report_multiseed.md            rendered as a table
```

### 2. Recompute the statistics from the raw predictions

Every per-item prediction is in `results/`, one file per model × seed ×
condition:

```
results/<MODEL>_<PEFT>_<DATASET>[_seed<N>]_<CONDITION>.json
```

with the schema

```json
{"model": "...", "dataset": "...", "condition": "C0",
 "accuracy": 0.0, "n": 0, "n_correct": 0,
 "predictions": [{"qid": 0, "predicted": "...", "gold": "...", "correct": true}]}
```

Rebuild every paired comparison and aggregate:

```bash
python -m src.stats.backfill_vs_c0 --results-dir results
python -m src.analysis.build_multiseed_matrix --results-dir results --output-dir results/multiseed
python -m src.analysis.build_multiseed_summary_matrix
python -m src.stats.report --multiseed-matrix results/summary_matrix_multiseed.json --output results/report_multiseed.md
```

Counterfactual Consistency Rate for one configuration:

```bash
python -m src.analysis.ccr --model SMOLVLM_LORA_KVASIR_seed0 --dataset KVASIR --condition C5
```

The slices and subsets of Section 5.4, over every configuration and seed:

```bash
python -m src.analysis.ccr_slices   --dataset KVASIR --condition C5
python -m src.analysis.ccr_slices   --dataset SLAKE  --condition C6
python -m src.analysis.ccr_sameproc --subset same
```

`data/swapmap.json` ships precomputed, so CCR is verifiable without
regenerating any intervention data.

**Corrected donor map (Kvasir-VQA-x1, C5 and C6).** The Kvasir block of
`data/swapmap.json` carries the corrected donor assignments of
`results/C5_manifest_KVASIR_clean.json` (611 donors) and
`results/C6_manifest_KVASIR_clean.json` (777 donors), the manifests the paper
counts on: |S_T| = 8,934, |I| = 5,216, |U| = 2,886, |N| = 832 under C5 and
10,265 questions under C6. The earlier map, recovered from the intervened
arrows by matching image bytes, listed six phantom rows (six images with one
question each that received no donor in the manifests) and one image whose
donor differs; it is kept unchanged as `data/swapmap_v3.json` for the record.
The six removed rows change the informative set by two items (qids 2000405 and
2000414) and the CCR range by at most 0.1 pp.
Quantities the submitted paper computed on the earlier map move by at most
0.1 pp under the corrected one, and four of them change at one decimal:
VDD(C5) of Qwen2-VL DoRA 46.9 -> 47.0 and VDD(C6) of Qwen2-VL LoRA
42.3 -> 42.4 (Table 3), the C6 spread among configurations 1.66 -> 1.67 pp
(Section 5.4) and the largest joint pair accuracy 0.668 -> 0.669
(Section 5.5); the same-procedure partition of Section 5.2 has 240 same and
369 cross pairs with 2, not 8, pairs lacking the procedure label. The tests in
`tests/analysis/` pin the corrected values.

### 3. Regenerate everything from the datasets (needs downloads and a GPU)

See `docs/REPRODUCING.md`.

---

## Layout

```
data/
  kvasir_clean_split/   split_map.csv, qid_map.csv — the leakage-free split
  swapmap.json          donor of every swapped image + reference answers (for CCR); corrected Kvasir C5/C6 map
  swapmap_v3.json       the earlier map recovered from the arrows (six phantom rows; kept for the record)
  fig_data.json         aggregated data behind Figures 2–5
src/
  prepro/               construction of the clean split and of the domain/finding labels
  interventions/        the C1–C6 ladder
  stats/                VDD, McNemar, Cohen's g, bootstrap, multi-seed
  analysis/             CCR, VDA, summary matrices, figures
  inference/            running a model on one condition
  baselines/            M0, the question-only baseline
configs/                training hyperparameters (12 model × dataset × PEFT combinations)
results/                per-item predictions, paired statistics, swap manifests
tests/                  unit tests, including the zero-overlap assertion
docs/                   reproduction guide, leakage verification, noise floor
```

## A note on naming

The paper calls the metric **VDD** (Visual Dependency Drop). In the JSON
result files the key is historically named `"ldv"` and holds the VDD value.
The key was not renamed because doing so would invalidate the 500+ result
files already generated; the code and documentation use VDD throughout.

## License

Code is released under the MIT License (`LICENSE`). The derived data in
`data/` and `results/` is subject to the terms of the original datasets —
see `DATA_NOTICE.md`.
