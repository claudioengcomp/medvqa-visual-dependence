# Destabilized or Redirected
### What the Standard Probes of Visual Dependence Measure in Med-VQA

Companion repository for the paper of the same name: code, the image-disjoint
split of Kvasir-VQA-x1, the per-item predictions of every model, seed and
condition, the swap manifests, the pre-registration and the feature-space
analyses.

It contains everything needed to verify the paper's claims: the image-disjoint
split, the intervention code that defines the conditions C0-C7, the statistics
(VDD, McNemar, Cohen's *g*, bootstrap, multi-seed aggregation), the identity of
the swap cost and its decomposition (Eq. 2), the coverage analysis that computes
phi without running any model, the five neutral rulers and the analyses of the
feature space, and **the per-item predictions of every model, seed and
condition** — 557 prediction files, 464 paired comparisons against the intact
condition (493 comparison files on disk, 29 of them the question-only baseline,
whose predictions do not change and which has b = c = 0 in every condition),
and 145 multi-seed aggregates. Every count here is asserted by
`tests/test_published_counts.py`, against the directory rather than prose.

No dataset images are redistributed. Only identifiers, hashes, model
predictions, derived statistics and code (`DATA_NOTICE.md`).

---

## Quick start

```bash
pip install -r requirements.txt
pytest --ignore=tests/test_circle_data_bridge.py
```

Nothing above needs a GPU or a dataset download. `tests/test_circle_data_bridge.py`
exercises the data bridge of the inference code and needs the `datasets`
package; the remaining tests, including the zero-overlap assertion on the
split, run without it.

---

## Where each claim of the paper lives

| Paper | Claim | Where |
|---|---|---|
| §3.1 | The conditions C0-C7 (Figure 2) | `src/interventions/`, `src/featurespace/make_c7.py` |
| §3.2, §5.2 | The swap cost: the identity of the CCR and its decomposition (Eq. 2) over S_T = I + U + N | `src/analysis/identity.py`, `src/analysis/ccr.py` |
| §3.3, Table 1 | The five neutral rulers, validated by leave-one-out kNN on the intact test images | `src/featurespace/g0_embed.py`, `docs/v4/g0_results.json` |
| §3.3, §5.1 | The rulers organise the training bank as well; distance to the bank for training (leave-one-out) and test (C0) images | `src/featurespace/train_rulers.py`, `docs/v4/TRAIN_RULERS.md` |
| §3.3 | Model-derived rulers (frozen and adapted encoders) against the neutral ones | `docs/v4/model_rulers.json` |
| §3.3 | Analyses A-C of the feature space and their gates G0-G3 | `src/featurespace/analysis_a.py`, `analysis_b.py`, `analysis_c.py`, `g*_check.py` |
| §3.3 | Pre-registration of the feature-space analyses and its dated amendments | `docs/preregistration/`, `docs/PROVENANCE.md` |
| §4.1 | 3,821 of 4,058 official Kvasir-VQA-x1 test images appear in its training set | `docs/LEAKAGE.md` |
| §4.1 | Release of a stratified, image-disjoint split, and *"a test asserts that no image appears in two partitions"* | `data/kvasir_clean_split/`, `tests/prepro/test_build_kvasir_clean_split.py` |
| §4.1 | The content audit of the split: exact copies and pHash neighbours between test and training | `src/featurespace/dup_audit.py`, `docs/v4/duplicate_audit.json` |
| §4.2 | Training hyperparameters of every fine-tuned configuration | `configs/` |
| §4.3 | Counts: 435 paired comparisons of the fine-tuned rows and 464 in all | `tests/test_published_counts.py` |
| §4.3 | The per-row manifest of every swap: donor and SHA-256 of the bytes each row received | `results/manifests/` |
| §5.1, Table 3 | Destabilisation: accuracy under C0 and VDD under C1, C2, C5, C6; the whole table, rebuilt from the predictions | `src/analysis/build_table1.py`, `results/` |
| §5.1 | VDD over the whole test set, and **restricted to the questions an intervention touches** (what Table 3 reports for C5 and C6) | `src/stats/compute_vdd.py`, `src/stats/conditional_vdd.py` |
| §5.1 | McNemar, Cohen's *g*, paired bootstrap CI; multi-seed aggregation (mean ± sd; *k*/3 significance) | `src/stats/compute_vdd.py`, `src/stats/compute_multiseed.py` |
| §5.1, §5.4 | The question-only floor (M0): SLAKE and VQA-RAD on closed questions; Kvasir-VQA-x1 on every question, 65.74% (7,086/10,779) | `src/baselines/question_only.py`, `src/baselines/question_only_kvasir.py` |
| §5.2, Table 4 | The Counterfactual Consistency Rate, the fraction phi, and phi on six benchmarks without running any model | `src/analysis/ccr.py`, `src/analysis/coverage.py`, `docs/COVERAGE.md` |
| §5.2 | CCR by answer slice and by procedure (same- and cross-procedure donors) | `src/analysis/ccr_slices.py`, `src/analysis/ccr_sameproc.py`, `src/analysis/subset_vdd.py` |
| §5.3, Figures 4 and 5 | Distance to the donor: VDD against the kNN distance to the training bank, and flip rate against the distance to the donor | `src/featurespace/fig3_vdd_vs_distance.py`, `src/featurespace/fig5_flip_vs_distance.py`, `docs/v4/` |
| §5.4, Table 6 | The hard negative: the swap that moves the least in the rulers, and the spread among configurations (permutation test, Welch's ANOVA) | `src/featurespace/analysis_e2.py`, `src/stats/spread_tests.py` |
| §5.4 | Monotonicity of VDD with severity under occlusion and noise | `src/analysis/severity.py` |
| §5.5 | Overlap: complementary-pair accuracy and the departure from independence | `src/analysis/complementary_pairs.py` |
| §5.6 | Limitations: the noise floor of the same environment against across environments, and of batch composition | `src/analysis/noise_floor.py`, `docs/NOISE_FLOOR.md` |
| Figure 3 | The colour histogram as a ruler (the floor a ruler must beat) | `src/featurespace/g0_embed.py`, `docs/v4/g0_results.json` |
| every number | The sheet the paper's numbers are read from, one line per quantity with the JSON that carries it | `docs/v4/numbers_v4.md`, `src/featurespace/make_numbers.py` |

The module is called `build_table1.py` because the main results table was
Table 1 in an earlier draft; it is Table 3 in the paper. The name is kept
because it appears in the provenance of generated files, and renaming it would
break the link between them and the code that produced them.

Sections of the paper: §3.1 conditions, §3.2 the cost of a swap, §3.3 rulers
and the analyses of the feature space, §4.1 split and content audit, §4.2
models, §4.3 counts, §5.1 destabilisation, §5.2 identity and Eq. 2, §5.3
distance to the donor, §5.4 the hard negative, §5.5 overlap, §5.6 limitations.
Tables: 1 rulers, 2 the roles of the conditions, 3 main results, 4 phi, 5 flip
rate, 6 the hard negative. Figures: 1 the two axes, 2 the ladder of conditions,
3 the colour ruler, 4 VDD against kNN distance, 5 flip rate against distance.

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

The rule never merges distinct reference answers of the same question: within a
question template, no two reference answers of the three benchmarks compare as
equal (0 of 260 reference strings in Kvasir-VQA-x1, 0 of 130 in SLAKE, 0 of 120
in VQA-RAD), so no prediction can match both the original reference and the
donor's, and the informative sets are exactly the denominators the tables
report. `tests/analysis/` pins those denominators.

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
are regenerated from the re-scored files.

---

## Condition labels: paper vs. code

**Read this before matching any number to the paper.** The paper's ladder has
eight conditions, C0 to C7; the code uses seven condition prefixes, because the
two parametric corruptions share one generator and one naming scheme.

| Paper | Code label | What it does |
|---|---|---|
| C0 | `C0` | intact — the original image |
| C1 | `C1` | image replaced by a constant mid-gray frame |
| C2 | `C2` | free swap — fixed-seed derangement over the whole test set |
| **C3** | **`C4_OCC_25`, `C4_OCC_50`, `C4_OCC_75`** | occlusion — centered black square covering 25/50/75% of the image **area** |
| **C4** | **`C4_NOISE_25`, `C4_NOISE_50`, `C4_NOISE_75`** | noise — zero-mean Gaussian, σ = 25/50/75% of the 8-bit dynamic range |
| C5 | `C5` | restricted swap — donor of the same imaging domain, **different** finding |
| C6 | `C6` | restricted swap — donor of the same imaging domain, finding **free** |
| C7 | `C7` | patch shuffle — the 32×32 patches of the image are permuted with a fixed seed; the remainder strips narrower than a patch stay in place, every pixel is conserved; stored as lossless PNG at the original resolution (`src/featurespace/make_c7.py`, manifest in `results/manifests/C7_manifest.json`). C7 enters only the feature-space analyses (§3.3, §5.3), never Table 3 |

The two rows in bold are the only place where the labels diverge. The paper
separates occlusion (C3) from noise (C4) as distinct rungs; the code generates
both from a single script (`src/interventions/make_c4_severity.py --kind
occlusion|noise`) and labels both under the `C4_` prefix, distinguished by
`_OCC_` and `_NOISE_`.

**The code labels are frozen on purpose.** They appear in the names of the
result files, inside their `"condition"` field, in the keys of
`data/swapmap.json`, and in the whitelist that validates every result record.
Renaming them would break the correspondence with the data already generated,
so this table — not a rename — is the mapping of record.

Consequently, **every filename and every `"condition"` field in `results/`
follows the code labels above**, never the paper's C3/C4 numbering. A file
named `SMOLVLM_LORA_KVASIR_seed0_C4_OCC_50.json` holds the paper's **C3 at
50% severity**.

Everywhere else the two agree: C0, C1, C2, C5, C6 and C7 are the same string in
both.

### One quantity is named differently too

| Paper | Code label | What it is |
|---|---|---|
| VDD | `"ldv"` — a JSON key in `results/*_vs_C0.json`, and the label printed by `src/stats/compute_vdd.py` | The **Visual Dependency Drop**. Same quantity, same values, under the key name the generated files carry. |

As with the condition labels, the correspondence is this table rather than a
rename, so the shipped result files are exactly what the runs produced.

---

## Models and the visual encoder

The training hyperparameters of the twelve fine-tuned configurations are in
`configs/` (rank 16, alpha 32, learning rate 2e-4, 3 epochs, batch size 4 with
gradient accumulation of 4 on Kvasir-VQA-x1 and 8 on SLAKE and VQA-RAD). The
adapters target `q_proj`, `k_proj`, `v_proj` and `o_proj`. In SmolVLM2 the first
three names also match the attention of the SigLIP vision tower, so its visual encoder is
**adapted**; in Qwen2-VL the tower's attention is a fused `attn.qkv` /
`attn.proj` that the names do not match, so its visual encoder stays
**frozen**. The classifier (M3AE) is fine-tuned end to end. This is the
difference Section 4.2 and the model-derived rulers of Section 3.3 rely on.

---

## The split and its content audit

The official release of Kvasir-VQA-x1 assigns image-question pairs to
partitions, and 3,821 of its 4,058 test images also appear in training
(`docs/LEAKAGE.md`). The split released in `data/kvasir_clean_split/` is
**image-disjoint**: no image identifier appears in two partitions, and a test
asserts it. Disjoint identifiers are not disjoint content. The audit the paper
reports (§4.1; `docs/v4/duplicate_audit.json`, `docs/v4/nn1_to_bank.json`)
finds, between the test images of this split and its training and validation
images, 36 exact byte copies and 351 near neighbours at a perceptual-hash
distance of at most 4, out of 1,291 test images; in VQA-RAD all 203 test images
have a byte-identical copy in training; in SLAKE 24 of the 96 test images have
a perceptual-hash neighbour in training or validation. The distances of §5.3
and the rulers of §3.3 are computed with these images in place; the audit is
what makes that visible.

---

## Reproducing the numbers

Three levels, in increasing cost.

### 1. Read the aggregated results (no computation)

```
results/multiseed/                     mean ± sd of accuracy and VDD, k/3 McNemar,
                                       Cohen's g, per configuration and condition
results/summary_matrix_multiseed.json  the same, consolidated
results/report_multiseed.md            rendered as a table
docs/v4/numbers_v4.md                  every number of the paper, one line each,
                                       with the JSON it is read from
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

Counterfactual Consistency Rate and the identity of Section 5.2 for one
configuration:

```bash
python -m src.analysis.ccr --model SMOLVLM_LORA_KVASIR_seed0 --dataset KVASIR --condition C5
python -m src.analysis.identity --dataset KVASIR --condition C5
```

The slices and subsets of Section 5.2, over every configuration and seed:

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

See `docs/REPRODUCING.md`. The feature-space layer (`src/featurespace/`) reads
the embeddings from a local cache that is not redistributed; the JSONs it
produced are in `docs/v4/`.

---

## Layout

```
data/
  kvasir_clean_split/   split_map.csv, qid_map.csv — the image-disjoint split
  swapmap.json          donor of every swapped image + reference answers (for CCR); corrected Kvasir C5/C6 map
  swapmap_v3.json       the earlier map recovered from the arrows (six phantom rows; kept for the record)
  fig_data.json         aggregated data behind the figures of an earlier draft (kept; the paper's figures read docs/v4/)
src/
  prepro/               construction of the image-disjoint split and of the domain/finding labels
  interventions/        the conditions C1-C6
  featurespace/         rulers, embeddings, C7, the analyses A-E of the feature space, gates, audits, the numbers sheet
  stats/                VDD, McNemar, Cohen's g, bootstrap, multi-seed, spread tests
  analysis/             CCR, identity, complementary pairs, coverage, summary matrices
  inference/            running a model on one condition
  baselines/            M0, the question-only baseline
configs/                training hyperparameters (12 model × dataset × PEFT combinations)
results/                per-item predictions, paired statistics, swap manifests
results/manifests/      per-row manifests of every swap (ids and SHA-256), C7 manifest
docs/                   reproduction guide, leakage verification, noise floor, coverage
docs/preregistration/   the pre-registration of the feature-space analyses and its amendments
docs/v4/                the JSONs of the feature-space analyses and the numbers sheet
tests/                  unit tests, including the zero-overlap assertion
figures/                the figures of the paper and the additional material
```

`src/analysis/vda.py` (a harmonic mean of accuracy and VDD from an earlier
draft) is kept because the summary matrices carry its column; it is **not used
in the paper**.

## A note on naming

The paper calls the metric **VDD** (Visual Dependency Drop). In the JSON
result files the key is historically named `"ldv"` and holds the VDD value.
The key was not renamed because doing so would invalidate the 500+ result
files already generated; the code and documentation use VDD throughout.

## License

Code is released under the MIT License (`LICENSE`). The derived data in
`data/`, `results/` and `docs/v4/` is subject to the terms of the original
datasets — see `DATA_NOTICE.md`.
