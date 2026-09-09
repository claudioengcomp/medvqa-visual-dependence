# How Much Do They Really See? — code, clean split and predictions

Companion repository for the paper *"How Much Do They Really See? Quantifying
Visual Dependence in Med-VQA with Paired Image Interventions"*.

It contains everything needed to verify the paper's claims: the leakage-free
split of Kvasir-VQA-x1, the intervention code that defines the C1–C6 ladder,
the statistics (VDD, McNemar, Cohen's *g*, bootstrap, multi-seed aggregation),
the Counterfactual Consistency Rate, and **the per-item predictions of every
model, seed and condition** — 549 prediction files, 492 paired comparisons
against the intact condition, and 145 multi-seed aggregates.

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
| Abstract, §1, §4.1 | 94% of the official Kvasir-VQA-x1 test images appear in its training set | `docs/LEAKAGE.md` |
| Abstract, §1, §4.1 | Release of a stratified, leakage-free split | `data/kvasir_clean_split/` |
| §4.1 | *"a test asserts that no image appears in two partitions"* | `tests/prepro/test_build_kvasir_clean_split.py` |
| §3.2 | The intervention ladder C1, C2, C4, C5, C6 | `src/interventions/` |
| §3.3 | VDD — accuracy drop over the whole test set | `src/stats/compute_vdd.py` |
| §3.3 | VDD **restricted to the questions an intervention touches** (this is what Table 1 reports for C5/C6) | `src/stats/conditional_vdd.py` |
| §3.4 | McNemar, Cohen's *g*, paired bootstrap CI | `src/stats/compute_vdd.py` |
| §3.5 | Counterfactual Consistency Rate | `src/analysis/ccr.py` |
| §5.4 | CCR by answer slice — aggregate, single-valued, list | `src/analysis/ccr_slices.py` |
| §5.3, §5.4 | The same-procedure and cross-procedure C5 subsets | `src/analysis/ccr_sameproc.py` |
| §3.6 | Multi-seed aggregation (mean ± sd; *k*/3 significance) | `src/stats/compute_multiseed.py` |
| §4.2 | Training hyperparameters of every fine-tuned configuration | `configs/` |
| §5, Table 1 | All numbers | `results/` (see below) |
| Figures 2–5 | Plots | `src/analysis/make_figs.py` + `data/fig_data.json` |
| Figure 4 | The data behind it, rebuilt from the predictions | `src/analysis/build_ccr_fig_data.py` |
| §5.6 | Noise floor of the protocol | `docs/NOISE_FLOOR.md` |

---

## One ruler for accuracy, VDD and the CCR

An audit of this code found that the CCR was scored with a normalization of its
own, which keeps punctuation, while accuracy and VDD used `answer_matches` from
`src/inference/process_text.py`. Two rulers in the same table. The CCR is now
scored with `answer_matches` everywhere — for the informative test and for all
three answer buckets — and the numbers below are the ones this repository
produces. Accuracy, VDD, Cohen's *g*, McNemar and the bootstrap intervals are
untouched: the correction is not on the path that produces the `correct` field,
and the 25 cells of Table 1 are reproduced digit for digit before and after it.

| Quantity, Kvasir-VQA-x1 under C5 | Value |
|---|---|
| CCR over all 5,218 informative questions per seed | **79.3 – 80.7%** |
| … on the 3,936 with single-valued answers | **89.7 – 90.9%** |
| … on the 1,282 whose answer is a list | **46.6 – 49.6%** |
| answers keeping the original reference | **2.6 – 4.4%** |
| same-procedure subset (2,026 questions per seed) | **82.4 – 84.2%** |
| standard deviation across seeds | up to **1.0 pp** |
| SLAKE under C6 (92 informative questions per seed) | 56.9 – 69.6%, unchanged by the correction |
| VQA-RAD | fewer than four informative questions; not reportable |

Two checks bound what the correction can do. It cannot merge the original
reference with the donor's and turn "followed the donor" into a false positive:
0 of 3,936 single-valued and 0 of 1,282 list questions change status. And it
cannot merge two distinct reference answers of the same question: no such pair
exists in any question template of the three datasets (0 of 260 reference
strings in Kvasir-VQA-x1, 0 of 130 in SLAKE, 0 of 120 in VQA-RAD). The
informative sets are therefore identical under both rules, which is what makes
the before/after comparison one of like with like. `tests/analysis/` pins those
denominators.

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
| — | `"ldv"` (JSON key in `results/*_vs_C0.json`, and the label printed by `src/stats/compute_vdd.py`) | **the Visual Dependency Drop (VDD)** under its earlier working name. Same quantity, same values; the key was left unchanged so that the shipped result files stay byte-identical to the ones the runs produced. |
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

### 3. Regenerate everything from the datasets (needs downloads and a GPU)

See `docs/REPRODUCING.md`.

---

## Layout

```
data/
  kvasir_clean_split/   split_map.csv, qid_map.csv — the leakage-free split
  swapmap.json       donor of every swapped image + reference answers (for CCR)
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
