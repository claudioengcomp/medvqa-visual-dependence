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
| §3.6 | Multi-seed aggregation (mean ± sd; *k*/3 significance) | `src/stats/compute_multiseed.py` |
| §4.2 | Training hyperparameters of every fine-tuned configuration | `configs/` |
| §5, Table 1 | All numbers | `results/` (see below) |
| Figures 2–5 | Plots | `src/analysis/make_figs.py` + `data/fig_data.json` |
| §5.6 | Noise floor of the protocol | `docs/NOISE_FLOOR.md` |

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
