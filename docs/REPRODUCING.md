# Reproducing the results

Three levels, in increasing cost. Level 1 needs nothing but this repository;
level 3 needs the datasets and a GPU.

---

## Level 1 — verify the statistics from the shipped predictions

**Cost: seconds. No GPU, no dataset download.**

Every per-item prediction is in `results/`, so every number in Table 1, every
McNemar test and every CCR value can be recomputed here.

```bash
pip install -r requirements.txt

# paired statistics (VDD, bootstrap CI, McNemar, Cohen's g) for every
# model x seed x condition that does not have them yet
python -m src.stats.backfill_vs_c0 --results-dir results

# aggregate the three seeds of each configuration
python -m src.analysis.build_multiseed_matrix --results-dir results --output-dir results/multiseed
python -m src.analysis.build_multiseed_summary_matrix

# render the table
python -m src.stats.report \
    --multiseed-matrix results/summary_matrix_multiseed.json \
    --output results/report_multiseed.md
```

The scripts are idempotent: they skip whatever is already computed, so running
them over the shipped `results/` should report zero new files and leave the
numbers unchanged. That in itself is a check.

### Restricted swaps: conditional vs. unconditional VDD

For C1, C2 and the parametric corruptions (C4_OCC_*, C4_NOISE_* -- the
paper's C3 and C4) the intervention touches every test item, so the plain
accuracy difference *is* the VDD, and `results/multiseed/` already holds the
values reported in Table 1.

**C5 and C6 are partial by construction.** Images without an eligible finding
label keep their original image, and a question whose image did not change
produces the same prediction under C0 and under the intervention. Table 1
therefore reports VDD restricted to the subset the intervention actually
touches (Section 3.3); the coverage is printed in each group header of the
table (on Kvasir-VQA-x1: 8,934 questions for C5, 10,265 for C6, out of 10,779).

The aggregates in `results/multiseed/` are **unconditional** — computed over
all 10,779 questions. They are smaller than Table 1 by exactly the coverage
factor |S_T|/N. To reproduce the published numbers:

```bash
python -m src.stats.conditional_vdd --model M3AE_KVASIR          --dataset KVASIR --condition C5
python -m src.stats.conditional_vdd --model SMOLVLM_LORA_KVASIR  --dataset KVASIR --condition C5
```

Expected for the first: 8,934 questions in S_T, VDD 46.4 +/- 0.3, McNemar
significant in 3/3 seeds — the M3AE row of Table 1.

### Counterfactual Consistency Rate

```bash
python -m src.analysis.ccr --model M3AE_KVASIR_seed0 --dataset KVASIR --condition C5
python -m src.analysis.ccr --model SMOLVLM_LORA_KVASIR_seed0 --dataset KVASIR --condition C5
```

Expected: 5,216 informative questions per seed on Kvasir-VQA-x1 under C5, and a
CCR of 79.3% to 80.8% across the five fine-tuned configurations, matching
Table 1. For the slices and subsets of Section 5.4:

```bash
python -m src.analysis.ccr_slices   --dataset KVASIR --condition C5
python -m src.analysis.ccr_sameproc --subset same
```

This uses `data/swapmap.json`, which ships precomputed, so no intervention
data has to be regenerated.

### Figures

```bash
python src/analysis/make_figs.py     # writes to figures/
```

### Noise floor

See [NOISE_FLOOR.md](NOISE_FLOOR.md).

---

## Level 2 — rebuild the clean split and the interventions

**Cost: minutes to hours. Needs dataset downloads, no GPU.**

### Verify the leakage claim

```bash
python -m src.prepro.verify_leakage
```

See [LEAKAGE.md](LEAKAGE.md) for the expected output.

### Rebuild the leakage-free split

```bash
python -m src.prepro.build_kvasir_clean_split --output-dir data/finetune_arrows --seed 42
```

Produces the three arrow files plus `kvasir_clean_split_map.csv`. Compare that
file against the shipped `data/kvasir_clean_split/split_map.csv`: they must
agree image by image. The zero-overlap property is asserted by

```bash
pytest tests/prepro/test_build_kvasir_clean_split.py
```

### Regenerate the intervention ladder

Each condition is materialized once per dataset with a fixed seed, so every
model answers exactly the same perturbed items.

```bash
ARROWS=data/finetune_arrows

# C1 - image replaced by a constant frame
python -m src.interventions.make_c1_blank_images \
    --input $ARROWS/vqa_kvasir_test_clean.arrow --output $ARROWS/vqa_kvasir_test_c1.arrow

# C2 - free swap (fixed-seed derangement over the whole test set)
python -m src.interventions.make_c2_shuffled_pairs \
    --input $ARROWS/vqa_kvasir_test_clean.arrow --output $ARROWS/vqa_kvasir_test_c2.arrow

# Parametric corruption, two families x three severities.
# C4_OCC_* is the paper's C3 (occlusion); C4_NOISE_* is the paper's C4 (noise).
# See the correspondence table in ../README.md.
for s in 0.25 0.50 0.75; do
  python -m src.interventions.make_c4_severity --kind occlusion --severity $s \
      --input $ARROWS/vqa_kvasir_test_clean.arrow --output $ARROWS/vqa_kvasir_test_c4_occ_${s/0./}.arrow
  python -m src.interventions.make_c4_severity --kind noise --severity $s \
      --input $ARROWS/vqa_kvasir_test_clean.arrow --output $ARROWS/vqa_kvasir_test_c4_noise_${s/0./}.arrow
done
```

C5 and C6 additionally need the domain and finding labels per image
(`src/prepro/build_kvasir_domain_membership.py`,
`src/prepro/build_slake_pathology_labels.py`,
`src/prepro/build_vqa_rad_organ_membership.py`); see each script's `--help`.

Rebuilding the swap map after regenerating the interventions:

```bash
python -m src.analysis.build_swapmap --arrow-dir $ARROWS --output data/swapmap.json
```

---

## Level 3 — retrain and re-run inference

**Cost: days of GPU time.**

### Training

The fine-tuning itself uses a standard LoRA/DoRA recipe; the exact
hyperparameters of all 12 model x dataset x PEFT combinations are in
`configs/`. Every fine-tuned configuration is trained with seeds 0, 1 and 2,
which control initialization of the trainable parameters, data order and
dropout.

Summary of the generative recipe (identical across datasets):

| | |
|---|---|
| Base models | `Qwen/Qwen2-VL-2B-Instruct`, `HuggingFaceTB/SmolVLM2-2.2B-Instruct` |
| PEFT | LoRA and DoRA, rank 16, scaling 32, adapters on `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Learning rate | 2e-4, warm-up ratio 0.05 |
| Epochs | 3 |
| Batch | 4, gradient accumulation 8 |
| Max sequence length | 2,048 tokens |

The M3AE classifier follows the training recipe of its original authors. The
zero-shot model (`UCSC-VLAA/MedVLThinker-7B-RL_PMC`) is not trained.

### Inference

```bash
python -m src.inference.extract_predictions_circle \
    --model SMOLVLM --peft LORA --dataset KVASIR --condition C5 \
    --condition-arrow $ARROWS/vqa_kvasir_test_c5.arrow \
    --adapter-dir path/to/adapter \
    --output results/SMOLVLM_LORA_KVASIR_seed0_C5.json
```

`extract_predictions_m3ae.py` does the same for the classifier, which needs its
own older dependency stack.

The M0 question-only baseline needs no GPU. It reads the dataset's own
train/test JSON files (it never touches an image), fits TF-IDF plus logistic
regression on the training questions, and is applied to closed questions only:

```bash
python -m src.baselines.question_only \
    --dataset SLAKE --condition C0 \
    --train_json path/to/slake/train.json \
    --test_json  path/to/slake/test.json \
    --output results/M0_question_only_SLAKE_C0.json
```

Because it never sees the image, its predictions are identical in every
condition and its VDD is zero by construction — which is why it serves as the
floor of the analysis.

### A note on determinism

Decoding is greedy in effect for all three generative models (see
[NOISE_FLOOR.md](NOISE_FLOOR.md)), so re-running inference in the *same*
environment reproduces predictions exactly. Across environments, expect 2.7% to
5.9% of predictions to differ without any bias in either direction.
