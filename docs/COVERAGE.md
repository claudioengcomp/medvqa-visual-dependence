# Reproducing the coverage analysis (Section 5.2)

The coverage analysis answers, for a benchmark and a swap, **where φ is
estimable and at what scale**. It runs **no model** and needs **no GPU**: φ and
|V| are properties of the test set and of the swap, computed from questions and
reference answers alone.

    V_T   swapped questions whose donor image carries the SAME question
    I_T   the subset of V_T whose reference answer differs
    φ     |I_T| / |V_T|

Everything is in `src/analysis/coverage.py`, and the scoring rule is imported
from `src/analysis/ccr.py` - the same `question_key` / `answers_match` pair that
decides accuracy, VDD and the CCR. No normalization is defined in the coverage
module.

## The three benchmarks that ship here

SLAKE, VQA-RAD and Kvasir-VQA-x1 are read from `data/swapmap.json`, which is in
this repository. Nothing to download:

```bash
python -m src.analysis.coverage --all --skip-missing --draws 1000
```

This reproduces, for those three, the duplication table, the blocking
validation, the collision counts and the φ rows of the table below.

## φ at full scale and at 100 items

Estimate and 95% interval over 1,000 derangements of the test items, on the
full test set and on subsamples of 100 items, with the fraction of draws below
the reporting floor `|V| >= 20`. A dash: every draw, or nearly every draw, is
below the floor.

| Benchmark | φ, full test set | φ, 100 items |
|---|---|---|
| Kvasir-VQA-x1 | 48.7 [47.9, 49.6] | 48.6 [37.5, 60.7] (0% below) |
| SLAKE | 59.5 [51.5, 67.8] | - (98% below) |
| VQA-RAD | - | - |
| PathVQA | 94.3 [93.2, 95.4] | 94.5 (14% below) |
| PMC-VQA (both files) | - | - |
| ProbMed | 40.4 [39.7, 41.2] | 40.3 [20.7, 59.1] (10% below) |

φ is estimable on four of six benchmarks at full scale and on three at 100
items, where the interval no longer separates 40% from 60%. This is the table
the paper summarises in one sentence of Section 5.2.

## The three that must be obtained first

They are **not redistributed here**. Place the files in `data/coverage/`
(git-ignored) under exactly these names:

| Benchmark | Source | File(s) expected in `data/coverage/` |
|---|---|---|
| PathVQA | `flaviagiammarino/path-vqa` on the Hugging Face Hub, split `test` | `path-vqa__test-0000{0,1,2}-of-00003-*.parquet` |
| PMC-VQA | `RadGenome/PMC-VQA` | `PMC-VQA__test_clean.csv`, `PMC-VQA__test.csv` |
| ProbMed | `rippleripple/ProbMed`, file `test/test.json` | `ProbMed__test.json` |

**ProbMed is a gated release.** Access is granted by its maintainers; the paper
records this in Table 2. Run `huggingface-cli login` with an account that has
been granted access before downloading. Without the file, the module stops and
says so; pass `--skip-missing` to run the rest.

`python -m src.analysis.fetch_coverage_data` downloads what it can and prints
the exact instructions for what it cannot.

## Two choices that change the numbers

**The floor.** φ is reported only when `|V| >= 20` in **more than half** of the
draws. A cell where 3 of 1,000 draws clear the floor has a φ computed on those
3, and showing it would contradict the floor rule. The value stays in the JSON
under `phi_mean`; `reportable_draws` says how many draws it rests on. The table
below gives, for each row, the fraction of draws below the floor for this reason.

**Derangement over items, not over images.** Section 5.2 draws derangements of
the test *items*, so an image with 25 questions is 25 times more likely to be a
donor than one with a single question. The recorded swap maps of C2/C5/C6
derange *images* instead. On Kvasir-VQA-x1 the two give coverages |V|/|S| of
81.6% and 52.1% - but φ of 48.7% and 47.8%. **φ is insensitive to the
construction; |V| is not**, and the floor applies to |V|. The paper reports both
numbers for this reason.

## Collision counts, and why they differ from `ccr.py`

Section 5.2 reports that the scoring rule merges no two reference strings
within a question template: 0 of 353 on Kvasir-VQA-x1, 0 of 268 on SLAKE, 0 of
13 on VQA-RAD, 0 of 7,858 on ProbMed, against 2 of 881 on PathVQA and 2 of 98
and 176 of 6,578 on the two PMC-VQA files.

The docstring of `src/analysis/ccr.py` reports **0 of 260, 0 of 130 and 0 of
120** for the same three benchmarks. The numerator is the same zero; the
denominator is not, because the two count different sets:

- `ccr.py` counts the reference strings of the templates that enter the
  **informative computation** of the CCR;
- `coverage.py` counts them over the **whole test set**, which is the set φ is
  defined on.

Both are correct for what they measure. Section 5.2 quotes the second.

## What the module prints

1. **Step 1 - question duplication.** Deterministic, and it bounds `E[|V|]`. If
   almost no question appears on two images, |V| collapses and that is already
   the answer for that benchmark.
2. **Step 2 - validation.** Reproduces SLAKE C2 (φ = 56.0% over 125 verifiable)
   and VQA-RAD (|I| = 1, below the minimum of four). **The module stops if
   these do not reproduce**: the fault would be in the pipeline, not the data.
3. **Step 3 - φ per draw**, at both scales, with strata where the benchmark has
   them (open/closed on PathVQA, diagnostic category on ProbMed).
4. **Step 4 - reference collisions.**

## Determinism

Master seed `20260910`, 1,000 draws. The same seed reproduces every number in
the table below bit for bit. The seed is ours on purpose: the goal is the *expected* φ
of the benchmark under a free swap, independent of any one implementation,
which is a stronger claim than reproducing one particular draw.
