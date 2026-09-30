# Pre-registration of analyses E1 and E2

Written and committed **before the first number of either**, as §0 of the plan
requires and as B and C were. Neither analysis runs VQA inference: E1 needs
embeddings only, E2 needs no model at all.

Inherited rules: amendment 7.6 (per-cell checkpointing, per-cell seeds), 7.7
(cache keyed by sha256 with lookup, fp32, TF32 off, deterministic algorithms,
the position control at exactly zero before any array is written), canonical
order by image_id, the host recorded in every JSON, and no `nice`.

Rulers: the five of the manifest — MedSigLIP (primary), DINOv3, BiomedCLIP,
DINOv2, and the 64-bin RGB colour histogram (the counterfactual of amendment
7.4). Benchmarks: Kvasir-VQA-x1, SLAKE, VQA-RAD.

---

## E1 — patch shuffle as an embedding-only condition (C7)

### What C7 is

Each image is divided into a grid of 32×32 patches and the patches are permuted
uniformly at random. The image keeps its pixels and loses its arrangement.

**Border rule, fixed here before anything is generated.** An image whose height
or width is not a multiple of 32 is divided into ⌊H/32⌋ × ⌊W/32⌋ **complete**
patches; only those are permuted. The remainder strips on the right and at the
bottom, each fewer than 32 pixels wide, **stay where they are**. Their size is
recorded per benchmark in the manifest.

This rule is chosen, among the alternatives, because it **conserves every
pixel**. Cropping would delete pixels and would make prediction (ii) below
untestable: a colour histogram of a cropped image is not the histogram of the
original, and the prediction would fail for a reason that has nothing to do with
patch shuffling.

Seed 20260918, derived per `image_id`, so the permutation of an image does not
depend on how many images preceded it. Original resolution, lossless PNG, one
arrow per benchmark, sha256 of every arrow in the manifest.

### Predictions, fixed before running

**(i)** Under the four content rulers, `knn(C7)` to the training bank falls
**between C3-50 and C1**. Shuffling destroys spatial arrangement while keeping
local texture, so it should cost more than covering half the image and less than
replacing the image with a blank frame.

**(ii)** Under the colour histogram, the per-item distance `d(C0, C7)` is
**exactly 0**, and `knn_colour(C7) = knn_colour(C0)`. A histogram counts pixels
and ignores where they are, so permuting patches cannot move it.

**Prediction (ii) is a generation test, not a finding.** If it does not come out
exactly zero, the arrows were not generated as specified — interpolation, a
resize, a lossy re-encode — and the defect is fixed before anything else is
read. It is stated as a prediction so that failing it cannot be reported as a
result about colour.

### What is computed

`knn` to the training bank, sliced Wasserstein against the bank with the
permutation test, and the per-item displacement C0→C7, under all five rulers;
plus the per-item colour distance C0→C7.

### Where C7 appears

As a point in Figure 3 on the x axis only, marked **"embedding only, no
inference"**, because no model was ever run on it and it has no VDD. As a row in
`tab_g1_distance.tex`, and as entries in the number sheet.

---

## E2 — the φ of CORAL's hard negative, with no model

### The map

For each test item (x_i, q_i, y_i), the hard negative is the pool image x_j
nearest to x_i by cosine of embedding, subject to `label(x_j) ≠ y_i`, by exact
search (inner product on unit vectors; FAISS `IndexFlatIP` or the equivalent in
NumPy). The pool is the **test images of the same benchmark**, and x_j = x_i is
excluded.

`label(x_j)` is the gold answer of item j to **its own** question, as CORAL
defines it. Where x_j carries several questions, each pair (x_j, q_j) is a
separate candidate, as their pool does.

Two encoders for the search, both embedded into the same cache in fp32:
**BiomedCLIP**, already in the manifest, and **CLIP ViT-L/14**
(`openai/clip-vit-large-patch14`, open weights).

### What is measured, without any model

- **|V|_HN** — items whose donor carries the same question q_i;
- **φ_HN = |I|/|V|** — among those, the fraction where the reference of q_i on
  the donor differs from y_i;
- the fraction of items whose donor does **not** carry q_i (the N group);
- the fraction of donors byte-identical to the original (sha256) and within
  pHash ≤ 4;
- mean cosine and its range, to sit beside CORAL's reported 0.967 and
  [0.63, 1.00];
- the distribution of d under MedSigLIP, against the terciles of C2.

Reported in one table beside C2 and C5 (φ, |V|, mean d).

### Reading rule, fixed before the numbers

- **φ_HN ≥ 0.9 and |V|_HN ≥ 20 in more than half the items** → the HN-Gap is
  interpretable on this benchmark, and the text says so.
- **|V|_HN below that floor, or φ_HN < 0.7** → the criticism of the
  `label(x_j) ≠ y_i` restriction gains a number rather than staying an argument.

### What is excluded and why

**VQA-RAD** is out: its informative set is a single question, which the CCR code
has documented since v3, so φ has no denominator there. **PathVQA** is out: it
has no predictions and no question-level gold structure in this repository. Both
exclusions are declared here rather than discovered later.

### Sensitivity, one line

The same HN map under the **correct** restriction — the reference of q_i on the
donor differs from y_i, evaluated only where that reference is defined — and how
many items survive it. That is C5 built by neighbourhood instead of by label,
and it is what §2 needs to make the point.

---

## Amendment E.1 — 2026-09-18: PathVQA rejoins E2, and why that is not a free choice

The pre-registration excluded PathVQA from E2 with a stated reason: "no
question-level gold structure in this repository". **That reason stopped being
true.** E1 built `vqa_path_vqa_test.arrow` from the released parquet shards with
the questions and answers grouped per image, so the structure E2 needs now
exists — 858 images, 6,719 questions, 2,913 distinct templates.

An exclusion lifted after the fact is a degree of freedom, and it is declared
here rather than taken quietly. Three things bound it:

1. **The reading rule does not change.** The same thresholds apply as to Kvasir
   and SLAKE: φ_HN ≥ 0.9 with |V|_HN ≥ 20 in more than half the items makes the
   HN-Gap interpretable; |V|_HN below the floor, or φ_HN < 0.7, gives the
   criticism a number.
2. **The measurement does not change.** Same pool construction, same two search
   encoders, same quantities, same correct-restriction sensitivity.
3. **The result is not predetermined.** PathVQA's three generic templates —
   *what is present?*, *what does this image show?*, *where is this?* — carry
   2,244 of its 6,719 questions, and |V| requires the donor to carry the same
   question, so |V|_HN will be dominated by them. Those templates have diverse
   answers across images, so **φ_HN may well come out high here**, against
   0.19 on Kvasir and SLAKE. That would make PathVQA a contrast rather than a
   third example, and it is the reason to run it.

**VQA-RAD stays excluded**, on its original ground: a single informative
question, which the CCR code has documented since v3.

### Scope of the sentence E2 licenses

CORAL reports on PMC-VQA, PathVQA, SLAKE and VQA-RAD. We measure Kvasir-VQA-x1,
SLAKE and now PathVQA. The claim holds **for the benchmarks measured, with their
construction reproduced**; for PMC-VQA it is extrapolation and the text says so.
SLAKE is the overlap between their set and ours, and it is where the comparison
is direct.
