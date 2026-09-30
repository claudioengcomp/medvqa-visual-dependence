# Pre-registration of analysis B — donor distance

Written and committed **before the first number of B is computed**, as §0 of the
plan requires. Section 2.3 of `PLANO_ARTIGO_VISAPP2027_v4.md` still lists
PathVQA among the benchmarks with metadata and is superseded here; the G2 line
of §4 is the one that holds. This document is the consolidated definition, and
it is what the script implements.

Host: **host-B**. Inherited rules: amendment 7.6 (per-cell checkpointing, per-cell
seeds), amendment 7.7 (canonical order by image_id, vectors resolved by sha256
lookup and never recomputed), the host recorded in every JSON, and no `nice`.

## 1. Input

For every swapped row of C2, C5 and C6 — the rows in S_T — the pair
(original image_id, donor image_id) is read from the C2/C5/C6 manifest. **Rows
of C5 that keep their own image have no donor and are excluded**; they are not
counted as pairs at distance zero, which would drag every summary toward zero
for a reason that has nothing to do with donor choice.

The embeddings of both members of a pair come from the **C0 cache, resolved by
the sha256 of the image bytes** (amendment 7.7). Nothing is recomputed for B.

## 2. Distance

    d = 1 − cos( e(x), e(x') )

computed per ruler, on all five: **MedSigLIP** primary; DINOv3, BiomedCLIP and
DINOv2 as robustness; **colour histogram as the counterfactual** of contribution
2, under amendment 7.4.

The derangement is over images, so **d is one value per pair of images**, not
per question. Both summaries are reported: the distribution **per image**, and
the distribution **weighted by question count**, since an image carrying eleven
questions weighs eleven times as much in any accuracy the paper reports.

## 3. Domain buckets

The buckets are the ones already used in §4.1 of the paper:

| benchmark | bucket | count |
|---|---|---|
| SLAKE | modality × location | 9 |
| VQA-RAD | organ | 3 — chest, abdomen, head |
| Kvasir-VQA-x1 | procedure type | the domain of C6 |

**near** = the donor is in the same bucket as the original. **far** = a
different bucket.

## 4. Test for benchmarks with metadata (G2)

Wasserstein-1 between the distribution of d in the near group and in the far
group, with a permutation test: **1,000 shuffles of the group labels**,
significance at **p < 0.01**. Reported alongside: the **fraction of C2 donors in
the near group**.

The sliced Wasserstein between the feature clouds of the donors of the two
groups is reported **without a verdict**.

## 5. Benchmarks without metadata

**PathVQA**: Hartigan's dip test on the d of C2, significant at **p < 0.05**.

**HEAL**: the same dip test, and it is a **negative control** — it must come out
**unimodal, p ≥ 0.05**. A bimodal HEAL fails G2. HEAL has no domain metadata to
separate near from far, so a clean split of its donor distances would mean the
bimodality is being produced by something other than domain, and the reading of
every other benchmark would be in doubt.

## 6. Verdict of G2

Separation in **at least 3 of the 4** benchmarks (Kvasir-VQA-x1, SLAKE, VQA-RAD,
PathVQA) **and** HEAL unimodal.

## 7. Check before any test

**C5 and C6 must have 100% of donors in the near group**, because both swaps are
constructed within the bucket. Only C2 mixes. Anything below 100% is a defect in
the manifest or in the bucket assignment, and it is diagnosed before a single
test is run.

The **near fraction of C2** is the number contribution 2 uses.

## 8. Figure 4

UMAP of C0 under MedSigLIP, coloured by bucket, with original→donor segments for
a sample of C2. **Illustration only: no number in the paper comes from it.**

## 9. Order of execution

Kvasir-VQA-x1, SLAKE, VQA-RAD first — their embeddings are already in the host-B
cache, 180 files with sha256 verified against host-A. PathVQA and HEAL follow, as
soon as their embeddings arrive from host-A with sha256 checked.

## 10. Report

Per benchmark and per ruler: the near fraction of C2, W1 and its p-value, the
per-image and per-question distributions, the dip test on PathVQA and on HEAL,
and Figure 4.

---

## Amendment B.1 — 2026-09-17, before the first distance: check 7 restated, and how the Kvasir bucket is derived

Check 7 as first written — "C5 and C6 must have 100% of donors in the near
group, because both swaps are constructed within the bucket" — is true of SLAKE
and VQA-RAD and false of Kvasir-VQA-x1. The error is in the assertion, not in
the data: §4.1 of the paper says what the Kvasir swaps are built on, and it is
not the procedure.

### What C5 and C6 are on Kvasir

**C5 pairs within the endoscopic modality — a single bucket — with a different
finding category** (polyp, ulcerative colitis, oesophagitis). The procedure is
not a restriction. §4.1 records the 617 donor assignments: **240 within the same
procedure, 369 crossing it, 8 without a label.**

**C6 keeps the C5 pairs and completes the rest by procedure**, so its within-
procedure count is 240 inherited plus 166 from the completion, **406**.

The inventory measured 240 of 609 for C5 and 406 of 775 for C6, from the
manifests, before this text was written. Those are the published numbers.
**Check 7 passes on all three benchmarks**, against the expectation the paper
states rather than against the one the first draft assumed:

| benchmark | condition | expected within-bucket | source |
|---|---|---|---|
| Kvasir-VQA-x1 | C5 | **240 of 609** (369 crossing, 8 unlabelled) | §4.1 |
| Kvasir-VQA-x1 | C6 | **406 of 775** (240 inherited + 166 completed) | §4.1 |
| SLAKE | C5, C6 | 100% | construction |
| VQA-RAD | C6 | 100% | construction |

**The expectation comes from the paper, not from the manifests.** Deriving the
buckets from the C6 sub-manifests themselves returns 100% on Kvasir for both
conditions — and it is a tautology, since every C6 pair lies inside one
sub-manifest by construction. That near-miss is recorded in commit `6134d57`
with the script that produced it, and it is the reason this amendment reads the
expectation off §4.1.

### Note for the v4 rewrite

The v3 sentence **"C6 uses the procedure type as domain"** holds only for the
completion. The pairs C6 inherits from C5 cross procedures, and 369 of 777 of
them do. The sentence has to say so.

### The Kvasir bucket for analysis B

The bucket remains the **procedure**, derived from two sources:

1. **the procedure question**, where the benchmark asks it;
2. **the landmark answer**, where it does not, under this declared map:

   | landmark | procedure |
   |---|---|
   | cecum, ileum, retroflex-rectum | colonoscopy |
   | z-line, pylorus, retroflex-stomach | gastroscopy |

The 515 test images with no procedure label are the ones with no pathology
annotation, and the procedure was derived from the pathology. They are the
**anatomical landmark** images — z-line, pylorus, cecum, retroflexions — which
carry the landmark questions the placebo excluded. Their procedure is in the
benchmark, in a different question.

**Coverage is reported.** If it falls below 90%, Kvasir enters G2 with the
coverage written down and the near fraction of C2 computed over the classifiable
pairs, with the denominator declared **before any distance is computed**.

### Order of execution, revised

SLAKE and VQA-RAD run now, in parallel: their buckets cover 93 of 96 and 203 of
203, and check 7 passes at 100% on both. Kvasir follows as soon as the bucket
derivation above is measured. PathVQA and HEAL when their embeddings arrive from
host-A with sha256 checked.

---

## Amendment B.2 — 2026-09-17, before the Kvasir distances: the declared denominator, and a bias test instead of a caveat

### The 515 have a name, and it comes from the benchmark

Amendment B.1 expected the landmark answer to supply the procedure of the 515
Kvasir test images that carry no pathology annotation, and predicted coverage
near 100%. **It supplied none.** The derivation is sound where it applies — the
two sources agree on 195 of 195 images, and the result matches
`kvasir_clean_bucket_*.json` on 776 of 776 — but the 515 have no landmark
question, no procedure question and no pathology label. Coverage stays at **776
of 1,291, 60.1%**. No file in the repository carries their procedure;
`kvasir_clean_split_map.csv` holds only `img_id` and `split`.

What they do have is one question each, and it has an answer:

| | `does this image contain any finding?` |
|---|---|
| the 515 without a bucket | asked of 514; answered **"no"** in **514 of 514** |
| the 776 with a bucket | **never asked** |

So the subpopulation is named by the benchmark, not by us: it is the
**no-finding** class. These are annotated images — annotated as having nothing
to find — and the pathology label is absent because there is no pathology. That
gives them an identity, and it still gives no procedure, so **they do not enter
the domain bucket.**

This also explains a result the duplicate audit reported without a mechanism.
`does this image contain any finding?` has a VDD of 0.00 and every model answers
it perfectly. It is asked **only of images whose answer is "no"**. A question the
benchmark poses exclusively when the answer is negative is answerable without
the image, by construction of the dataset rather than by any property of the
model.

### The declared denominator for Kvasir's G2

| | images | questions |
|---|---|---|
| test split | 1,291 | 10,779 |
| **C2 pairs classifiable into near/far** | **478 (37.0%)** | **6,367 (59.1%)** |
| images without a bucket | 515 (39.9%) | 520 (**4.8%**) |

The 515 are 40% of the images and 4.8% of the questions. The loss is not mainly
theirs: **298 pairs are excluded because the donor alone falls in the 515**, the
original being labelled. That is what turns 776 labelled originals into 478
classifiable pairs.

Kvasir enters G2 on those **478 pairs**, with this denominator stated here,
before any distance is computed.

### The bias is measured, not declared

Excluding a pair because of its donor selects on a property of the donor, and
the excluded donors are a distinct subpopulation. Whether that moves the
distance distribution is an empirical question with a cheap answer, so it is
answered rather than hedged.

**Test:** Wasserstein-1 between the distribution of d over the **298 pairs whose
donor is one of the 515** and the distribution over the **478 classified
pairs**, per ruler, with 1,000 shuffles of the group labels. Both distances are
computable — every image has a vector; only the bucket is missing.

- If the two distributions agree, the exclusion does not bias the near fraction,
  and the report says so with the number.
- If they differ, the report states **the direction** — whether donors from the
  no-finding class sit systematically nearer or further — and the paper carries
  it.

This is one extra line of W1 and it replaces a caveat with a measurement.

### Note on the colour counterfactual

That a 64-bin histogram separates CT from X-ray was expected, and analysis A
already showed colour moving opposite to the content rulers along the ladder.
The normalisation of W1 by the pooled standard deviation stays in the report,
labelled as chosen after seeing the result.

What decides whether contribution 2 is about content is **analysis C**, exactly
as amendment 7.4 anticipated: the flip curve against d under colour versus under
MedSigLIP. B shows that both see the domain; C shows which of them predicts what
the model does.

### PathVQA and HEAL, status

**HEAL**: the eleven condition arrows exist on host-A; **no embeddings exist**.
Embedding needs a GPU and explicit authorisation.

**PathVQA**: there is no arrow at all — only training configs. It needs the
arrow built, the C2 derangement generated by image with seed 20260910 and
recorded in a manifest, and then embedding in the five rulers. It is the longest
of the remaining paths, and the G2 verdict needs it: separation in at least 3 of
4 benchmarks, and PathVQA is one of the four.

---

## Amendment B.3 — 2026-09-17: a sentence for Section 5, and the colour control C inherits from Kvasir

### For Section 5, beside the question-only floor

> The benchmark asks `does this image contain any finding?` **only of images
> whose answer is "no"**: it appears for 514 of the 515 test images that carry
> no pathology annotation, is answered "no" in 514 of 514, and never appears for
> any of the 776 annotated images. A visual dependency drop of zero on this
> template is therefore a property of the dataset's construction, not of the
> model, and it is the clean explanation of why the question-only baseline
> scores 1.000 on it.

Measured in `harness/v4/kvasir_procedure_bucket.py` and the answer tally beside
it; the template's VDD of 0.00 was already reported by the duplicate audit
(commit `8aad98c`), without a mechanism. This is the mechanism.

### Colour is not a uniform control across benchmarks

Analysis B, C2, Wasserstein-1 between near and far, normalised by the pooled
standard deviation of the two groups:

| ruler | SLAKE | VQA-RAD | Kvasir-VQA-x1 |
|---|---|---|---|
| MedSigLIP | 1.52 | 3.32 | **1.46** |
| DINOv3 | 2.35 | 2.61 | 0.95 |
| BiomedCLIP | 2.64 | 3.03 | 0.75 |
| DINOv2 | 1.62 | 2.34 | 0.54 |
| **colour histogram** | **0.77** | **0.85** | **1.25** |

In radiology the colour histogram is the weakest separator by a wide margin. On
Kvasir it is the **second strongest**, ahead of three of the four content rulers
and behind only MedSigLIP. Kvasir's domain is the procedure, and colonoscopy
against gastroscopy is in part a chromatic distinction.

Contribution 2 states this **of endoscopy specifically**, not as a general
caveat. The normalisation by pooled standard deviation is descriptive and was
chosen after seeing the result; the pre-registered criterion remains W1 with its
permutation p, which all five rulers pass on all three benchmarks.

### What analysis C must run on Kvasir, declared now

On Kvasir the flip curve against d is also computed **within procedure** — the
control of §2.4 — because that is where colour loses its domain label and only
content is left to explain the flip.

The reading is fixed here, before the curve exists:

- **MedSigLIP rises within procedure and colour does not** → contribution 2
  stands on endoscopy: what predicts the flip is content, not colour.
- **both rise equally** → it is colour, and the text says so.

This is declared for Kvasir because Kvasir is where colour behaves as a domain
ruler. SLAKE and VQA-RAD keep the ordinary C protocol.

### G2, partial

Three of three benchmarks with metadata separate at p < 0.01 on all five rulers:
Kvasir-VQA-x1, SLAKE, VQA-RAD. The verdict of G2 as registered needs **3 of 4**
plus HEAL unimodal, and the fourth is PathVQA. Until PathVQA and HEAL run, the
result is recorded as **partial: 3 of 3 separating, verdict pending**.

---

## Amendment B.4 — 2026-09-17: what the duplicate collapse in the swapmap changed, and the single aggregation rule

### Origin

`harness/analysis/build_swapmap.py` resolves a donor through `md5` of the image
bytes into a dictionary built with `setdefault`, so images that are
byte-identical collapse onto whichever identifier was read first. Kvasir-VQA-x1
has seven such groups in its test split; SLAKE and VQA-RAD have none, and show
**zero divergence** in every condition.

The direction of the map is sound — the original is resolved through
`question_id`, which stays in place, and the donor through the bytes, which move
— so this is not the inversion corrected in commit `150b7a3`. It is a distinct
defect with two effects:

- **C5 and C6: six phantom swaps each.** The image kept itself — same
  `image_id`, same bytes between C0 and the condition — but being a duplicate,
  the hash returns the twin's identifier, `donor != original` becomes true, and
  the row enters the map as though it had swapped.
- **C2: seven donors named as their twin.** The swap is real; only the
  identifier is wrong.

### What changed

| quantity | v3 | v4 |
|---|---|---|
| C5 donor assignments | 617 | **611** |
| C6 donor assignments | 783 | **777** |
| C5 \|S_T\| / \|I\| / \|U\| / \|N\| / \|V\| | 8,940 / 5,218 / 2,892 / 830 / 8,110 | **8,934 / 5,216 / 2,886 / 832 / 8,102** |
| C6 \|S_T\| / \|I\| / \|U\| / \|N\| / \|V\| | 10,271 / 5,511 / 3,866 / 894 / 9,377 | **10,265 / 5,509 / 3,860 / 896 / 9,369** |
| φ (C5) | 64.3% | **64.4%** |
| d_N (C5) | 18.2–21.8 pp | **18.5–21.7 pp** |
| d_N (C6) | — | **17.6–20.4 pp** |
| observed spread (C6) | 1.66 pp | **1.67 pp** |

`|N|` rises while `|S_T|` falls, which is not a contradiction: the six phantoms
leave the set entirely, and separately the one genuinely renamed donor of C5/C6
points at a twin that does not carry two of the questions, so two questions stop
being verifiable.

The `d_N` figure changes for two reasons at once — the correction, and the
aggregation rule below. Under the per-cell range it would read 18.3–21.9; the
value above is the one the single rule produces.

### What did not change

**C2 in its entirety**: |S_T|, |I|, |U|, |N|, |V|, φ, CCR and VDD are identical
to the fourth decimal in all fifteen cells. The renamed donors are
byte-identical twins carrying the same questions with the same answers, so no
classification moves.

**a_I** (81.3–82.6 on C5), **k** (2.6–4.4), **VDD/φ** (0.72–0.75), and the
**spread prediction of Equation 3**, 1.81 pp on C5 and 1.72 pp on C6, under both
sources.

The identity of Equation 3 holds exactly: the three terms sum to the measured
conditional VDD within 1.665e-16 across sixty cells.

### The aggregation rule, single from here on

Every range the paper reports is **the mean per configuration over its three
seeds, then the minimum and maximum across the five configurations** — the
multi-seed protocol of §3.8.

This was already how a_I, k and VDD/φ were computed: they reproduce their
published values exactly under it and only under it. `d_N` was published as a
per-cell range over the fifteen, a second aggregation inside the same sentence.
It moves to the single rule.

Where a per-cell range is genuinely useful it is labelled **"per cell"**, and it
never shares a sentence with a per-configuration one.

### Expected k

v3 defines the expected k as the rate implied by "each model's own distribution
of predictions on the same question template". No routine for it exists in
`harness/analysis`, so the reading was reconstructed operationally: for each
item of I, how often the model predicts that item's reference under C0 among the
items of the same template, averaged over I. It reproduces the published
39.5–41.0% on C5 exactly, which is what identifies it as the reading that
produced the number. It now lives in `harness/v4/eq3_recompute.py`.

### Source of the definitions

a_I, k, t_U, d_N and Equation 3 are transcribed from `main_v3.tex` §3.5 and
§5.2. They appear nowhere in this repository, and nothing here was reconstructed
from the numbers.
