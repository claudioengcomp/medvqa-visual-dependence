# Full test set against the strict (deduplicated) subset

Sensitivity of the reported quantities to content duplicates in the clean
Kvasir-VQA-x1 split (Section 4.1 of the paper). Removing the 351 test images
with a neighbour in training (pHash <= 4; 36 of them exact copies) and 7
copies internal to the test set leaves 933 of 1,291 images and 10,203 of 10,779
questions. Over all templates the VDD is about 1 pp higher on the strict
subset; with the single trivial template excluded from both sides the
difference falls to between -0.01 and +0.14 pp, below the spread over seeds.
The CCR and phi do not move materially.

Two earlier readings of these numbers did not survive their controls. They are
kept below because the corrected reading is easier to check next to what it
replaces.

## Two readings the controls ruled out

**Draft 1: "near-duplicates are 9 points easier, consistently and unanimously."**

Accuracy on the 556 questions of near-duplicate images was 0.95–0.96 against
0.855–0.872 on the rest, in all 15 configurations. Read as evidence that content
overlap helps the model.

It was composition. The near group is 1.6 questions per image; the rest is 11.
**336 of the 556 near questions are a single template**, `does this image
contain any finding?`, which every model answers perfectly. The rest carries the
full mix of 18 templates, hard ones included.

Controlling within template leaves **one** template with n ≥ 20 in both groups,
and there the gap is **0.00 pp** (1.000 against 1.000). The within-template test
has no power to say more than that, and it is reported as such rather than as a
null result.

**Draft 2: "the presence of duplicates dampens VDD."**

VDD was about 1 pp higher on the strict subset, in every condition and
configuration, exceeding the seed spread in 15 of 15 cells for the swaps.

It was the same template. VDD on `does this image contain any finding?` is
**0.00** - the answer does not depend on the image at all. The strict subset
removes a set of questions that is disproportionately made of it, so the average
VDD rises by arithmetic.

Excluding that template from both sides:

| condition | full − strict, all templates | full − strict, without the trivial one | seed sd |
|---|---|---|---|
| C1 | −1.15 | **−0.01** | 4.65 |
| C2 | −0.89 | **+0.14** | 0.68 |
| C5 | −1.24 | **+0.03** | 0.62 |
| C6 | −1.29 | **+0.03** | 0.70 |

Every difference collapses to within a rounding of zero and falls **below** the
seed spread. There is no VDD effect of removing content duplicates on
Kvasir-VQA-x1.

Supporting numbers: VDD on the 576 removed questions is 14.8–17.8 pp against
35–41 pp on the full set - the removed set is dominated by questions the image
barely moves.

## CCR and phi

`compute_ccr` returns `ccr` and the informative-set breakdown; phi = |I|/|V| is
computed separately from the same swap map with the repository ruler.

| | CCR full | CCR strict | phi full | phi strict | \|V\| full | \|V\| strict |
|---|---|---|---|---|---|---|
| Kvasir C2 | 0.7818 | 0.7784 | 0.4778 | 0.4897 | 5,613 | 5,324 |
| Kvasir C5 | 0.7968 | 0.7948 | 0.6438 | 0.6456 | 8,102 | 7,896 |
| Kvasir C6 | 0.7872 | 0.7854 | 0.5880 | 0.5886 | 9,369 | 9,149 |
| SLAKE C2 | 0.7533 | 0.7071 | 0.5600 | 0.5315 | 125 | 111 |
| SLAKE C5 | 0.3580 | 0.3733 | 0.3600 | 0.3562 | 75 | 73 |
| SLAKE C6 | 0.6275 | 0.6260 | 0.3151 | 0.3231 | 292 | 260 |

Neither moves materially on Kvasir. On SLAKE the verifiable sets are 75 to 292
questions, so the changes are within what such denominators produce.

## What survives

Content overlap between partitions of the clean Kvasir split is real and
measured - 36 exact copies, 351 near neighbours in 1,291 test images - and it
does **not** measurably change any quantity the paper reports. That is the claim
the evidence supports: the overlap exists, and it is not load-bearing.

The strict subset stays in the repository as a selection list, so the claim can
be checked rather than taken on trust.

SLAKE differs in kind: its near neighbours are slices of the same volume, its
removed questions carry as many questions as the kept ones (10.5 against 11.3),
and its seed spread (2.7 to 5.8 pp) is larger than any difference measured here.
Nothing about SLAKE is decided by this comparison.
