# Pre-registration of analysis C — does distance predict the flip?

Written and committed **before the first number of C**, as §0 of the plan
requires and as B and G0 were. G1 and G2 are closed; C was held shut until both
verdicts had been read.

Inherited rules: amendment 7.6 (per-cell checkpointing, per-cell seeds), 7.7
(canonical order by image_id, vectors resolved by sha256 lookup and never
recomputed), the host recorded per cell, no `nice`.

## 1. Unit and quantities

The unit is **one swapped row × checkpoint × seed** — a row of C2, C5 or C6, as
predicted by one trained checkpoint under one training seed.

    flip = 1[ prediction under the swap ≠ prediction under C0 ]

The distance **d of the row is the distance of its (original, donor) image
pair**, taken from analysis B, per ruler: MedSigLIP primary, the colour
histogram as the counterfactual of amendment 7.4, DINOv3, BiomedCLIP and DINOv2
as robustness.

The derangement is over images, so every row of a given image carries the same
d. That is a property of the design, not an approximation, and it means the
effective sample for the distance axis is the image, not the row — reported
alongside the row count so a reader can see both.

## 2. P(flip | d)

Two readings of the same relation, one non-parametric and one parametric.

- **Binned.** Five bins by **quantiles of d computed per benchmark**, so the bin
  edges do not import one benchmark's scale into another. Bootstrap confidence
  interval per bin.
- **Logistic.** `flip ~ d`, fitted **per checkpoint × seed**, coefficient
  reported with its confidence interval. Seeds are never pooled.

## 3. Where the donor carries the question

On the verifiable set V of v3 — the rows where the donor image carries the same
question, so the swapped item has a gold answer — also report
**P(prediction = y′ | flip, d)** per bin: given that the answer changed, how
often it changed *to the donor's answer* rather than to something else. A flip
toward y′ is evidence that the model read the donor image; a flip elsewhere is
evidence only that it was disturbed.

## 4. VDD by tercile of d

VDD computed on the near, middle and far terciles of d. On Kvasir and on SLAKE's
X-ray/lung bucket, the **near tercile is compared against VDD(C5)**, which is
the within-modality swap: if the near tercile of C2 reproduces C5, then C5 is
what a near donor already was, and the ladder's two rungs are measuring one
thing.

## 5. Controls

**(a) Within bucket.** The curve is recomputed restricted to pairs whose donor
is in the same bucket — same procedure on Kvasir (amendment B.3), same
modality × location bucket on SLAKE. This removes the domain contrast from the
distance axis and leaves only within-domain variation.

**(b) Colour beside MedSigLIP.** The same curve under the colour histogram,
plotted next to the primary ruler. **The reading is fixed here, before the curve
exists:**

- **MedSigLIP rises and colour does not** → what predicts the flip is content.
  Contribution 2 stands.
- **both rise equally** → it is colour, and the text says so.

This control matters most on Kvasir, where analysis B found the colour histogram
to be the second strongest separator of the domain, ahead of three of the four
content rulers.

## 6. G3

The gate: the **logistic coefficient is positive with a confidence interval
excluding zero in at least 12 of the 15 cells** of configuration × benchmark —
Kvasir-VQA-x1, SLAKE and VQA-RAD, five configurations, and the three seeds
**never pooled**.

HEAL enters when the C2 inference of its checkpoints exists (item 6c of the
plan), and **outside G3**: its gate was G2's negative control, and adding it to
G3 after seeing B would be changing the denominator after the fact.

## 7. Where each part runs

Kvasir, SLAKE and VQA-RAD on **host-B** — predictions and embeddings copied with
sha256 verified, never recomputed on the destination. HEAL on **host-A**, where
the declared limitation of §4 keeps VinDr images and embeddings; only
aggregates, curves and tables leave.

## 8. What C is not being asked

C does not ask whether the model is right. It asks whether the **distance
between two images predicts that the model's answer changes**. A model that
flips on every far donor and none of the near ones scores perfectly here and may
still be wrong about every one of them — accuracy is reported separately and is
not part of G3.
