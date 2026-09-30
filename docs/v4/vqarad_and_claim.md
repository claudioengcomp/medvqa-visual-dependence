# Item 5 — VQA-RAD, and the sentence the audit supports

## VQA-RAD: 203 of 203, by bytes

Every one of the 203 test images of VQA-RAD has a byte-identical copy in the
training or validation split, after decoding to RGB. Not a near neighbour: the
same pixels.

| measurement | value |
|---|---|
| test images | 203 |
| with an exact copy in train/val | **203 (100%)** |
| with a pHash neighbour ≤ 4 | 203 (100%), 327 pairs |
| identical pairs inside the test split | 0 |

Evidence: `harness/v4/dup_audit.py`, hashes cached at
`v4_cache/image_hashes.json`, results in `docs/v4/duplicate_audit.json`.
The hash is sha256 of the decoded RGB buffer, so it is invariant to container
and compression metadata and answers "the same picture", not "the same file".

**No strict subset is possible.** Removing from the test split every image with
a copy in train or validation empties it. There is nothing left to evaluate on,
so the full-against-strict comparison that Kvasir and SLAKE received cannot be
run here, and its absence is a property of VQA-RAD rather than a gap in the
analysis.

The paper currently cites Yang et al. 2024 for "202 of its 203 test images
appear in training". The measurement here gives **203 of 203**, obtained by
pixel identity after decoding rather than by identifier. The citation stays; the
number becomes ours.

## The sentence for the paper

> The clean split is disjoint by image. A content audit finds 36 exact copies
> and 351 perceptual near-duplicates (pHash ≤ 4) between test and
> training/validation; on the strict subset that removes them (933 images,
> 10,203 questions), Acc(C0) moves 0.5 pp, VDD less than 0.2 pp once template
> composition is controlled, and CCR and φ in the third decimal. The overlap
> exists and carries no result. On VQA-RAD, 203 of 203 test images have a
> byte-identical copy in training; on SLAKE, at least 25% of test images are
> slices of volumes present in training.

Three properties of this sentence are worth keeping when it is edited.

It is **measured, not cited**: every number in it comes from a script in this
repository, run on the released files, with the hashes cached so the numbers can
be recomputed rather than trusted.

It is **controlled**: "VDD less than 0.2 pp once template composition is
controlled" is the corrected reading. The uncontrolled difference is about 1 pp
and would have been reported as an effect of duplication; it is the removal of
one template whose VDD is 0.00, and the control is what distinguishes the two.

It is **stronger than "leakage-free"**, because it concedes the overlap and
shows it does not matter, instead of asserting a cleanliness that holds only for
identifiers. A reviewer who runs the audit finds what the sentence says.

The "at least" on SLAKE is load-bearing: an image-level hash bounds
patient-level overlap from below, since slices of one volume that look different
are not caught by pHash at all.
