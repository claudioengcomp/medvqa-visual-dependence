# Provenance of the feature-space layer

The pre-registration in `docs/preregistration/` and the JSONs in `docs/v4/` were produced in the authors' source repository, whose history this public repository does not carry. This file records the commits of that repository the paper relies on, as hash and UTC date, with their messages translated from the Portuguese originals. Hashes identify commits, not people.

## Milestones of Section 3.3

| commit | date (UTC) | what | message |
|---|---|---|---|
| `531cb78` | 2026-09-17T07:17:17+00:00 | pre-registration of the feature-space analyses, before any embedding | v4: pre-registration of the feature-space analysis, before any embedding |
| `3e7c63e` | 2026-09-17T07:18:30+00:00 | amendment 7.1 | v4: amendment 7.1 to the pre-registration: the labels G0 can actually use, before running |
| `5e5231b` | 2026-09-17T07:27:09+00:00 | amendment 7.2 | v4: amendment 7.2: G0 now tests content, not label; colour floor. Before running |
| `47af0be` | 2026-09-17T19:52:31+00:00 | gate G1 and analysis A; amendment 7.9 | v4: G1 PASSES: analysis A (160 cells) and amendment 7.9, which delimits what is a criterion |

The pre-registration (`531cb78`, 07:17 UTC on 17 September 2026) precedes the first embedding run and the first gate (`47af0be`, 19:52 UTC the same day); the amendments between them are dated in the file itself (Section 7 of `preregistration_v4.md`).

## Commits stamped in the numbers sheet

`docs/v4/numbers_v4.md` cites, under every number, the JSON it is read from and the commit of the source repository that last changed that JSON. Those commits, in date order:

| commit | date (UTC) | message |
|---|---|---|
| `9ce13ee` | 2026-09-17T14:37:29+00:00 | v4: captions per case directory; the reading of each benchmark recorded in its own words |
| `8aad98c` | 2026-09-17T15:43:45+00:00 | v4: full against strict: the content overlap exists and moves no number |
| `be515b8` | 2026-09-17T17:18:00+00:00 | v4: occlusion collapses distinct inputs; noise degrades without collapsing |
| `1b6eade` | 2026-09-17T17:31:23+00:00 | v4: distance to the nearest neighbour in the bank: the duplicate audit by a second instrument |
| `47af0be` | 2026-09-17T19:52:31+00:00 | v4: G1 PASSES: analysis A (160 cells) and amendment 7.9, which delimits what is a criterion |
| `6134d57` | 2026-09-17T20:46:46+00:00 | v4: input inventory of B: check 7 passes on SLAKE and VQA-RAD and fails on Kvasir |
| `affb321` | 2026-09-17T20:53:03+00:00 | v4: analysis B on SLAKE and VQA-RAD: both separate, and colour separates with them |
| `7a43c6e` | 2026-09-17T20:56:55+00:00 | v4: B on Kvasir over the 478 declared pairs, and the bias of the exclusion measured |
| `1c187c6` | 2026-09-17T21:59:31+00:00 | v4: G2 PASSES narrowly: 3 of 4 separate, HEAL unimodal, and the bridge to the phi of v3 |
| `8f28260` | 2026-09-17T22:58:32+00:00 | v4: d per ruler and benchmark, with ranking within each ruler, and the earlier reading corrected |
| `1546fb0` | 2026-09-18T00:03:57+00:00 | v4: recomputation of the v3 quantities from the manifests: the phi of C5 changes in the first decimal |
| `ef10c9d` | 2026-09-18T00:21:24+00:00 | v4: Eq. 3 recomputed from the manifests: the identity closes, 1.81/1.72 do not change, d_N changes |
| `7003769` | 2026-09-18T01:13:01+00:00 | v4: G3 PASSES, 14 of 15, and the control within the bucket inverts the picture |
| `7a7a093` | 2026-09-18T01:34:47+00:00 | v4: where the pairs of C5 and C6 fall on the distance axis; the corresponding sentence corrected |
| `d84b4f1` | 2026-09-18T01:47:55+00:00 | v4: a_I - k per template: the difference is NOT template composition |
| `7c4d128` | 2026-09-18T01:58:08+00:00 | v4: D with support per bin and both weightings: the inversion does NOT hold by the rule |
| `91b9a0c` | 2026-09-18T02:48:01+00:00 | v4: seen against unseen (section 5): the comparison is structurally impossible, and that is the finding |
| `e0ee640` | 2026-09-18T10:27:46+00:00 | v4: E1: patch shuffle (C7) as an embedding-only condition, and the two predictions scored |
| `81469aa` | 2026-09-18T12:12:15+00:00 | E2 on PathVQA: amendment E.1, and the contrast it predicted |
| `323ba67` | 2026-09-20T03:47:13+00:00 | fig3: y axis in VDD (pp), not in relative drop |
| `6c9a867` | 2026-09-20T04:41:43+00:00 | fig5: 1x3, one panel per ruler and one curve per benchmark |
| `904c6d5` | 2026-09-20T14:23:29+00:00 | v4: |V| of C6 on SLAKE, destination of the changed answer, terciles in pp, C1 on closed questions, caption of fig5, table clean-up |
| `c4b6762` | 2026-09-20T23:43:33+00:00 | v4: MedVLThinker re-scored by E7 in Table 3; CCR identity 30/30 and the two sources of non-determinism |
| `62686ea` | 2026-09-21T00:33:12+00:00 | v4: the three open items measured: |I| under the two donor maps, recount of the prediction files, definition of the 30 cells |
| `3542fbc` | 2026-09-21T00:42:09+00:00 | v4: corrected manifest adopted in the CCR, the five tokens in fine-tuned output, the 30 cells, and why the identity does not close for the zero-shot model |
| `922b3b6` | 2026-09-29T23:58:07-03:00 | v4: E7 applied to the zero-shot Kvasir C6 file; Table 3 cell 7.3 -> 4.2; rulers on the training bank stamped in the sheet |

Inside this repository the sheet is regenerated by `src/featurespace/make_numbers.py`, which stamps each line with the commit of this repository instead; the table above is the bridge between the two histories.
