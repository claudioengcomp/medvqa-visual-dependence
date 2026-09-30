# Figures

Plots of the paper and of the additional material that contain no dataset
image; the data behind each one is in `docs/v4/figs/` with the same stem.

| file | paper | data |
|---|---|---|
| `fig3_vdd_vs_distance.pdf` | Figure 4: VDD against the kNN distance to the training bank, per condition and configuration | `docs/v4/figs/fig3_vdd_vs_distance.json` |
| `fig5_flip_vs_distance.pdf` | Figure 5: flip rate against the distance to the donor, one panel per ruler | `docs/v4/figs/fig5_flip_vs_distance.json` |
| `fig4_umap_supplement.pdf` | additional material: UMAP of the conditions (illustration; no number is read from it) | `docs/v4/figs/fig4_umap_supplement.json` |

The file names follow the scripts that draw them (`src/featurespace/fig3_vdd_vs_distance.py`,
`fig5_flip_vs_distance.py`, `fig4_umap.py`), numbered as in an earlier draft.

Not included, because they show dataset images and this repository
redistributes none (`DATA_NOTICE.md`): Figure 2 (the ladder of conditions on
one Kvasir-VQA-x1 image), Figure 3 (the colour ruler, with the original and
donor images of three pairs) and the compiled additional material, whose
Figure S3 shows the same pairs. They are in the paper and its supplement.
