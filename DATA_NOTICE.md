# Data notice

This repository redistributes **no images** from any dataset.

What it does contain, and under what terms:

| Content | What it is | Terms |
|---|---|---|
| `data/kvasir_clean_split/` | Image identifiers (`img_id`) and question identifiers of Kvasir-VQA-x1, mapped to a split. No images, no answer text. | Derived identifiers from `SimulaMet/Kvasir-VQA-x1`; obtain the dataset from its original source. |
| `data/swapmap.json` | For each swapped image, the identifier of the donor image, plus the question text and reference answer of each test question. Needed to compute the Counterfactual Consistency Rate. | Derived from the test splits of SLAKE, VQA-RAD and Kvasir-VQA-x1; subject to each dataset's own license. |
| `results/` | Model predictions, together with the reference answer of each question. | Model outputs (this work); reference answers derived from the datasets above. |
| `results/manifests/` | For every swapped row: question id, original and donor image identifiers, SHA-256 of both images' bytes. No image. | Identifiers and hashes derived from the test splits above. |
| `results/reruns/` | Model predictions of three cells re-inferred in one environment or with batch size 1 (Section 5.6). | Model outputs (this work). |
| `docs/v4/` | Derived statistics of the feature-space analyses: kNN and Frechet distances, sliced Wasserstein and permutation p-values, dip statistics, counts, tables and the numbers sheet. **No embedding, no image, no study identifier of any dataset.** | Derived from the datasets below; subject to each dataset's own license. |
| `docs/preregistration/`, `docs/PROVENANCE.md` | The pre-registration of the analyses and the record of the source commits. | This work. |
| `figures/` | Plots without dataset images. The figures of the paper that show dataset images (Figures 2 and 3, and the additional material) are not redistributed. | This work. |
| `configs/`, `src/`, `tests/` | Code and configuration. | MIT License, see `LICENSE`. |

The split of Kvasir-VQA-x1 released here is identifier-disjoint; its content audit (36 exact copies and 351 near neighbours between test and training images, under distinct identifiers) is in the README and in `docs/v4/duplicate_audit.json`.

The datasets themselves must be obtained from their original distributors:

- **SLAKE** — Liu et al., ISBI 2021.
- **VQA-RAD** — Lau et al., Scientific Data 2018.
- **Kvasir-VQA-x1** — Gautam et al., DEMI 2025 (MICCAI Workshop); `SimulaMet/Kvasir-VQA-x1` on the HuggingFace Hub.
- **HEAL-MedVQA / VinDr-CXR** (PhysioNet, credentialed access) — used only in the feature-space analyses of Section 5.3. This repository holds **no image, no embedding and no study identifier** from it: only distributions of distances, dip statistics, counts and p-values (`docs/v4/`).
- **PathVQA**, **PMC-VQA** (HuggingFace Hub) and **ProbMed** (credentialed release) — used only in the coverage analysis of Section 5.2 (phi); this repository holds only the coverage counts (`docs/COVERAGE.md`, `docs/v4/`).

If you are a rights holder and believe any file here exceeds fair reuse for
research reproducibility, please open an issue and it will be removed.
