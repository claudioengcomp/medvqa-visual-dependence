# Data notice

This repository redistributes **no images** from any dataset.

What it does contain, and under what terms:

| Content | What it is | Terms |
|---|---|---|
| `data/kvasir_clean_split/` | Image identifiers (`img_id`) and question identifiers of Kvasir-VQA-x1, mapped to a split. No images, no answer text. | Derived identifiers from `SimulaMet/Kvasir-VQA-x1`; obtain the dataset from its original source. |
| `data/swapmap.json` | For each swapped image, the identifier of the donor image, plus the question text and reference answer of each test question. Needed to compute the Counterfactual Consistency Rate. | Derived from the test splits of SLAKE, VQA-RAD and Kvasir-VQA-x1; subject to each dataset's own license. |
| `results/` | Model predictions, together with the reference answer of each question. | Model outputs (this work); reference answers derived from the datasets above. |
| `configs/`, `src/`, `tests/` | Code and configuration. | MIT License, see `LICENSE`. |

The datasets themselves must be obtained from their original distributors:

- **SLAKE** — Liu et al., ISBI 2021.
- **VQA-RAD** — Lau et al., Scientific Data 2018.
- **Kvasir-VQA-x1** — Gautam et al., DEMI 2025 (MICCAI Workshop); `SimulaMet/Kvasir-VQA-x1` on the HuggingFace Hub.

If you are a rights holder and believe any file here exceeds fair reuse for
research reproducibility, please open an issue and it will be removed.
