# Noise floor of the protocol (Section 5.6)

Inference is not bit-exact across environments. To know how much of an
observed VDD could be noise rather than signal, the intact condition (C0) was
re-run and the predictions compared.

## Measurements

| Configuration | Same environment (run 1 vs run 2) | Across environments (original vs re-run) | McNemar (across) |
|---|---|---|---|
| SmolVLM2-2.2B + LoRA, SLAKE | **0.00%** (0/1,061) | 5.94% (63/1,061) | b=20, c=18 |
| Qwen2-VL-2B + LoRA, SLAKE | **0.00%** (0/1,061) | 3.30% (35/1,061) | b=9, c=12 |
| Qwen2-VL-2B + LoRA, VQA-RAD | **0.00%** (0/451) | 2.66% (12/451) | b=4, c=4 |

Percentages are the fraction of predictions that differ between the two runs.

## Reading

**Within one environment the pipeline is deterministic**: zero differing
predictions out of 1,061, in two different architectures, reached through two
different decoding configurations (Qwen2-VL uses `do_sample=True` with
`top_k=1`; SmolVLM2 falls back to the transformers default `do_sample=False`).
Both are greedy in effect.

**Across environments, 2.7% to 5.9% of predictions differ**, with McNemar's
b and c almost equal in all three cases. The disagreements have no direction:
they add variance to the paired tests without biasing VDD. On VQA-RAD the
aggregate accuracy did not move at all (49.67% in both runs) despite 12
predictions changing.

An internal consistency check supports the "same environment" row
independently: for each configuration, comparing the original C0 against run 1
and against run 2 gives *identical* counts (same number of differences, same
b, same c), which is only possible if the two runs agree exactly.

## The two sources, measured on the paper's own cells (Section 5.6)

Across environments: the C6 predictions of the SLAKE classifier (M3AE_SCRATCH)
re-run in one session differ from the published file in **5.94%** of the
rows (63 of 1,061; `results/reruns/M3AE_SCRATCH_SLAKE_C6_sameenv.json`
against `results/M3AE_SCRATCH_SLAKE_C6.json`), the top of the 2.7-5.9% band
above.

Batch composition, same host and same arrow: re-running C0 with batch size 1
instead of 4 changes **0.71%** of the predictions of SmolVLM2 LoRA seed 2
(77 of 10,779; 36 change their `correct` flag) and **0.59%** of SmolVLM2 DoRA
seed 0 (64 of 10,779; 32), on Kvasir-VQA-x1
(`results/reruns/*_bs1.json`; `docs/v4/ccr_identity_bytes.json`). Decoding is
greedy but not invariant to how a batch is composed.

These are the three cells where the identity of Section 5.2 fails on the
published files and holds on the re-runs (`python -m src.analysis.identity
--all --reruns`).

## Consequence for the paper

The instability reported between C5 and C6 on their shared questions
(6–16% on SLAKE) is **environmental, not stochastic decoding**: the SLAKE
batteries were executed across different machines on different dates, whereas
the Kvasir-VQA-x1 batteries ran in a single environment and produce identical
predictions on the shared questions for all five fine-tuned configurations.

An effect smaller than this band is interpreted only when it replicates across
seeds.

## Reproducing

The repeated runs are shipped as `results/<PREFIX>_C0_REP1.json` and
`results/<PREFIX>_C0_REP2.json`. Compare any pair with:

```python
import json
def load(p): return {x["qid"]: x for x in json.load(open(p))["predictions"]}
a = load("results/SMOLVLM_LORA_SLAKE_C0_REP1.json")
b = load("results/SMOLVLM_LORA_SLAKE_C0_REP2.json")
shared = [q for q in a if q in b]
print(sum(a[q]["predicted"] != b[q]["predicted"] for q in shared), "/", len(shared))
```
