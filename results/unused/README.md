# Manifests not used in the paper

`C5_manifest_SLAKE_MRI.json`: a matched-pathology swap (C5) restricted to the
MRI images of SLAKE, 13 pairs. The paper reports that on SLAKE only the
X-ray/lung bucket admits a balanced derangement, so this manifest never entered
any number; it is kept here for the record and excluded from the counts of
`tests/test_published_counts.py`, which reads only `results/*.json`.
