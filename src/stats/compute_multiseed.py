"""Aggregates results from N independent seeds of the SAME model x dataset x
condition combination -- accuracy/VDD as mean +/- standard deviation
(standard practice), and McNemar/Cohen's g computed per seed with a
consistency report. Predictions from different seeds are never pooled into a
single paired comparison: that would be pseudo-replication and would
artificially inflate the sample size, so significance is reported as
"significant in k of N seeds" instead.

VDA is not aggregated here -- it is not part of the output schema of
compute_summary_vs_baseline, so there is nothing to aggregate; it is left to
the consumer of this module to derive it from accuracy_c0_mean/vdd_mean,
which are already sufficient for that.

Takes a list of summaries already computed by
src.stats.compute_vdd.compute_summary_vs_baseline (one per seed) -- it
recomputes no statistics, it only aggregates what was already computed.
"""
from __future__ import annotations

import statistics

_COMBO_KEYS = ("model", "dataset", "condition")


def _validate_same_combo(summaries: list[dict]) -> None:
    """Checks that all summaries belong to the SAME model/dataset/condition
    combination -- only for the keys that are actually present (opt-in; it
    does not require a stricter schema than the minimum used by the tests).
    Mirrors the validation that compute_summary/compute_summary_vs_baseline
    already perform between C0 and the other condition, now applied across
    seeds of the same combination.

    It does NOT anchor on summaries[0]: the expected value for each key comes
    from the FIRST summary in the list that has it (not necessarily index 0)
    -- if summaries[0] lacks 'model' but summaries[1] and summaries[2] carry
    DIFFERENT values, that must still be caught (bug found in code review:
    anchoring on summaries[0] caused the whole check to be skipped in that
    case)."""
    for key in _COMBO_KEYS:
        expected = next((s[key] for s in summaries if key in s), None)
        if expected is None and not any(key in s for s in summaries):
            continue  # no summary has this key -- nothing to validate
        for s in summaries:
            if key in s and s[key] != expected:
                raise ValueError(
                    f"summaries have inconsistent {key!r} across seeds: "
                    f"{expected!r} vs {s[key]!r} -- aggregate_multiseed_summary "
                    "expects N seeds of the SAME model/dataset/condition combo"
                )


def aggregate_multiseed_summary(
    summaries: list[dict], alpha: float = 0.05
) -> dict:
    """`summaries` is a list of dicts, each one the output of
    compute_summary_vs_baseline for a single seed (it must carry at least the
    keys accuracy_c0, ldv, mcnemar.p_value, mcnemar.cohens_g; "ldv" is the
    historical JSON key that holds the VDD value). If the keys
    model/dataset/condition are present, it validates that all summaries
    belong to the same combination (raising ValueError otherwise). Returns:
        {
            "n_seeds": int,
            "accuracy_c0_mean": float, "accuracy_c0_std": float,
            "vdd_mean": float, "vdd_std": float,
            "mcnemar_significant_count": int, "mcnemar_total_seeds": int,
            "cohens_g_mean": float | None, "cohens_g_std": float | None,
            "cohens_g_n": int,
        }
    The standard deviation is the SAMPLE standard deviation (ddof=1,
    statistics.stdev). This is a deliberate choice: reporting "mean +/- std"
    over N random seeds is the common convention in ML papers, and the seeds
    are a sample of the possible initializations rather than the whole
    population. The standard deviation of a single seed is 0.0
    (statistics.stdev raises with only one element -- one run is not a
    sample, so there is nothing meaningful to compute).

    cohens_g may be None for a given seed (see src.stats.compute_vdd.cohens_g
    -- b+c=0, no discordant pairs, effect size undefined). The Nones are
    filtered out before computing cohens_g_mean/cohens_g_std; cohens_g_n
    reports how many of the n_seeds had a defined cohens_g, so that consumers
    know the mean does not necessarily cover every seed. If NO seed has a
    defined cohens_g, cohens_g_mean/std come back as None (a legitimate "no
    effect size available" state, not an error)."""
    if not summaries:
        raise ValueError("aggregate_multiseed_summary needs at least 1 summary")

    _validate_same_combo(summaries)

    n = len(summaries)
    accuracies = [s["accuracy_c0"] for s in summaries]
    vdds = [s["ldv"] for s in summaries]  # "ldv" is the historical JSON key name; the metric is VDD
    cohens_gs = [s["mcnemar"]["cohens_g"] for s in summaries]
    p_values = [s["mcnemar"]["p_value"] for s in summaries]

    def _mean_std(values: list[float]) -> tuple[float, float]:
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return mean, std

    acc_mean, acc_std = _mean_std(accuracies)
    vdd_mean, vdd_std = _mean_std(vdds)

    valid_cohens_gs = [g for g in cohens_gs if g is not None]
    if valid_cohens_gs:
        g_mean, g_std = _mean_std(valid_cohens_gs)
    else:
        g_mean, g_std = None, None

    return {
        "n_seeds": n,
        "accuracy_c0_mean": acc_mean,
        "accuracy_c0_std": acc_std,
        "vdd_mean": vdd_mean,
        "vdd_std": vdd_std,
        "cohens_g_mean": g_mean,
        "cohens_g_std": g_std,
        "cohens_g_n": len(valid_cohens_gs),
        "mcnemar_significant_count": sum(1 for p in p_values if p < alpha),
        "mcnemar_total_seeds": n,
    }
