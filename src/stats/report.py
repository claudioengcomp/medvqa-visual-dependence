"""Renders a Markdown report either from the summaries produced by
src/stats/compute_vdd.py (per-seed) or from the final multi-seed matrix
consolidated by src/analysis/build_multiseed_summary_matrix.py.

Usage (per-seed, C0 vs C1 only):
    python -m src.stats.report --summaries results/*_vs_C0.json --output results/report.md

Usage (consolidated multi-seed, all conditions):
    python -m src.stats.report --multiseed-matrix results/summary_matrix_multiseed.json \
        --output results/report_multiseed.md
"""
from __future__ import annotations

import argparse
import glob
import json

from src.analysis.build_summary_matrix import _COND_ORDER


def _fmt_ci(ci) -> str:
    if ci is None:
        return "-"
    lo, hi = ci
    return f"[{lo:.3f}, {hi:.3f}]"


def _fmt_aggregate_ci(ci_c0, ci_c1) -> str:
    """Fallback for when no paired CI is available: approximates the VDD
    confidence interval from the two independent CIs (C0 and C1) by
    adding/subtracting their endpoints. This is NOT a rigorous paired CI
    (it does not capture item-by-item correlation) -- it is labelled with
    '~' to make that explicit to the reader of the report."""
    if ci_c0 is None or ci_c1 is None:
        return "-"
    lo = ci_c0[0] - ci_c1[1]
    hi = ci_c0[1] - ci_c1[0]
    return f"~[{lo:.3f}, {hi:.3f}]"


def render_report(summaries: list[dict]) -> str:
    """Per-seed report, C0 vs C1 only (one row per summary produced by
    compute_summary/compute_summary_vs_baseline)."""
    lines = [
        "# VDD per model",
        "",
        "| Modelo | Dataset | Acc(C0) | Acc(C1) | VDD | IC95%(VDD) | McNemar p |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        ci = s.get("ldv_ci_paired")
        ci_label = _fmt_ci(ci) if ci is not None else _fmt_aggregate_ci(s.get("ci_c0"), s.get("ci_c1"))
        p_value = s.get("mcnemar", {}).get("p_value")
        p_str = f"{p_value:.4g}" if p_value is not None else "-"
        lines.append(
            f"| {s['model']} | {s['dataset']} | {s['accuracy_c0']:.4f} | "
            f"{s['accuracy_c1']:.4f} | {s['ldv']:.4f} | {ci_label} | {p_str} |"
        )
    return "\n".join(lines) + "\n"


def render_multiseed_report(matrix: dict) -> str:
    """Final multi-seed report -- one row per model_base x condition, built
    from the output of
    src.analysis.build_multiseed_summary_matrix.build_multiseed_summary_matrix().
    Acc(C0)/VDD as mean +/- standard deviation across seeds; VDA only on the
    C1 row (it is defined over C1, not recomputed per condition); McNemar as
    a "k/N significant" count (seeds are never pooled into a single CI);
    Cohen's g as mean +/- standard deviation over the seeds in which it is
    defined (n)."""
    lines = [
        "# Multi-seed VDD/VDA per model",
        "",
        "| Model | Condition | n seeds | Acc(C0) | VDD | VDA(C1) | McNemar sig. | Cohen's g |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model_base in sorted(matrix):
        entry = matrix[model_base]
        vda_c1 = entry.get("vda_c1")
        for cond in _COND_ORDER[1:]:
            agg = entry["conditions"].get(cond)
            if agg is None:
                continue
            acc_str = f"{agg['accuracy_c0_mean']:.4f} ± {agg['accuracy_c0_std']:.4f}"
            vdd_str = f"{agg['vdd_mean']:.4f} ± {agg['vdd_std']:.4f}"
            vda_str = f"{vda_c1:.4f}" if cond == "C1" and vda_c1 is not None else "-"
            mcnemar_str = f"{agg['mcnemar_significant_count']}/{agg['mcnemar_total_seeds']}"
            g_mean, g_std, g_n = agg.get("cohens_g_mean"), agg.get("cohens_g_std"), agg.get("cohens_g_n")
            g_str = f"{g_mean:.4f} ± {g_std:.4f} (n={g_n})" if g_mean is not None else "-"
            lines.append(
                f"| {model_base} | {cond} | {agg['n_seeds']} | {acc_str} | "
                f"{vdd_str} | {vda_str} | {mcnemar_str} | {g_str} |"
            )
    return "\n".join(lines) + "\n"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", nargs="+",
                         help="Paths (or globs) to the *_vs_C0.json files (per-seed mode)")
    parser.add_argument("--multiseed-matrix",
                         help="Path to summary_matrix_multiseed.json (consolidated multi-seed mode)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.summaries and not args.multiseed_matrix:
        parser.error("passe --summaries OU --multiseed-matrix")
    return args


if __name__ == "__main__":
    args = _parse_args()
    if args.multiseed_matrix:
        with open(args.multiseed_matrix) as fp:
            matrix = json.load(fp)
        report = render_multiseed_report(matrix)
    else:
        paths = []
        for pattern in args.summaries:
            paths.extend(sorted(glob.glob(pattern)))
        summaries = []
        for p in paths:
            with open(p) as fp:
                summaries.append(json.load(fp))
        report = render_report(summaries)
    with open(args.output, "w") as fp:
        fp.write(report)
    print(report)
