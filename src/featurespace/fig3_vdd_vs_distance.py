"""Figure 3: the drop against the distance the intervention actually moves.

Every point is one condition under one configuration: on x, how far that
condition pushes the images from the training bank; on y, how much accuracy it
costs. The two axes come from different instruments - an encoder and a set of
predictions - and the figure exists to show whether they agree.

x   knn to the training bank under MedSigLIP, **relative to C0**, from
    `g1_results.json`. The relative form is what the pre-registration fixed for
    this figure: C0 is not at zero, it is the distance from the intact test set
    to the bank, and plotting absolute distance would have a reader asking why
    the intact condition is off-distribution.

y   VDD in percentage points, Acc(C0) - Acc(T) on the same items (Eq. 1 of the
    paper), the mean per configuration over its three seeds. Swaps are
    conditional on the swapped subset, as in Table 3 of v3, with Kvasir's
    subsets taken from the corrected manifests (611 rows in C5, 777 in C6, not
    the swap map's 617 and 783). Occlusion and noise are over all questions, as
    in Table 4. An earlier version of this script divided by the C0 correct
    count, which is the relative drop VDD/Acc(C0), not VDD; the axis, the
    legend and the JSON all said VDD. Fixed 2026-09-20.

C1 is drawn apart. It replaces every image with the same blank frame, so the
condition holds one distinct image and its Fréchet is undefined (amendment 7.8);
this figure uses knn throughout, which is defined for it.

The swaps sit at x = 0 by construction, not by measurement: they carry the
images of C0 permuted among the rows, so the set is unchanged and its distance
to the bank is C0's. G1 verified that to 2.7e-8. Their spread on y against no
spread on x is the whole point of the 2x2 map.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402
import numpy as np                       # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.analysis.ccr import load_predictions            # noqa: E402
from src.stats.conditional_vdd import eligible_qids      # noqa: E402
from src.featurespace import paths                                 # noqa: E402

V4 = REPO / "docs/v4"
FIGS = V4 / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
BENCH = ["KVASIR", "SLAKE", "VQARAD"]
NICE = {"KVASIR": "Kvasir-VQA-x1", "SLAKE": "SLAKE", "VQARAD": "VQA-RAD"}

# paper name -> (file suffix, family, conditional on the swapped subset?)
COND = {
    "C1": ("C1", "removal", False),
    "C2": ("C2", "swap", True),
    "C5": ("C5", "swap", True),
    "C6": ("C6", "swap", True),
    "C3-25": ("C4_OCC_25", "occlusion", False),
    "C3-50": ("C4_OCC_50", "occlusion", False),
    "C3-75": ("C4_OCC_75", "occlusion", False),
    "C4-25": ("C4_NOISE_25", "noise", False),
    "C4-50": ("C4_NOISE_50", "noise", False),
    "C4-75": ("C4_NOISE_75", "noise", False),
}
MARK = {"swap": "s", "occlusion": "o", "noise": "^", "removal": "X"}
MANIFEST = {
    ("KVASIR", "C2"): "C2_manifest_KVASIR.json",
    ("KVASIR", "C5"): "C5_manifest_KVASIR_clean.json",
    ("KVASIR", "C6"): "C6_manifest_KVASIR_clean.json",
    ("SLAKE", "C2"): "C2_manifest_SLAKE.json",
    ("SLAKE", "C5"): "C5_manifest_SLAKE.json",
    ("SLAKE", "C6"): "C6_manifest_SLAKE.json",
    ("VQARAD", "C2"): "C2_manifest_VQARAD.json",
    ("VQARAD", "C6"): "C6_manifest_VQARAD.json",
}
SM = json.loads((REPO / "data/swapmap.json").read_text())
G1 = json.loads((V4 / "g1_results.json").read_text())
OCC = json.loads((V4 / "occlusion_collapse.json").read_text())
E1P = V4 / "analysis_e1.json"
E1 = json.loads(E1P.read_text()) if E1P.exists() else {"cells": {}}


def commit(rel):
    r = subprocess.run(["git", "log", "-1", "--format=%h", "--", rel],
                       cwd=REPO, capture_output=True, text=True)
    return r.stdout.strip() or "sem commit"


def label(i):
    p = [x for x in str(i).replace("//", "/").split("/") if x]
    if p and p[-1].lower() in ("source.jpg", "source.png"):
        return p[-2] if len(p) > 1 else p[-1]
    return str(i)


X = {(r["benchmark"], r["condition"]): r for r in G1["figure3_relative_to_c0"]
     if r["ruler"] == "medsiglip"}


def donor_map(b, c):
    d = json.loads((paths.RESULTS / MANIFEST[(b, c)]).read_text())
    rows = d["pairs"] if isinstance(d, dict) else d
    return {str(e["qid_image"]): str(e["donor_image"]) for e in rows}


def vdd(prefix, b, cond, conditional):
    c0 = load_predictions(str(paths.RESULTS), prefix, "C0")
    cx = load_predictions(str(paths.RESULTS), prefix, COND[cond][0])
    if c0 is None or cx is None:
        return None, 0
    QM = {int(k): v for k, v in SM[b]["qid"].items()}
    keep = None
    if conditional and (b, cond) in MANIFEST:
        keep = eligible_qids(QM, donor_map(b, cond))
    qs = [q for q in c0 if q in cx and (keep is None or q in keep)]
    if not qs:
        return None, 0
    a0 = sum(c0[q]["correct"] for q in qs)
    a1 = sum(cx[q]["correct"] for q in qs)
    # VDD in percentage points: Acc(C0) - Acc(T) on the same items (Eq. 1),
    # NOT (a0 - a1) / a0, which is the relative drop VDD / Acc(C0)
    return 100.0 * (a0 - a1) / len(qs), len(qs)


meta = {"host": os.environ.get("RUN_HOST", "unspecified"),
        "x_axis": "knn to the training bank under MedSigLIP, relative to C0",
        "x_note": ("relative, not absolute: C0 is the distance from the intact "
                   "test set to the bank and is not zero"),
        "y_axis": ("VDD in percentage points, Acc(C0) - Acc(T) on the same items "
                   "(Eq. 1), mean per configuration over three seeds; swaps "
                   "conditional on the swapped subset (Table 3 of v3), "
                   "occlusion and noise over all questions (Table 4)"),
        "kvasir_subsets": "corrected manifests: 611 rows in C5, 777 in C6",
        "sources": {"x": "docs/v4/g1_results.json (%s)"
                         % commit("docs/v4/g1_results.json"),
                    "y": "results/*_C0.json and results/*_<cond>.json",
                    "collapse": "docs/v4/occlusion_collapse.json (%s)"
                                % commit("docs/v4/occlusion_collapse.json")},
        "c7": {"source": "docs/v4/analysis_e1.json (%s)"
                         % commit("docs/v4/analysis_e1.json"),
               "note": "x only: no model was run on C7, so it has no VDD"},
        "points": []}

CONFIGS = sorted({c["config"] for c in paths.cells()})
# colour and marker follow the family of the intervention; C5 is a diamond with a
# light fill; the classifier carries a black edge (legend entries below)
COL = {"swap": "#2A62B8", "occlusion": "#D9822B", "noise": "#2E8B57", "removal": "#7B4FA0"}
C5_FILL = "#9DB7E3"
CLASSIFIER = ("M3AE", "M3AE_SCRATCH")

fig, axes = plt.subplots(1, len(BENCH), figsize=(7.4, 2.7), dpi=200, sharey=True)
for ax, b in zip(axes, BENCH):
    cfgs = paths.CONFIGS[b]
    for cfg in cfgs:
        prefixes = [c["prefix"] for c in paths.cells(b) if c["config"] == cfg]
        for cname, (suffix, fam, cond_only) in COND.items():
            if (b, cname) not in X:
                continue
            vals, ns = [], []
            for pf in prefixes:
                v, n = vdd(pf, b, cname, cond_only)
                if v is not None:
                    vals.append(v)
                    ns.append(n)
            if not vals:
                continue
            x = X[(b, cname)]["knn_minus_c0"]
            y = float(np.mean(vals))
            is_c5 = cname == "C5"
            is_cls = cfg in CLASSIFIER
            ax.scatter(x, y, marker="D" if is_c5 else MARK[fam],
                       s=26 if fam != "removal" else 40,
                       color=C5_FILL if is_c5 else COL[fam],
                       edgecolor="black" if is_cls else (COL[fam] if is_c5 else "none"),
                       linewidths=0.7, alpha=.95, zorder=4 if is_c5 else 3)
            meta["points"].append(
                {"benchmark": b, "config": cfg, "condition": cname, "family": fam,
                 "x_knn_minus_c0": x, "x_knn_abs": X[(b, cname)]["knn_abs"],
                 "y_vdd_mean": y, "y_vdd_seeds": vals, "n_questions": ns[0],
                 "conditional_on_swapped_subset": cond_only,
                 "y_unit": "percentage points"})
    # C7 has an x and no y: no model was ever run on it, so it is a vertical
    # line rather than a point; the legend says so
    c7 = E1["cells"].get("%s|medsiglip" % b)
    xc7 = None
    if c7 and c7.get("knn_c0") is not None:
        xc7 = c7["knn_c7"] - c7["knn_c0"]
        ax.axvline(xc7, color=COL["removal"], lw=1.0, ls="--", zorder=1)
    ax.axvline(0, color="0.82", lw=.8, zorder=0)
    ax.axhline(0, color="0.82", lw=.8, zorder=0)
    ax.set_title(NICE[b], fontsize=9)
    ax.set_xlabel("knn to bank, relative to C0", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=.25, lw=.4)
    for s_ in ax.spines.values():
        s_.set_color("0.85")
    xs = [p["x_knn_minus_c0"] for p in meta["points"] if p["benchmark"] == b]
    if xc7 is not None:
        xs.append(xc7)
    ax.set_xlim(-0.03, max(xs) + 0.03)
    # the SLAKE occlusion cells no longer hold 96 distinct inputs; anchored to
    # their clusters with a short leader
    if b == "SLAKE":
        arr = dict(arrowstyle="-", color="0.45", lw=.6, shrinkA=0, shrinkB=2)
        for cname, n, off, ha, va, pick in (
                ("C3-50", OCC["SLAKE"]["C3-50"]["distinct_images"], (-22, 18), "right", "bottom", max),
                ("C3-75", OCC["SLAKE"]["C3-75"]["distinct_images"], (-2, -22), "center", "top", min)):
            pts = [p for p in meta["points"] if p["benchmark"] == b and p["condition"] == cname]
            if pts:
                ax.annotate("%d of 96\ndistinct inputs" % n,
                            (pts[0]["x_knn_minus_c0"], pick(p["y_vdd_mean"] for p in pts)),
                            textcoords="offset points", xytext=off, ha=ha, va=va,
                            fontsize=6.5, color="0.35", arrowprops=arr)
axes[0].set_ylabel("VDD (pp)", fontsize=8)
axes[0].set_ylim(-4, None)

proxies = [
    Line2D([], [], ls="", marker="s", color=COL["swap"], ms=5, label="swap (C2, C6)"),
    Line2D([], [], ls="", marker="D", color=C5_FILL, markeredgecolor=COL["swap"], ms=5,
           label="swap, different finding (C5)"),
    Line2D([], [], ls="", marker="o", color=COL["occlusion"], ms=5, label="occlusion (C3, 25/50/75%)"),
    Line2D([], [], ls="", marker="^", color=COL["noise"], ms=5, label="noise (C4, 25/50/75%)"),
    Line2D([], [], ls="", marker="X", color=COL["removal"], ms=6, label="removal (C1), one distinct image"),
    Line2D([], [], color=COL["removal"], lw=1.0, ls="--", label="patch shuffle (C7), embedded only"),
    Line2D([], [], ls="", marker="o", color="white", markeredgecolor="black", ms=5,
           label="black edge: classifier (M3AE)"),
]
fig.legend(handles=proxies, loc="upper center", ncol=4, fontsize=6.8,
           frameon=False, bbox_to_anchor=(.5, 1.02), columnspacing=1.2,
           handletextpad=.4)
fig.tight_layout(rect=(0, 0, 1, .88))
for ext in ("png", "pdf"):
    fig.savefig(FIGS / ("fig3_vdd_vs_distance.%s" % ext), bbox_inches="tight")
(FIGS / "fig3_vdd_vs_distance.json").write_text(json.dumps(meta, indent=1))
print("  %d pontos" % len(meta["points"]))
for b in BENCH:
    n = len([p for p in meta["points"] if p["benchmark"] == b])
    print("    %-8s %d pontos" % (b, n))
print("\n  gravado %s/fig3_vdd_vs_distance.{png,pdf,json}" % FIGS)
