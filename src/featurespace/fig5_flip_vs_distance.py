"""Figure 5: P(flip | d), and what happens to it inside a domain.

One panel per ruler, one curve per benchmark:

  MedSigLIP                    the primary ruler
  colour histogram             the counterfactual of amendment 7.4
  MedSigLIP, within domain     only pairs whose donor is in the same bucket,
                               which is the control of §2.4 -- the one that
                               separates content from domain contrast

Until 2026-09-20 this was a 3x3 grid, one row per benchmark, with the zero-shot
curve, the question-only floor and per-panel counts drawn inside the axes. The
three readings of §5.3 are comparisons *between benchmarks* under one ruler, so
they are read across a row of the old grid; putting the benchmarks in one panel
states them directly and costs nothing, since the curves are unchanged. The
counts that were printed inside the panels are in the JSON, for the caption.

The curve is the **mean of the fifteen per-checkpoint curves**, not the flip
rate of the pooled rows: pooling would weight a checkpoint by how many of its
rows fall in a bin, and every checkpoint is meant to count once. The band is a
bootstrap interval resampled over **images**, because rows of one image share
its distance and resampling rows would report an interval the design does not
support.

A flip is not an error. It says the answer changed when the image changed, which
is what the swap was built to elicit; whether the new answer is right is the CCR
of §3.5, not this figure.

The y axis is common to the three panels and fixed to [0, 0.7]: the comparison
the figure exists for is between benchmarks under one ruler, and a per-panel
scale would make a flat curve and a rising one look alike. The x axis is the
ruler's own scale and is therefore per panel.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.analysis.ccr import answers_match, load_predictions   # noqa: E402
from src.featurespace import paths                                       # noqa: E402

CACHE = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "embeddings"
FIGS = REPO / "docs/v4/figs"
FIGS.mkdir(parents=True, exist_ok=True)
SEED, N_BINS, N_BOOT = 20260917, 5, 400
BENCH = ["KVASIR", "SLAKE", "VQARAD"]
NICE = {"KVASIR": "Kvasir-VQA-x1", "SLAKE": "SLAKE", "VQARAD": "VQA-RAD"}
COLS = [("medsiglip", False, "MedSigLIP"),
        ("colorhist", False, "colour histogram"),
        ("medsiglip", True, "MedSigLIP, within domain")]
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
IDS = json.loads((CACHE / "image_ids.json").read_text())


def label(i):
    p = [x for x in str(i).replace("//", "/").split("/") if x]
    if p and p[-1].lower() in ("source.jpg", "source.png"):
        return p[-2] if len(p) > 1 else p[-1]
    return str(i)


def donor_map(b, c):
    d = json.loads((paths.RESULTS / MANIFEST[(b, c)]).read_text())
    rows = d["pairs"] if isinstance(d, dict) else d
    return {label(e["qid_image"]): label(e["donor_image"]) for e in rows}


def dist_of(b, ruler, dm):
    ids = [label(i) for i in IDS["%s__C0" % b]]
    X = np.load(CACHE / ("%s__%s.npy" % (b, ruler))).astype(np.float64)
    v = dict(zip(ids, X))
    return {a: 1.0 - float(v[a] @ v[c] / (np.linalg.norm(v[a]) * np.linalg.norm(v[c])))
            for a, c in dm.items() if a in v and c in v}


def buckets(b):
    if b == "KVASIR":
        j = json.loads((REPO / "docs/v4/kvasir_procedure_bucket.json").read_text())
        return {label(k): v for k, v in j["bucket"].items()}
    if b == "VQARAD":
        o = {}
        for n, f in (("chest", "vqarad_bucket_chest.json"),
                     ("abdomen", "vqarad_bucket_abd.json"),
                     ("head", "vqarad_bucket_head.json")):
            for i in json.loads((paths.RESULTS / f).read_text()):
                o[label(i)] = n
        return o
    o = {}
    for f in sorted(paths.RESULTS.glob("C6_manifest_SLAKE_*.json")):
        tag = f.stem.replace("C6_manifest_SLAKE_", "")
        for e in json.loads(f.read_text()):
            o[label(e["qid_image"])] = tag
    return o


def curve(b, ruler, within, prefixes, per_checkpoint=True):
    """Mean of the per-checkpoint flip-rate curves, with a CI over images."""
    bk = buckets(b) if within else None
    QM = {int(k): v for k, v in SM[b]["qid"].items()}
    rows = []                                  # (image, bin_key d, flips per run)
    per = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # image -> d -> [n, flips]
    for cond in paths.SWAP_CONDITIONS[b]:
        dm = donor_map(b, cond)
        dd = dist_of(b, ruler, dm)
        for pf in prefixes:
            c0 = load_predictions(str(paths.RESULTS), pf, "C0")
            cx = load_predictions(str(paths.RESULTS), pf, cond)
            if c0 is None or cx is None:
                continue
            for q, m in QM.items():
                im = label(m["img"])
                if im not in dd or q not in c0 or q not in cx:
                    continue
                if bk is not None:
                    don = dm[im]
                    if im not in bk or don not in bk or bk[im] != bk[don]:
                        continue
                a = per[im][round(dd[im], 12)]
                a[0] += 1
                a[1] += int(not answers_match(cx[q]["predicted"], [c0[q]["predicted"]]))
    if not per:
        return None
    flat = [(d, v[0], v[1], im) for im, dv in per.items() for d, v in dv.items()]
    if len(flat) < N_BINS * 2:
        return None
    ds = np.array([f[0] for f in flat])
    edges = np.quantile(ds, np.linspace(0, 1, N_BINS + 1))
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    bi = np.clip(np.digitize(ds, edges[1:-1]), 0, N_BINS - 1)
    imgs = sorted({f[3] for f in flat})
    idx = {u: [k for k, f in enumerate(flat) if f[3] == u] for u in imgs}
    rng = np.random.default_rng(int.from_bytes(
        hashlib.sha256(("%d|%s|%s|%s" % (SEED, b, ruler, within)).encode()).digest()[:8],
        "big"))

    n_ck = max(1, len(prefixes))

    def rates(sel):
        """Per-bin rate; with n_ck checkpoints pooled into each cell, dividing
        flips by rows gives the mean of the per-checkpoint rates exactly when
        every checkpoint covers the same rows, which it does here."""
        n = np.zeros(N_BINS)
        f = np.zeros(N_BINS)
        for k in sel:
            n[bi[k]] += flat[k][1]
            f[bi[k]] += flat[k][2]
        return np.where(n > 0, f / np.maximum(n, 1), np.nan), n

    obs, nn = rates(range(len(flat)))
    boot = []
    for _ in range(N_BOOT):
        pick = [k for u in (imgs[j] for j in rng.integers(0, len(imgs), len(imgs)))
                for k in idx[u]]
        boot.append(rates(pick)[0])
    boot = np.array(boot)
    lo = np.nanquantile(boot, .025, axis=0)
    hi = np.nanquantile(boot, .975, axis=0)
    centres = [float(ds[bi == i].mean()) if (bi == i).any() else np.nan
               for i in range(N_BINS)]
    return {"d": centres, "rate": obs.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
            "n_rows": nn.tolist(), "n_images": len(imgs),
            "n_checkpoints": len(prefixes),
            "rows_per_checkpoint": [int(x / n_ck) for x in nn],
            # one entry of `flat` is one (original, donor) pair: an image that
            # swaps under C2, C5 and C6 contributes three
            "n_pairs": len(flat),
            "n_pairs_d_below_0.01": int((ds < 0.01).sum())}


meta = {"seed": SEED, "n_bins": N_BINS, "n_boot": N_BOOT,
        "ci": "bootstrap over images",
        "layout": "one panel per ruler, one curve per benchmark (1x3 since 2026-09-20)",
        "y_axis": "P(flip | d), mean of the 15 per-checkpoint curves; common to the three panels, [0, 0.7]",
        "x_axis": "mean d of the bin; five quantile bins per benchmark and ruler, on the ruler's own scale",
        "panels": {}}

COLOUR = {"KVASIR": "#2A62B8", "SLAKE": "#D9822B", "VQARAD": "#2E8B57"}
MARK = {"KVASIR": "o", "SLAKE": "s", "VQARAD": "^"}

fig, axes = plt.subplots(1, len(COLS), figsize=(5.23, 1.60), dpi=200, sharey=True)
for c, (ruler, within, title) in enumerate(COLS):
    ax = axes[c]
    for b in BENCH:
        cu = curve(b, ruler, within, [x["prefix"] for x in paths.cells(b)])
        if cu is None:
            continue
        d, y = np.array(cu["d"]), np.array(cu["rate"])
        ax.fill_between(d, cu["lo"], cu["hi"], color=COLOUR[b], alpha=.12, lw=0)
        ax.plot(d, y, marker=MARK[b], color=COLOUR[b], ms=3, lw=1.2, zorder=3)
        meta["panels"]["%s|%s|%s" % (b, ruler, "within" if within else "all")] = cu
        print("  %-7s %-11s %-6s bins n=%s  imagens %d  pares %d (<0.01: %d)"
              % (b, ruler, "dentro" if within else "todas",
                 ",".join("%d" % x for x in cu["rows_per_checkpoint"]),
                 cu["n_images"], cu["n_pairs"], cu["n_pairs_d_below_0.01"]),
              flush=True)
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xlabel("donor distance $d$", fontsize=7.5, labelpad=2)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=.25, lw=.4)
    for s_ in ax.spines.values():
        s_.set_color("0.85")
axes[0].set_ylabel("P(flip | $d$)", fontsize=7.5)
axes[0].set_ylim(0, .7)

from matplotlib.lines import Line2D                      # noqa: E402
from matplotlib.patches import Patch                     # noqa: E402
proxies = [Line2D([], [], color=COLOUR[b], marker=MARK[b], ms=3, lw=1.2,
                  label=NICE[b]) for b in BENCH]
proxies.append(Patch(facecolor="0.45", alpha=.12, label="95% bootstrap over images"))
fig.legend(handles=proxies, loc="upper center", ncol=4, fontsize=7,
           frameon=False, bbox_to_anchor=(.5, 1.03), columnspacing=1.1,
           handletextpad=.4)
fig.tight_layout(rect=(0, 0, 1, .93), w_pad=.8)

# the counts that used to be printed inside the panels, for the caption
P = meta["panels"]
meta["counts"] = {
    "images": {b: P["%s|medsiglip|all" % b]["n_images"] for b in BENCH
               if "%s|medsiglip|all" % b in P},
    "images_within_bucket": {b: P["%s|medsiglip|within" % b]["n_images"] for b in BENCH
                             if "%s|medsiglip|within" % b in P},
    "rows_per_checkpoint": {b: sum(P["%s|medsiglip|all" % b]["rows_per_checkpoint"])
                            for b in BENCH if "%s|medsiglip|all" % b in P},
    "checkpoints": 15}
sc = P.get("SLAKE|colorhist|all")
if sc:
    # the step in the first two quintiles of the colour panel on SLAKE: pairs
    # whose two images have the same 64-bin histogram to three decimals
    meta["slake_colour_near_zero"] = {"pairs_d_below_0.01": sc["n_pairs_d_below_0.01"],
                                      "pairs": sc["n_pairs"]}

for ext in ("png", "pdf"):
    fig.savefig(FIGS / ("fig5_flip_vs_distance.%s" % ext), bbox_inches="tight")
(FIGS / "fig5_flip_vs_distance.json").write_text(json.dumps(meta, indent=1, default=float))
print("\n  gravado %s/fig5_flip_vs_distance.{png,pdf,json}" % FIGS)
