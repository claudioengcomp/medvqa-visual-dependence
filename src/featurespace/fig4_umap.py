"""Figure 4: where the donors come from, under MedSigLIP and under colour.

Illustration only. **No number in the paper is read from this figure**, and the
JSON beside it records the parameters so the picture can be regenerated rather
than trusted: UMAP is stochastic and its geometry is not a measurement.

One row per benchmark, two columns: the primary ruler and the colour
counterfactual. Points are the C0 images. Segments join an original to its C2
donor for a sample of pairs, so the reader sees whether a swap stays inside a
region or crosses between them — which is the claim analysis B tested with W1
and the dip, and which this only shows.

Per benchmark:
  Kvasir      coloured by procedure; the no-finding class, which carries no
              procedure label, is drawn in grey rather than dropped -- it is
              40% of the images and hiding it would make the split look tidier
              than it is
  SLAKE       nine modality x location buckets
  VQA-RAD     three organs
  PathVQA     one colour: it publishes no domain label
  HEAL        points and segments, no thumbnails, generated on host-A under the
              licence of section 4. The licence is about pixels, not about lines
              between points: the segments carry no image, and they are what
              shows the regime -- all short, inside one cloud, against VQA-RAD's
              long crossings

The sample of segments is drawn with a seed derived per cell (amendment 7.6),
so the same pairs are drawn every time the figure is regenerated.
"""
from __future__ import annotations

import hashlib
import json
import socket
import sys
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import umap                              # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "embeddings"
FIGS = REPO / "docs/v4/figs"
FIGS.mkdir(parents=True, exist_ok=True)

SEED = 20260917
N_NEIGHBORS = 15
MIN_DIST = 0.1
N_SEGMENTS = 60
# the paper carries three rows -- one per regime: colour separates the domain,
# colour does not, everything near -- and the supplement carries all five
ROWS = {"paper": ["KVASIR", "VQARAD", "HEAL"],
        "supplement": ["KVASIR", "SLAKE", "VQARAD", "PATHVQA", "HEAL"]}
WHICH = sys.argv[1] if len(sys.argv) > 1 else "paper"
BENCH = ROWS[WHICH]
RULERS = [("medsiglip", "MedSigLIP"), ("colorhist", "colour histogram")]
MANIFEST = {"KVASIR": "C2_manifest_KVASIR.json", "SLAKE": "C2_manifest_SLAKE.json",
            "VQARAD": "C2_manifest_VQARAD.json", "PATHVQA": "C2_manifest_PATHVQA.json",
            "HEAL": "C2_manifest_HEAL.json"}
NO_BUCKET = {"KVASIR": "no finding", "PATHVQA": None, "HEAL": None}
# the paper's names, not the cache keys
NICE = {"KVASIR": "Kvasir-VQA-x1", "SLAKE": "SLAKE", "VQARAD": "VQA-RAD",
        "PATHVQA": "PathVQA", "HEAL": "HEAL"}


def cell_rng(*parts):
    h = hashlib.sha256("|".join(str(p) for p in (SEED,) + parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def label(i):
    p = [x for x in str(i).replace("//", "/").split("/") if x]
    if p and p[-1].lower() in ("source.jpg", "source.png"):
        return p[-2] if len(p) > 1 else p[-1]
    return str(i)


def buckets(bench):
    R = REPO / "results"
    if bench == "KVASIR":
        b = json.loads((REPO / "docs/v4/kvasir_procedure_bucket.json").read_text())
        return {label(k): v for k, v in b["bucket"].items()}
    if bench == "VQARAD":
        out = {}
        for name, f in (("chest", "vqarad_bucket_chest.json"),
                        ("abdomen", "vqarad_bucket_abd.json"),
                        ("head", "vqarad_bucket_head.json")):
            for i in json.loads((R / f).read_text()):
                out[label(i)] = name
        return out
    if bench == "SLAKE":
        out = {}
        for f in sorted(R.glob("C6_manifest_SLAKE_*.json")):
            tag = f.stem.replace("C6_manifest_SLAKE_", "")
            for e in json.loads(f.read_text()):
                out[label(e["qid_image"])] = tag
        return out
    return {}


IDS = json.loads((CACHE / "image_ids.json").read_text())
meta = {"host": os.environ.get("RUN_HOST", "unspecified"), "illustration_only": True,
        "no_number_is_read_from_this_figure": True,
        "umap": {"n_neighbors": N_NEIGHBORS, "min_dist": MIN_DIST, "metric": "cosine",
                 "seed_base": SEED, "seed_per_cell": "sha256(SEED|bench|ruler)"},
        "n_segments": N_SEGMENTS, "panels": {}}

fig, axes = plt.subplots(len(BENCH), len(RULERS),
                         figsize=(9.2, 2.45 * len(BENCH) + .4), dpi=200)
for r, bench in enumerate(BENCH):
    bk = buckets(bench)
    key = "%s__C0" % bench
    ids = [label(i) for i in IDS[key]]
    man = json.loads((REPO / "results" / MANIFEST[bench]).read_text())
    rows = man["pairs"] if isinstance(man, dict) else man
    pair = {label(e["qid_image"]): label(e["donor_image"]) for e in rows}

    for c, (ruler, nice) in enumerate(RULERS):
        ax = axes[r, c]
        p = CACHE / ("%s__C0__%s.npy" % (bench, ruler))
        if not p.exists():
            p = CACHE / ("%s__%s.npy" % (bench, ruler))
        X = np.load(p).astype(np.float32)
        rng = cell_rng(bench, ruler)
        red = umap.UMAP(n_neighbors=min(N_NEIGHBORS, len(X) - 1),
                        min_dist=MIN_DIST, metric="cosine",
                        random_state=SEED).fit_transform(X)
        pos = {i: red[k] for k, i in enumerate(ids)}

        groups = sorted({bk[i] for i in ids if i in bk})
        cmap = plt.get_cmap("tab10" if len(groups) <= 10 else "tab20")
        for gi, g in enumerate(groups):
            m = np.array([i in bk and bk[i] == g for i in ids])
            ax.scatter(red[m, 0], red[m, 1], s=5, linewidths=0,
                       color=cmap(gi % cmap.N), label=g, alpha=.85)
        unl = np.array([i not in bk for i in ids])
        if unl.any():
            ax.scatter(red[unl, 0], red[unl, 1], s=5, linewidths=0,
                       color="0.72", label=NO_BUCKET.get(bench) or "no label", alpha=.7)

        if True:
            keys = [a for a in pair if a in pos and pair[a] in pos]
            take = rng.choice(len(keys), size=min(N_SEGMENTS, len(keys)), replace=False)
            for t in take:
                a = keys[int(t)]
                x0, y0 = pos[a]
                x1, y1 = pos[pair[a]]
                ax.plot([x0, x1], [y0, y1], color="0.25", lw=.35, alpha=.55, zorder=0)

        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("0.85")
        if r == 0:
            ax.set_title(nice, fontsize=9)
        if c == 0:
            ax.set_ylabel(NICE[bench], fontsize=9)
        if c == 1 and groups:
            ax.legend(fontsize=5, markerscale=1.6, frameon=False,
                      loc="center left", bbox_to_anchor=(1.01, .5),
                      handletextpad=.3, labelspacing=.25)
        meta["panels"]["%s|%s" % (bench, ruler)] = {
            "n_points": int(len(X)), "n_buckets": len(groups),
            "without_bucket": int(unl.sum()),
            "segments_drawn": int(min(N_SEGMENTS, len(keys)))}
        print("  %-8s %-11s %4d pontos, %d buckets, %d sem rotulo"
              % (bench, ruler, len(X), len(groups), int(unl.sum())), flush=True)

fig.suptitle("Where the donors come from", fontsize=10, y=.999)
fig.text(.44, .965,
         "Illustration only: no number in the paper is read from this figure. "
         "UMAP with n_neighbors=%d, min_dist=%.2f, cosine metric, seed %d; "
         "%d original\u2192donor segments of C2 per panel."
         % (N_NEIGHBORS, MIN_DIST, SEED, N_SEGMENTS),
         ha="center", fontsize=6.3, color="0.35")
fig.tight_layout(rect=(0, 0, .88, .955))
stem = "fig4_umap" if WHICH == "paper" else "fig4_umap_supplement"
meta["variant"] = WHICH
meta["benchmarks"] = BENCH
for ext in ("png", "pdf"):
    fig.savefig(FIGS / ("%s.%s" % (stem, ext)), bbox_inches="tight")
(FIGS / ("%s.json" % stem)).write_text(json.dumps(meta, indent=1))
print("\n  gravado %s/%s.{png,pdf,json}  (%s)" % (FIGS, stem, ", ".join(BENCH)))
