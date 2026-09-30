"""G0: embeddings of the C0 test sets and the kNN sanity check.

Runs only after the pre-registration and its two amendments were committed
(531cb78, 3e7c63e, 5e5231b). Reads no result and tunes nothing: k is 10, the
thresholds come from the pre-registration, and every classification carries the
denominator it was computed on.

The colour histogram is not a ruler. It is the floor: a ruler that does not
beat it is reading colour.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import time
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

import numpy as np
import pyarrow as pa
import torch
from PIL import Image

ROOT = REPO.parent
P0 = ROOT / "med-vqa-icdip-p0"
CACHE = Path(os.environ.get("V4_CACHE", str(REPO / "cache"))) / "embeddings"
CACHE.mkdir(parents=True, exist_ok=True)
DEV = "cuda"
K = 10

ARROWS = {
    "KVASIR": P0 / "data/finetune_arrows/vqa_kvasir_test_clean.arrow",
    "SLAKE": ROOT / "M3AE/data/finetune_arrows/vqa_slack_test.arrow",
    "VQARAD": ROOT / "M3AE/data/finetune_arrows/vqa_vqa_rad_test.arrow",
}


def load_images(path):
    with pa.memory_map(str(path), "rb") as src:
        t = pa.ipc.open_file(src).read_all()
    ids = [str(x) for x in t.column("image_id").to_pylist()]
    raw = t.column("image").to_pylist()
    return ids, raw


def to_pil(b):
    return Image.open(io.BytesIO(b)).convert("RGB")


# ------------------------------------------------------------------- rulers

def _as_vec(o):
    """transformers 5.x devolve objeto em alguns caminhos; extrai o vetor."""
    if torch.is_tensor(o):
        return o
    for a in ("pooler_output", "image_embeds", "last_hidden_state"):
        v = getattr(o, a, None)
        if v is not None:
            return v[:, 0] if a == "last_hidden_state" and v.dim() == 3 else v
    raise TypeError("sem vetor em %s" % type(o).__name__)


def embed_medsiglip(images, bs=32):
    from transformers import AutoImageProcessor, AutoModel
    mid = "google/medsiglip-448"
    proc = AutoImageProcessor.from_pretrained(mid)  # imagem so: dispensa o tokenizer de texto
    # fp32 like every other ruler: the pre-registration admits no exception.
    model = AutoModel.from_pretrained(mid, dtype=torch.float32).to(DEV).eval()
    out = []
    for i in range(0, len(images), bs):
        px = proc(images=images[i:i + bs], return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            f = _as_vec(model.get_image_features(pixel_values=px.to(DEV, torch.float32)))
        out.append(f.float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)


def embed_dinov3(images, bs=32):
    from transformers import AutoImageProcessor, AutoModel
    mid = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    proc = AutoImageProcessor.from_pretrained(mid)
    # fp32: em fp16 o forward do DINOv3 produziu NaN em todas as imagens (17/09).
    model = AutoModel.from_pretrained(mid, dtype=torch.float32).to(DEV).eval()
    out = []
    for i in range(0, len(images), bs):
        px = proc(images=images[i:i + bs], return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            o = model(pixel_values=px.to(DEV, torch.float32))
        f = _as_vec(o)
        out.append(f.float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)


def embed_biomedclip(images, bs=32):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    model = model.to(DEV).eval()
    out = []
    for i in range(0, len(images), bs):
        px = torch.stack([preprocess(im) for im in images[i:i + bs]]).to(DEV)
        with torch.no_grad():
            f = _as_vec(model.encode_image(px))
        out.append(f.float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)


def embed_colorhist(images, bins=64):
    """The floor. 64 bins per RGB channel, L1-normalised. No model."""
    out = np.zeros((len(images), bins * 3), dtype=np.float32)
    for i, im in enumerate(images):
        a = np.asarray(im)
        v = [np.histogram(a[:, :, c], bins=bins, range=(0, 256))[0] for c in range(3)]
        h = np.concatenate(v).astype(np.float32)
        out[i] = h / max(h.sum(), 1.0)
    return out




def embed_dinov2(images, bs=32):
    """Robustness ruler. DINOv3 was approved, so DINOv2-L moves to robustness."""
    from transformers import AutoImageProcessor, AutoModel
    mid = "facebook/dinov2-large"
    proc = AutoImageProcessor.from_pretrained(mid)
    model = AutoModel.from_pretrained(mid, dtype=torch.float32).to(DEV).eval()
    out = []
    for i in range(0, len(images), bs):
        px = proc(images=images[i:i + bs], return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            o = model(pixel_values=px.to(DEV, torch.float32))
        out.append(_as_vec(o).float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)

RULERS = {"medsiglip": embed_medsiglip, "dinov3": embed_dinov3,
          "biomedclip": embed_biomedclip, "dinov2": embed_dinov2,
          "colorhist": embed_colorhist}


# --------------------------------------------------------------------- kNN
def knn_loo(X, y, k=K):
    """Leave-one-out kNN on cosine similarity. Returns accuracy and n."""
    y = np.asarray(y)
    if len(y) < k + 1 or len(set(y)) < 2:
        return None, len(y)
    Z = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    S = Z @ Z.T
    np.fill_diagonal(S, -np.inf)          # leave one out
    idx = np.argsort(-S, axis=1)[:, :k]
    hit = 0
    for i in range(len(y)):
        vals, cnt = np.unique(y[idx[i]], return_counts=True)
        hit += int(vals[np.argmax(cnt)] == y[i])
    return hit / len(y), len(y)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ labels
def build_labels(ids):
    R = P0 / "results"
    L = {}
    kid = set(ids["KVASIR"])

    colon = set(json.load(open(R / "kvasir_clean_bucket_colonoscopy.json")))
    gastro = set(json.load(open(R / "kvasir_clean_bucket_gastroscopy.json")))
    proc = {i: ("colonoscopy" if i in colon else "gastroscopy")
            for i in ids["KVASIR"] if i in colon or i in gastro}
    L["kvasir_procedure"] = ("KVASIR", proc)

    path = json.load(open(P0 / "data/finetune_arrows/vqa_kvasir_clean_pathology_labels.json"))
    pure = {i: path[i] for i in ids["KVASIR"] if i in path and ";" not in path[i]}
    composite = sum(1 for i in ids["KVASIR"] if i in path and ";" in path[i])
    L["kvasir_finding"] = ("KVASIR", pure)
    L["_composite_dropped"] = composite

    for pname, bucket in [("colonoscopy", colon), ("gastroscopy", gastro)]:
        sub = {i: v for i, v in pure.items() if i in bucket}
        L["kvasir_finding_in_" + pname] = ("KVASIR", sub)

    organ = {}
    for name, f in [("chest", "vqarad_bucket_chest.json"),
                    ("abdomen", "vqarad_bucket_abd.json"),
                    ("head", "vqarad_bucket_head.json")]:
        for i in json.load(open(R / f)):
            organ[str(i)] = name
    L["vqarad_organ"] = ("VQARAD", {i: organ[i] for i in ids["VQARAD"] if i in organ})

    mod = {}
    for f in sorted(R.glob("C6_manifest_SLAKE_*.json")):
        tag = f.stem.replace("C6_manifest_SLAKE_", "")
        m = tag.split("_")[0]                       # CT / MRI / XRAY
        for e in json.load(open(f)):
            mod[str(e["qid_image"])] = m
    L["slake_modality"] = ("SLAKE", {i: mod[i] for i in ids["SLAKE"] if i in mod})
    return L


def main():
    t0 = time.time()
    ids, imgs = {}, {}
    for b, p in ARROWS.items():
        i, raw = load_images(p)
        ids[b] = i
        imgs[b] = [to_pil(x) for x in raw]
        print("  carregado %-7s %d imagens" % (b, len(i)), flush=True)

    manifest = {"k": K, "cache": str(CACHE), "arrays": {}}
    emb = {}
    for rname, fn in RULERS.items():
        emb[rname] = {}
        for b in ARROWS:
            f = CACHE / ("%s__%s.npy" % (b, rname))
            if f.exists():
                emb[rname][b] = np.load(f)
            else:
                t = time.time()
                emb[rname][b] = fn(imgs[b])
                np.save(f, emb[rname][b])
                print("    %-11s %-7s %s  %.1fs" % (rname, b, emb[rname][b].shape,
                                                    time.time() - t), flush=True)
            manifest["arrays"]["%s__%s" % (b, rname)] = {
                "shape": list(emb[rname][b].shape), "sha256": sha(f)}

    L = build_labels(ids)
    composite = L.pop("_composite_dropped")
    res = {"composite_labels_dropped": composite, "results": {}}
    for task, (bench, lab) in L.items():
        keep = [j for j, i in enumerate(ids[bench]) if i in lab]
        y = [lab[ids[bench][j]] for j in keep]
        res["results"][task] = {"benchmark": bench, "n": len(keep),
                                "classes": sorted(set(y)), "acc": {}}
        for rname in RULERS:
            a, n = knn_loo(emb[rname][bench][keep], y)
            res["results"][task]["acc"][rname] = a

    out = P0 / "docs/v4/g0_results.json"
    out.write_text(json.dumps({"manifest": manifest, **res}, indent=1))
    (CACHE / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("\n  gravado %s  (%.1f min)" % (out, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
