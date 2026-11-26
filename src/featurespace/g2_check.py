"""G2, against the criterion registered before any of B was computed.

    separation in at least 3 of the 4 benchmarks (Kvasir-VQA-x1, SLAKE, VQA-RAD,
    PathVQA) AND HEAL unimodal.

Two instruments, because the benchmarks differ in what they publish. Where a
domain label exists, separation means Wasserstein-1 between the near and far
distances with a permutation test at p < 0.01. Where it does not, it means
Hartigan's dip at p < 0.05 - a weaker instrument, which the report states rather
than letting the two be read as equivalent.

The verdict is read off the primary ruler, MedSigLIP. The other four are
reported beside it; the colour histogram is the counterfactual of amendment 7.4
and never decides anything.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
B = json.loads((REPO / "docs/v4/analysis_b.json").read_text())["cells"]
D = json.loads((REPO / "docs/v4/analysis_b_dip.json").read_text())["cells"]
BRIDGE = json.loads((REPO / "docs/v4/bridge_v3_v4.json").read_text())["benchmarks"]

FOUR = ["KVASIR", "SLAKE", "VQARAD", "PATHVQA"]
PRIMARY = "medsiglip"
RULERS = ["medsiglip", "dinov3", "biomedclip", "dinov2", "colorhist"]

out = {"criterion": "separation in >= 3 of 4, and HEAL unimodal",
       "verdict_read_on": PRIMARY, "benchmarks": {}}

print("=== SEPARACAO, POR BASE (regua primaria: %s) ===" % PRIMARY)
sep = []
for b in FOUR:
    c = B.get("%s|C2|%s" % (b, PRIMARY))
    if c and c.get("separated") is not None:
        ok = bool(c["separated"])
        rec = {"instrument": "W1 near vs far, permutation", "separated": ok,
               "w1": c["w1"], "p": c["w1_p"], "near_fraction": c["near_fraction"],
               "pairs": c["classifiable"],
               "note": "domain label available; two-sample test"}
        print("  %-8s W1 %.5f  p %.4f  fracao perto %.4f  em %d pares   %s"
              % (b, c["w1"], c["w1_p"], c["near_fraction"], c["classifiable"],
                 "SEPARA" if ok else "nao separa"))
    else:
        c = D.get("%s|%s" % (b, PRIMARY))
        ok = bool(c["bimodal_at_005"])
        rec = {"instrument": "Hartigan dip", "separated": ok,
               "dip": c["dip"], "p": c["dip_p"], "pairs": c["n_pairs"],
               "mean_d": c["mean"],
               "note": ("no domain label; the dip only sees domain structure when "
                        "it produces two clear modes, so a negative here is weak "
                        "evidence and not evidence of absence")}
        print("  %-8s dip %.5f  p %.4f  media d %.4f  em %d pares   %s"
              % (b, c["dip"], c["dip_p"], c["mean"], c["n_pairs"],
                 "bimodal" if ok else "unimodal -- NAO separa"))
    rec["phi_v3"] = BRIDGE.get(b, {}).get("phi_v3")
    out["benchmarks"][b] = rec
    sep.append(ok)

n_sep = sum(sep)
print("  %d de 4 separam" % n_sep)

h = D.get("HEAL|%s" % PRIMARY)
heal_unimodal = not h["bimodal_at_005"]
print("\n=== CONTROLE NEGATIVO: HEAL ===")
for r in RULERS:
    c = D.get("HEAL|%s" % r)
    if c:
        print("  %-11s media d %.4f  dip %.5f  p %.4f  %s"
              % (r, c["mean"], c["dip"], c["dip_p"],
                 "unimodal" if not c["bimodal_at_005"] else "BIMODAL"))
out["heal"] = {"unimodal_primary": heal_unimodal, "dip": h["dip"], "p": h["dip_p"],
               "mean_d": h["mean"], "all_rulers_unimodal":
               all(not D["HEAL|%s" % r]["bimodal_at_005"] for r in RULERS if "HEAL|%s" % r in D)}

passed = n_sep >= 3 and heal_unimodal
out["separating"] = n_sep
out["passed"] = bool(passed)
out["margin"] = "at the threshold: 3 of 4" if n_sep == 3 else "above the threshold"

print("\n=== REGIMES DA DISTRIBUICAO DE d ===")
print("  %-8s %-11s %8s %8s   %s" % ("base", "regua", "media", "mediana", "regime"))
for b in ["HEAL"] + FOUR:
    for r in (PRIMARY, "dinov3"):
        s = BRIDGE.get(b, {}).get("rulers", {}).get(r)
        if not s:
            continue
        c = D.get("%s|%s" % (b, r))
        shape = ("bimodal" if c and c["bimodal_at_005"] else
                 "unimodal" if c else "mistura (rotulo de dominio)")
        print("  %-8s %-11s %8.4f %8.4f   %s" % (b, r, s["mean"], s["median"], shape))

print("\n=== VEREDITO G2 ===")
print("  separam: %d de 4 (%s)" % (n_sep, ", ".join(b for b, s in zip(FOUR, sep) if s)))
print("  nao separa: %s" % ", ".join(b for b, s in zip(FOUR, sep) if not s))
print("  HEAL unimodal: %s (p %.4f)" % ("sim" if heal_unimodal else "NAO", h["dip_p"]))
print("  G2: %s  -- %s" % ("PASSA" if passed else "FALHA", out["margin"]))

p = REPO / "docs/v4/g2_results.json"
p.write_text(json.dumps(out, indent=1))
print("\n  gravado %s" % p)
