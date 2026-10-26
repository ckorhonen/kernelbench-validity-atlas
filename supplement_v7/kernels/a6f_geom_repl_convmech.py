"""A6f — leg 2 continuation (geometry replication at two more K) + conv mechanism discriminator.

Predictions frozen here BEFORE the run:
  G1. The geometry term found in a6e (fixed-65536 rho rising with output size at K=3632) REPLICATES at K=1024
      and K=8192: fixed-n spread >=0.5 at both => promote the geometry coordinate from hypothesis to measured.
      If flat (<0.2) at K=1024 but present at K=8192, the term is depth-gated and stated as such.
  G2. The median (not the max) rises with output geometry at all K => the effect is distributional, not a
      tail artifact.

  M1 (conv mechanism discriminator). If the conv rho at C=256 changes when cudnn is DISABLED entirely
      (torch.backends.cudnn.enabled=False, fallback path = a different algorithm), then the effect is
      algorithm-class; if rho is unchanged, the fp32 accumulation tree of the same family is the source and
      "algorithm" is not the axis at all. Prediction: CHANGES (>10%), because cd_V1 already showed the
      transform-class path is what diverges; if unchanged, that attribution also dies.
"""
import json, os, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "a6f_geom_repl_convmech.json")
ATOL = RTOL = 1e-4
dev = "cuda"


def stats(a, b, fixed_n=65536):
    d = (a.double() - b).abs()
    den = ATOL + RTOL * b.abs()
    r = (d / den).flatten()
    return {"rho_max": float(r.max()), "rho_fixed": float(r[:fixed_n].max()),
            "med": float(r.median()), "frac_over": float((r > 1).double().mean())}


geom = []
for K in (1024, 8192):
    for M, N in ((256, 256), (1024, 1024), (4096, 4096)):
        torch.manual_seed(9200 + K)
        A = torch.randn(M, K, device=dev)
        B = torch.randn(K, N, device=dev)
        s = stats(A @ B, A.double() @ B.double())
        s.update({"K": K, "M": M, "N": N})
        geom.append(s)
        del A, B
        torch.cuda.empty_cache()

per_k = {}
for K in (1024, 8192):
    rs = [r for r in geom if r["K"] == K]
    fx = [r["rho_fixed"] for r in rs]
    md = [r["med"] for r in rs]
    per_k[K] = {"fixed_rho": fx, "spread": (max(fx) - min(fx)) / max(fx),
                "med": md, "med_ratio_max_over_min": md[-1] / md[0] if md[0] else None}
g1 = {"per_k": per_k, "replicates_both": bool(all(v["spread"] >= 0.5 for v in per_k.values())),
      "flat_at_1024": bool(per_k[1024]["spread"] < 0.2)}

# conv mechanism discriminator
mech = []
prev_enabled = torch.backends.cudnn.enabled
prev_tf32 = torch.backends.cudnn.allow_tf32
for C in (16, 256):
    torch.manual_seed(1234 + 5 * 100 + 2 * 10 + C)
    x = torch.randn(2, C, 5, 5, device=dev)
    w = torch.randn(C, C, 3, 3, device=dev)
    o64 = F.conv2d(x.double(), w.double(), padding=1)
    o_cudnn = F.conv2d(x, w, padding=1)
    torch.backends.cudnn.enabled = False
    o_fallback = F.conv2d(x, w, padding=1)
    torch.backends.cudnn.enabled = prev_enabled
    torch.backends.cudnn.allow_tf32 = prev_tf32
    rc, rf = stats(o_cudnn, o64), stats(o_fallback, o64)
    mech.append({"C": C, "rho_cudnn": rc["rho_max"], "rho_fallback": rf["rho_max"],
                 "bit_equal": bool(torch.equal(o_cudnn, o_fallback)),
                 "delta_frac": abs(rf["rho_max"] - rc["rho_max"]) / rc["rho_max"]})
    del x, w, o64, o_cudnn, o_fallback
    torch.cuda.empty_cache()

c256 = [m for m in mech if m["C"] == 256][0]
m1 = {"c256_delta": c256["delta_frac"], "changes": bool(c256["delta_frac"] > 0.10),
      "bit_equal": c256["bit_equal"]}

out = {"header": {"date": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()), "torch": torch.__version__,
                  "gpu": torch.cuda.get_device_name(0),
                  "prereg": "understand_plan_2026-10-24 leg 2 continuation; predictions in docstring"},
       "geometry": geom, "g1": g1, "conv_mechanism": mech, "m1": m1}
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({"per_k": {k: {"spread": round(v["spread"], 3),
                                "fixed": [round(x, 3) for x in v["fixed_rho"]],
                                "med_ratio": round(v["med_ratio_max_over_min"], 2) if v["med_ratio_max_over_min"] else None}
                           for k, v in per_k.items()},
                  "g1": {"replicates_both": g1["replicates_both"], "flat_at_1024": g1["flat_at_1024"]},
                  "conv_mech": mech, "m1": m1}, indent=1))
