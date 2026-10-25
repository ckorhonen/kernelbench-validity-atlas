"""A6b — bounded extension: is rho a function of reduction depth K?

DECLARED AS EXTENSION, not part of the original prereg (lab/a6_refdistance_prereg.md @ dbfd600).
The original prereg's two family predictions both failed at the shapes it declared (conv rho 0.74 vs
predicted >=2; matmul rho 2.95 vs predicted <0.1). The failure has an obvious candidate explanation that
the prereg did not test: the elementwise-relative budget statistic scales with reduction depth, because an
fp32 accumulation over K terms carries its own rounding against an fp64 anchor regardless of TF32 policy.
This extension tests that, with its own predictions written BEFORE the run.

Predictions (frozen here, before running):
  P1. rho is monotone increasing in K for matmul fp32 (TF32 off) across K in {128, 512, 2048, 8192}.
  P2. rho crosses 1.0 somewhere in 512..8192; the crossing is a property of fp32 arithmetic, not of any
      precision-default setting (matmul allow_tf32 is False and float32_matmul_precision is 'highest').
  P3. conv rho at matched effective K (K_eff = C*9) tracks the matmul curve within 3x at the same K_eff.
Kill: if rho is flat in K (spread <10% across the sweep), the K hypothesis dies and rho is a per-op
property; report the flat table with no law.
"""
import json, os, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "a6b_ksweep.json")
DATE = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())
ATOL = RTOL = 1e-4
dev = "cuda"


def rho(a_def, a_64):
    d = (a_def.double() - a_64).abs()
    denom = ATOL + RTOL * a_64.abs()
    r = (d / denom).flatten()
    return {"rho": float(r.max()), "med": float(r.median()), "abs_max": float(d.max()),
            "ref64_absmax": float(a_64.abs().max()), "frac_over": float((r > 1).double().mean())}


rows = []
hdr = {"date": DATE, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
       "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
       "f32_prec": torch.get_float32_matmul_precision(),
       "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
       "prereg": "extension of lab/a6_refdistance_prereg.md @ dbfd600 (predictions frozen in this docstring)"}

# matmul K sweep, fixed M=N=1024
for K in (128, 512, 2048, 8192):
    torch.manual_seed(9000 + K)
    A = torch.randn(1024, K, device=dev)
    B = torch.randn(K, 1024, device=dev)
    d32 = A @ B
    d64 = A.double() @ B.double()
    r = rho(d32, d64)
    r.update({"family": "matmul", "K": K, "M": 1024, "N": 1024})
    rows.append(r)
    del A, B, d32, d64
    torch.cuda.empty_cache()

# conv sweep: K_eff = C*9 (3x3 kernel), fixed N=2 H=5
for C in (16, 64, 256):
    torch.manual_seed(1234 + 5 * 100 + 2 * 10 + C)
    x = torch.randn(2, C, 5, 5, device=dev)
    w = torch.randn(C, C, 3, 3, device=dev)
    d32 = F.conv2d(x, w, padding=1)
    d64 = F.conv2d(x.double(), w.double(), padding=1)
    r = rho(d32, d64)
    r.update({"family": "conv2d", "C": C, "K_eff": C * 9, "N": 2, "H": 5})
    rows.append(r)
    del x, w, d32, d64
    torch.cuda.empty_cache()

mm = [r for r in rows if r["family"] == "matmul"]
cv = [r for r in rows if r["family"] == "conv2d"]
mono = all(mm[i]["rho"] <= mm[i + 1]["rho"] for i in range(len(mm) - 1))
spread = (max(r["rho"] for r in rows) - min(r["rho"] for r in rows)) / max(r["rho"] for r in rows)
crossed = [r["K"] for r in mm if r["rho"] > 1]

out = {"header": hdr, "rows": rows, "matmul_monotone_in_K": mono, "killed": bool(spread < 0.10),
       "k_where_over_budget": crossed, "spread_frac": spread}
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({"matmul": [(r["K"], round(r["rho"], 3)) for r in mm],
                  "conv": [(r["K_eff"], round(r["rho"], 3)) for r in cv],
                  "monotone": mono, "killed": out["killed"]}, indent=1))
