"""A6d — leg 2 of understand_plan_2026-10-24: MEASURE the matmul rho=1 crossing rather than interpolate it.

Predictions frozen in this docstring BEFORE the run (plan leg 2 as submitted):
  P1. rho=1 crossing lies in K in (2048, 8192); interpolation from a6b gave K*~2875, so the measured
      crossing should be within 2x of that (i.e. inside [1438, 5750]) or the log-log model is retracted.
  P2. The crossing does NOT move with M/N tile geometry: at fixed K, varying M,N over 256/1024/4096 should
      change rho by <2x. If it moves, accumulation depth is not the sole mechanism and tile geometry joins
      it as a coordinate.
Bisection: integer K grid, 8 steps, halving; rho computed as in a6 (max elementwise budget ratio vs fp64).
Also records whether rho depends on M,N at the crossing K.
"""
import json, os, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "a6d_crossing.json")
ATOL = RTOL = 1e-4
dev = "cuda"


def rho_at(M, K, N, seed=9100):
    torch.manual_seed(seed)
    A = torch.randn(M, K, device=dev)
    B = torch.randn(K, N, device=dev)
    d32 = A @ B
    d64 = A.double() @ B.double()
    r = ((d32.double() - d64).abs() / (ATOL + RTOL * d64.abs())).flatten()
    out = {"M": M, "K": K, "N": N, "rho": float(r.max()), "med": float(r.median()),
           "frac_over": float((r > 1).double().mean())}
    del A, B, d32, d64
    torch.cuda.empty_cache()
    return out


hdr = {"date": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()), "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0),
       "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
       "f32_prec": torch.get_float32_matmul_precision(),
       "prereg": "understand_plan_2026-10-24 leg 2 (predictions in docstring, frozen pre-run)"}

# --- bisection on K at fixed M=N=1024
lo, hi = 2048, 8192
steps = []
for _ in range(8):
    mid = (lo + hi) // 2
    row = rho_at(1024, mid, 1024)
    steps.append(row)
    if row["rho"] < 1.0:
        lo = mid
    else:
        hi = mid
    if hi - lo <= 8:
        break
lo_row = rho_at(1024, lo, 1024)
hi_row = rho_at(1024, hi, 1024)

# --- geometry test at the crossing K
geom = [rho_at(M, hi, N) for (M, N) in ((256, 256), (1024, 1024), (4096, 4096))]
g_rho = [g["rho"] for g in geom]
geom_spread = (max(g_rho) - min(g_rho)) / max(g_rho) if max(g_rho) else None

pred_center, pred_band = 2875.0, (1438.0, 5750.0)
p1 = {"measured_crossing_lo": lo, "measured_crossing_hi": hi,
      "interpolated_center": pred_center, "inside_2x_band": bool(pred_band[0] <= hi <= pred_band[1]),
      "model_retracted": bool(not (pred_band[0] <= hi <= pred_band[1]))}
p2 = {"geometry_spread_frac": geom_spread, "holds": bool(geom_spread is not None and geom_spread < 0.5)}

out = {"header": hdr, "bisection": steps, "lo_row": lo_row, "hi_row": hi_row,
       "geometry_at_crossing": geom, "p1_crossing": p1, "p2_geometry": p2}
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({"bisect": [(s["K"], round(s["rho"], 3)) for s in steps],
                  "lo": lo, "hi": hi, "p1": p1, "geom": [(g["M"], round(g["rho"], 3)) for g in geom],
                  "p2": p2}, indent=1))
