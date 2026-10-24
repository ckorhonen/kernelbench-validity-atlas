"""A6e — leg 3 of understand_plan_2026-10-24 (algorithm attribution) + the max-statistic control
that leg 2 surfaced.

Leg 2 result: matmul rho (max-over-elements) rose 0.648 -> 1.232 -> 2.777 across M=256/1024/4096 at fixed
K=3632, so P2 as frozen FAILED (spread 0.767). But rho is a maximum over output elements, and max-of-N grows
with N (65k -> 16.7M elements). That is a definitional confound, not necessarily a geometry effect. This run
separates them: same rho AND median AND p99 AND a fixed-size element subsample (first 65536 elements) so the
sample size is held constant.

Predictions frozen here BEFORE the run:
  C1. Fixed-size subsample rho is geometry-flat (spread <0.5) across M=256/1024/4096 at K=3632 => the leg-2
      geometry "failure" is a max-of-N artifact and rho must be reported with a fixed sample size.
  C2. The median at M=4096 stays ~2x the median at M<=1024 (as leg 2 showed) => there IS a real geometry
      term in accumulation depth beyond the max statistic. If the median is flat instead, C2 fails and
      geometry enters only through the max statistic.

Leg 3 (algorithm attribution), predictions frozen:
  A1. Forcing cudnn off benchmark mode + deterministic=True at conv C=256 changes rho by >10% => the conv
      non-monotonicity is heuristic dispatch, and "algorithm selection" is the named coordinate.
  A2. At conv C=16, rho with allow_tf32=False stays near the 0.022 already measured regardless of benchmark
      mode => the C=16 effect is a precision default, not dispatch.
"""
import json, os, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "a6e_geometry_algo.json")
ATOL = RTOL = 1e-4
dev = "cuda"


def stats(a, b, fixed_n=None):
    d = (a.double() - b).abs()
    den = ATOL + RTOL * b.abs()
    r = (d / den).flatten()
    out = {"rho_max": float(r.max()), "med": float(r.median()), "n_elems": int(r.numel())}
    try:
        out["p99"] = float(torch.quantile(r[: min(r.numel(), 4_000_000)].float(), 0.99))
    except Exception:
        out["p99"] = None
    if fixed_n is not None:
        out["rho_fixed_n"] = float(r[:fixed_n].max())
    return out


rows = []

# ---- C1/C2: geometry at fixed K with fixed-size subsample ----
for M, N in ((256, 256), (1024, 1024), (4096, 4096)):
    torch.manual_seed(9100)
    A = torch.randn(M, 3632, device=dev)
    B = torch.randn(3632, N, device=dev)
    s = stats(A @ B, A.double() @ B.double(), fixed_n=65536)
    s.update({"M": M, "N": N, "K": 3632, "kind": "matmul_geometry"})
    rows.append(s)
    del A, B
    torch.cuda.empty_cache()

fixed = [r["rho_fixed_n"] for r in rows]
c1_spread = (max(fixed) - min(fixed)) / max(fixed)
meds = [r["med"] for r in rows]
c1 = {"fixed_n_spread": c1_spread, "holds": bool(c1_spread < 0.5),
      "interpretation": "leg2 geometry failure is a max-of-N artifact" if c1_spread < 0.5 else "real geometry term"}
c2 = {"med_256": meds[0], "med_1024": meds[1], "med_4096": meds[2],
      "ratio_4096_over_1024": meds[2] / meds[1] if meds[1] else None,
      "holds": bool(meds[1] and meds[2] / meds[1] > 1.5)}

# ---- A1/A2: conv algorithm attribution ----
algo_rows = []
prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
prev_bench = torch.backends.cudnn.benchmark
prev_det = torch.backends.cudnn.deterministic
for C in (16, 256):
    torch.manual_seed(1234 + 5 * 100 + 2 * 10 + C)
    x = torch.randn(2, C, 5, 5, device=dev)
    w = torch.randn(C, C, 3, 3, device=dev)
    o64 = F.conv2d(x.double(), w.double(), padding=1)
    # state 1: defaults as found
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
    o_def = F.conv2d(x, w, padding=1)
    # state 2: forced deterministic algo
    torch.backends.cudnn.deterministic = True
    o_det = F.conv2d(x, w, padding=1)
    # state 3: tf32 off
    torch.backends.cudnn.allow_tf32 = False
    o_ieee = F.conv2d(x, w, padding=1)
    torch.backends.cudnn.benchmark = prev_bench
    torch.backends.cudnn.deterministic = prev_det
    torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
    r_def, r_det, r_ie = stats(o_def, o64), stats(o_det, o64), stats(o_ieee, o64)
    algo_rows.append({"C": C, "rho_default": r_def["rho_max"], "rho_deterministic": r_det["rho_max"],
                      "rho_tf32off": r_ie["rho_max"],
                      "bit_eq_default_det": bool(torch.equal(o_def, o_det)),
                      "bit_eq_default_ieee": bool(torch.equal(o_def, o_ieee)),
                      "delta_det_over_default": abs(r_det["rho_max"] - r_def["rho_max"]) / r_def["rho_max"]})
    del x, w, o64, o_def, o_det, o_ieee
    torch.cuda.empty_cache()

c256 = [r for r in algo_rows if r["C"] == 256][0]
c16 = [r for r in algo_rows if r["C"] == 16][0]
a1 = {"delta_frac": c256["delta_det_over_default"], "holds": bool(c256["delta_det_over_default"] > 0.10)}
a2 = {"c16_rho_tf32off": c16["rho_tf32off"], "holds": bool(c16["rho_tf32off"] < 0.1)}

out = {"header": {"date": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()), "torch": torch.__version__,
                  "gpu": torch.cuda.get_device_name(0),
                  "prereg": "understand_plan_2026-10-24 legs 2-3; predictions frozen in this docstring"},
       "geometry_rows": rows, "c1": c1, "c2": c2, "algo_rows": algo_rows, "a1": a1, "a2": a2}
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({"geom": [{"M": r["M"], "rho_max": round(r["rho_max"], 3),
                            "rho_fixed_n": round(r["rho_fixed_n"], 3), "med": r["med"]} for r in rows],
                  "c1": c1, "c2": c2, "algo": algo_rows, "a1": a1, "a2": a2}, indent=1))
