"""A7b — precision-policy discriminator for the non-winograd cudnn engines (frozen BEFORE run).

Motivated by a7 (results/a7_kernelclass.json, run 07:25Z): profiler named the engines.
  R1 3x3 s1 C256 -> cudnn::winograd_nonfused 4x4 (+ cutlass simt sgemm), rho 10.195
  R2 3x3 s2 C256 -> nchwToNhwc + cutlass__5x_cudnn tensorop s1688fprop, rho 251.69
  R3 3x3 d2 C256 -> same tensorop engine, rho 231.86
  R4 1x1  C256 -> cutlass_80_tensorop_s1688gemm, rho 110.85
  R5 explicit im2col + torch.matmul (allow_tf32=False) -> 0.657
  R6 cudnn disabled -> at::native im2col + cutlass simt sgemm, rho 0.192
"tensorop_s1688" is the TF32 MMA shape, so the excess may be a precision-default effect in engines that
winograd-eligible shapes never reach. a6c's "flag-inert, bit-identical at C>=32" was measured ONLY on the
3x3 s1 family, i.e. only on winograd. That generalization is now suspect.

PREDICTIONS FROZEN HERE, BEFORE THE RUN:
 Q1. At s=2, d=2 and k=1 (C=256) rho(allow_tf32=False)/rho(allow_tf32=True) < 0.10 with bit-INequal outputs
     => those engines honor the precision flag; the excess there is a TF32 tensor-op default, NOT an
     arithmetic-tree property. This RETRACTS "flag-invisible divergence" as a general claim and confines it
     to the winograd-eligible family.
 Q2. The 3x3 s1 winograd row stays bit-identical and rho-identical across the flag (replicates a6c).
 Q3. torch.matmul K=2304 with allow_tf32=True reaches rho >= 1 (TF32 matmul is out of budget) while
     allow_tf32=False stays < 1 — positive control that the flag works at all on this build.
 If Q1 fails (ratio ~1, bit-identical), the tensorop engines consume budget regardless of the flag and the
 excess is an engine-intrinsic property: then "precision flag cannot see it" survives, generalized.

Post-hoc: per-row kernel names again, to tie each rho to a named engine.
"""
import json, os, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "a7b_precision_policy.json")
ATOL = RTOL = 1e-4
dev = "cuda"


def stats(a, b):
    d = (a.double() - b).abs()
    den = ATOL + RTOL * b.abs()
    r = (d / den).flatten()
    return {"rho_max": float(r.max()), "med": float(r.median()),
            "frac_over": float((r > 1).double().mean()), "n_elem": int(r.numel())}


def kernels(fn, warm=2, iters=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    return {e.key: int(e.count) for e in prof.key_averages() if int(e.count) > 0}


rows = []
for C in (256, 64):
    for tag, k, s, d in (("winograd_s1", 3, 1, 1), ("s2", 3, 2, 1), ("d2", 3, 1, 2), ("k1", 1, 1, 1)):
        torch.manual_seed(1234 + 5 * 100 + 2 * 10 + C)
        x = torch.randn(2, C, 5, 5, device=dev)
        w = torch.randn(C, C, k, k, device=dev)
        p = (k - 1) // 2 * d
        prev = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = True
        o_on = F.conv2d(x, w, padding=p, stride=s, dilation=d)
        kn = kernels(lambda: F.conv2d(x, w, padding=p, stride=s, dilation=d))
        torch.backends.cudnn.allow_tf32 = False
        o_off = F.conv2d(x, w, padding=p, stride=s, dilation=d)
        torch.backends.cudnn.allow_tf32 = prev
        o64 = F.conv2d(x.double(), w.double(), padding=p, stride=s, dilation=d)
        r_on, r_off = stats(o_on, o64), stats(o_off, o64)
        rows.append({"C": C, "tag": tag, "k": k, "s": s, "d": d, "rho_on": r_on["rho_max"], "rho_off": r_off["rho_max"],
                     "med_on": r_on["med"], "med_off": r_off["med"], "frac_over_on": r_on["frac_over"],
                     "frac_over_off": r_off["frac_over"], "bit_equal": bool(torch.equal(o_on, o_off)),
                     "ratio_off_over_on": (r_off["rho_max"] / r_on["rho_max"]) if r_on["rho_max"] else float("nan"),
                     "engines": [k2 for k2 in kn if "winograd" in k2 or "s1688" in k2 or "sgemm" in k2 or "Nhwc" in k2]})
        del x, w, o_on, o_off, o64
        torch.cuda.empty_cache()

mm = []
for flag in (True, False):
    torch.manual_seed(9200 + 2304)
    A = torch.randn(128, 2304, device=dev)
    B = torch.randn(2304, 100, device=dev)
    torch.backends.cuda.matmul.allow_tf32 = flag
    o = A @ B
    o64 = A.double() @ B.double()
    mm.append({"allow_tf32": flag, "rho_max": stats(o, o64)["rho_max"], "med": stats(o, o64)["med"]})
    del A, B, o, o64
    torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = False

q1rows = [r for r in rows if r["tag"] in ("s2", "d2", "k1") and r["C"] == 256]
q1 = {"ratios": {r["tag"]: r["ratio_off_over_on"] for r in q1rows},
      "bit_equal": {r["tag"]: r["bit_equal"] for r in q1rows},
      "pass": all(r["ratio_off_over_on"] < 0.10 and not r["bit_equal"] for r in q1rows)}
w1 = [r for r in rows if r["tag"] == "winograd_s1" and r["C"] == 256][0]
q2 = {"bit_equal": w1["bit_equal"], "rho_on": w1["rho_on"], "rho_off": w1["rho_off"],
      "pass": bool(w1["bit_equal"] and abs(w1["rho_off"] - w1["rho_on"]) / w1["rho_on"] < 0.01)}
q3 = {"tf32_rho": mm[0]["rho_max"], "fp32_rho": mm[1]["rho_max"],
      "pass": mm[0]["rho_max"] >= 1.0 and mm[1]["rho_max"] < 1.0}

out = {"header": {"date": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()), "torch": torch.__version__,
                  "gpu": torch.cuda.get_device_name(0),
                  "prereg": "lab/a7b_precision_prereg.md (frozen pre-run)", "upstream": "results/a7_kernelclass.json"},
       "conv_rows": rows, "matmul_rows": mm, "Q1": q1, "Q2": q2, "Q3": q3}
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)

print(json.dumps({"conv": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "engines"} | {"engine_top": r["engines"][:2]} for r in rows],
                  "matmul": mm, "Q1": q1, "Q2": q2, "Q3": q3}, indent=1))
