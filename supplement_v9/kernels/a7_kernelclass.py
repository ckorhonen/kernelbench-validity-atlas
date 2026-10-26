"""A7 — kernel-class discriminator for the conv rho excess.

Prereg: lab/a7_kernelclass_prereg.md (commit e1526ee, frozen BEFORE this run).
Predictions P1-P5 are copied verbatim into the docstring below; the run does not add new bars.

P1 (depth): rho(R5 im2col fp32 @ K_eff=2304) < 2 at C=256.
P2 (transform): rho(R2 s=2)/rho(R1) < 0.2 AND rho(R3 d=2)/rho(R1) < 0.2 at C=256.
P3 (no-transform): rho(R4 1x1) < 1 at C=256.
P4 (dispatch): |rho(R8 benchmark=True) - rho(R1)|/rho(R1) < 0.10.
P5 (depth gating): same ordering at C=64.

Post-hoc additions (labeled): geometry-matched toggles (H=W=9 so output is 5x5 like R1) because rho is a
max statistic and output element count differs across the prereg rows; and per-row CUDA kernel names from
torch.profiler as direct evidence of which cudnn kernel ran.
"""
import json, os, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "a7_kernelclass.json")
ATOL = RTOL = 1e-4
dev = "cuda"


def stats(a, b):
    d = (a.double() - b).abs()
    den = ATOL + RTOL * b.abs()
    r = (d / den).flatten()
    n = r.numel()
    return {"rho_max": float(r.max()), "med": float(r.median()),
            "frac_over": float((r > 1).double().mean()), "n_elem": int(n),
            "rho_fixed_4608": float(r[:4608].max()), "abs_max": float(d.max())}


def kernel_names(fn, warm=1, iters=3):
    """Best-effort: run fn under profiler and return the CUDA kernel names seen."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    names = {}
    for e in prof.key_averages():
        if e.device_type == torch.autograd.DeviceType.CUDA or True:
            names[e.key] = names.get(e.key, 0) + int(e.count)
    return names


def conv_row(C, H, k, s, d, tag, cudnn_enabled=True, benchmark=False, seed=None):
    torch.manual_seed(seed if seed is not None else 1234 + 5 * 100 + 2 * 10 + C)
    x = torch.randn(2, C, H, H, device=dev)
    w = torch.randn(C, C, k, k, device=dev)
    p = (k - 1) // 2 * d  # keep output spatial size = H where possible
    prev_en, prev_tf, prev_bm = (torch.backends.cudnn.enabled, torch.backends.cudnn.allow_tf32,
                                 torch.backends.cudnn.benchmark)
    torch.backends.cudnn.enabled = cudnn_enabled
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.allow_tf32 = True  # A6 runtime fact; conv rows are flag-inert at C>=32
    o = F.conv2d(x, w, padding=p, stride=s, dilation=d)
    kn = kernel_names(lambda: F.conv2d(x, w, padding=p, stride=s, dilation=d))
    torch.backends.cudnn.enabled, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = prev_en, prev_tf, prev_bm
    o64 = F.conv2d(x.double(), w.double(), padding=p, stride=s, dilation=d)
    st = stats(o, o64)
    st.update({"row": tag, "C": C, "H": H, "k": k, "s": s, "d": d, "pad": p,
               "k_eff": int(C * k * k), "out_shape": list(o.shape),
               "cudnn_enabled": cudnn_enabled, "benchmark": benchmark,
               "kernels": kn})
    del x, w, o, o64
    torch.cuda.empty_cache()
    return st


def im2col_row(C, H, k, tag, seed=None):
    torch.manual_seed(seed if seed is not None else 1234 + 5 * 100 + 2 * 10 + C)
    x = torch.randn(2, C, H, H, device=dev)
    w = torch.randn(C, C, k, k, device=dev)
    p = (k - 1) // 2
    cols = F.unfold(x, k, padding=p)                       # (2, C*9, L)
    wm = w.reshape(C, C * k * k)                           # (C_out, C*9)
    o = torch.matmul(wm, cols)                             # (2, C, L)
    o64 = torch.matmul(wm.double(), cols.double())
    st = stats(o, o64)
    st.update({"row": tag, "C": C, "H": H, "k": k, "k_eff": int(C * k * k),
               "out_shape": list(o.shape), "note": "explicit im2col + fp32 matmul, no cudnn conv"})
    del x, w, cols, wm, o, o64
    torch.cuda.empty_cache()
    return st


def matmul_row(K, M, N, tag, seed=None):
    torch.manual_seed(seed if seed is not None else 9200 + K)
    A = torch.randn(M, K, device=dev)
    B = torch.randn(K, N, device=dev)
    o = A @ B
    o64 = A.double() @ B.double()
    st = stats(o, o64)
    st.update({"row": tag, "K": K, "M": M, "N": N, "k_eff": K})
    del A, B, o, o64
    torch.cuda.empty_cache()
    return st


runtime = {"cudnn_enabled": torch.backends.cudnn.enabled, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
           "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
           "float32_precision": torch.get_float32_matmul_precision(),
           "deterministic": torch.backends.cudnn.deterministic}

rows = []
# --- prereg rows, C=256 primary ---
rows.append(conv_row(256, 5, 3, 1, 1, "R1_cudnn_3x3_s1"))
rows.append(conv_row(256, 5, 3, 2, 1, "R2_cudnn_3x3_s2"))
rows.append(conv_row(256, 5, 3, 1, 2, "R3_cudnn_3x3_d2"))
rows.append(conv_row(256, 5, 1, 1, 1, "R4_cudnn_1x1"))
rows.append(im2col_row(256, 5, 3, "R5_im2col_matmul"))
rows.append(conv_row(256, 5, 3, 1, 1, "R6_cudnn_disabled", cudnn_enabled=False))
rows.append(matmul_row(2304, 128, 100, "R7_matmul_K2304"))
rows.append(conv_row(256, 5, 3, 1, 1, "R8_cudnn_benchmark_true", benchmark=True))
# --- post-hoc geometry-matched toggles (out 5x5) ---
rows.append(conv_row(256, 9, 3, 2, 1, "R2g_s2_geom_matched", seed=77))
rows.append(conv_row(256, 9, 3, 1, 2, "R3g_d2_geom_matched", seed=78))
# --- C=64 secondary depth check (P5) ---
rows.append(conv_row(64, 5, 3, 1, 1, "R1_c64"))
rows.append(conv_row(64, 5, 3, 2, 1, "R2_c64"))
rows.append(conv_row(64, 5, 3, 1, 2, "R3_c64"))
rows.append(conv_row(64, 5, 1, 1, 1, "R4_c64"))
rows.append(im2col_row(64, 5, 3, "R5_c64"))

by = {r["row"]: r for r in rows}


def ratio(a, b):
    denom = by[b]["rho_max"]
    return by[a]["rho_max"] / denom if denom else float("nan")


verdicts = {
    "P1_depth_im2col_lt2_C256": {"value": by["R5_im2col_matmul"]["rho_max"], "pass": by["R5_im2col_matmul"]["rho_max"] < 2.0},
    "P2_transform_s2_ratio": {"value": ratio("R2_cudnn_3x3_s2", "R1_cudnn_3x3_s1"),
                              "pass": ratio("R2_cudnn_3x3_s2", "R1_cudnn_3x3_s1") < 0.2},
    "P2_transform_d2_ratio": {"value": ratio("R3_cudnn_3x3_d2", "R1_cudnn_3x3_s1"),
                              "pass": ratio("R3_cudnn_3x3_d2", "R1_cudnn_3x3_s1") < 0.2},
    "P3_1x1_lt1_C256": {"value": by["R4_cudnn_1x1"]["rho_max"], "pass": by["R4_cudnn_1x1"]["rho_max"] < 1.0},
    "P4_dispatch_delta": {"value": (p4 := abs(by["R8_cudnn_benchmark_true"]["rho_max"] - by["R1_cudnn_3x3_s1"]["rho_max"]) / by["R1_cudnn_3x3_s1"]["rho_max"]),
                          "pass": p4 < 0.10},
    "P5_c64_ordering": {"s2_ratio": ratio("R2_c64", "R1_c64"), "d2_ratio": ratio("R3_c64", "R1_c64"),
                        "im2col": by["R5_c64"]["rho_max"],
                        "pass": (ratio("R2_c64", "R1_c64") < 0.2 and ratio("R3_c64", "R1_c64") < 0.2
                                 and by["R5_c64"]["rho_max"] < 2.0)},
}
# post-hoc geometry-matched reading
posthoc = {"s2_geom_matched_ratio": ratio("R2g_s2_geom_matched", "R1_cudnn_3x3_s1"),
           "d2_geom_matched_ratio": ratio("R3g_d2_geom_matched", "R1_cudnn_3x3_s1"),
           "label": "post-hoc; geometry-matched so rho maxima are compared at equal output size"}

out = {"header": {"date": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()), "torch": torch.__version__,
                  "gpu": torch.cuda.get_device_name(0), "prereg": "lab/a7_kernelclass_prereg.md e1526ee",
                  "runtime_pre": runtime},
       "rows": rows, "verdicts": verdicts, "posthoc": posthoc}

with open(OUT, "w") as f:
    json.dump(out, f, indent=1)

print(json.dumps({"runtime_pre": runtime,
                  "rows": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                            if k in ("row", "rho_max", "med", "frac_over", "n_elem", "rho_fixed_4608", "out_shape", "k_eff")}
                           for r in rows],
                  "verdicts": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in verdicts.items()},
                  "posthoc": posthoc}, indent=1))
print("KERNELS:")
for r in rows:
    if "kernels" in r:
        print(r["row"], "->", [k for k in r["kernels"] if "conv" in k.lower() or "winograd" in k.lower() or "gemm" in k.lower()][:6])
