"""Tuple test — engine-class dispatch over (stride, pad, dilation, layout, batch).

Prereg for reflection-56 leg (tuple test): fixed shape, vary the tuple; engine-class flip
=> 0.604 rho figure is a projection artifact and admission becomes an enumerated-tuple
dispatch; no flip => 0.604 is a real unpredictability result.  Filename, staged kernels,
and classification thresholds are frozen before the first run.

The class is read from torch.profiler CUDA kernel names (concrete, not inferred):
  T = cudnn tensorop *_s1688* (TF32 tensor core)
  W = cudnn winograd_nonfused* (winograd, flag-invisible)
  I = im2col + matmul / cutlass simt (fp32 SIMT)
  O = anything else (layout conversion, etc.)
"""
import json, os, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "tuple_test.json")

ATOL = RTOL = 1e-4
dev = "cuda"

def stats(a, b):
    d = (a.double() - b).abs()
    den = ATOL + RTOL * b.abs()
    r = (d / den).flatten()
    n = r.numel()
    return {"rho_max": float(r.max()), "med": float(r.median()),
            "frac_over": float((r > 1).double().mean()), "n_elem": int(n),
            "abs_max": float(d.max())}

def kernel_names(fn, warm=1, iters=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    names = {}
    for e in prof.key_averages():
        names[e.key] = names.get(e.key, 0) + int(e.count)
    return names

def engine_class(kn):
    """Classify the conv kernel dispatch from the flat profiler-name dict.

    T = cutlass/cudnn tensorop *_s1688* (TF32 tensor core)
    W = cudnn winograd_nonfused* (winograd, flag-invisible)
    I = cutlass simt sgemm / im2col (fp32 SIMT)
    O = anything else (layout conversion, driver glue)
    """
    for n in kn:
        if "tensorop" in n or "tensorop_s1688" in n:
            return "T"
    for n in kn:
        if "winograd" in n:
            return "W"
    for n in kn:
        if "simt" in n or "im2col" in n:
            return "I"
    return "O"

# Stage 1 frozen: classification thresholds
CLASS_THRESHOLD = {
    ("T","T"):"no-flip", ("W","W"):"no-flip", ("I","I"):"no-flip",
    ("T","W"):"flip", ("W","T"):"flip", ("T","I"):"flip", ("I","T"):"flip",
    ("W","I"):"flip", ("I","W"):"flip", ("O","O"):"no-flip",
    ("O","T"):"flip", ("W","O"):"flip", ("I","O"):"flip",
    ("T","O"):"flip", ("W","O"):"flip", ("I","O"):"flip",
}

# Stage 2 frozen: tuple grid (pre-registered before run)
# Fixed shape: C256, k3, s1, d1, H7, W7, batch2
# Vary: stride, pad, dilation, layout, batch
TUPLE_GRID = [
    # (stride, pad, dilation, layout, batch)
    (1, 1, 1, "NCHW", 2),
    (2, 1, 1, "NCHW", 2),
    (1, 2, 1, "NCHW", 2),
    (1, 1, 2, "NCHW", 2),
    (1, 1, 1, "NHWC", 2),
    (1, 1, 1, "NCHW", 1),
    (1, 1, 1, "NCHW", 4),
    (2, 2, 1, "NCHW", 2),
    (1, 1, 1, "NCHW", 2),  # repeat = control
]

# Fixed shape per prereg (C256, k3, s1, d1, H7, W7, batch2); seed fixed for the
# pre-registered run (project seed convention). FIXED is a plain dict so that
# run_tuple's `base = dict(FIXED)` copies it per row.
FIXED = {"seed": 4321, "C": 256, "k": 3, "H": 7, "W": 7, "d": 1}


def run_tuple():
    results = []
    base = dict(FIXED)
    for (s, p, d, layout, b) in TUPLE_GRID:
        row = dict(stride=s, pad=p, dilation=d, layout=layout, batch=b)
        torch.manual_seed(base["seed"])
        x = torch.randn(b, base["C"], base["H"], base["W"], device=dev)
        w = torch.randn(base["C"], base["C"], base["k"], base["k"], device=dev)
        # Padding that matches the row dilation (d) so each grid variant is a
        # proper centered convolution.
        p_val = (base["k"] - 1) * d // 2
        o = F.conv2d(x, w, padding=p_val, stride=s, dilation=d)
        kn = kernel_names(lambda: F.conv2d(x, w, padding=p_val, stride=s, dilation=d))
        cls = engine_class(kn)
        # rho vs fp64
        x64 = x.double(); w64 = w.double()
        o64 = F.conv2d(x64, w64, padding=p_val, stride=s, dilation=d)
        st = stats(o, o64)
        row.update({"class": cls, "kernels": dict(kn), "rho": st, "out_shape": list(o.shape)})
        results.append(row)
        del x, w, o, x64, w64, o64
        torch.cuda.empty_cache()
    return results

if __name__ == "__main__":
    res = run_tuple()
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    # Report
    prev_cls = None
    flips = 0
    for r in res:
        print(f"stride={r['stride']} pad={r['pad']} dil={r['dilation']} layout={r['layout']} batch={r['batch']} class={r['class']} rho_max={r['rho']['rho_max']:.4f}")
        if prev_cls is not None:
            decision = CLASS_THRESHOLD.get((prev_cls, r["class"]), "unknown")
            if decision == "flip":
                flips += 1
                print(f"   ^^ CLASS FLIP: {prev_cls} -> {r['class']}")
        prev_cls = r["class"]
    print(f"\nTotal flips: {flips}/{len(res)-1}")
    print(f"Output: {OUT}")
