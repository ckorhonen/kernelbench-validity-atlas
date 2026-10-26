"""Tuple test — engine-class dispatch over (stride, pad, dilation, layout, batch).

Prereg for reflection-56 leg (tuple test): fixed shape, vary the tuple; engine-class flip
=> 0.604 rho figure is a projection artifact and admission becomes an enumerated-tuple
dispatch; no flip => 0.604 is a real unpredictability result. Filename, staged kernels,
and classification thresholds are frozen before the first run.

The class is read from torch.profiler CUDA kernel names (concrete, not inferred):
  T = cudnn tensorop *_s1688* (TF32 tensor core)
  W = cudnn winograd_nonfused* (winograd, flag-invisible)
  I = im2col + matmul / cutlass simt (fp32 SIMT)
  O = anything else (layout conversion, driver glue).

BUG FIXED (the tuple test artifact shipped in the round-7 submission carried this):
  - padding must be the row's own `pad` field; the earlier line `(k-1)*d//2` collapsed
    every row to zero padding for k=3,d=1, so rows 3/4/8 of the shipped
    results/tuple_test.json have output shapes that contradict the convolution formula,
    and the pad-2 rows repeated the pad-1 error summaries exactly. The corrected script
    applies the row pad, asserts each output shape against
    H_out = floor((H + 2p - d*(k-1) - 1)/s) + 1, and records effective parameters
    (s,p,d,layout,batch), tensor strides and, for every row, a paired precision-flag
    measurement at H=7 (rho/frac_over with allow_tf32=False, plus bit-equality vs the
    default) so the W-class flag-invariance claim is measured at this height, not only
    at the H=5 height of the A7b ratios (which the round-7 review is right to flag).
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
    O = anything else (layout conversion, driver glue).
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

def expected_h(s, p, d):
    """H_out = floor((H + 2p - d*(k-1) - 1)/s) + 1, with H=7, k=3, W parallel."""
    return (FIXED["H"] + 2*p - d*(FIXED["k"]-1) - 1)//s + 1

def run_tuple():
    results = []
    base = dict(FIXED)
    for (s, p, d, layout, b) in TUPLE_GRID:
        row = {"stride": s, "pad": p, "dilation": d, "layout": layout, "batch": b,
               "expected_h": expected_h(s, p, d)}
        torch.manual_seed(base["seed"])
        x = torch.randn(b, base["C"], base["H"], base["W"], device=dev)
        w = torch.randn(base["C"], base["C"], base["k"], base["k"], device=dev)
        # The row padding field is the one that defines a centered convolution on this
        # grid; do NOT recompute a dilation-matched padding, and do NOT use
        # (k-1)*d//2, which is not the row pad.
        p_val = p
        o = F.conv2d(x, w, padding=p_val, stride=s, dilation=d)
        assert list(o.shape[-2:]) == [expected_h(s, p, d), expected_h(s, p, d)], (
            f"output shape {list(o.shape[-2:])} != expected {expected_h(s,p,d)} "
            f"for (s={s},p={p},d={d})")
        kn = kernel_names(lambda: F.conv2d(x, w, padding=p_val, stride=s, dilation=d))
        cls = engine_class(kn)
        # default reference
        x64 = x.double(); w64 = w.double()
        o64 = F.conv2d(x64, w64, padding=p_val, stride=s, dilation=d)
        st = stats(o, o64)
        # paired precision-flag measurement at H=7 (the H=5 A7b ratios are not
        # transferable to this height, per the round-7 review):
        #   rho_off, frac_over_off, bit_equal = exact element equality vs default.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            x2 = x.clone(); w2 = w.clone()
            o_off = F.conv2d(x2, w2, padding=p_val, stride=s, dilation=d)
            x64o = x2.double(); w64o = w2.double()
            o64_off = F.conv2d(x64o, w64o, padding=p_val, stride=s, dilation=d)
            st_off = stats(o_off, o64_off)
            bit_equal = bool(torch.equal(o, o_off))
        finally:
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        row.update({"class": cls, "kernels": dict(kn),
                    "rho": st, "rho_flagoff": st_off,
                    "flagoff_bit_equal": bit_equal,
                    "flagoff_rho_ratio": st_off["rho_max"] / max(st["rho_max"], 1e-30),
                    "out_shape": [int(v) for v in o.shape[-2:]],
                    "strides": [int(v) for v in o.stride()]})
        results.append(row)
        del x, w, o, x64, w64, o64, x2, w2, o_off, x64o, w64o, o64_off
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
        print(f"stride={r['stride']} pad={r['pad']} dil={r['dilation']} layout={r['layout']} "
              f"batch={r['batch']} class={r['class']} rho={r['rho']['rho_max']:.4f} "
              f"flagoff={r['rho_flagoff']['rho_max']:.4f} bit_eq={r['flagoff_bit_equal']} "
              f"exp_h={r['expected_h']} out={r['out_shape']}")
        if prev_cls is not None:
            decision = CLASS_THRESHOLD.get((prev_cls, r["class"]), "unknown")
            if decision == "flip":
                flips += 1
                print(f"   ^^ CLASS FLIP: {prev_cls} -> {r['class']}")
        prev_cls = r["class"]
    print(f"\nTotal flips: {flips}/{len(res)-1}")
    print(f"Output: {OUT}")
