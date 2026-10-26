"""A8 — compression test for (rho_max, frac_over) across families.

Prereg: lab/a8_compress_prereg.md, commit c56c0a0, frozen BEFORE this run.
No bar changes; the frozen outcome rule fires as written.

Runtime facts recorded per row: cudnn.allow_tf32, matmul.allow_tf32, float32_precision,
cudnn.benchmark. Primary grid leaves defaults untouched.
"""
import json, os, sys
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "a8_compress_v2.json")
ATOL = RTOL = 1e-4
dev = "cuda"

ENGINE_MAP = [("winograd", "W"), ("tf32", "TF32"), ("sgemm", "GEMM"), ("gemm", "GEMM")]


def engine_class(names):
    keys = " ".join(names).lower()
    for pat, cls in ENGINE_MAP:
        if pat in keys:
            return cls
    return "OTHER"


def stats(a, b):
    d = (a.double() - b).abs()
    den = ATOL + RTOL * b.abs()
    r = (d / den).flatten()
    return {"rho_max": float(r.max()), "med": float(r.median()),
            "frac_over": float((r > 1).double().mean()), "n_elem": int(r.numel()),
            "abs_max": float(d.max())}


def kernel_names(fn, warm=1, iters=2):
    """Return (device_kernels, runtime_calls). DEVICE list = actual GPU kernels, untruncated.
    v2 (declared harness fix, found by a8b: truncating the raw list to 8 entries hid the conv
    kernel behind nchwToNhwc/nhwcToNchw layout conversions)."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    device, runtime = {}, {}
    for e in prof.key_averages():
        key = e.key
        is_kernel = ("void " in key) or ("cutlass" in key) or ("cudnn" in key) or \
                    ("implicit_convolve" in key) or ("tensorop" in key) or ("Kernel" in key)
        (device if is_kernel else runtime)[key] = int(e.count)
    return device, runtime


def conv_row(idx, C, H, k, s, d, cudnn_enabled=True, tf32=None):
    seed = 7300 + idx
    torch.manual_seed(seed)
    x = torch.randn(2, C, H, H, device=dev)
    w = torch.randn(C, C, k, k, device=dev)
    p = (k - 1) // 2 * d
    prev = (torch.backends.cudnn.enabled, torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.benchmark, torch.get_float32_matmul_precision())
    torch.backends.cudnn.enabled = cudnn_enabled
    if tf32 is not None:
        torch.backends.cudnn.allow_tf32 = tf32
    fn = lambda: F.conv2d(x, w, padding=p, stride=s, dilation=d)
    o = fn()
    dev_names, rt_names = kernel_names(fn)
    names = sorted(dev_names)
    o2 = fn()
    det = bool(torch.equal(o, o2))
    (torch.backends.cudnn.enabled, torch.backends.cudnn.allow_tf32,
     torch.backends.cudnn.benchmark, _) = prev
    o64 = F.conv2d(x.double(), w.double(), padding=p, stride=s, dilation=d)
    st = stats(o, o64)
    st.update({"row": f"conv_C{C}_H{H}_k{k}_s{s}_d{d}_cudnn{cudnn_enabled}_tf32{tf32}",
               "family": "conv2d", "C": C, "H": H, "k": k, "s": s, "d": d, "pad": p,
               "k_eff": int(C * k * k), "C_eff": C, "out_shape": list(o.shape),
               "cudnn_enabled": cudnn_enabled, "allow_tf32": tf32,
               "engine": engine_class(names), "kernels": names,
               "runtime_calls": sorted(rt_names),
               "deterministic_2run": det, "seed": seed})
    del x, w, o, o2, o64
    torch.cuda.empty_cache()
    return st


def im2col_row(idx, C, H, k):
    seed = 7300 + idx
    torch.manual_seed(seed)
    x = torch.randn(2, C, H, H, device=dev)
    w = torch.randn(C, C, k, k, device=dev)
    cols = F.unfold(x, k, padding=(k - 1) // 2)
    o = (w.view(C, -1) @ cols).view(2, C, H, H)
    o64 = F.conv2d(x.double(), w.double(), padding=(k - 1) // 2)
    st = stats(o, o64)
    st.update({"row": f"im2col_C{C}_H{H}_k{k}", "family": "im2col_gemm", "C": C, "H": H, "k": k,
               "k_eff": int(C * k * k), "C_eff": C, "out_shape": list(o.shape),
               "engine": "GEMM", "kernels": [], "seed": seed, "deterministic_2run": None})
    del x, w, cols, o, o64
    torch.cuda.empty_cache()
    return st


def matmul_row(idx, M, K):
    seed = 7400 + idx
    torch.manual_seed(seed)
    a = torch.randn(M, K, device=dev)
    b = torch.randn(K, M, device=dev)
    fn = lambda: a @ b
    o = fn()
    dev_names, rt_names = kernel_names(fn)
    names = sorted(dev_names)
    o2 = fn()
    det = bool(torch.equal(o, o2))
    o64 = (a.double() @ b.double())
    st = stats(o, o64)
    st.update({"row": f"mm_M{M}_K{K}", "family": "matmul", "M": M, "K": K,
               "k_eff": K, "C_eff": M, "out_shape": list(o.shape),
               "engine": engine_class(names), "kernels": names,
               "runtime_calls": sorted(rt_names),
               "deterministic_2run": det, "seed": seed})
    del a, b, o, o2, o64
    torch.cuda.empty_cache()
    return st


def main():
    facts = {"cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
             "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
             "float32_precision": torch.get_float32_matmul_precision(),
             "cudnn_benchmark": torch.backends.cudnn.benchmark,
             "torch": torch.__version__, "device": torch.cuda.get_device_name(0)}
    rows = []
    idx = 0
    for C in (16, 64, 256):
        for (k, s, d) in ((3, 1, 1), (3, 2, 1), (3, 1, 2), (1, 1, 1)):
            for H in (7, 13, 21):
                idx += 1
                rows.append(conv_row(idx, C, H, k, s, d))
                print("row", idx, rows[-1]["row"], rows[-1]["rho_max"], rows[-1]["frac_over"],
                      rows[-1]["engine"], flush=True)
    for M in (256, 1024, 4096):
        for K in (128, 512, 2048, 8192):
            idx += 1
            rows.append(matmul_row(idx, M, K))
            print("row", idx, rows[-1]["row"], rows[-1]["rho_max"], rows[-1]["frac_over"],
                  rows[-1]["engine"], flush=True)
    controls = []
    idx += 1
    controls.append(im2col_row(idx, 256, 13, 3))
    idx += 1
    controls.append(conv_row(idx, 256, 13, 3, 1, 1, cudnn_enabled=False))
    idx += 1
    controls.append(conv_row(idx, 256, 13, 3, 1, 1, tf32=False))
    idx += 1
    controls.append(conv_row(idx, 256, 13, 3, 2, 1, tf32=False))
    out = {"header": {"prereg": "lab/a8_compress_prereg.md @ c56c0a0", "atol": ATOL, "rtol": RTOL,
                      "runtime_facts": facts, "n_primary": len(rows), "n_control": len(controls)},
           "rows": rows, "controls": controls}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE", OUT, "rows", len(rows), "controls", len(controls))


if __name__ == "__main__":
    main()
