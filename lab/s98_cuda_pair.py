#!/usr/bin/env python3
"""s98: E2 of the exploitation plan — reference-order statistics + tiled affine (V1) vs fast-statistics +
same affine (V2) on GB10, against the LayerNorm reference (A) with torch.compile (C) as timing comparator.

Preregistered in lab/s98_prereg.md (commit defa76c, frozen before this run). The two statistics kernels are
mine, in one CUDA extension; their merge trees are declared in the prereg. Also probes A's launch geometry
from a Kineto chrome trace.
"""
import json, os, statistics, torch, torch.nn.functional as F
import torch.utils.cpp_extension

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

OUT = os.environ.get("S98_OUT", "results/s98_cuda_pair.jsonl")
SHAPES = [("S1_official", (16, 64, 256, 256)), ("S2_reduced", (16, 32, 128, 128))]
DRAWS = 16
SEGS = 32

cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// V1 declared tree: one block per row, 128 threads, per-thread sequential strided accumulation
// (i += blockDim.x), shared-memory fan-in-2 over 128 lanes (levels 64..1). Estimator = sum/sumsq.
__global__ void v1_stats(const float* __restrict__ x, float* __restrict__ mean, float* __restrict__ rstd,
                         const long N, const float eps){
  const long row = blockIdx.x;
  const float* xr = x + row*N;
  float s = 0.f, q = 0.f;
  for (long i = threadIdx.x; i < N; i += blockDim.x){ float v = xr[i]; s += v; q += v*v; }
  __shared__ float ss[128], qq[128];
  ss[threadIdx.x] = s; qq[threadIdx.x] = q;
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1){
    if (threadIdx.x < stride){ ss[threadIdx.x] += ss[threadIdx.x+stride]; qq[threadIdx.x] += qq[threadIdx.x+stride]; }
    __syncthreads();
  }
  if (threadIdx.x == 0){
    float m = ss[0] / (float)N;
    float v = qq[0] / (float)N - m*m;
    mean[row] = m; rstd[row] = rsqrtf(v + eps);
  }
}

// V2 declared tree: grid M*segs, 256 threads, contiguous segment, fan-in-2 x 8 within segment,
// then a single-thread sequential fan-in-32 merge. Estimator = sum/sumsq.
__global__ void v2_partial(const float* __restrict__ x, float* __restrict__ ps, float* __restrict__ pq,
                           const long N, const int segs){
  const long row = blockIdx.x / segs;
  const int seg = blockIdx.x % segs;
  const long per = N / segs;
  const float* xr = x + row*N + (long)seg*per;
  float s = 0.f, q = 0.f;
  for (long i = threadIdx.x; i < per; i += blockDim.x){ float v = xr[i]; s += v; q += v*v; }
  __shared__ float ss[256], qq[256];
  ss[threadIdx.x] = s; qq[threadIdx.x] = q;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1){
    if (threadIdx.x < stride){ ss[threadIdx.x] += ss[threadIdx.x+stride]; qq[threadIdx.x] += qq[threadIdx.x+stride]; }
    __syncthreads();
  }
  if (threadIdx.x == 0){ ps[row*segs + seg] = ss[0]; pq[row*segs + seg] = qq[0]; }
}

__global__ void v2_merge(const float* __restrict__ ps, const float* __restrict__ pq, float* __restrict__ mean,
                         float* __restrict__ rstd, const long N, const int segs, const float eps){
  const int row = blockIdx.x;
  if (threadIdx.x == 0){
    float s = 0.f, q = 0.f;
    for (int i = 0; i < segs; i++){ s += ps[row*segs + i]; q += pq[row*segs + i]; }
    float m = s / (float)N;
    float v = q / (float)N - m*m;
    mean[row] = m; rstd[row] = rsqrtf(v + eps);
  }
}

// Shared tiled affine for both variants: float4 per thread, grid = ceil(total/4/256).
__global__ void affine_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                              const float* __restrict__ mean, const float* __restrict__ rstd,
                              float* __restrict__ y, const long total, const long WN){
  const long idx = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx >= total) return;
  const long row = idx / WN;
  const float m = mean[row], r = rstd[row];
  const long off = idx - row*WN;
  const float4 xv = *reinterpret_cast<const float4*>(x + idx);
  const float4 wv = *reinterpret_cast<const float4*>(w + off);
  const float4 bv = *reinterpret_cast<const float4*>(b + off);
  float4 o;
  o.x = (xv.x - m) * r * wv.x + bv.x;
  o.y = (xv.y - m) * r * wv.y + bv.y;
  o.z = (xv.z - m) * r * wv.z + bv.z;
  o.w = (xv.w - m) * r * wv.w + bv.w;
  *reinterpret_cast<float4*>(y + idx) = o;
}

std::vector<torch::Tensor> stats_v1(torch::Tensor x, double eps){
  auto M = x.size(0); long N = x.numel() / M;
  auto mean = torch::empty({M}, x.options());
  auto rstd = torch::empty({M}, x.options());
  v1_stats<<<(unsigned)M, 128>>>((const float*)x.data_ptr<float>(), mean.data_ptr<float>(),
                                 rstd.data_ptr<float>(), N, (float)eps);
  return {mean, rstd};
}

std::vector<torch::Tensor> stats_v2(torch::Tensor x, double eps, int64_t segs){
  auto M = x.size(0); long N = x.numel() / M;
  auto ps = torch::empty({M, segs}, x.options());
  auto pq = torch::empty({M, segs}, x.options());
  auto mean = torch::empty({M}, x.options());
  auto rstd = torch::empty({M}, x.options());
  v2_partial<<<(unsigned)(M*segs), 256>>>((const float*)x.data_ptr<float>(), ps.data_ptr<float>(),
                                          pq.data_ptr<float>(), N, (int)segs);
  v2_merge<<<(unsigned)M, 1>>>(ps.data_ptr<float>(), pq.data_ptr<float>(), mean.data_ptr<float>(),
                               rstd.data_ptr<float>(), N, (int)segs, (float)eps);
  return {mean, rstd};
}

torch::Tensor affine(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor mean, torch::Tensor rstd){
  auto y = torch::empty_like(x);
  long total = x.numel(); long WN = w.numel();
  long nblk = ((total/4) + 255) / 256;
  affine_kernel<<<(unsigned)nblk, 256>>>((const float*)x.data_ptr<float>(), (const float*)w.data_ptr<float>(),
                                         (const float*)b.data_ptr<float>(), (const float*)mean.data_ptr<float>(),
                                         (const float*)rstd.data_ptr<float>(), y.data_ptr<float>(), total, WN);
  return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> stats_v1(torch::Tensor x, double eps);
std::vector<torch::Tensor> stats_v2(torch::Tensor x, double eps, int64_t segs);
torch::Tensor affine(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor mean, torch::Tensor rstd);
"""

ext = torch.utils.cpp_extension.load_inline(
    name="s98_kernels", cpp_sources=cpp_src, cuda_sources=cuda_src,
    functions=["stats_v1", "stats_v2", "affine"], verbose=False,
    extra_cuda_cflags=["-O3"],
)

def bits(t):
    return t.view(torch.int32)

def exact_frac(a, b):
    return float((bits(a) == bits(b)).float().mean())

def eval_ratio(cand, ref):
    # evaluator-exact: |cand-ref| / (atol + rtol*|ref|), atol=rtol=1e-4; returns max and frac over
    d = (cand.double() - ref.double()).abs()
    den = 1e-4 + 1e-4 * ref.double().abs()
    r = d / den
    return float(r.max()), float((r > 1).double().mean()), float((r > 100).double().mean())

def med_time(fn, reps=20):
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
        t0.record(); fn(); t1.record(); torch.cuda.synchronize()
        ts.append(t0.elapsed_time(t1))
    return statistics.median(ts), min(ts), max(ts)

def launch_geometry(shape, w, b):
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    for _ in range(3):
        F.layer_norm(x, shape[1:], w, b, 1e-5)
    torch.cuda.synchronize()
    path = "/tmp/s98_trace.json"
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        F.layer_norm(x, shape[1:], w, b, 1e-5)
        torch.cuda.synchronize()
    prof.export_chrome_trace(path)
    tr = json.load(open(path))
    out = []
    for e in tr.get("traceEvents", []):
        nm = e.get("name", "")
        args = e.get("args", {}) or {}
        if "layer_norm" in nm or "vectorized" in nm:
            out.append({"name": nm, "grid": args.get("grid"), "block": args.get("block"),
                        "regs": args.get("registers per thread"), "smem": args.get("shared memory")})
    os.remove(path)
    return out

rows = []
for name, shape in SHAPES:
    M = shape[0]
    N = shape[1] * shape[2] * shape[3]
    for d in range(DRAWS):
        torch.manual_seed(9800 + d)
        x = torch.randn(shape, device="cuda", dtype=torch.float32)
        w = torch.randn(shape[1:], device="cuda", dtype=torch.float32)
        b = torch.randn(shape[1:], device="cuda", dtype=torch.float32)
        A = F.layer_norm(x, shape[1:], w, b, 1e-5)
        X64 = F.layer_norm(x.double(), shape[1:], w.double(), b.double(), 1e-5)
        m1, r1 = ext.stats_v1(x, 1e-5)
        V1 = ext.affine(x, w, b, m1, r1)
        m2, r2 = ext.stats_v2(x, 1e-5, SEGS)
        V2 = ext.affine(x, w, b, m2, r2)
        torch.cuda.synchronize()
        rec = {"shape_name": name, "shape": list(shape), "M": M, "N": N, "draw": d, "seed": 9800 + d,
               "arms": {}, "stats_delta": {"mean_v1_minus_xbar": float((m1.double() - x.mean(dim=(1, 2, 3)).double()).abs().max()),
                                           "mean_v2_minus_xbar": float((m2.double() - x.mean(dim=(1, 2, 3)).double()).abs().max()),
                                           "xbar": [float(v) for v in x.mean(dim=(1, 2, 3))[:3]]}}
        for arm, cand in (("V1_ref_order", V1), ("V2_fast_stats", V2), ("A_reference", A)):
            rho, f1, f100 = eval_ratio(cand, A)
            rho64, _, _ = eval_ratio(cand, X64)
            rec["arms"][arm] = {
                "exact_frac_vs_A": exact_frac(cand, A),
                "max_abs_vs_A": float((cand.double() - A.double()).abs().max()),
                "rho_vs_ref": rho, "frac_over_1e4": f1, "frac_over_1e2": f100,
                "rho_vs_fp64": rho64,
            }
        if d == 0:
            tA, a0, a1 = med_time(lambda: F.layer_norm(x, shape[1:], w, b, 1e-5))
            t1_, b0, b1_ = med_time(lambda: ext.affine(x, w, b, *ext.stats_v1(x, 1e-5)))
            t2_, c0, c1_ = med_time(lambda: ext.affine(x, w, b, *ext.stats_v2(x, 1e-5, SEGS)))
            rec["timing"] = {"A": [tA, a0, a1], "V1": [t1_, b0, b1_], "V2": [t2_, c0, c1_]}
        rows.append(rec)
        print(f"{name} d{d}: V1 exact {rec['arms']['V1_ref_order']['exact_frac_vs_A']:.4f} rho {rec['arms']['V1_ref_order']['rho_vs_ref']:.4f} "
              f"| V2 exact {rec['arms']['V2_fast_stats']['exact_frac_vs_A']:.4f} rho {rec['arms']['V2_fast_stats']['rho_vs_ref']:.4f} "
              f"| A rho64 {rec['arms']['A_reference']['rho_vs_fp64']:.4f}", flush=True)

# compile comparator on draw-0 tensors of each shape (timing only)
comp_rows = []
for name, shape in SHAPES:
    torch.manual_seed(9800)
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    w = torch.randn(shape[1:], device="cuda", dtype=torch.float32)
    b = torch.randn(shape[1:], device="cuda", dtype=torch.float32)
    f = torch.compile(lambda xx, ww, bb: F.layer_norm(xx, shape[1:], ww, bb, 1e-5), dynamic=False)
    for _ in range(3):
        yc = f(x, w, b)
    torch.cuda.synchronize()
    tc, c0, c1 = med_time(lambda: f(x, w, b), reps=10)
    A = F.layer_norm(x, shape[1:], w, b, 1e-5)
    comp_rows.append({"shape_name": name, "compile_med_ms": tc, "compile_min_ms": c0, "compile_max_ms": c1,
                      "compile_exact_frac_vs_A": exact_frac(yc, A),
                      "compile_rho_vs_A": eval_ratio(yc, A)[0]})
    print(name, "compile", comp_rows[-1], flush=True)

geo = {}
for name, shape in SHAPES:
    torch.manual_seed(9800)
    w = torch.randn(shape[1:], device="cuda", dtype=torch.float32)
    b = torch.randn(shape[1:], device="cuda", dtype=torch.float32)
    geo[name] = launch_geometry(shape, w, b)
print("GEOMETRY", json.dumps(geo)[:800], flush=True)

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")

# frozen verdict computation
def q(vals, p):
    vals = sorted(vals)
    k = (len(vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)

summary = {"env": {"torch": torch.__version__, "device": torch.cuda.get_device_name(0),
                   "cc": torch.cuda.get_device_capability(0)}, "segs": SEGS, "shapes": {}}
for name, shape in SHAPES:
    rs = [r for r in rows if r["shape_name"] == name]
    v1r = [r["arms"]["V1_ref_order"]["rho_vs_ref"] for r in rs]
    v2r = [r["arms"]["V2_fast_stats"]["rho_vs_ref"] for r in rs]
    v1e = [r["arms"]["V1_ref_order"]["exact_frac_vs_A"] for r in rs]
    v2e = [r["arms"]["V2_fast_stats"]["exact_frac_vs_A"] for r in rs]
    d = {"n_draws": len(rs),
         "V1_rho": {"median": q(v1r, .5), "q25": q(v1r, .25), "q75": q(v1r, .75), "min": min(v1r), "max": max(v1r)},
         "V2_rho": {"median": q(v2r, .5), "q25": q(v2r, .25), "q75": q(v2r, .75), "min": min(v2r), "max": max(v2r)},
         "V1_exact_median": q(v1e, .5), "V2_exact_median": q(v2e, .5),
         "A_rho_fp64_median": q([r["arms"]["A_reference"]["rho_vs_fp64"] for r in rs], .5),
         "frac_over_1e4_median": {"V1": q([r["arms"]["V1_ref_order"]["frac_over_1e4"] for r in rs], .5),
                                  "V2": q([r["arms"]["V2_fast_stats"]["frac_over_1e4"] for r in rs], .5)},
         "timing_draw0": next((r.get("timing") for r in rs if r.get("timing")), None)}
    disjoint = d["V1_rho"]["q75"] < d["V2_rho"]["q25"] or d["V2_rho"]["q75"] < d["V1_rho"]["q25"]
    if q(v1e, .5) >= 0.999 and d["V1_rho"]["median"] <= 0.25 * d["V2_rho"]["median"] and disjoint:
        verdict = "E2_CONTRAST_REPRODUCED"
    elif not disjoint:
        verdict = "E2_CONTRAST_ABSENT"
    else:
        verdict = "E2_CONTRAST_PARTIAL"
    d["verdict"] = verdict
    d["kill_gate_binding"] = (d["V1_rho"]["median"] > 1 and d["V2_rho"]["median"] > 1)
    d["estimator_caveat_fired"] = q(v1e, .5) < 0.999
    summary["shapes"][name] = d
    print(name, "VERDICT", verdict, json.dumps(d)[:500], flush=True)

summary["compile_comparator"] = comp_rows
summary["launch_geometry_A"] = geo
with open(os.path.splitext(OUT)[0] + "_summary.json", "w") as fh:
    json.dump(summary, fh, indent=1)
print("DONE")
