#!/usr/bin/env python3
"""s100: per-thread sequential depth ladder at FIXED shape (S1 official) + s100b Welford V1 arm.

Prereg frozen pre-run in lab/s100_prereg.md (commit e9c8301). Everything below follows that text:
levels T in {128,256,512,768,1024} => D = N/T in {32768,16384,8192,5461,4096}, one block per row,
declared fan-in-2 shared reduction, sum/sumsq estimator, s98's shared tiled float4 affine; same-run V2
fast-statistics as the contrast denominator; K1 tie-out to s98's T=128 rows on seeds 9800+d.

Implementation note (mechanical, no threshold touched): the fan-in-2 loop handles non-power-of-2 T by
halving with ceil(active/2) so no lane is dropped; for power-of-2 T this is identical to s98's
"for stride = T/2 .. 1, lane < stride" loop, which is what K1 checks.
"""
import json, os, statistics, torch, torch.nn.functional as F
import torch.utils.cpp_extension

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

OUTDIR = os.environ.get("S100_OUT", "results")
LEVELS = [128, 256, 512, 768, 1024]
S1 = (16, 64, 256, 256)
S2 = (16, 32, 128, 128)
DRAWS = 16
SEGS = 32
EPS = 1e-5

cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Declared tree: one block per row, blockDim=T threads, per-thread strided sequential accumulation
// (i += blockDim.x) of depth N/T, then shared-memory fan-in-2 over the T lanes. Estimator sum/sumsq.
__global__ void depth_stats(const float* __restrict__ x, float* __restrict__ mean, float* __restrict__ rstd,
                            const long N, const float eps){
  extern __shared__ float sm[];
  const long row = blockIdx.x;
  const float* xr = x + row*N;
  const int T = blockDim.x;
  float s = 0.f, q = 0.f;
  for (long i = threadIdx.x; i < N; i += T){ float v = xr[i]; s += v; q += v*v; }
  sm[threadIdx.x] = s;
  sm[T + threadIdx.x] = q;
  __syncthreads();
  int active = T;
  while (active > 1){
    int stride = (active + 1) / 2;
    int limit = active - stride;
    if ((int)threadIdx.x < limit){
      sm[threadIdx.x] += sm[threadIdx.x + stride];
      sm[T + threadIdx.x] += sm[T + threadIdx.x + stride];
    }
    __syncthreads();
    active = stride;
  }
  if (threadIdx.x == 0){
    float m = sm[0] / (float)N;
    float v = sm[T] / (float)N - m*m;
    mean[row] = m;
    rstd[row] = rsqrtf(v + eps);
  }
}

// Same tree, Welford estimator with Chan's pairwise merge (counts carried in shared memory).
__global__ void welford_stats(const float* __restrict__ x, float* __restrict__ mean, float* __restrict__ rstd,
                              const long N, const float eps){
  extern __shared__ float sm[];   // [0..T) mean, [T..2T) M2, [2T..3T) count
  const long row = blockIdx.x;
  const float* xr = x + row*N;
  const int T = blockDim.x;
  long cnt = 0; float m = 0.f, M2 = 0.f;
  for (long i = threadIdx.x; i < N; i += T){
    float v = xr[i]; cnt++;
    float d = v - m;
    m += d / (float)cnt;
    M2 += d * (v - m);
  }
  sm[threadIdx.x] = m;
  sm[T + threadIdx.x] = M2;
  sm[2*T + threadIdx.x] = (float)cnt;
  __syncthreads();
  int active = T;
  while (active > 1){
    int stride = (active + 1) / 2;
    int limit = active - stride;
    if ((int)threadIdx.x < limit){
      float ma = sm[threadIdx.x], M2a = sm[T + threadIdx.x], na = sm[2*T + threadIdx.x];
      float mb = sm[threadIdx.x + stride], M2b = sm[T + threadIdx.x + stride], nb = sm[2*T + threadIdx.x + stride];
      float n = na + nb;
      float delta = mb - ma;
      float mn = ma + delta * nb / n;
      float M2n = M2a + M2b + delta * delta * na * nb / n;
      sm[threadIdx.x] = mn;
      sm[T + threadIdx.x] = M2n;
      sm[2*T + threadIdx.x] = n;
    }
    __syncthreads();
    active = stride;
  }
  if (threadIdx.x == 0){
    mean[row] = sm[0];
    rstd[row] = rsqrtf(sm[T] / (float)N + eps);
  }
}

// s98's shared tiled affine, verbatim: float4 per thread, grid = ceil(total/4/256).
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

// V2 fast-statistics, s98 verbatim: grid M*segs, 256 threads, contiguous segment, fan-in-2 x 8,
// then a single-thread sequential fan-in-32 merge. Estimator sum/sumsq.
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

std::vector<torch::Tensor> stats_depth(torch::Tensor x, double eps, int64_t threads){
  auto M = x.size(0); long N = x.numel() / M;
  auto mean = torch::empty({M}, x.options());
  auto rstd = torch::empty({M}, x.options());
  size_t smem = (size_t)2 * threads * sizeof(float);
  depth_stats<<<(unsigned)M, (unsigned)threads, smem>>>((const float*)x.data_ptr<float>(),
      mean.data_ptr<float>(), rstd.data_ptr<float>(), N, (float)eps);
  return {mean, rstd};
}

std::vector<torch::Tensor> stats_welford(torch::Tensor x, double eps, int64_t threads){
  auto M = x.size(0); long N = x.numel() / M;
  auto mean = torch::empty({M}, x.options());
  auto rstd = torch::empty({M}, x.options());
  size_t smem = (size_t)3 * threads * sizeof(float);
  welford_stats<<<(unsigned)M, (unsigned)threads, smem>>>((const float*)x.data_ptr<float>(),
      mean.data_ptr<float>(), rstd.data_ptr<float>(), N, (float)eps);
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
std::vector<torch::Tensor> stats_depth(torch::Tensor x, double eps, int64_t threads);
std::vector<torch::Tensor> stats_welford(torch::Tensor x, double eps, int64_t threads);
std::vector<torch::Tensor> stats_v2(torch::Tensor x, double eps, int64_t segs);
torch::Tensor affine(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor mean, torch::Tensor rstd);
"""

ext = torch.utils.cpp_extension.load_inline(
    name="s100_kernels", cpp_sources=cpp_src, cuda_sources=cuda_src,
    functions=["stats_depth", "stats_welford", "stats_v2", "affine"], verbose=False,
    extra_cuda_cflags=["-O3"],
)

def bits(t):
    return t.view(torch.int32)

def exact_frac(a, b):
    return float((bits(a) == bits(b)).float().mean())

def eval_ratio(cand, ref):
    d = (cand.double() - ref.double()).abs()
    den = 1e-4 + 1e-4 * ref.double().abs()
    r = d / den
    return float(r.max()), float((r > 1).double().mean()), float((r > 1e2).double().mean())

def med_time(fn, reps=10):
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
        t0.record(); fn(); t1.record(); torch.cuda.synchronize()
        ts.append(t0.elapsed_time(t1))
    return statistics.median(ts)

def profile_kernels(fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn(); torch.cuda.synchronize()
    names = {}
    for e in prof.key_averages():
        if e.device_type == torch.autograd.DeviceType.CUDA or True:
            names[e.key] = names.get(e.key, 0) + e.count
    return names

def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0]*len(v); i = 0
        while i < len(v):
            j = i
            while j+1 < len(v) and v[order[j+1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j+1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs); mx = sum(rx)/n; my = sum(ry)/n
    num = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    den = (sum((a-mx)**2 for a in rx) * sum((b-my)**2 for b in ry)) ** 0.5
    return num/den if den else 0.0

def q(vals, p):
    vals = sorted(vals); k = (len(vals)-1)*p
    lo, hi = int(k), min(int(k)+1, len(vals)-1)
    return vals[lo] + (vals[hi]-vals[lo])*(k-lo)

os.makedirs(OUTDIR, exist_ok=True)
env = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0),
       "cc": torch.cuda.get_device_capability(0), "segs": SEGS, "eps": EPS}

# ---------- s100: depth ladder ----------
M1, N1 = S1[0], S1[1]*S1[2]*S1[3]
ladder_rows = []
v2_rows = []
geom = {}
for T in LEVELS:
    D = N1 // T
    for d in range(DRAWS):
        torch.manual_seed(9800 + d)
        x = torch.randn(S1, device="cuda", dtype=torch.float32)
        w = torch.randn(S1[1:], device="cuda", dtype=torch.float32)
        b = torch.randn(S1[1:], device="cuda", dtype=torch.float32)
        A = F.layer_norm(x, S1[1:], w, b, EPS)
        X64 = F.layer_norm(x.double(), S1[1:], w.double(), b.double(), EPS)
        m, r = ext.stats_depth(x, EPS, T)
        Y = ext.affine(x, w, b, m, r)
        torch.cuda.synchronize()
        rho, f1, f100 = eval_ratio(Y, A)
        rho64, _, _ = eval_ratio(Y, X64)
        xbar = x.mean(dim=(1, 2, 3)).double()
        rec = {"exp": "s100_ladder", "shape": list(S1), "M": M1, "N": N1, "T": T, "D": D, "draw": d,
               "seed": 9800 + d, "rho_vs_ref": rho, "frac_over_1e4": f1, "frac_over_1e2": f100,
               "rho_vs_fp64": rho64, "exact_frac_vs_A": exact_frac(Y, A),
               "max_abs_vs_A": float((Y.double()-A.double()).abs().max()),
               "mean_dev_rel": float((m.double()-xbar).abs().max()/xbar.abs().max()),
               "ref_rho_vs_fp64": eval_ratio(A, X64)[0]}
        if d == 0:
            rec["timing_ms"] = med_time(lambda: ext.affine(x, w, b, *ext.stats_depth(x, EPS, T)))
            geom[str(T)] = profile_kernels(lambda: ext.affine(x, w, b, *ext.stats_depth(x, EPS, T)))
        ladder_rows.append(rec)
        print(f"T{T} D{D} d{d}: rho {rho:.4f} f1 {f1:.2e} exact {rec['exact_frac_vs_A']:.4f} rho64 {rho64:.4f}", flush=True)
        if T == LEVELS[0] and d < DRAWS:
            pass
# same-run V2 contrast denominator
for d in range(DRAWS):
    torch.manual_seed(9800 + d)
    x = torch.randn(S1, device="cuda", dtype=torch.float32)
    w = torch.randn(S1[1:], device="cuda", dtype=torch.float32)
    b = torch.randn(S1[1:], device="cuda", dtype=torch.float32)
    A = F.layer_norm(x, S1[1:], w, b, EPS)
    m, r = ext.stats_v2(x, EPS, SEGS)
    Y = ext.affine(x, w, b, m, r)
    torch.cuda.synchronize()
    rho, f1, f100 = eval_ratio(Y, A)
    v2_rows.append({"exp": "s100_v2", "shape": list(S1), "draw": d, "seed": 9800+d, "rho_vs_ref": rho,
                    "frac_over_1e4": f1, "exact_frac_vs_A": exact_frac(Y, A)})
print("V2 done", flush=True)

# ---------- s100b: Welford arm ----------
welf_rows = []
for name, shape in (("S1_official", S1), ("S2_reduced", S2)):
    Mx = shape[0]; Nx = shape[1]*shape[2]*shape[3]
    for d in range(DRAWS):
        torch.manual_seed(9800 + d)
        x = torch.randn(shape, device="cuda", dtype=torch.float32)
        w = torch.randn(shape[1:], device="cuda", dtype=torch.float32)
        b = torch.randn(shape[1:], device="cuda", dtype=torch.float32)
        A = F.layer_norm(x, shape[1:], w, b, EPS)
        X64 = F.layer_norm(x.double(), shape[1:], w.double(), b.double(), EPS)
        mw, rw = ext.stats_welford(x, EPS, 128)
        W = ext.affine(x, w, b, mw, rw)
        torch.cuda.synchronize()
        rho, f1, f100 = eval_ratio(W, A)
        var64 = x.double().var(dim=(1, 2, 3), unbiased=False)
        varW = (1.0 / rw.double()**2) - EPS
        rec = {"exp": "s100b_welford", "shape_name": name, "shape": list(shape), "M": Mx, "N": Nx,
               "T": 128, "D": Nx // 128, "draw": d, "seed": 9800+d, "rho_vs_ref": rho,
               "frac_over_1e4": f1, "frac_over_1e2": f100, "rho_vs_fp64": eval_ratio(W, X64)[0],
               "exact_frac_vs_A": exact_frac(W, A), "max_abs_vs_A": float((W.double()-A.double()).abs().max()),
               "welford_var_vs_fp64_rel": float(((varW - var64).abs() / var64.abs()).max())}
        if d == 0:
            rec["timing_ms"] = med_time(lambda: ext.affine(x, w, b, *ext.stats_welford(x, EPS, 128)))
        welf_rows.append(rec)
        print(f"Welford {name} d{d}: rho {rho:.4f} exact {rec['exact_frac_vs_A']:.4f} varrel {rec['welford_var_vs_fp64_rel']:.2e}", flush=True)

# ---------- write artifacts ----------
def dump(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

dump(os.path.join(OUTDIR, "s100_depth_ladder.jsonl"), ladder_rows)
dump(os.path.join(OUTDIR, "s100_v2_contrast.jsonl"), v2_rows)
dump(os.path.join(OUTDIR, "s100b_welford.jsonl"), welf_rows)

# ---------- frozen verdicts ----------
summary = {"env": env, "prereg": "lab/s100_prereg.md commit e9c8301", "s100": {}, "s100b": {}}
levels = {}
for T in LEVELS:
    rs = [r for r in ladder_rows if r["T"] == T]
    rhos = [r["rho_vs_ref"] for r in rs]
    levels[str(T)] = {"T": T, "D": N1//T, "n": len(rs), "rho_median": q(rhos, .5), "rho_q25": q(rhos, .25),
                      "rho_q75": q(rhos, .75), "rho_min": min(rhos), "rho_max": max(rhos),
                      "frac_over_1e4_median": q([r["frac_over_1e4"] for r in rs], .5),
                      "exact_frac_median": q([r["exact_frac_vs_A"] for r in rs], .5),
                      "rho_vs_fp64_median": q([r["rho_vs_fp64"] for r in rs], .5),
                      "mean_dev_rel_max": max(r["mean_dev_rel"] for r in rs),
                      "timing_ms_draw0": next((r.get("timing_ms") for r in rs if r.get("timing_ms")), None)}
v2med = q([r["rho_vs_ref"] for r in v2_rows], .5)
summary["s100"]["levels"] = levels
summary["s100"]["v2_rho_median"] = v2med
summary["s100"]["v2_exact_median"] = q([r["exact_frac_vs_A"] for r in v2_rows], .5)
summary["s100"]["launch_geometry_kernels"] = geom
# M1 monotonicity
meds = [levels[str(T)]["rho_median"] for T in LEVELS]   # descending D
Ds = [levels[str(T)]["D"] for T in LEVELS]
rho_s = spearman(Ds * 1, meds)
row_rho_s = spearman([r["D"] for r in ladder_rows], [r["rho_vs_ref"] for r in ladder_rows])
strict_inc = all(meds[i] < meds[i+1] for i in range(len(meds)-1))  # D descending => rho must fall
summary["s100"]["monotonicity"] = {"level_medians_by_D_desc": dict(zip([str(d) for d in Ds], meds)),
                                   "spearman_level_medians": rho_s, "spearman_rows": row_rho_s,
                                   "strictly_increasing_in_D": strict_inc}
m1 = strict_inc and row_rho_s >= 0.60
summary["s100"]["M1"] = "M1_MONOTONE" if m1 else "DEPTH_NONMONOTONE"
# D* boundary at 0.25x V2
cand = [T for T in LEVELS if levels[str(T)]["rho_median"] <= 0.25*v2med]
if cand:
    Dstar = max(levels[str(T)]["D"] for T in cand)
    summary["s100"]["D_star"] = Dstar
    summary["s100"]["D_star_branch"] = "BOUNDARY_LOCATED"
elif all(levels[str(T)]["rho_median"] <= 0.25*v2med for T in LEVELS):
    summary["s100"]["D_star_branch"] = "BELOW_RANGE"
else:
    summary["s100"]["D_star_branch"] = "NO_BOUNDARY_IN_RANGE"
# kills
s98_tie = None
try:
    s98 = [json.loads(l) for l in open("results/s98_cuda_pair.jsonl")]
    s98_v1 = {(r["shape_name"], r["draw"]): r["arms"]["V1_ref_order"]["rho_vs_ref"] for r in s98}
    mine = {(r["T"], r["draw"]): r["rho_vs_ref"] for r in ladder_rows}
    diffs = [abs(mine[(128, d)] - s98_v1[("S1_official", d)]) for d in range(DRAWS)]
    s98_tie = {"max_abs_rho_diff": max(diffs), "n": len(diffs)}
except FileNotFoundError:
    s98_tie = {"status": "s98 artifact not present on spark"}
summary["s100"]["K1_s98_tieout"] = s98_tie
summary["s100"]["K2_engine_sets"] = {k: sorted(v.items(), key=lambda kv: -kv[1])[:4] for k, v in geom.items()}
summary["s100"]["K3_mean_dev_rel_max"] = max(levels[str(T)]["mean_dev_rel_max"] for T in LEVELS)

# s100b
for name in ("S1_official", "S2_reduced"):
    rs = [r for r in welf_rows if r["shape_name"] == name]
    ex = q([r["exact_frac_vs_A"] for r in rs], .5)
    band = ("ESTIMATOR_EXPLAINS" if ex >= 0.999 else "ESTIMATOR_DOMINANT" if ex >= 0.90
            else "PARTIAL" if ex >= 0.55 else "TREE_OWNS")
    summary["s100b"][name] = {"n": len(rs), "exact_frac_median": ex,
                              "rho_vs_ref_median": q([r["rho_vs_ref"] for r in rs], .5),
                              "rho_vs_ref_q25": q([r["rho_vs_ref"] for r in rs], .25),
                              "rho_vs_ref_q75": q([r["rho_vs_ref"] for r in rs], .75),
                              "rho_vs_fp64_median": q([r["rho_vs_fp64"] for r in rs], .5),
                              "frac_over_1e4_median": q([r["frac_over_1e4"] for r in rs], .5),
                              "welford_var_vs_fp64_rel_max": max(r["welford_var_vs_fp64_rel"] for r in rs),
                              "band": band, "timing_ms_draw0": next((r.get("timing_ms") for r in rs if r.get("timing_ms")), None)}

with open(os.path.join(OUTDIR, "s100_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print("SUMMARY", json.dumps({k: summary["s100"][k] for k in ("M1", "D_star_branch", "v2_rho_median")}), flush=True)
print("BANDS", json.dumps({k: v["band"] for k, v in summary["s100b"].items()}), flush=True)
print("DONE")
