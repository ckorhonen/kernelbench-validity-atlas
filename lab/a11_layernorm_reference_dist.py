"""A11 — reference-distance (rho) and delivery budget for wide LayerNorm across input distributions.
Tick #56, 2026-10-25. Exploitation lane, experiment A.

PREREG (frozen in this docstring BEFORE the run; committed before the spark run):
  Question: does PyTorch's fp32 LayerNorm reference at the official KernelBench L1/40 shape consume the
  benchmark's own tolerance budget, and is that consumption a function of the INPUT DISTRIBUTION
  (kappa = |mean|/std of the normalized axis) rather than of the shape?
  Claim under test (from the exploitation plan, derived analytically before this run): the split-merge
  error amplification is proportional to kappa*sqrt(n_splits); plain sequential summation is NOT the
  mechanism at this depth (its relative error ~1e-7 at n=4.19M).
  Measured on the planning host before this run: benchmark default get_inputs() is torch.rand = U[0,1),
  kappa 1.73; normal+100 kappa 99.98; normal+1000 kappa 1000.01.

  Arms per distribution (all on the SAME tensors, one seed, recorded):
    R_def  = nn.LayerNorm((64,256,256)) fp32 forward on cuda (the pinned evaluator's reference)
    R_64   = the same computation in fp64 (anchor; PyTorch fp64 LayerNorm, its own tree — an ANCHOR, not
             a structural twin; scope stated so nobody reads this as a bitwise comparison)
    C_fast = split-statistics prototype: mean/var over dim (1,2,3) with unbiased=False, then
             (x-mean)*rsqrt(var+eps)*gamma+beta, all fp32 — i.e. torch.var_mean + pointwise, which is what
             sylve_recursions' fast prototype does.
    C_seq  = the same math but with a sequential per-row fp32 reduction tree? NOT run here (cost); declared
             as a scope note. The ordered variant's bit identity is taken from sylve's own measurement.

  Metrics per (distribution, arm):
    rho_anchor  = max_elem |arm - R_64| / (1e-4 + 1e-4*|R_64|)      [arm's distance from exact arithmetic]
    frac_over_anchor = fraction of elements where that ratio > 1
    budget_vs_ref = max_elem |arm - R_def| / (1e-4 + 1e-4*|arm|)   [the evaluator's own predicate,
                    candidate-scaled: atol + rtol*|candidate|]
    pass_vs_ref  = allclose(R_def, arm, atol=1e-4, rtol=1e-4)      [binary, whole-tensor]
    pass_vs_anchor = allclose(R_64.float(), arm, atol=1e-4, rtol=1e-4)  [anchor contract, diagnostic]
    kappa = |mean|/std of the normalized axis, per row, reported as min/median/max over rows.

  FROZEN FALSIFIERS (bars are NOT moved after seeing the numbers):
    P1 reference anchor-clean at the benchmark default: rho_anchor(R_def) < 0.1 at U[0,1).
    P2 reference anchor-dirty at high kappa: rho_anchor(R_def) > 0.5 at normal+1000.
    P3 the split prototype is admissible on the benchmark default (pass_vs_ref True at U[0,1)) and
       inadmissible at normal+1000 (pass_vs_ref False).
    P4 rho_anchor(R_def) increases monotonically in kappa across the six distributions.
    P5 determinism: R_def recomputed on the same inputs is bit-equal.
  Readings: if P1,P2,P3 hold => the school's binary choice is a ROBUSTNESS choice, not a benchmark
  requirement, and the fast prototype's domain is bounded in kappa. If P1 fires and P2 does not => bit
  identity is vindicated at the benchmark shape and the distribution axis is not load-bearing. If P4 fails
  => rho is not monotone in kappa and the domain predicate needs more than one coordinate (the A10 lesson
  repeating on a new axis, which would be reported as such).

  Scope: one architecture (GB10/sm_120), one torch build, fp32 in / fp32 out, affine params default
  (gamma=1, beta=0) plus one random-affine arm, one seed per distribution (seeds recorded), inference only.
  fp64 LayerNorm is an anchor, not a structural twin: "distance from fp64" measures distance from AN exact
  answer, which is the right quantity for a conditioning question and the wrong quantity for a bitwise one.

Run on spark: python3 a11_layernorm_reference_dist.py -> results/a11_layernorm_reference_dist.json
"""
import json, os, time
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
OUT = os.path.join(RES, "a11_layernorm_reference_dist.json")
DATE = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())
dev = "cuda"
ATOL = RTOL = 1e-4
EPS = 1e-5
B, C, H, W = 16, 64, 256, 256
NORM = (C, H, W)


def ratio_stats(a, b):
    """ratio of |a-b| to the budget (1e-4 + 1e-4*|b|), b-scaled as the evaluator does for its second arg."""
    d = (a.double() - b.double()).abs()
    denom = ATOL + RTOL * b.double().abs()
    ratio = (d / denom).flatten()
    return {
        "max_ratio": float(ratio.max()),
        "med_ratio": float(ratio.median()),
        "p99_ratio": float(torch.quantile(ratio[: min(ratio.numel(), 4_000_000)].float(), 0.99)),
        "frac_over": float((ratio > 1).double().mean()),
        "abs_max": float(d.max()),
        "n_elems": int(ratio.numel()),
    }


def kappa_of(x):
    xf = x.float()
    m = xf.mean(dim=(1, 2, 3), dtype=torch.float64)
    v = xf.var(dim=(1, 2, 3), unbiased=False, dtype=torch.float64)
    sd = v.sqrt()
    k = (m.abs() / sd).tolist()
    return {"min": float(min(k)), "med": float(sorted(k)[len(k) // 2]), "max": float(max(k)),
            "mean_abs": float(m.abs().mean()), "std_med": float(sd.median())}


def make(dist, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if dist == "uniform_benchmark_default":
        return torch.rand(B, C, H, W, generator=g)
    if dist == "normal":
        return torch.randn(B, C, H, W, generator=g)
    if dist == "normal_x100":
        return torch.randn(B, C, H, W, generator=g) * 100.0
    if dist == "normal_plus100":
        return torch.randn(B, C, H, W, generator=g) + 100.0
    if dist == "normal_plus1000":
        return torch.randn(B, C, H, W, generator=g) + 1000.0
    if dist == "constant_10000":
        return torch.full((B, C, H, W), 10000.0)
    raise ValueError(dist)


def fast_proto(x, gamma, beta):
    """split-statistics prototype: var_mean over the normalized axes in fp32, then pointwise affine."""
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    v = x.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
    inv = torch.rsqrt(v + EPS)
    return (x - m) * inv * gamma + beta


DISTS = ["uniform_benchmark_default", "normal", "normal_x100", "normal_plus100", "normal_plus1000",
         "constant_10000"]

rows = []
hdr = {
    "date": DATE,
    "torch": torch.__version__,
    "device": torch.cuda.get_device_name(0),
    "cap": list(torch.cuda.get_device_capability(0)),
    "shape": [B, C, H, W],
    "norm_axis": list(NORM),
    "norm_elems_per_row": C * H * W,
    "prereg": "docstring of this file, frozen before run",
    "anchor_scope": "fp64 LayerNorm is an anchor, not a structural twin of the fp32 path",
    "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
    "float32_matmul_precision": torch.get_float32_matmul_precision(),
}

ln = nn.LayerNorm(NORM).to(dev).eval()
torch.manual_seed(20261025)
gamma_rand = torch.randn(NORM, device=dev) * 0.5 + 1.0
beta_rand = torch.randn(NORM, device=dev) * 0.5

for di, dist in enumerate(DISTS):
    seed = 20261025 + 1000 * (di + 1)
    x_cpu = make(dist, seed)
    x = x_cpu.to(dev)
    kap = kappa_of(x_cpu)
    torch.backends.cudnn.deterministic = True
    with torch.no_grad():
        r_def = ln(x)
        r_def2 = ln(x)
        det = bool(torch.equal(r_def, r_def2))
        r_64 = torch.nn.functional.layer_norm(x.double(), NORM, None, None, EPS)
        c_fast = fast_proto(x, ln.weight, ln.bias)
        c_fast_rand = fast_proto(x, gamma_rand, beta_rand)
        r_def_rand = torch.nn.functional.layer_norm(x, NORM, gamma_rand, beta_rand, EPS)
        r_64_rand = torch.nn.functional.layer_norm(x.double(), NORM,
                                                   gamma_rand.double(), beta_rand.double(), EPS)

    row = {"distribution": dist, "seed": seed, "kappa": kap, "det_repeat_equal": det,
           "arms": {}}
    for name, a, anchor in [("R_def", r_def, r_64), ("C_fast", c_fast, r_64)]:
        st = ratio_stats(a, anchor)
        vs_ref = ratio_stats(a, r_def)
        st["pass_vs_ref"] = bool(torch.allclose(r_def, a, atol=ATOL, rtol=RTOL, equal_nan=False))
        st["pass_vs_anchor"] = bool(torch.allclose(r_64.float(), a, atol=ATOL, rtol=RTOL, equal_nan=False))
        st["budget_vs_ref_max"] = vs_ref["max_ratio"]
        st["frac_over_vs_ref"] = vs_ref["frac_over"]
        row["arms"][name] = st
    # random-affine control on the same tensor (does gamma/beta change the story?)
    row["arms"]["R_def_randaffine"] = ratio_stats(r_def_rand, r_64_rand)
    row["arms"]["C_fast_randaffine"] = ratio_stats(c_fast_rand, r_64_rand)
    rows.append(row)
    print(f"{dist:26s} kappa_med={kap['med']:.4g}  rho(R_def)={row['arms']['R_def']['max_ratio']:.4g} "
          f"frac_over={row['arms']['R_def']['frac_over']:.4g}  rho(C_fast)={row['arms']['C_fast']['max_ratio']:.4g} "
          f"fast_pass_vs_ref={row['arms']['C_fast']['pass_vs_ref']}", flush=True)
    del x, r_def, r_def2, r_64, c_fast, c_fast_rand, r_def_rand, r_64_rand, x_cpu
    torch.cuda.empty_cache()

# frozen verdicts
by = {r["distribution"]: r for r in rows}
p1 = by["uniform_benchmark_default"]["arms"]["R_def"]["max_ratio"] < 0.1
p2 = by["normal_plus1000"]["arms"]["R_def"]["max_ratio"] > 0.5
p3 = (by["uniform_benchmark_default"]["arms"]["C_fast"]["pass_vs_ref"] is True and
      by["normal_plus1000"]["arms"]["C_fast"]["pass_vs_ref"] is False)
rhos = [by[d]["arms"]["R_def"]["max_ratio"] for d in DISTS]
p4 = all(rhos[i] <= rhos[i + 1] for i in range(len(rhos) - 1))
p5 = all(r["det_repeat_equal"] for r in rows)

verdicts = {"P1_ref_anchor_clean_at_benchmark_default": bool(p1),
            "P2_ref_anchor_dirty_at_kappa1000": bool(p2),
            "P3_fast_admissible_on_benchmark_inadmissible_at_1000": bool(p3),
            "P4_ref_rho_monotone_in_kappa": bool(p4),
            "P5_determinism": bool(p5),
            "rho_by_distribution": dict(zip(DISTS, rhos)),
            "kappa_med_by_distribution": {d: by[d]["kappa"]["med"] for d in DISTS}}

if p1 and p2 and p3:
    reading = "ROBUSTNESS_DOMAIN"
elif p1 and not p2:
    reading = "BIT_IDENTITY_VINDICATED"
elif not p4:
    reading = "NON_MONOTONE_IN_KAPPA"
else:
    reading = "MIXED"
verdicts["frozen_reading"] = reading

out = {"header": hdr, "rows": rows, "verdicts": verdicts}
json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT, "rows", len(rows))
print(json.dumps(verdicts, indent=1))
