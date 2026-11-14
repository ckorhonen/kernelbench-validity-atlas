#!/usr/bin/env python3
"""s100c: POST-HOC refinement (declared after s100's numbers, no threshold changed).

s100 measured a 3.0x jump in rho_vs_ref between T=128 (0.0132) and T=256 (0.0401) at fixed shape, i.e. the
reference-order advantage appears to be a cliff at the reference's own realized block width (128 lanes) rather
than a smooth depth law. This leg adds intermediate, non-power-of-2 widths {144, 160, 192, 224} at S1 with 8
draws to distinguish a structural cliff at T=128 from a continuous decline. Descriptive; the frozen s100
verdicts are untouched. Reuses the already-built s100_kernels extension from the torch extension cache.
"""
import glob, importlib.util, json, os, statistics, sys, torch, torch.nn.functional as F
import torch.utils.cpp_extension as ce

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

so = glob.glob(os.path.join(ce._get_build_directory("s100_kernels", verbose=False), "s100_kernels.so"))
assert so, "s100_kernels.so not found in the extension cache"
spec = importlib.util.spec_from_file_location("s100_kernels", so[0])
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)

OUTDIR = os.environ.get("S100_OUT", "results")
S1 = (16, 64, 256, 256)
EPS = 1e-5
DRAWS = 8
LEVELS = [128, 144, 160, 192, 224, 256]

def bits(t):
    return t.view(torch.int32)

def exact_frac(a, b):
    return float((bits(a) == bits(b)).float().mean())

def eval_ratio(cand, ref):
    d = (cand.double() - ref.double()).abs()
    den = 1e-4 + 1e-4 * ref.double().abs()
    r = d / den
    return float(r.max()), float((r > 1).double().mean())

def q(vals, p):
    vals = sorted(vals); k = (len(vals)-1)*p
    lo, hi = int(k), min(int(k)+1, len(vals)-1)
    return vals[lo] + (vals[hi]-vals[lo])*(k-lo)

N = S1[1]*S1[2]*S1[3]
rows = []
for T in LEVELS:
    D = N // T
    for d in range(DRAWS):
        torch.manual_seed(9800 + d)
        x = torch.randn(S1, device="cuda", dtype=torch.float32)
        w = torch.randn(S1[1:], device="cuda", dtype=torch.float32)
        b = torch.randn(S1[1:], device="cuda", dtype=torch.float32)
        A = F.layer_norm(x, S1[1:], w, b, EPS)
        m, r = ext.stats_depth(x, EPS, T)
        Y = ext.affine(x, w, b, m, r)
        torch.cuda.synchronize()
        rho, f1 = eval_ratio(Y, A)
        rows.append({"exp": "s100c_refine", "T": T, "D": D, "draw": d, "seed": 9800+d, "rho_vs_ref": rho,
                     "frac_over_1e4": f1, "exact_frac_vs_A": exact_frac(Y, A)})
        print(f"T{T} D{D} d{d}: rho {rho:.4f} exact {rows[-1]['exact_frac_vs_A']:.4f}", flush=True)

with open(os.path.join(OUTDIR, "s100c_refine.jsonl"), "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")

summary = {"exp": "s100c_refine", "note": "post-hoc refinement declared after s100; no threshold changed",
           "levels": {}}
for T in LEVELS:
    rs = [r for r in rows if r["T"] == T]
    summary["levels"][str(T)] = {"T": T, "D": N//T, "n": len(rs),
                                 "rho_median": q([r["rho_vs_ref"] for r in rs], .5),
                                 "rho_min": min(r["rho_vs_ref"] for r in rs),
                                 "rho_max": max(r["rho_vs_ref"] for r in rs),
                                 "exact_frac_median": q([r["exact_frac_vs_A"] for r in rs], .5),
                                 "frac_over_1e4_median": q([r["frac_over_1e4"] for r in rs], .5)}
print(json.dumps(summary["levels"]))
with open(os.path.join(OUTDIR, "s100c_refine_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print("DONE")
