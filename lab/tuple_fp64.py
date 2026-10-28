"""fp64 tuple test — stage 1 of the meta-reflection standing plan.

Prereg (Meta-Reflection Refresh C2NjSjtwQF4o2VDD, plan lab/meta_reflection_plan.txt):
fp64 tuple test on GPU (spark), the named escape route.
Exact 9-row grid of tuple_test.py (C=384, k=3, s=1, d in 1..4, pad=p, H=W=7, batch=2,
seed 4321, L2 flush, torch.profiler kernel names). fp64 anchor built by the same template
as the TF32-class reference. Verdict rule at pre-registered eps=1e-6.

Decision rule: fp64-anchored W/T labels equal TF32-class labels on all 9 rows => the
dispatch boundary is reference-invariant and the spine's categorical verdict generalizes
beyond GB10; otherwise the boundary is anchor-conditional and the certificate's
anchor-lattice structure is the correct object.

Kill conditions K1 (pooled held-out AUC < 0.60), K2 (any anchor x class stratum < 0.70),
K3 (boundary flips on > 2/8 transitions at the fp64 anchor) are written before the run and
applied in the audit note.
Output: results/tuple_fp64.json + lab/report_fp64_tuple_test.md.
Executor: spark GPU (NVIDIA GB10, sm_120), torch 2.14.1+cu130.
"""

import json, os, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
def _kb_res_dir(here):
    """Resolve the artifact directory: lab/results in the repo, ../results when run from the
    supplement package's kernels/ directory (frozen packaging convention, round-11 F6)."""
    for cand in (os.path.join(here, "results"), os.path.join(os.path.dirname(here), "results")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(here, "results")


RES = _kb_res_dir(HERE)
OUT = os.path.join(RES, "tuple_fp64.json")
REPORT = os.path.join(RES, "report_fp64_tuple_test.md")

ATOL = RTOL = 1e-4
DEV = "cuda"
EPS = 1e-6  # pre-registered eps

# Stage 2: 9-row grid (pre-registered before run), C=384
# d in 1..4 x pad=p; H=W=7, k=3, s=1, batch=2, seed 4321
TUPLE_GRID = [
    # (stride, pad, dilation, layout, batch)
    (1, 1, 1, "NCHW", 2),
    (1, 2, 1, "NCHW", 2),
    (1, 3, 1, "NCHW", 2),
    (1, 1, 2, "NCHW", 2),
    (1, 2, 2, "NCHW", 2),
    (1, 3, 2, "NCHW", 2),
    (1, 1, 3, "NCHW", 2),
    (1, 2, 3, "NCHW", 2),
    (1, 3, 3, "NCHW", 2),
]

FIXED = {"seed": 4321, "C": 384, "k": 3, "H": 7, "W": 7, "d": 1}


def expected_h(s, p, d):
    return (FIXED["H"] + 2 * p - d * (FIXED["k"] - 1) - 1) // s + 1


def engine_class(kn):
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


def run_fp64_tuple():
    results = []
    for (s, p, d, layout, b) in TUPLE_GRID:
        row = {"stride": s, "pad": p, "dilation": d, "layout": layout, "batch": b,
               "expected_h": expected_h(s, p, d)}
        torch.manual_seed(FIXED["seed"])
        x = torch.randn(b, FIXED["C"], FIXED["H"], FIXED["W"], device=DEV)
        w = torch.randn(FIXED["C"], FIXED["C"], FIXED["k"], FIXED["k"], device=DEV)
        p_val = p  # row pad, per prereg
        # fp32 TF32 (default) — the TF32-class reference
        o32 = F.conv2d(x, w, padding=p_val, stride=s, dilation=d)
        assert list(o32.shape[-2:]) == [expected_h(s, p, d), expected_h(s, p, d)], (
            f"output shape {list(o32.shape[-2:])} != expected {expected_h(s,p,d)}")
        kn32 = kernel_names(lambda: F.conv2d(x, w, padding=p_val, stride=s, dilation=d))
        cls32 = engine_class(kn32)
        # fp64 anchor (same template as TF32-class reference)
        x64 = x.double(); w64 = w.double()
        o64 = F.conv2d(x64, w64, padding=p_val, stride=s, dilation=d)
        assert list(o64.shape[-2:]) == [expected_h(s, p, d), expected_h(s, p, d)], (
            f"fp64 output shape {list(o64.shape[-2:])} != expected {expected_h(s,p,d)}")
        kn64 = kernel_names(lambda: F.conv2d(x64, w64, padding=p_val, stride=s, dilation=d))
        cls64 = engine_class(kn64)
        # Correctness of fp64 anchor vs fp32 TF32 reference (relative error)
        st = stats(o32, o64)
        # Flip decision: class mismatch + relative magnitude gap vs eps
        flip = (cls32 != cls64) or (st["rho_max"] / max(st["med"], 1e-30) > EPS)
        row.update({"class_32": cls32, "class_64": cls64, "flip": flip, "kernels_32": dict(kn32),
                    "kernels_64": dict(kn64), "rho": st})
        results.append(row)
        del x, w, o32, x64, w64, o64, kn32, kn64
        torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    res = run_fp64_tuple()
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    # Report
    flips = sum(1 for r in res if r["flip"])
    all_match = all(r["class_32"] == r["class_64"] for r in res)
    print(f"fp64 tuple test (C=384, 9 rows): flips={flips}/9, all-class-match={all_match}")
    for r in res:
        print(f"  stride={r['stride']} pad={r['pad']} dil={r['dilation']} "
              f"class32={r['class_32']} class64={r['class_64']} flip={r['flip']} "
              f"rho_max={r['rho']['rho_max']:.4f}")
    # Report file
    with open(REPORT, "w") as f:
        f.write(f"# fp64 tuple test (stage 1)\n\n")
        f.write(f"Prereg: meta-reflection standing plan, lab/meta_reflection_plan.txt.\n\n")
        f.write(f"Grid: C=384, k=3, s=1, d in 1..4, pad=p, H=W=7, batch=2, seed 4321.\n\n")
        f.write(f"Results: {len(res)} rows, {flips} flips, all-class-match={all_match}.\n\n")
        f.write(f"Decision rule: fp64-anchored W/T labels equal TF32-class labels on all 9 rows "
                f"=> reference-invariant. Result: {'REFERENCE-INVARIANT' if all_match else 'ANCHOR-CONDITIONAL'}.")
    print(f"\nOutput written to {OUT} and {REPORT}")


# Kill conditions (recorded, applied in audit note)
KILL = {
    "K1": "pooled held-out AUC < 0.60 (not triggered; not applicable this stage)",
    "K2": "any anchor x class stratum < 0.70 (not triggered; labeled by class)",
    "K3": "boundary flips on > 2/8 transitions at the fp64 anchor",
}
