#!/usr/bin/env python
"""s77_l40_realization.py — E1 of the exploitation plan: realization matching + margin accounting on the
strongest community control (sylve post 5's family), at the official KernelBench L1/40 shape.

PREREG (frozen at commit before the run; do not edit after).

Population: KernelBench rev 423217d level1/40_LayerNorm.py, the task's own get_init_inputs/get_inputs, seed 0,
one model instance, one input draw. Shape (16,64,256,256), normalization over (64,256,256) = 4,194,304 fp32 per
row; 16 rows. This is an OUT-OF-POPULATION test of my s73/s74/s75 instrument: that population was conv-heavy and
biased to 1e-4-exceeding tasks; this family is norm/reduction at the official default shape.

Arms:
  A   reference = the task's Model (torch F.layer_norm), default flags (cudnn tf32 on, matmul tf32 off).
  X   fp64 anchor = the same model widened to float64 (32 MB per tensor; whole-tensor materialization is legal
      here, no chunking needed).
  C1  torch.compile(model, dynamic=False), fp32, default flags.
  C2  statistics-order variant: mean/variance accumulated SERIALLY across index-order chunks in fp32
      (chunk 65,536 elements, sequential accumulation across chunks), then normalize in fp32.
  C3  fast-statistics variant: torch.var_mean over the normalized dims in fp32, then normalize in fp32.
Declared limit: C2/C3 are torch-composed statistics variants, NOT sylve's fused kernel (their implementations may
be CuPy/version-specific). So this tests "a reduction in the documented order", not "their kernel". The
normalization epilogue in C2/C3 is a torch composition, so bit identity can fail for epilogue reasons unrelated to
statistics order; that is why bit identity is reported per arm rather than as a single school verdict.

Quantities per arm: bit-identical to A (max abs diff == 0), exceedance vs A at the evaluator predicate
(allclose(A, cand, atol=rtol=1e-4), candidate-scaled: thr = atol + rtol*|cand|) and at 1e-2; exceedance vs X at
the same predicates; failing-element decomposition (reference-induced = |A-X| > thr, else candidate-induced) for
each arm at both tolerances. Reference quantities: A_vs_X exceedance at 1e-4 and 1e-2, ref deviation as fraction
of budget (median/q90/q99/max, frac over budget), ref_max_abs.

PREDECLARED BRANCHES (choose exactly one after the run, by the first matching clause):
 (i)   C1 or C2 bit-identical to A AND reference over-budget fraction at 1e-4 is < 1% => rule works, family
       reference-clean.
 (ii)  C1 or C2 bit-identical to A AND reference over-budget fraction at 1e-4 >= 1% => rule works, passes on this
       family are reference-conferred.
 (iii) no arm bit-identical AND candidate-induced share of C2/C3 failing elements < 50% => statistics order is
       not the binding constraint on this family; rule unnecessary here.
 (iv)  no arm bit-identical AND candidate-induced share >= 50% => my composition of the documented order is wrong;
       the tree-equality claim is WITHDRAWN and the deliverable shrinks to the decomposition table.
KILL CLAUSE: branch (iv) fires => withdraw tree-equality, keep the decomposition, pivot to the reference-share
column (E2) and the benchmark-wide audit scoping doc.
Falsifier written in advance: if any single arm's bit-identity count is 0/16 while its 1e-4 exceedance count is
also 0, the arm is "numerically equal but not identical" and that is recorded as a distinct outcome, not folded
into either branch.
"""
import importlib.util, json, os, signal, time
import torch

KB = os.path.expanduser("~/kernelbench/KernelBench")
OUT = os.path.expanduser("~/kb-lab/results/s77_l40_realization.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
TASK = os.path.join(KB, "KernelBench", "level1", "40_LayerNorm.py")
TOL_BENCH = 1e-2
TOL_STRICT = 1e-4
CHUNK = 65536


def load_task(path):
    spec = importlib.util.spec_from_file_location("kb_l40", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ew(a, b, atol, rtol):
    """a,b same shape; returns count of |a-b| > atol + rtol*|b| and max abs diff."""
    d = (a.double() - b.double()).abs()
    thr = atol + rtol * b.double().abs()
    return {"count": int((d > thr).sum()), "total": int(d.numel()), "frac": float((d > thr).float().mean()),
            "maxabs": float(d.max())}


def decomp(cand, A, X, atol, rtol):
    """Failing elements of cand vs A split by whether the reference alone already exceeds the budget."""
    d = (cand.double() - A.double()).abs()
    thr = atol + rtol * cand.double().abs()
    fail = d > thr
    nfail = int(fail.sum())
    if nfail == 0:
        return {"n_fail": 0, "ref_induced": 0, "cand_induced": 0, "ref_share": None}
    dref = (A.double() - X.double()).abs()
    ref_over = dref > thr
    ref_share = float((fail & ref_over).sum()) / nfail
    return {"n_fail": nfail, "ref_induced": int((fail & ref_over).sum()),
            "cand_induced": int((fail & ~ref_over).sum()), "ref_share": ref_share}


def serial_stats(x):
    """fp32 mean/var accumulated serially across index-order chunks."""
    flat = x.reshape(x.shape[0], -1)
    n = flat.shape[1]
    s = torch.zeros(flat.shape[0], device=flat.device, dtype=torch.float32)
    s2 = torch.zeros_like(s)
    for i in range(0, n, CHUNK):
        c = flat[:, i:i + CHUNK]
        s = s + c.sum(dim=1)
        s2 = s2 + (c * c).sum(dim=1)
    mean = s / n
    var = s2 / n - mean * mean
    return mean, var.clamp_min(0)


def normalize(x, mean, var, eps, g, b):
    shape = [x.shape[0]] + [1] * (x.dim() - 1)
    xh = (x - mean.view(shape)) * torch.rsqrt(var.view(shape) + eps)
    return xh * g.view(1, *g.shape) + b.view(1, *b.shape)


def main():
    torch.cuda.init()
    dev = "cuda"
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    mod = load_task(TASK)
    init_args = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    inputs = mod.get_inputs()
    torch.manual_seed(0)
    model = mod.Model(*init_args).to(dev)
    model.eval()
    ln = model.ln
    eps = float(ln.eps)
    g = ln.weight.detach()
    b = ln.bias.detach()
    x = inputs[0].to(dev).float()
    t0 = time.time()
    with torch.no_grad():
        A = model(x)
        C1 = torch.compile(model, dynamic=False)(x)
        model.float()
        X = model.to(torch.float64)(x.double())
        model.float()
        mean_s, var_s = serial_stats(x)
        C2 = normalize(x, mean_s, var_s, eps, g, b)
        vm = torch.var_mean(x.reshape(x.shape[0], -1), dim=1, correction=0)
        C3 = normalize(x, vm[1], vm[0], eps, g, b)
    rec = {"task": "level1/40_LayerNorm", "shape": list(x.shape), "eps": eps, "wall_s": round(time.time() - t0, 2),
           "torch": torch.__version__, "arms": {}}
    for nm, cand in (("C1_compile", C1), ("C2_serial_stats", C2), ("C3_fast_stats", C3)):
        rec["arms"][nm] = {
            "bit_identical_to_A": bool(ew(cand, A, 0.0, 0.0)["maxabs"] == 0.0),
            "vs_A_1e-4": ew(cand, A, TOL_STRICT, TOL_STRICT),
            "vs_A_1e-2": ew(cand, A, TOL_BENCH, TOL_BENCH),
            "vs_X_1e-4": ew(cand, X, TOL_STRICT, TOL_STRICT),
            "decomp_1e-4": decomp(cand, A, X, TOL_STRICT, TOL_STRICT),
            "decomp_1e-2": decomp(cand, A, X, TOL_BENCH, TOL_BENCH),
        }
    ax = ew(A, X, TOL_STRICT, TOL_STRICT)
    axb = ew(A, X, TOL_BENCH, TOL_BENCH)
    rec["reference"] = {"A_vs_X_1e-4": ax, "A_vs_X_1e-2": axb, "ref_max_abs": float(X.abs().max()),
                        "over_budget_frac_1e-4": ax["frac"], "over_budget_frac_1e-2": axb["frac"]}
    # predeclared branch selection
    bit = [k for k, v in rec["arms"].items() if v["bit_identical_to_A"]]
    ref_dirty = ax["frac"] >= 0.01
    if bit and not ref_dirty:
        branch = "(i)"
    elif bit and ref_dirty:
        branch = "(ii)"
    else:
        shares = [v["decomp_1e-4"]["ref_share"] for v in rec["arms"].values() if v["decomp_1e-4"]["ref_share"] is not None]
        cand_share = 1.0 - (sum(shares) / len(shares)) if shares else 1.0
        branch = "(iv)" if cand_share >= 0.5 else "(iii)"
    rec["branch"] = branch
    rec["bit_identical_arms"] = bit
    json.dump(rec, open(OUT, "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("branch", "bit_identical_arms", "reference")}, indent=1))
    print(json.dumps(rec["arms"], indent=1))


if __name__ == "__main__":
    main()
