"""B5 stage-1: error-budget ownership decomposition, normalization family.

Pre-registration (written 2026-10-20 ~07:45Z BEFORE first run; no threshold below
edited after seeing results). Mirrors b3_stage1 stage-1 exactly, on the B5 family
per exploitation plan ("third transfer: broadcast/normalization-fold"). Implements
the #17/#20 conditionals: exposure re-entry AND ref-model-drop eligibility are
decided by transform-share on THIS family.

FAMILY (fixed ex ante): KernelBench L1/40 LayerNorm, default shape (16,64,256,256),
normalization over (64,256,256) (per sylve post 5 / task source 423217d).
  R = task Model(x)  (native fp32 LN, torch 2.14.1+cu130, GB10) -- the reference.
  T = split-statistics candidate: (mu, var) = torch.var_mean(x, dim, keepdim);
      T = (x-mu)/sqrt(var+eps)*gamma + beta, affine in fp32 -- the reassociation
      whose guard-adoption a B5 stage-2 would gate (sylve's 'fast statistics'
      class; known offset-failure structure on 2.5.1/3090, NOT re-derived here).
  E = full candidate path in fp64 (var_mean fp64 + fp64 affine): b3-parity
      definition. t=|T-E|, r=|R-E|, d=|T-R|, all elementwise fp64.
  E2 = fp64 native-LN path (secondary, descriptive): separates pure reference
      rounding; share_T2e2 recorded, NOT primary.

GRID (fixed ex ante): affine in {task-default, randn} x dist in
  {default get_inputs, normal, normal+1000, normal*100, const 1e4} x seeds {0,1,2}
  = 30 rows; per-row n = 16*64*256*256 = 67,108,864 elements.
  Offset/const dists included BECAUSE sylve post5 shows statistic-tree failures
  concentrate there (normal+1000: 6/6 fail at 1e-4) -- the regime the guard must
  cover; uniform alone would flatter the transform (B2 'ordinary inputs' lesson).

PRIMARY (decides both #17/#20 branches; b3 parity):
  share_T2 = sqrt(sum t^2) / (sqrt(sum t^2) + sqrt(sum r^2)), pooled over ALL
  30 rows (numerator/denominator summed across rows before the ratio).
  share_T2 >= 0.80 => transform-owned on family #2: grammar MAY drop the
    version-pinned reference-error model for this family (record, not ship) AND
    #20's exposure re-entry condition is MET (b5_stage1.json is the receipt).
  share_T2 <  0.80 => reference-error model load-bearing on a 2nd family
    (cross-family generality of b3's 0.383); exposure re-entry NOT met.
SECONDARY (descriptive, no branch): per-dist and per-affine shares; zero-total
  counts; path agreement |E.fp32path - E2.fp32path| style diagnostics as in b3
  (E vs E2 max abs); fraction of d owned by t among failing elements at
  atol+rtol*cand (evaluator-exact direction, per 2026-07:53 direction audit).
Known confounds (named ex ante): single task/shape/dtype; 'transform side'
includes the statistic-tree difference by construction (E = candidate-fp64);
fp64 var_mean on CUDA is its own implementation, not the fp32 tree lifted.

Provenance: spark GB10 sm_120, torch 2.14.1+cu130, TF32 off, matmul highest,
grad off. Results -> results/b5_stage1.json. Correctness only, no timing.
"""
import json, math, pathlib, time
import torch

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
ATOL = RTOL = 1e-4

ROOT = pathlib.Path("/home/ckorhonen/kb-lab")
RES = ROOT / "results"; RES.mkdir(exist_ok=True)

def load_task(n):
    import importlib.util
    p = next(pathlib.Path("/home/ckorhonen/kernelbench/KernelBench/KernelBench/level1").glob(f"{n}_*.py"))
    spec = importlib.util.spec_from_file_location(f"task{n}", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

m40 = load_task(40)
model = m40.Model(*m40.get_init_inputs()).cuda().eval()
lnmods = [m for m in model.modules() if isinstance(m, torch.nn.LayerNorm)]
assert len(lnmods) == 1, len(lnmods)
ln = lnmods[0]
N = tuple(ln.normalized_shape)
EPS = float(ln.eps)
torch.manual_seed(123)  # fixed probe for the shape only
SHAPE = tuple(m40.get_inputs()[0].shape)
print("L1/40 normalized_shape:", N, "eps:", EPS, "input shape:", SHAPE)
DIST_IDX = {"default": 0, "normal": 1, "norm_p1000": 2, "norm_x100": 3, "const1e4": 4}

def cand_fp32(x, g, b):
    var, mu = torch.var_mean(x, dim=tuple(range(1, x.ndim)), keepdim=True)  # NB: (var, mean)!
    return (x - mu) / torch.sqrt(var + EPS) * g + b

def cand_fp64(x, g, b):
    xd = x.double()
    var, mu = torch.var_mean(xd, dim=tuple(range(1, x.ndim)), keepdim=True)  # NB: (var, mean)!
    return (xd - mu) / torch.sqrt(var + EPS) * g.double() + b.double()

def ref_fp64(x, g, b):
    xd = x.double()
    mu = xd.mean(dim=tuple(range(1, x.ndim)), keepdim=True)
    var = ((xd - mu) ** 2).mean(dim=tuple(range(1, x.ndim)), keepdim=True)
    return (xd - mu) / torch.sqrt(var + EPS) * g.double() + b.double()

def rowtag(aff, dist, seed): return f"{aff}|{dist}|s{seed}"

rows = []; sums = {"t2": 0.0, "r2": 0.0}
for aff in ("task_default", "randn"):
    for seed in (0, 1, 2):
        for dist in ("default", "normal", "norm_p1000", "norm_x100", "const1e4"):
            torch.manual_seed(seed * 10 + DIST_IDX[dist])  # stable per (seed,dist)
            if dist == "default":
                x = m40.get_inputs()[0].cuda().float()
            elif dist == "normal":
                x = torch.randn(*SHAPE, device="cuda")
            elif dist == "norm_p1000":
                x = torch.randn(*SHAPE, device="cuda") + 1000.0
            elif dist == "norm_x100":
                x = torch.randn(*SHAPE, device="cuda") * 100.0
            else:
                x = torch.full(SHAPE, 1e4, device="cuda", dtype=torch.float32)
            g = ln.weight.detach().clone().float()
            b = ln.bias.detach().clone().float()
            if aff == "randn":
                gg = torch.Generator().manual_seed(seed + 77)
                g = torch.randn(ln.weight.shape, generator=gg, device="cpu").float().cuda()
                b = torch.randn(ln.bias.shape, generator=gg, device="cpu").float().cuda()
            R = model(x) if aff == "task_default" else torch.nn.functional.layer_norm(x, N, g, b, EPS)
            T = cand_fp32(x, g, b)
            E = cand_fp64(x, g, b)
            E2 = ref_fp64(x, g, b)
            Td, Rd = T.double(), R.double()
            t = (Td - E).abs(); r = (Rd - E).abs(); d = (Td - Rd).abs()
            lim = ATOL + RTOL * T.abs()  # evaluator-exact direction (candidate-scaled)
            fail = d > lim
            st2 = float((t ** 2).sum()); sr2 = float((r ** 2).sum())
            sums["t2"] += st2; sums["r2"] += sr2
            # crash-fix (degenerate all-constant row: t=r=0 exactly => 0/0); no threshold touched
            share_row = (math.sqrt(st2) / (math.sqrt(st2) + math.sqrt(sr2))) if (st2 + sr2) > 0 else 0.5
            tot = t + r; nz = tot > 0
            n_fail = int(fail.sum())
            row = dict(tag=rowtag(aff, dist, seed), n=int(t.numel()),
                share_T2=share_row,
                median_elem_share=float((t[nz] / tot[nz]).median()) if bool(nz.any()) else float("nan"),
                zero_total=int((~nz).sum()), n_fail=n_fail,
                path_agreement_E_E2=float((E - E2).abs().max()),
                d_max=float(d.max()), budget_ratio_max=float((d / lim).max()),
                frac_towner_on_fail=(float((t[fail] > lim[fail]).float().mean()) if n_fail else None))
            rows.append(row)
            print(row["tag"], "share=%.4f" % row["share_T2"], "n_fail=%d" % n_fail,
                  "mbr=%.3g" % row["budget_ratio_max"], flush=True)
            del x, R, T, E, E2, Td, Rd, t, r, d, lim, fail
        torch.cuda.empty_cache()
share_pooled = math.sqrt(sums["t2"]) / (math.sqrt(sums["t2"]) + math.sqrt(sums["r2"]))
verdict = ("transform_owned_may_drop_ref_model_AND_exposure_reentry_met" if share_pooled >= 0.80
           else "ref_model_load_bearing_exposure_reentry_not_met")
out = dict(when=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           prereg="docstring written before run 2026-10-20 ~07:45Z",
           family="L1/40 LayerNorm default shape; T=var_mean split-statistics fp32",
           primary_share_T2_pooled=share_pooled, verdict=verdict, per_row=rows,
           env=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                    tf32=False, atol=ATOL, rtol=RTOL, direction="candidate-scaled (evaluator-exact)"))
print("POOLED share_T2 =", round(share_pooled, 4), "| verdict:", verdict)
(RES / "b5_stage1.json").write_text(json.dumps(out, indent=1, default=float))
print("saved", RES / "b5_stage1.json")
