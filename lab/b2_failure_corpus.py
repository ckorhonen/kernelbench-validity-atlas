"""B2 failure corpus — guarded-reassociation lane (exploitation_plan.md 5L69uAkllDQYQObB).

Pre-registration (written 2026-10-19 ~14:45Z BEFORE first run; verdict rules below are
final; no threshold edited after seeing results).

Arms
  A. L2/14 fold replicate, seeds 0-4, official init/inputs (task source read on spark,
     rev 423217d): ref = ((x@W.T)/2).sum(1,keepdim)*sf  vs
     fold32 = (x@W.sum(0)).unsqueeze(1)*(sf/2)   [sylve expression, verbatim algebra]
     fold64 = (x@W.sum(0).double()).unsqueeze(1)*(sf/2) cast float
  B. seed-43 constructed cancellation (L2/51), sylve construction + DRAW REPLICATION:
     scales {1,1e2,1e3,1e4,1e5,1e6} x 20 seeded row draws (seed 4300+si*100+i):
     row=rand(1,K), row[0,0]=0, x=expand; W=base*scale; subtract=gemm(x)[0]; ref=m(x).
     Arms: fold32 torch_factored, fold64cast. Diagnostic (non-gating): gemm-subtract
     residual, ref_first/cand_first.
  C. ordinary-distribution magnification (L2/51): 25 log-spaced scales 1e0..1e6 x
     seeds {44,45}, official input per seed, W=base*scale, fold32/fold64 vs ref.

Features per row (B3 candidates, named now): family, task, seed, scale, B,K,H,
collapse_work_ratio = B*K*H/(H*K+K) [fold work / ref GEMM work, +],
w_max,w_mean_abs,w_rel_spread=std/|mean| over W, phi_frac_ref_lt_atol on final ref,
gelu_in_abs_max (fold scalar pre-activation where defined), dtype, tf32, tol, method.

VERDICT RULES (pre-registered)
  A: L2/14 family fails on GB10/torch2.14.1 iff >=3/5 seeds have fold32 bad>0.
     fold64 bad-count recorded (reference-rounding lesson check).
  B: construction-admitted iff >=12/20 draws fold32 max_budget_ratio>1 at scale=1e6.
     Per-draw transition scale = largest scale with pass. Residual diagnostic recorded.
  C: boundary-signal-present iff >=1/2 seeds show a pass->fail transition across the
     sweep (fold32). Descriptive; NO monotone threshold claimed (per source).
  Corpus usable-for-B3 iff (>=10 failing AND >=10 passing fold32 rows overall).
     Else: corpus-empty negative recorded, guard has no failure corpus on GB10 today.

Provenance: spark GB10 sm_120, torch 2.14.1+cu130 (source used 2.5.1+cu121/3090 --
version-pinned reference-error model is the object of study). TF32 off, grad off.
FP32 strict; correctness only, no timing (check-cost timing is B4's business).
"""
import json, time, pathlib
import torch

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ATOL = RTOL = 1e-4

ROOT = pathlib.Path("/home/ckorhonen/kb-lab")
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)

def summarize(a, b):
    d = (a - b).abs(); limit = ATOL + RTOL * a.abs()
    ratio = d / limit
    q = torch.quantile(ratio.flatten().float(), torch.tensor([0.5, 0.9, 0.99], device=a.device))
    return {"pass": bool((d <= limit).all()), "max_abs": d.max().item(),
            "bad": int((d > limit).sum()), "max_budget_ratio": ratio.max().item(),
            "ratio_q50": q[0].item(), "ratio_q90": q[1].item(), "ratio_q99": q[2].item()}

def load_task(n):
    import importlib.util
    p = next(pathlib.Path("/home/ckorhonen/kernelbench/KernelBench/KernelBench/level2").glob(f"{n}_*.py"))
    spec = importlib.util.spec_from_file_location(f"task{n}", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def feats(**kw):
    return kw

rows = []
def row(**kw):
    rows.append(kw)

def wstats(w):
    return dict(w_max=w.abs().max().item(), w_mean_abs=w.abs().mean().item(),
                w_rel_spread=(w.std() / w.abs().mean()).item())

# ---------------- A. L2/14 replicate ----------------
m14 = load_task(14)
armA = []
for seed in range(5):
    torch.manual_seed(seed)
    model = m14.Model(*m14.get_init_inputs()).cuda().eval()
    x = m14.get_inputs()[0].cuda()
    W = model.weight; sf = model.scaling_factor
    ref = model(x)
    ref64 = ((x.double() @ W.double().T) / 2).sum(1, keepdim=True) * sf
    f32 = (x @ W.sum(0)).unsqueeze(1) * (sf / 2)
    f64 = (x.double() @ W.sum(0).double()).unsqueeze(1) * (sf / 2)
    r = dict(seed=seed, ref_rounding_vs64=(ref.double() - ref64).abs().max().item(),
             **{f"fold32_{k}": v for k, v in summarize(ref, f32).items()},
             **{f"fold64_{k}": v for k, v in summarize(ref, f64).items()},
             **wstats(W))
    armA.append(r); row(family="l2_14_fold", method="fold32", label=r["fold32_pass"],
                        B=1024, K=8192, H=8192, seed=seed, scale=1.0,
                        collapse_work_ratio=1024*8192*8192/(8192*8192+8192),
                        phi=None, fan_in_sum=8192, dtype="fp32", tf32=False,
                        tol="1e-4", **{k: v for k, v in r.items() if k != "seed"})
    row(family="l2_14_fold", method="fold64cast", label=r["fold64_pass"],
        B=1024, K=8192, H=8192, seed=seed, scale=1.0, fan_in_sum=8192,
        dtype="fp32/fold64", tf32=False, tol="1e-4",
        **{k: v for k, v in r.items() if k != "seed"})
    del model, x, ref, ref64, f32, f64
torch.cuda.empty_cache()

# ---------------- B. seed-43 constructed cancellation, draw-replicated ----------------
m51 = load_task(51)
torch.manual_seed(43)
m = m51.Model(*m51.get_init_inputs()).cuda().eval()
base_w = m.gemm.weight.detach().clone(); base_b = m.gemm.bias.detach().clone()
K = m51.get_inputs()[0].shape[1]
scales = [1.0, 100.0, 1e3, 1e4, 1e5, 1e6]
armB = []
for si, scale in enumerate(scales):
    m.gemm.weight.copy_(base_w * scale); m.gemm.bias.copy_(base_b)
    fails32 = 0; trans = []
    for i in range(20):
        torch.manual_seed(4300 + si * 100 + i)
        r_ = torch.rand(1, K, device="cuda"); r_[0, 0] = 0
        x = r_.expand(2048, K).contiguous()
        m.subtract.copy_(m.gemm(x)[0])
        resid = (m.gemm(x) - m.subtract).abs().max().item()
        ref = m(x)
        v = m.gemm.weight.mean(0); c = m.gemm.bias.mean() - m.subtract.mean()
        pre = (x @ v + c)
        f32 = torch.nn.functional.gelu(pre.unsqueeze(1)) + x
        f64 = torch.nn.functional.gelu((x.double() @ m.gemm.weight.double().mean(0)
              + (m.gemm.bias.double().mean() - m.subtract.double().mean())).float().unsqueeze(1)) + x
        s32, s64 = summarize(ref, f32), summarize(ref, f64)
        if not s32["pass"]: fails32 += 1
        armB.append(dict(scale=scale, draw=i, gemm_sub_resid=resid,
                         ref_first=ref[0, 0].item(), cand_first=f32[0, 0].item(),
                         pre_absmax=pre.abs().max().item(),
                         **{"fold32_" + k: v for k, v in s32.items()},
                         **{"fold64_" + k: v for k, v in s64.items()}))
        row(family="l2_51_cancel", method="fold32", label=s32["pass"], B=2048, K=8192,
            H=8192, seed=43, scale=scale, fan_in_sum=8192,
            collapse_work_ratio=2048*8192*8192/(8192*8192+8192), draw=i,
            phi=float((ref.abs() < ATOL).float().mean()),
            gelu_in_abs_max=pre.abs().max().item(), dtype="fp32", tf32=False, tol="1e-4",
            **wstats(m.gemm.weight), **{k: v for k, v in s32.items()})
    armB.append(dict(scale=scale, fails32_of_20=fails32))
    del x
torch.cuda.empty_cache()

# ---------------- C. ordinary-distribution magnification ----------------
armC = []
for seed in (44, 45):
    torch.manual_seed(seed)
    m = m51.Model(*m51.get_init_inputs()).cuda().eval()
    x = m51.get_inputs()[0].cuda()
    base_w = m.gemm.weight.detach().clone(); base_b = m.gemm.bias.detach().clone()
    seq = []
    for si in range(25):
        scale = float(10 ** (6.0 * si / 24))
        m.gemm.weight.copy_(base_w * scale); m.gemm.bias.copy_(base_b)
        ref = m(x)
        v = m.gemm.weight.mean(0); c = m.gemm.bias.mean() - m.subtract.mean()
        pre = x @ v + c
        f32 = torch.nn.functional.gelu(pre.unsqueeze(1)) + x
        f64 = torch.nn.functional.gelu((x.double() @ m.gemm.weight.double().mean(0)
              + (m.gemm.bias.double().mean() - m.subtract.double().mean())).float().unsqueeze(1)) + x
        s32, s64 = summarize(ref, f32), summarize(ref, f64)
        seq.append(dict(scale=scale, **s32, fold64_pass=s64["pass"], fold64_mbr=s64["max_budget_ratio"]))
        row(family="l2_51_ordinary_mag", method="fold32", label=s32["pass"], B=2048,
            K=8192, H=8192, seed=seed, scale=scale, fan_in_sum=8192,
            collapse_work_ratio=2048*8192*8192/(8192*8192+8192),
            phi=float((ref.abs() < ATOL).float().mean()),
            gelu_in_abs_max=pre.abs().max().item(), dtype="fp32", tf32=False, tol="1e-4",
            **wstats(m.gemm.weight), **{k: v for k, v in s32.items()})
    armC.append(dict(seed=seed, sweep=seq))
    del m, x
    torch.cuda.empty_cache()

# ---------------- verdicts (pre-registered rules, applied mechanically) ----------------
failsA = sum(1 for r in armA if r["fold32_bad"] > 0)
failsA64 = sum(1 for r in armA if r["fold64_bad"] > 0)
b1e6 = [r for r in armB if r.get("draw") is not None and r["scale"] == 1e6]
admittedB = sum(1 for r in b1e6 if r["fold32_max_budget_ratio"] > 1)
transB = []
for si, scale in enumerate(scales):
    passmax = max((r["draw"] for r in armB
                   if r.get("draw") is not None and r["scale"] == scale and r["fold32_pass"]),
                  default=-1)
    transB.append({"scale": scale, "passing_draws": sum(1 for r in armB if r.get("draw") is not None and r["scale"] == scale and r["fold32_pass"])})
transC = []
for a in armC:
    s = a["sweep"]; last = None; t = None
    for pt in s:
        if last and last["pass"] and not pt["pass"] and t is None: t = pt["scale"]
        last = pt
    transC.append({"seed": a["seed"], "first_fail_scale": t,
                   "all_pass": all(p["pass"] for p in s)})
n_fail_rows = sum(1 for r in rows if r["method"] == "fold32" and not r["label"])
n_pass_rows = sum(1 for r in rows if r["method"] == "fold32" and r["label"])
verdict = {
    "A_l2_14_fails_gb10": failsA >= 3, "A_failing_seeds": failsA, "A_fold64_failing_seeds": failsA64,
    "B_admitted_1e6": admittedB, "B_admitted_rule_ge12": admittedB >= 12,
    "C_boundary_signal": any(t["first_fail_scale"] is not None for t in transC), "C_detail": transC,
    "corpus_usable_for_B3": (n_fail_rows >= 10 and n_pass_rows >= 10),
    "n_fold32_rows": len([r for r in rows if r['method'] == 'fold32']),
    "n_fail_rows": n_fail_rows, "n_pass_rows": n_pass_rows,
}
out = {"date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "env": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
               "cuda": torch.version.cuda, "tf32": False, "tol": [ATOL, RTOL],
               "commit": "423217d (task sources read on spark)"},
       "armA": armA, "armB_summary": [r for r in armB if "fails32_of_20" in r],
       "armB_rows": [r for r in armB if "draw" in r], "armC": armC,
       "verdict": verdict}
(RESULTS / "b2_failure_corpus.json").write_text(json.dumps(out, indent=2))
with open(RESULTS / "b2_failure_corpus.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(json.dumps(verdict, indent=2))
print("A per-seed:", [(r['seed'], r['fold32_bad'], r['fold64_bad']) for r in armA])
print("B fails-of-20 by scale:", [(r['scale'], r['fails32_of_20']) for r in armB if 'fails32_of_20' in r])
