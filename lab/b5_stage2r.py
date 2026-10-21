"""B5 stage-2 REPAIR (R1) -- frozen-guard transfer rerun with corrected references.

Repairs per rejection_triage_R5 R1 (paper R5c3ENyOqeuCfg0t rejected 2026-10-21T04:47:25Z).
The published b5_stage2 compared a scaled candidate against an UNscaled reference
(b5_stage2.py:155 g_base vs :157-158 scaled g => T-R=(s-1)*z*gamma0; 276/288 rows
non-unit-scale) -- the calibration-transfer result as published is invalidated. This
rerun changes EXACTLY the flagged constructions and keeps the frozen guard machinery
verbatim; no threshold, gate, orientation, or feature recomputation.

REPAIRS (declared before the run; commit-dated):
  R1a reference: F.layer_norm(x, N, g, b_base, EPS) with the SAME scaled (g, b) as the
      candidate at EVERY point. Task-model provenance preserved by a one-time check
      model(x) == F.layer_norm(x, N, g0_task, b0_task) at scale 1 (max abs diff recorded).
  R1b variance ddof reconciliation: candidate uses population variance
      (correction=0) == nn.LayerNorm semantics. The deployed stage-1 form's correction=1
      differs by the factor N/(N-1), N=C=4194304 => relative 2.384e-7; one-time empirical
      max rel diff recorded (expected << tolerance; choice non-decisional, recorded).
  R1c tolerance direction evaluator-exact: KernelBench eval.py:804
      torch.allclose(output_reference, output_candidate, atol, rtol) =>
      |ref-cand| <= atol + rtol*|cand| (candidate-magnitude). The published run used
      reference-magnitude; b4 v1 was direction-invariant (max ratio delta 0.59% recorded
      there); here both the reference scaling and direction change, so the evaluator-exact
      form is used and the reference-magnitude ratio is also recorded per row.

GUARD (verbatim b4 v1 / b5_stage2): results/guard_v1.json; kept features
[w_mean_abs, w_max, collapse_work_ratio]; rank within frozen corpus-column++[value]
(n=176), z=(r-(n+1)/2)/sqrt((n^2-1)/12); orientation {w_mean_abs:F,w_max:F,cwr:T};
score=mean; ACCEPT iff score >= t=-0.6134445630270532. corpus sha256 re-checked.

FAMILY (verbatim b5_stage1/2): L1/40 default shape; gamma0 {task_default, randn} x
xdist {randu, randn, randn+1000} x scale 24 log steps 1e0..1e6 on gamma ONLY x
seeds {50,51} = 288 rows; x fixed per (xdist,seed) seed 9000+seed*10+dist_idx; randn
gamma0 cpu Generator 1234. cwr = B*C*C/(C*C+C) constant on family (declared degeneracy
channel; z recorded per row). ATOL=RTOL=1e-4.

METRICS / KILLS / VERDICT: verbatim b5_stage2 (PRIMARY pooled AUC>=0.80; K2 stratum
AUC<0.70; K3 accepted frac<0.20; KC descriptive). Per-row z_raw by kept feature recorded
(postmortem channel) -- the controlled score-component attribution runs OFFLINE on the
jsonl (cwr z is a constant offset on this family; no refit, no threshold move).

Provenance: spark GB10 sm_120, torch 2.14.1+cu130, TF32 off, matmul highest, grad off.
Results -> results/b5_stage2r.{json,jsonl} (published b5_stage2.* left untouched).
"""
import json, time, hashlib, pathlib
import torch

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
ATOL = RTOL = 1e-4

ROOT = pathlib.Path("/home/ckorhonen/kb-lab")
RES = ROOT / "results"; RES.mkdir(exist_ok=True)

# ---------------- frozen guard (load; re-verify corpus sha) ----------------
FZ = json.loads((RES / "guard_v1.json").read_text())
KEPT = FZ["kept_features"]; ORIENT = FZ["orientation"]; T = FZ["threshold"]
COLS = {k: FZ["corpus_columns"][k] for k in KEPT}
sha = hashlib.sha256((RES / "b2_failure_corpus.jsonl").read_bytes()).hexdigest()
assert sha == FZ["corpus_sha256"], "corpus drift vs frozen guard"
assert FZ["env"]["torch"].startswith("2.14.1") or "2.14" in FZ["env"]["torch"], FZ["env"]

def rankdata(v):
    idx = sorted(range(len(v)), key=lambda i: v[i]); out = [0.0] * len(v); i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]: j += 1
        m = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1): out[idx[k]] = m
        i = j + 1
    return out

def auc(score, lab_):
    r = rankdata(score); n1 = sum(lab_); n0 = len(lab_) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    s1 = sum(rk for rk, l in zip(r, lab_) if l)
    return (s1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)

def deployed_scores(eval_rows, feats):
    n = len(next(iter(COLS.values()))) + 1
    sd = ((n ** 2 - 1) / 12.0) ** 0.5; mu = (n + 1) / 2
    out = []
    for r in eval_rows:
        zs = []
        for f in feats:
            rk = rankdata(COLS[f] + [r[f]])[-1]
            z = (rk - mu) / sd
            r.setdefault("z_raw", {})[f] = z
            if not ORIENT[f]: z = -z
            zs.append(z)
        out.append(sum(zs) / len(zs))
    return out

# ---------------- family (b5_stage1 verbatim) ----------------
import importlib.util
p = next(pathlib.Path("/home/ckorhonen/kernelbench/KernelBench/KernelBench/level1").glob("40_*.py"))
spec = importlib.util.spec_from_file_location("task40", p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
model = mod.Model(*mod.get_init_inputs()).cuda().eval()
ln = [m for m in model.modules() if isinstance(m, torch.nn.LayerNorm)][0]
N = tuple(ln.normalized_shape); EPS = float(ln.eps)
torch.manual_seed(123)
SHAPE = tuple(mod.get_inputs()[0].shape)
C = 1
for d in N: C *= d
B = SHAPE[0]
CWR = B * C * C / (C * C + C)  # declared mapping; constant on this family
g0_task = ln.weight.detach().clone().float()
b0_task = ln.bias.detach().clone().float()
gg = torch.Generator().manual_seed(1234)
g0_randn = torch.randn(ln.weight.shape, generator=gg).float().cuda()
b0_randn = torch.randn(ln.bias.shape, generator=gg).float().cuda()
print("shape", SHAPE, "C", C, "B", B, "cwr", CWR,
      "| task gamma w_max/mean %.6g/%.6g" % (g0_task.abs().max(), g0_task.abs().mean()), flush=True)

# ---- R1a provenance check: task module == functional layer_norm at scale 1 ----
torch.manual_seed(9000 + 50 * 10 + 1)
x_chk = torch.randn(SHAPE, device="cuda")
m_chk = model(x_chk)
f_chk = torch.nn.functional.layer_norm(x_chk, N, g0_task, b0_task, EPS)
MODEL_CHK = float((m_chk - f_chk).abs().max())
del m_chk, f_chk, x_chk
# ---- R1b ddof reconciliation: one-time empirical bound, population vs correction=1 ----
torch.manual_seed(1234)
x_dd = torch.randn(SHAPE, device="cuda")
v0, _ = torch.var_mean(x_dd, dim=tuple(range(1, x_dd.ndim)), keepdim=True, correction=0)
v1, _ = torch.var_mean(x_dd, dim=tuple(range(1, x_dd.ndim)), keepdim=True, correction=1)
DDOF_MAXREL = float(((v1 - v0).abs() / v0).max())
del v0, v1, x_dd
print("model-vs-functional maxabs %.3g | ddof maxrel %.3g" % (MODEL_CHK, DDOF_MAXREL), flush=True)

def cand(x, g, b):
    var, mu = torch.var_mean(x, dim=tuple(range(1, x.ndim)), keepdim=True, correction=0)  # R1b population
    return (x - mu) / torch.sqrt(var + EPS) * g + b

def summarize(ref, cand_):
    """Evaluator-exact (R1c): allclose(ref, cand) => limit = atol + rtol*|cand|."""
    d = (ref - cand_).abs(); limit = ATOL + RTOL * cand_.abs()
    ratio = d / limit
    d_ref = (ref - cand_).abs(); limit_ref = ATOL + RTOL * ref.abs()
    return {"pass": bool((d <= limit).all()),
            "max_budget_ratio": ratio.max().item(),
            "max_budget_ratio_refscaled": (d_ref / limit_ref).max().item()}

XD = {"randu": 0, "randn": 1, "randn_p1000": 2}
rows = []
t0 = time.time()
for gi, gname in enumerate(("task_default", "randn")):
    g_base, b_base = (g0_task, b0_task) if gi == 0 else (g0_randn, b0_randn)
    for seed in (50, 51):
        for dname, didx in XD.items():
            torch.manual_seed(9000 + seed * 10 + didx)
            if dname == "randu": x = torch.rand(SHAPE, device="cuda")
            elif dname == "randn": x = torch.randn(SHAPE, device="cuda")
            else: x = torch.randn(SHAPE, device="cuda") + 1000.0
            for si in range(24):
                scale = float(10 ** (6.0 * si / 23))
                g = g_base * scale
                R_ref = torch.nn.functional.layer_norm(x, N, g, b_base, EPS)  # R1a same scaled (g,b)
                Tn = cand(x, g, b_base)
                s_ = summarize(R_ref, Tn)
                a = g.abs()
                rows.append(dict(arm=dname, gamma0=gname, seed=seed, scale=scale,
                                 label=s_["pass"], max_budget_ratio=s_["max_budget_ratio"],
                                 max_budget_ratio_refscaled=s_["max_budget_ratio_refscaled"],
                                 w_max=float(a.max()), w_mean_abs=float(a.mean()),
                                 w_rel_spread=float(g.std() / a.mean()),
                                 collapse_work_ratio=CWR, B=B, K=C, H=C, fan_in_sum=C))
            del x
        torch.cuda.empty_cache()
print("rows %d in %.1fs" % (len(rows), time.time() - t0), flush=True)

# ---------------- score as-is, mechanical verdict (verbatim) ----------------
hs = deployed_scores(rows, KEPT)
for r, s in zip(rows, hs):
    r["score"] = s; r["pred_pass"] = bool(s >= T)

pooled_auc = auc([r["score"] for r in rows], [r["label"] for r in rows])
by_arm = {a: auc([r["score"] for r in rows if r["arm"] == a],
                 [r["label"] for r in rows if r["arm"] == a]) for a in XD}
single_class_arm = [a for a, v in by_arm.items() if v != v]
eval_arms = {a: v for a, v in by_arm.items() if v == v}
acc = [r for r in rows if r["pred_pass"]]
prec = (sum(1 for r in acc if r["label"]) / len(acc)) if acc else None
metrics = dict(n=len(rows), n_fail=sum(1 for r in rows if not r["label"]),
               n_pass=sum(1 for r in rows if r["label"]),
               pooled_auc=pooled_auc, auc_by_arm=by_arm,
               single_class_arms=single_class_arm,
               collapse_flags={a: v for a, v in eval_arms.items() if v < 0.70},
               accepted_n=len(acc), accepted_precision=prec,
               accepted_frac=len(acc) / len(rows),
               accepted_frac_by_arm={a: (sum(1 for r in acc if r["arm"] == a) /
                                          max(1, sum(1 for r in rows if r["arm"] == a))) for a in XD})
kills = dict(K1_primary=bool(pooled_auc < 0.80),
             K2_collapse=bool(metrics["collapse_flags"]),
             K3_silent=bool(metrics["accepted_frac"] < 0.20))
# descriptive check cost (gamma stats; KC not kill-bearing, declared)
torch.manual_seed(999); gcost = torch.randn(*ln.weight.shape, device="cuda")
for _ in range(3): (gcost.abs().mean()).item()
torch.cuda.synchronize(); ts = []
for _ in range(20):
    t1 = time.perf_counter()
    a = gcost.abs(); s = torch.stack((a.max(), a.mean(), gcost.std())).cpu()
    _ = float(s[0]) + float(s[1]) + float(s[2])
    ts.append(time.perf_counter() - t1)
ts.sort(); cost_s = ts[len(ts) // 2]
verdict = dict(alive=(not kills["K1_primary"]) and (not kills["K2_collapse"]) and (not kills["K3_silent"]),
               **kills)
out = dict(when=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           prereg=("R1 repair per rejection_triage_R5; repairs R1a/R1b/R1c declared in docstring "
                   "BEFORE run; guard machinery verbatim b4 v1 / b5_stage2"),
           repairs=dict(reference="same scaled (g,b) both sides (R1a)",
                        var="population correction=0 == LayerNorm; ddof maxrel recorded (R1b)",
                        tolerance="evaluator-exact allclose(ref,cand) => rtol*|cand| (R1b/R1c; KernelBench eval.py:804)",
                        model_check_maxabs=MODEL_CHK, ddof_maxrel=DDOF_MAXREL),
           guard=dict(path="results/guard_v1.json", t=T, kept=KEPT, corpus_sha256=sha),
           family=dict(task="L1/40", shape=list(SHAPE), C=C, B=B, cwr=CWR,
                       task_gamma_w_max=float(g0_task.abs().max()),
                       task_gamma_w_mean=float(g0_task.abs().mean()),
                       task_gamma_rel_spread=float((g0_task.std() / g0_task.abs().mean()))),
           metrics=metrics, kills=kills, verdict=verdict, check_cost_s=cost_s,
           env=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                    tf32=False, atol=ATOL, rtol=RTOL,
                    direction="evaluator-exact candidate-magnitude; refscaled ratio recorded per row"))
(RES / "b5_stage2r.json").write_text(json.dumps(out, indent=1, default=float))
with open(RES / "b5_stage2r.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(json.dumps(dict(pooled_auc=pooled_auc, by_arm=by_arm, acc_frac=metrics["accepted_frac"],
                      acc_prec=prec, kills=kills, alive=verdict["alive"],
                      check_cost_s=cost_s), indent=1, default=str))
