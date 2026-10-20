"""B5 stage-2: frozen-guard transfer to the normalization family (F3, zero-rederivation).

Pre-registration written 2026-10-20 ~10:55Z BEFORE any run; no threshold, gate,
orientation, or column below edited after seeing results. Implements plan block
B5 leg 2 / falsifier F3: "third transfer with zero rederivation"; FINALITY per
b4 v1: this is the lane's last measurement -- kill => B6 wrap with the negative
triple {B4v0 kill, B5 postmortem, mechanism}; pass => B6 wrap documents the
positive triple. Either way B6 next; no v2, no retune.

GUARD (as-is): results/guard_v1.json loaded, score machinery verbatim b4 v1 --
kept features [w_mean_abs, w_max, collapse_work_ratio], per-feature rank of the
instance value within frozen corpus-column++[value] (n=176), z=(r-(n+1)/2)/
sqrt((n^2-1)/12), orientation frozen {w_mean_abs:False,w_max:False,cwr:True},
score=mean, ACCEPT iff score >= t=-0.6134445630270532. No gate, no refit, no
threshold move, no feature recomputation. corpus sha256 re-checked vs artifact.

FEATURE REFERENTS (declared ex ante; the only family mapping in this program):
  fold W-role = the reassociation's multiplicative parameter => LN gamma.
    w_max=|g|.max; w_mean_abs=|g|.mean; w_rel_spread=g.std()/|g|.mean (recorded, not kept).
  fold (B,K,H) roles: B=batch rows; K=contracted/reduced extent; H=per-row output
    extent. LN default shape (16,64,256,256), norm over last 3 dims:
    B=16, K=C=64*256*256=4194304, H=C => collapse_work_ratio=B*C*C/(C*C+C)
    = 16*C/(C+1) ~= 16.0. CONSTANT on this family (like H/K/fan_in in train) --
    the declared degeneracy channel; its z is recorded per row, not repaired.
PREDICTION (pre-registered, tick-25 note): on task-default gamma the w_* columns
  degenerate (ones: w_rel_spread==0, w_max==w_mean_abs duplicated); the
  discriminative load falls on gamma magnitude; cwr z is extreme-low by
  construction (LN 16 << corpus 1024..2048) and drags scores down via
  orientation True. The test is whether the FROZEN basis separates LN pass/fail
  anyway; degeneracy observed != repair undertaken.

CORPUS (ex ante): L1/40 default shape; R = task Model(x) (fp32, native LN);
  T = var_mean-split fp32 + affine (the b5_stage1 candidate; sylve 'fast
  statistics' class). gamma0 in {task_default (the task's own affine pair),
  randn} x xdist in {randu (b4-as-is torch.rand), randn, randn+1000} x
  scale in 24 log steps 1e0..1e6 applied to gamma ONLY (beta, x untouched;
  b4 ordmag verbatim structure) x seeds {50,51}.
  = 2*3*24*2 = 288 rows. x fixed per (xdist,seed): torch.manual_seed
  (9000+seed*10+dist_idx); randn gamma0: cpu Generator seed 1234+idx (b5_stage1
  pattern). label per evaluator-exact summarize: d=|T-R|, limit=atol+rtol*|T|
  (candidate-scaled per the 2026-10-20 direction audit; b4 v1 metrics were
  identical under both directions, max ratio delta 0.59% -- recorded, choice
  non-decisional). ATOL=RTOL=1e-4.
STRATA = xdist arms (a-priori regimes; stage-1: LN failures concentrate at
  normal+1000, mbr<=17.2). A stratum with a single class => K2 not evaluable
  there, recorded as such (no silent pass).

METRICS / KILLS (mirror b4 v1 verbatim, mechanical):
  PRIMARY  pooled AUC >= 0.80.
  K2       any evaluable stratum AUC < 0.70.
  K3       pooled accepted fraction < 0.20 (precious-but-silent).
  KC       NOT kill-bearing here (declared): no LN paired-saving artifact exists
           to form N*; check-cost (gamma stats) timed 20 reps, descriptive only.
  Accepted-set precision, per-stratum accept/precision, per-cell AUC, per-row
  z_raw by kept feature (postmortem channel) recorded.
VERDICT mechanical: alive = PRIMARY and not K2 and not K3.

Provenance: spark GB10 sm_120, torch 2.14.1+cu130, TF32 off, matmul highest,
grad off. Results -> results/b5_stage2.{json,jsonl}. Correctness only.
"""
import json, math, time, hashlib, pathlib
import torch

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float3p_matmul_precision = None
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

def cand(x, g, b):
    var, mu = torch.var_mean(x, dim=tuple(range(1, x.ndim)), keepdim=True)  # NB (var, mean)
    return (x - mu) / torch.sqrt(var + EPS) * g + b

def summarize(a, b_):
    d = (a - b_).abs(); limit = ATOL + RTOL * a.abs()
    return {"pass": bool((d <= limit).all()),
            "max_budget_ratio": (d / limit).max().item()}

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
            R_task = model(x) if gi == 0 else torch.nn.functional.layer_norm(x, N, g_base, b_base, EPS)
            for si in range(24):
                scale = float(10 ** (6.0 * si / 23))
                g = g_base * scale
                Tn = cand(x, g, b_base)
                s_ = summarize(R_task, Tn)
                a = g.abs()
                rows.append(dict(arm=dname, gamma0=gname, seed=seed, scale=scale,
                                 label=s_["pass"], max_budget_ratio=s_["max_budget_ratio"],
                                 w_max=float(a.max()), w_mean_abs=float(a.mean()),
                                 w_rel_spread=float(g.std() / a.mean()),
                                 collapse_work_ratio=CWR, B=B, K=C, H=C, fan_in_sum=C))
            del x, R_task
        torch.cuda.empty_cache()
print("rows %d in %.1fs" % (len(rows), time.time() - t0), flush=True)

# ---------------- score as-is, mechanical verdict ----------------
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
           prereg="docstring written before run 2026-10-20 ~10:55Z",
           guard=dict(path="results/guard_v1.json", t=T, kept=KEPT, corpus_sha256=sha),
           family=dict(task="L1/40", shape=list(SHAPE), C=C, B=B, cwr=CWR,
                       task_gamma_w_max=float(g0_task.abs().max()),
                       task_gamma_w_mean=float(g0_task.abs().mean()),
                       task_gamma_rel_spread=float((g0_task.std() / g0_task.abs().mean()))),
           metrics=metrics, kills=kills, verdict=verdict, check_cost_s=cost_s,
           env=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                    tf32=False, atol=ATOL, rtol=RTOL, direction="candidate-scaled"))
(RES / "b5_stage2.json").write_text(json.dumps(out, indent=1, default=float))
with open(RES / "b5_stage2.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(json.dumps(dict(pooled_auc=pooled_auc, by_arm=by_arm, acc_frac=metrics["accepted_frac"],
                      acc_prec=prec, kills=kills, alive=verdict["alive"],
                      check_cost_s=cost_s), indent=1, default=str))
