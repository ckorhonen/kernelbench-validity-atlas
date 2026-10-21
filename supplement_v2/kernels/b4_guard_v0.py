"""B4 guard v0 — fit -> freeze -> held-out fresh shapes (exploitation_plan.md B4).

Pre-registration written 2026-10-20 ~00:40Z, BEFORE the run; kills, thresholds and
grids below are final. Degraded tier per B3 leg 3: NO exposure feature; the guard
is the derived 8-coordinate joint rank score + empirical paired-error envelope;
reference term pinned torch 2.14.1 (stage-1: ref model load-bearing, share 0.383).

FIT (train only; results/b2_failure_corpus.jsonl method==fold32, 175 rows, labels
per B3): features DERIVED = [collapse_work_ratio, fan_in_sum, w_max, w_mean_abs,
w_rel_spread, B, K, H] (b3_leg23 verbatim; fan_in_sum==K duplication kept for
continuity). Per-feature orientation = pass-class rank-mean higher (b3
`orientation`, full corpus). DEPLOYED SCORE for instance x: rank x's value within
concat(corpus column, [x]) (b3 rankdata, n=176), z=(r-(n+1)/2)/sqrt((n^2-1)/12),
orientation-signed, averaged over the 8 features. A single deployed instance is
scored against the frozen corpus columns -- the rank basis travels with the guard.
Threshold t: on deployed-scores of the 175 train rows (near-circular by
construction, declared), PRECISION_TARGET=0.90: t=min{s: mean(pass | >=s)>=0.90};
if unreachable, t=smallest s achieving max precision; train precision@t and
accept fraction recorded. Envelope: train-accepted max_budget_ratio
{q50,q90,q99,max} -- empirical paired-error envelope, deployment metadata.

FREEZE: results/guard_v0.json {features, orientation, per-feature corpus columns,
t, precision@t, envelope, corpus sha256, env pin} written BEFORE the held-out
eval; evaluation loads the artifact back and scores from it (freeze is
load-bearing, not ceremonial).

HELD-OUT (fresh shapes; no corpus grid/seed reused): cells (H,B,K) =
H{64,512,4096} x B{256,2048} x K{1024,8192} (12 cells); seeds {46,47} (corpus
used 0-4,43,44,45). Graphs verbatim b1_gatezero.py == 423217d level2/51
(correctness cross-checked in B1): ref = Linear -> -s -> mean(1) -> logsumexp ->
gelu -> +x; fold = gelu(x@mean(W,0)+(b-s).mean())+x. W,b: nn.Linear(K,H) seeded
`seed`; s=randn(H) seed+1; x=rand(B,K) seed+2, FIXED within (cell,seed) for the
ordmag arm (declared deviation: benchmark rand, not official input -- shape
generalization requires custom B). ordmag: 24 log-scales 1e0..1e6
(10**(6*si/23)), W=base*scale. cancel: scales {1e4,1e6} x 5 draws, seed
4600+ci*100+i, r_[0,0]=0, x=expand(B,K).contiguous(), s=gemm(x)[0] per draw (B2
construction with cell B). label/cand per B2 `summarize`; ATOL=RTOL=1e-4; features
from the POST-scale W (as B2). Strata: ordmag / cancel.

METRICS. PRIMARY alive iff pooled held-out AUC >= 0.80. KILL1 additionally if any
stratum AUC < 0.70 (collapse; 0.70-0.80 = recorded weakness, non-killing).
Accepted set (s>=t): precision, accept fraction, per stratum (deployment
metadata; envelope violation == 1-precision on the same predicate).

COST. Guard one-time eval = fused stats (a=W.abs(); torch.stack((a.max(),
a.mean(), W.std())); single .cpu()) + rank-inclusion; timed 20 reps (end-to-end
incl sync) at EACH B1 weight geometry (K=1024; H 8/128/2048/8192).
N*_cell = median_cost_s / paired_saving_s; savings = per-cell paired medians from
the COMMITTED results/b1_gatezero.jsonl (read from kb-lab results; sha printed).
KILL-COST (tick promise verbatim): N* > 100 at the min-saving cell (H8,B64,
0.0102720 ms) => check cost eats the speed margin. Full per-geometry N* table
recorded; geometries with N*>100 flagged deployment-restricted (descriptive).
Accepted-path per-call overhead = 0 by construction (bind-time decision; cfac IS
the accepted path, B1).

VERDICT mechanical; results/b4_guard_v0.json. Run on spark; small GPU.
"""
import json, math, os, time, hashlib, pathlib, statistics
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
ATOL = RTOL = 1e-4
PRECISION_TARGET = 0.90
REPS_COST = 20

ROOT = pathlib.Path("/home/ckorhonen/kb-lab")
RES = ROOT / "results"; RES.mkdir(exist_ok=True)
DERIVED = ["collapse_work_ratio", "fan_in_sum", "w_max", "w_mean_abs", "w_rel_spread", "B", "K", "H"]

def summarize(a, b):
    d = (a - b).abs(); limit = ATOL + RTOL * a.abs()
    ratio = d / limit
    return {"pass": bool((d <= limit).all()), "max_abs": d.max().item(),
            "max_budget_ratio": ratio.max().item()}

def ref_fwd(x, W, b, s):
    x0 = x
    y = F.linear(x, W, b) - s
    y = torch.mean(y, dim=1, keepdim=True)
    y = torch.logsumexp(y, dim=1, keepdim=True)
    y = F.gelu(y)
    return y + x0

def fac_fwd(x, W, b, s):
    x0 = x
    y = torch.matmul(x, W.mean(dim=0)) + (b - s).mean()
    y = y.reshape(-1, 1)
    y = torch.logsumexp(y, dim=1, keepdim=True)
    y = F.gelu(y)
    return y + x0

# ---------------- rank machinery (b3 verbatim) ----------------
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

def featval(r, f):
    return math.log10(r["scale"]) if f == "log10_scale" else float(r[f])

# ---------------- FIT on train corpus ----------------
corpus_path = RES / "b2_failure_corpus.jsonl"
rows = [json.loads(l) for l in open(corpus_path)]
rows = [r for r in rows if r.get("method") == "fold32"]
for r in rows: r["_lab"] = bool(r.get("fold32_pass", r.get("label")))
assert len(rows) == 175, len(rows)
sha = hashlib.sha256(corpus_path.read_bytes()).hexdigest()

def orientation(train, f):
    rk = rankdata([featval(r, f) for r in train])
    n1 = sum(1 for r in train if r["_lab"])
    mp = sum(x for x, r in zip(rk, train) if r["_lab"]) / max(n1, 1)
    mf = sum(x for x, r in zip(rk, train) if not r["_lab"]) / max(len(train) - n1, 1)
    return bool(mp >= mf)

ORIENT = {f: orientation(rows, f) for f in DERIVED}

def deployed_scores(eval_rows, cols, orient):
    """rank within corpus-column ++ [x]; z; orient; mean."""
    out = []
    n = len(next(iter(cols.values()))) + 1
    sd = ((n ** 2 - 1) / 12.0) ** 0.5; mu = (n + 1) / 2
    for r in eval_rows:
        zs = []
        for f in DERIVED:
            rk = rankdata(cols[f] + [featval(r, f)])[-1]
            z = (rk - mu) / sd
            if not orient[f]: z = -z
            zs.append(z)
        out.append(sum(zs) / len(zs))
    return out

cols_fit = {f: [featval(r, f) for r in rows] for f in DERIVED}
train_scores = deployed_scores(rows, cols_fit, ORIENT)
pairs = sorted(zip(train_scores, [r["_lab"] for r in rows]), key=lambda t: -t[0])
best_prec, best_k = 0.0, 1
for k in range(1, len(pairs) + 1):
    prec = sum(l for _, l in pairs[:k]) / k
    if prec > best_prec: best_prec, best_k = prec, k
t = None; prec_at_t = None
for k in range(1, len(pairs) + 1):  # largest prefix meeting target
    prec = sum(l for _, l in pairs[:k]) / k
    if prec >= PRECISION_TARGET:
        t, prec_at_t = pairs[k - 1][0], prec
if t is None:
    t, prec_at_t = pairs[best_k - 1][0], best_prec
accept_frac_train = sum(1 for s, _ in pairs if s >= t) / len(pairs)

env_train = sorted(r["max_budget_ratio"] for r, s in zip(rows, train_scores) if s >= t)
def q(xs, p):
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None
ENVELOPE = {"q50": q(env_train, .5), "q90": q(env_train, .9), "q99": q(env_train, .99),
            "max": env_train[-1] if env_train else None}

# ---------------- FREEZE ----------------
freeze = dict(features=DERIVED, orientation=ORIENT, corpus_columns=cols_fit,
              threshold=t, precision_train=prec_at_t, precision_target=PRECISION_TARGET,
              precision_max_train=best_prec, accept_frac_train=accept_frac_train,
              envelope=ENVELOPE, corpus_sha256=sha, n_train=len(rows),
              env=dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                       tf32=False, atol=ATOL, rtol=RTOL),
              when=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
FREEZE_PATH = RES / "guard_v0.json"
FREEZE_PATH.write_text(json.dumps(freeze, indent=1))
print("FROZEN ->", FREEZE_PATH, f"t={t:.4f} prec@t={prec_at_t:.3f} accept={accept_frac_train:.3f}")

fz = json.loads(FREEZE_PATH.read_text())  # eval ONLY from re-read artifact
ORIENT_L, COLS_L, T_L = fz["orientation"], fz["corpus_columns"], fz["threshold"]

# ---------------- HELD-OUT fresh shapes ----------------
CELLS = [(h, b, k) for h in (64, 512, 4096) for b in (256, 2048) for k in (1024, 8192)]
heldout = []
for (H, B, K) in CELLS:
    for seed in (46, 47):
        torch.manual_seed(seed)
        lin = torch.nn.Linear(K, H)
        W0 = lin.weight.detach().float().cuda(); b0 = lin.bias.detach().float().cuda()
        g = torch.Generator().manual_seed(seed + 1)
        s0 = torch.randn(H, generator=g).float().cuda()
        x_ord = torch.rand(B, K, generator=torch.Generator().manual_seed(seed + 2)).float().cuda().cuda()
        base = dict(cell=f"H{H}_B{B}_K{K}", seed=seed, B=B, K=K, H=H,
                    collapse_work_ratio=B * K * H / (H * K + K), fan_in_sum=K)
        for si in range(24):  # ordmag arm
            scale = float(10 ** (6.0 * si / 23)); W = W0 * scale
            ref = ref_fwd(x_ord, W, b0, s0); T = fac_fwd(x_ord, W, b0, s0)
            s_ = summarize(ref, T); a = W.abs()
            heldout.append(dict(base, arm="ordmag", scale=scale, label=s_["pass"],
                                max_budget_ratio=s_["max_budget_ratio"],
                                w_max=float(a.max()), w_mean_abs=float(a.mean()),
                                w_rel_spread=float(W.std() / a.mean())))
        for ci, scale in ((0, 1e4), (1, 1e6)):  # cancel arm, B2 construction
            W = W0 * scale
            for i in range(5):
                torch.manual_seed(4600 + ci * 100 + i)  # B2-verbatim global-seed draw
                r_ = torch.rand(1, K, device="cuda"); r_[0, 0] = 0
                x = r_.expand(B, K).contiguous()
                s_0 = F.linear(x, W, b0)[0].contiguous()  # subtract per draw
                ref = ref_fwd(x, W, b0, s_0); T = fac_fwd(x, W, b0, s_0)
                s_ = summarize(ref, T); a = W.abs()
                heldout.append(dict(base, arm="cancel", scale=scale, draw=i,
                                    label=s_["pass"],
                                    max_budget_ratio=s_["max_budget_ratio"],
                                    w_max=float(a.max()), w_mean_abs=float(a.mean()),
                                    w_rel_spread=float(W.std() / a.mean())))
        del W0, b0, s0, x_ord
        torch.cuda.empty_cache()
n_arm = {a: sum(1 for r in heldout if r["arm"] == a) for a in ("ordmag", "cancel")}
print("held-out rows:", len(heldout), n_arm)

hs = deployed_scores(heldout, COLS_L, ORIENT_L)
for r, s in zip(heldout, hs):
    r["score"] = s; r["pred_pass"] = bool(s >= T_L)

def sel_auc(sel):
    return auc([s for s, _ in sel], [l for _, l in sel])

pooled_auc = sel_auc([(r["score"], r["label"]) for r in heldout])
auc_by_arm = {a: sel_auc([(r["score"], r["label"]) for r in heldout if r["arm"] == a])
              for a in ("ordmag", "cancel")}
acc = [r for r in heldout if r["pred_pass"]]
def _prec(sel):
    return (sum(1 for r in sel if r["label"]) / len(sel)) if sel else None
metrics = dict(
    n_heldout=len(heldout), n_fail=sum(1 for r in heldout if not r["label"]),
    n_pass=sum(1 for r in heldout if r["label"]),
    pooled_auc=pooled_auc, auc_by_arm=auc_by_arm,
    collapse_flags={a: v for a, v in auc_by_arm.items() if v < 0.70},
    accepted_n=len(acc), accepted_precision=_prec(acc),
    accepted_frac=len(acc) / len(heldout),
    accepted_precision_by_arm={a: _prec([r for r in acc if r["arm"] == a])
                               for a in ("ordmag", "cancel")},
    accepted_frac_by_arm={a: (sum(1 for r in acc if r["arm"] == a) /
                              max(1, n_arm[a])) for a in ("ordmag", "cancel")})
cells_auc = {c: sel_auc([(r["score"], r["label"]) for r in heldout if r["cell"] == c])
             for c in {r["cell"] for r in heldout}}
metrics["cells_auc"] = cells_auc

# ---------------- COST ----------------
def guard_cost(H, K, reps=REPS_COST):
    torch.manual_seed(999)
    W = torch.randn(H, K, device="cuda")
    for _ in range(3):
        a = W.abs(); torch.stack((a.max(), a.mean(), W.std())).cpu()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        a = W.abs()
        stats = torch.stack((a.max(), a.mean(), W.std())).cpu()
        _ = stats[0].item() + stats[1].item() + stats[2].item()
        t1 = time.perf_counter()
        ts.append(t1 - t0)
    del W; torch.cuda.empty_cache()
    ts.sort(); return ts[len(ts) // 2]

b1 = [json.loads(l) for l in open(RES / "b1_gatezero.jsonl")]
sav = {}
for r in b1: sav.setdefault((r["H"], r["B"]), []).append(r["cref_ms"] - r["cfac_ms"])
sav = {k: statistics.median(v) for k, v in sav.items()}
assert len(sav) == 13, sorted(sav)
cost = {f"H{h}_K1024": guard_cost(h, 1024) for h in (8, 128, 2048, 8192)}
cost_largest_heldout = guard_cost(4096, 8192)
nstar = {}
for (H, B), s_ms in sorted(sav.items()):
    c_s = cost[f"H{H}_K1024"]
    nstar[f"H{H}_B{B}"] = dict(cost_s=c_s, saving_ms=s_ms, nstar=c_s / (s_ms / 1e3))
min_cell = min(sav, key=lambda k: sav[k])
min_nstar = nstar[f"H{min_cell[0]}_B{min_cell[1]}"]["nstar"]
cost_block = dict(cost_by_geom_s=cost, nstar=nstar,
                  min_saving_cell=list(min_cell), min_saving_ms=sav[min_cell],
                  nstar_min_cell=min_nstar,
                  cost_largest_heldout_s=cost_largest_heldout,
                  kill_cost=bool(min_nstar > 100),
                  deployment_restricted=[k for k, v in nstar.items() if v["nstar"] > 100])

# ---------------- VERDICT (mechanical) ----------------
verdict = dict(
    primary_alive=bool(pooled_auc >= 0.80),
    kill1_collapse=bool(metrics["collapse_flags"]),
    kill_cost=cost_block["kill_cost"],
    alive=bool(pooled_alive := pooled_auc >= 0.80) and not metrics["collapse_flags"]
          and not cost_block["kill_cost"])
out = dict(when=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           prereg="docstring written before run 2026-10-20 ~00:40Z",
           freeze=dict(path="results/guard_v0.json", threshold=T_L, corpus_sha256=sha),
           heldout_metrics=metrics, cost=cost_block, verdict=verdict,
           env=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, tf32=False))
(RES / "b4_guard_v0.json").write_text(json.dumps(out, indent=1, default=float))
with open(RES / "b4_guard_v0.jsonl", "w") as f:
    for r in heldout: f.write(json.dumps(r) + "\n")
print(json.dumps(dict(pooled_auc=pooled_auc, auc_by_arm=auc_by_arm,
                      accepted_precision=metrics["accepted_precision"],
                      accepted_frac=metrics["accepted_frac"],
                      nstar_min_cell=min_nstar, verdict=verdict), indent=1, default=str))
