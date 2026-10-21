"""D2 -- conv third-family test (plan v2 item 3). GPU harness + mechanical verdict.

PRE-REGISTRATION: lab/d2_freeze.py docstring + results/d2_freeze.json (written
21:00Z, BEFORE this run; kills/signals/conjecture bound frozen there and
quoted verbatim below). This file adds NO new freedom: guard score machinery
b4 v1 verbatim (kept feats, single-sided appended rank, orientation frozen),
Youden t* d1 convention verbatim (plateau midpoint), label = b4 summarize
ATOL=RTOL=1e-4. Env pin: torch fp32, TF32 off, cuda.

FROZEN CONSTANTS (d2_freeze.json): a_dag=-1.1052271376183762
gamma=+0.5876085080140739  b_conv_pred=-0.5707986843139425
pass bound |b_pred-b_meas| <= 0.27973625942385133 (2xRMSE_D1=0.279736)
t_frozen=-0.6134445630270532; kept=[w_mean_abs,w_max,cwr]; ori as frozen.

ARMS: ordmag 24 log-scales 1e0..1e6 on Wflat (b, x unscaled); cancel
{1e4,1e6} x 5 draws, W[0,0,0,0]=0, x=batch-expanded single sample.
Seeds {52,53}; geometry Cin64 Cout128 k3 8x8 valid; x randn per seed.
Contingency (declared in freeze): both arms single-class => ordmag 48 steps
1e0..1e8, same seeds, once.

VERDICTS (mechanical, from data only):
  K-a  prec(t_frozen) >= .90 and accept >= .20  => frozen-t sufficient (note)
  K-b  prec(T_law) < .80                        => hierarchy wrong
  K-c  pooled 3-family refit R2 < .50           => decomposition empty
  G    prec(T_law) >= .90 and accept >= .20     => construction signal GREEN
  rider |b_meas-b_pred| <= 0.279736              => offset law PASS on conv
Descriptive: per-arm AUC, pooled AUC, per-unit Youden table, envelope q's.

OUTPUT results/d2_conv.jsonl (rows) + d2_conv.json (verdict block).
Run on spark: ~/kernelbench/.venv/bin/python, from ~/kb-lab.
"""
import json, os, sys, hashlib, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ATOL = RTOL = 1e-4

# ---------------- frozen artifacts ----------------
g = json.load(open(os.path.join(RES, "guard_v1.json")))
COLS = {k: list(v) for k, v in g["corpus_columns"].items()}
ORIENT = g["orientation"]; KEPT = g["kept_features"]; T_FROZEN = g["threshold"]
sha = hashlib.sha256(open(os.path.join(RES, "b2_failure_corpus.jsonl"), "rb").read()).hexdigest()
assert sha == g["corpus_sha256"], "corpus drift"
fz = json.load(open(os.path.join(RES, "d2_freeze.json")))
A_DAG, GAMMA = fz["a_dag"], fz["gamma"]
B_PRED, BOUND = fz["b_conv_pred_F1"], fz["bound_2rmse"]
assert fz["corpus_sha"] == sha

# ---------------- b4-verbatim machinery ----------------
def rankdata(v):
    idx = sorted(range(len(v)), key=lambda i: v[i]); out = [0.0] * len(v); i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]: j += 1
        m = (i + j) / 2.0 + 1.0
        for k2 in range(i, j + 1): out[idx[k2]] = m
        i = j + 1
    return out

def featval(r, f):
    return float(r[f])

def deployed_z(r, record=None):
    n = len(next(iter(COLS.values()))) + 1
    sd = ((n ** 2 - 1) / 12.0) ** 0.5; mu = (n + 1) / 2
    zs = []
    for f in KEPT:
        rk = rankdata(COLS[f] + [featval(r, f)])[-1]
        z = (rk - mu) / sd
        if record is not None: record.setdefault("z_raw", {})[f] = z
        if not ORIENT[f]: z = -z
        zs.append(z)
    return sum(zs) / len(zs)

def summarize(a, b):
    d = (a - b).abs(); limit = ATOL + RTOL * a.abs()
    ratio = d / limit
    return {"pass": bool((d <= limit).all()), "max_abs": d.max().item(),
            "max_budget_ratio": ratio.max().item()}

def youden_tstar(scores, labels):
    pairs = sorted(zip(scores, labels)); n = len(pairs)
    P = sum(labels); N = n - P
    if P == 0 or N == 0: return None, None
    Js = []; idxs = []
    for i in range(-1, n):
        if i >= 0 and i < n - 1 and pairs[i][0] == pairs[i + 1][0]: continue
        acc = pairs[i + 1:]
        tp = sum(1 for s, l in acc if l); fp = len(acc) - tp
        Js.append((tp / P) - (fp / N)); idxs.append(i)
    m = max(Js)
    cuts = [idxs[k] for k, j in enumerate(Js) if abs(j - m) <= 1e-12]
    los = [pairs[c + 1][0] for c in cuts if c + 1 < n]
    his = [pairs[c][0] for c in cuts if c >= 0]
    lo = min(los) if los else (his[0] if his else 0.0)
    hi = max(his) if his else lo
    return (lo + hi) / 2.0, m

# ---------------- family forward ----------------
CIN, COUT, K, HIN, N = 64, 128, 3, 8, 8
OH = OW = HIN - K + 1

def ref_fwd(x, W, b):
    return F.conv2d(x, W, b)

def cand_fwd(x, W, b):
    P = x.unfold(2, K, 1).unfold(3, K, 1)          # (N,C,Oh,Ow,k,k)
    P = P.permute(0, 2, 3, 1, 4, 5).reshape(N * OH * OW, CIN * K * K)
    Wf = W.reshape(COUT, CIN * K * K)
    y = P @ Wf.t() + b
    return y.reshape(N, OH, OW, COUT).permute(0, 3, 1, 2).contiguous()

def build_rows(scale_steps):
    rows = []
    for seed in (52, 53):
        torch.manual_seed(1000 + seed)
        Wbase = torch.randn(COUT, CIN, K, K, dtype=torch.float32)
        b = torch.randn(COUT, dtype=torch.float32) * 0.1
        torch.manual_seed(2000 + seed)
        x0 = torch.randn(1, CIN, HIN, HIN, dtype=torch.float32)
        for arm in ("ordmag", "cancel"):
            if arm == "ordmag":
                specs = [("ordmag", 10 ** (6 * si / (len(scale_steps) - 1)), si)
                         for si in range(len(scale_steps))]
            else:
                specs = [("cancel", s, ci * 100 + i)
                         for ci, s in enumerate((1e4, 1e6)) for i in range(5)]
            for name, scale, si in specs:
                torch.manual_seed(5200 + seed * 100 + si)
                if name == "cancel":
                    W = torch.randn(COUT, CIN, K, K, dtype=torch.float32) * scale
                    W[0, 0, 0, 0] = 0.0
                    x = x0.expand(N, CIN, HIN, HIN).contiguous()
                else:
                    W = Wbase * scale
                    x = x0.expand(N, CIN, HIN, HIN).contiguous()
                R = ref_fwd(x.cuda(), W.cuda(), b.cuda())
                T = cand_fwd(x.cuda(), W.cuda(), b.cuda())
                Rc, Tc = R.cpu(), T.cpu()
                summ = summarize(Rc, Tc)
                Wf = W.reshape(COUT, -1)
                r = dict(src="d2", cell="C64_K576_Cout128_8x8_N4", arm=name,
                         seed=seed, scale=scale, si=si,
                         Cin=CIN, Cout=COUT, kk=K, Hin=HIN, N_batch=N, Oh=OH,
                         B=N * OH * OW, Krole=CIN * K * K, Hrole=COUT,
                         collapse_work_ratio=N * OH * OW * (CIN * K * K) * COUT / (COUT * (CIN * K * K) + CIN * K * K),
                         fan_in_sum=CIN * K * K,
                         w_max=Wf.abs().max().item(),
                         w_mean_abs=Wf.abs().mean().item(),
                         w_rel_spread=(Wf.std() / Wf.abs().mean()).item() if False else (Wf.abs().std() / Wf.abs().mean()).item(),
                         label=bool(summ["pass"]),
                         max_budget_ratio=summ["max_budget_ratio"],
                         max_abs=summ["max_abs"])
                rows.append(r)
    return rows

def prec_acc(rows, T):
    acc = [r for r in rows if r["score"] >= T]
    if not acc: return 0.0, 0.0
    return sum(1 for r in acc if r["label"]) / len(acc), len(acc) / len(rows)

def auc(pairs):
    lab = [1 if l else 0 for _, l in pairs]
    n1 = sum(lab); n0 = len(lab) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    rk = rankdata([s for s, _ in pairs])
    s1 = sum(x for x, l in zip(rk, lab) if l)
    return (s1 - n1 * (n1 + 1) / 2) / (n1 * n0)

def main():
    dev = "cuda"
    rows = build_rows([10 ** (6 * si / 23) for si in range(24)])
    # score on CPU-light path: features are scalars; deployed_z pure python
    for r in rows:
        r["score"] = deployed_z(r, record=r)
    single = [(a, sum(1 for r in rows if r["arm"] == a and r["label"]),
               sum(1 for r in rows if r["arm"] == a and not r["label"]))
              for a in ("ordmag", "cancel")]
    widened = False
    if all(p == 0 or n == 0 for _, p, n in single):
        rows = [r for r in rows if r["arm"] != "ordmag"] + \
               [r for r in build_rows_contingency()]
        widened = True
        for r in rows: r["score"] = deployed_z(r, record=r)
    for r in rows: r["score"] = r["score"]  # noop, kept for clarity

    # per-unit Youden (unit = arm; seeds pooled)
    units = {}
    for a in sorted({r["arm"] for r in rows}):
        rs = [r for r in rows if r["arm"] == a]
        t, J = youden_tstar([r["score"] for r in rs], [r["label"] for r in rs])
        m = st.fmean(sum((1 if not ORIENT[f] else 1) * 0 + v for v in [0]) for r in [rs[0]]) if False else None
        mean_signed = st.fmean(sum((r["z_raw"][f] if ORIENT[f] else -r["z_raw"][f]) for f in KEPT) / len(KEPT) for r in rs)
        units[a] = dict(n=len(rs), n_pos=sum(1 for r in rs if r["label"]),
                        youden_J=J, t_star=t, d_raw=mean_signed)
    conv_units = {a: u for a, u in units.items() if u["t_star"] is not None}
    d_mean = st.fmean(u["d_raw"] for u in conv_units.values()) if conv_units else 0.0
    for a, u in units.items():
        u["d_cen"] = u["d_raw"] - d_mean if a in conv_units else None
        if u["t_star"] is not None:
            u["T_law"] = A_DAG + GAMMA * u["d_cen"] + B_PRED

    # pooled thresholds
    prec_fr, acc_fr = prec_acc(rows, T_FROZEN)
    law_rows = [(r, units[r["arm"]].get("T_law")) for r in rows]
    prec_law = acc_law = None
    if all(t is not None for _, t in law_rows):
        acc = [r for r, t in law_rows if r["score"] >= t]
        prec_law = (sum(1 for r in acc if r["label"]) / len(acc)) if acc else 0.0
        acc_law = len(acc) / len(rows)
    # oracle
    oracle = [r for r in rows]
    prec_or = acc_or = None
    acc = [r for r, u in ((r, units[r["arm"]]) for r in rows)
           if u.get("t_star") is not None and r["score"] >= u["t_star"]]
    if acc:
        prec_or = sum(1 for r in acc if r["label"]) / len(acc); acc_or = len(acc) / len(rows)

    # K-c: pooled 3-family refit (b4/b5 committed + d2), M1 + conv indicator
    d1 = json.load(open(os.path.join(RES, "d1_cellaffine.json")))
    tab = list(d1["table"])
    X, y = [], []
    fammean = {}
    allu = [(u["src"], u["d_raw"], u["t_star"]) for u in tab]
    for src in ("b4", "b5", "d2"):
        du = [d for s, d, _ in allu if s == src] + \
             ([u["d_raw"] for u in conv_units.values()] if src == "d2" else [])
        fammean[src] = st.fmean(du) if du else 0.0
    for s, d, t in allu:
        X.append([1.0, 1.0 if s == "b5" else 0.0, 1.0 if s == "d2" else 0.0,
                  d - fammean[s]]); y.append(t)
    for u in conv_units.values():
        X.append([1.0, 0.0, 1.0, u["d_raw"] - fammean["d2"]]); y.append(u["t_star"])
    # OLS 4-param via normal equations
    import math
    p = 4
    A = [[sum(X[r][i] * X[r][j] for r in range(len(X))) for j in range(p)] for i in range(p)]
    b = [sum(X[r][i] * y[r] for r in range(len(X))) for i in range(p)]
    M = [A[i][:] + [b[i]] for i in range(p)]
    for c in range(p):
        piv = max(range(c, p), key=lambda rr: abs(M[rr][c]))
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        if abs(pv) < 1e-12: continue  # zero pivot (empty conv col): skip, verified vs d1 exact
        M[c] = [v / pv for v in M[c]]
        for rr in range(p):
            if rr != c and abs(M[rr][c]) > 0:
                f = M[rr][c]; M[rr] = [a1 - f * b1 for a1, b1 in zip(M[rr], M[c])]
    beta = [M[i][p] for i in range(p)]
    ybar = st.fmean(y)
    sstot = sum((v - ybar) ** 2 for v in y)
    ssres = sum((y[i] - sum(beta[j] * X[i][j] for j in range(p))) ** 2 for i in range(len(y)))
    R2_pool = 1 - ssres / sstot if sstot > 0 else float("nan")

    b_meas = (st.fmean(u["t_star"] for u in conv_units.values()) - A_DAG) if conv_units else None
    rider = None
    if b_meas is not None:
        rider = dict(b_meas=b_meas, b_pred=B_PRED, absdiff=abs(b_meas - B_PRED),
                     bound=BOUND, pass_=bool(abs(b_meas - B_PRED) <= BOUND))

    auc_all = auc([(r["score"], r["label"]) for r in rows])
    auc_arm = {a: auc([(r["score"], r["label"]) for r in rows if r["arm"] == a])
               for a in ("ordmag", "cancel")}

    # verdicts
    V = {}
    V["K-a_frozen_t_sufficient"] = bool(prec_fr is not None and prec_fr >= 0.90 and acc_fr is not None and acc_fr >= 0.20)
    V["K-b_repaired_below_0.80"] = bool(prec_law is not None and prec_law < 0.80)
    V["K-c_cellfit_R2_below_0.5"] = bool(R2_pool < 0.5)
    V["G_construction_signal"] = bool(prec_law is not None and prec_law >= 0.90 and acc_law is not None and acc_law >= 0.20)
    if rider: V["offset_law_rider_PASS"] = rider["pass_"]

    out = dict(when=__import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
               widened_contingency=widened, n_rows=len(rows),
               units=units, prec_frozen=prec_fr, acc_frozen=acc_fr,
               prec_law=prec_law, acc_law=acc_law,
               prec_oracle=prec_or, acc_oracle=acc_or,
               pooled_R2_3family=R2_pool, pooled_beta=beta,
               rider=rider, auc_all=auc_all, auc_arm=auc_arm,
               verdicts=V, env=dict(torch=torch.__version__, dev=dev,
                                    gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"),
               freeze_sha=hashlib.sha256(open(os.path.join(RES, "d2_freeze.json"), "rb").read()).hexdigest()[:16])
    with open(os.path.join(RES, "d2_conv.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items()}) + "\n")
    json.dump(out, open(os.path.join(RES, "d2_conv.json"), "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))

def build_rows_contingency():
    # declared: ordmag widened to 48 log steps 1e0..1e8, same seeds, cancel kept
    rows = []
    for seed in (52, 53):
        torch.manual_seed(1000 + seed)
        Wbase = torch.randn(COUT, CIN, K, K, dtype=torch.float32)
        b = torch.randn(COUT, dtype=torch.float32) * 0.1
        torch.manual_seed(2000 + seed)
        x0 = torch.randn(1, CIN, HIN, HIN, dtype=torch.float32)
        for si in range(48):
            scale = 10 ** (8 * si / 47)
            W = Wbase * scale
            x = x0.expand(N, CIN, HIN, HIN).contiguous()
            R = ref_fwd(x.cuda(), W.cuda(), b.cuda())
            T = cand_fwd(x.cuda(), W.cuda(), b.cuda())
            summ = summarize(R.cpu(), T.cpu())
            Wf = W.reshape(COUT, -1)
            rows.append(dict(src="d2", cell="C64_K576_Cout128_8x8_N4", arm="ordmag",
                             seed=seed, scale=scale, si=si,
                             Cin=CIN, Cout=COUT, kk=K, Hin=HIN, N_batch=N, Oh=OH,
                             B=N * OH * OW, Krole=CIN * K * K, Hrole=COUT,
                             collapse_work_ratio=N * OH * OW * (CIN * K * K) * COUT / (COUT * (CIN * K * K) + CIN * K * K),
                             fan_in_sum=CIN * K * K,
                             w_max=Wf.abs().max().item(),
                             w_mean_abs=Wf.abs().mean().item(),
                             w_rel_spread=(Wf.abs().std() / Wf.abs().mean()).item(),
                             label=bool(summ["pass"]),
                             max_budget_ratio=summ["max_budget_ratio"],
                             max_abs=summ["max_abs"]))
    return rows

if __name__ == "__main__":
    main()
