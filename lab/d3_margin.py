"""D3 runner -- conv margin decomposition + fp64-accum arm (spark GPU).

PRE-REGISTRATION: lab/d3_freeze.py docstring + results/d3_freeze.json
(committed 4492252 BEFORE this run). Adds no freedom: universe, metrics,
branch rule phi_cond>=0.50, cand64, T* axis, declared re-evaluation all
frozen there; this file only executes them. Tie-out vs committed
results/d2_conv.jsonl asserted BEFORE any d3 number is recorded.

Mechanism note: d2 builders are invoked VERBATIM via recorder wrappers
around d2_conv.ref_fwd/cand_fwd (wrappers call the originals and retain
inputs/outputs); seed/geometry logic is untouched and tie-out validates
the whole chain.

Run: cd ~/kb-lab && ~/kernelbench/.venv/bin/python d3_margin.py
"""
import json, os, sys, hashlib, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
sys.path.insert(0, HERE)

import torch
import d2_conv as D  # module level: loads guard_v1/b2 corpus/d2_freeze, asserts shas

FZ = json.load(open(os.path.join(RES, "d3_freeze.json")))
NEAR, ATOL, RTOL = FZ["near_band"], FZ["atol"], FZ["rtol"]
for key, fn in (("d2_freeze", "d2_freeze.json"), ("d2_conv", "d2_conv.jsonl"),
                ("guard_v1", "guard_v1.json")):
    s = hashlib.sha256(open(os.path.join(RES, fn), "rb").read()).hexdigest()
    assert s == FZ["shas"][key], (key, s, FZ["shas"][key])

# ---- verbatim row universe via recorder-wrapped builders ----
_rec = []
_of, _oc = D.ref_fwd, D.cand_fwd
def _ref_rec(x, W, b):
    R = _of(x, W, b); _rec.append(["R", x, W, b, R]); return R
def _cand_rec(x, W, b):
    T = _oc(x, W, b); _rec.append(["T", None, None, None, T]); return T
D.ref_fwd, D.cand_fwd = _ref_rec, _cand_rec

rows_all1 = D.build_rows([10 ** (6 * si / 23) for si in range(24)])   # d2 initial pass
rows_c = D.build_rows_contingency()                                   # d2 declared widen
D.ref_fwd, D.cand_fwd = _of, _oc
assert len(_rec) == 2 * (len(rows_all1) + len(rows_c))

def cand64_fwd(x, W, b):
    # d2 cand path, dtype raised; result cast fp32 (deployable object)
    P = x.double().unfold(2, D.K, 1).unfold(3, D.K, 1)
    P = P.permute(0, 2, 3, 1, 4, 5).reshape(D.N * D.OH * D.OW, D.CIN * D.K * D.K)
    Wf = W.double().reshape(D.COUT, -1)
    y = P @ Wf.t() + b.double()
    return y.reshape(D.N, D.OH, D.OW, D.COUT).permute(0, 3, 1, 2).contiguous().float()

all_rows = rows_all1 + rows_c
for i, r in enumerate(all_rows):
    r["_rec"] = 2 * i
kept = [r for r in all_rows if r["arm"] == "cancel"] + rows_c
assert len(kept) == 116 and {r["arm"] for r in kept} == {"ordmag", "cancel"}

# ---- tie-out vs committed d2_conv.jsonl ----
comm = {}
for line in open(os.path.join(RES, "d2_conv.jsonl")):
    r = json.loads(line)
    comm[(r["arm"], r["seed"], r["si"], round(float(r["scale"]), 12))] = r["max_budget_ratio"]
for r in kept:
    k = (r["arm"], r["seed"], r["si"], round(float(r["scale"]), 12))
    assert abs(comm[k] - r["max_budget_ratio"]) <= 1e-9 * abs(comm[k]), (k, comm[k], r["max_budget_ratio"])

# ---- frozen decomposition + arms ----
out_rows, nearR_f, nearR_p = [], [], []
agg = dict(n_fail=0, n_fail_near=0, n_pass=0, n_pass_near=0, n_elem=0, n_argmax_near=0)
for r in kept:
    _, x, W, b, R = _rec[r["_rec"]]
    T = _rec[r["_rec"] + 1][4]
    assert torch.equal(R, R) and R.is_cuda
    d = (T - R).abs(); lim = ATOL + RTOL * R.abs(); ratio = d / lim
    near = R.abs() < NEAR; fail = ratio > 1.0
    nfail = int(fail.sum()); nfn = int((fail & near).sum())
    npass = int((~fail).sum()); npn = int((~fail & near).sum())
    am = int(ratio.argmax()); am_near = bool(near.flatten()[am].item())
    t32 = (d / (1.0 + R.abs())).max().item()
    T64 = cand64_fwd(x, W, b)
    d64 = (T64 - R).abs(); lim64 = ATOL + RTOL * R.abs(); ratio64 = d64 / lim64
    label64 = bool((ratio64 <= 1.0).all().item())
    t64 = (d64 / (1.0 + R.abs())).max().item()
    nearR_f.append(R.abs().masked_select(fail)); nearR_p.append(R.abs().masked_select(~fail))
    out_rows.append(dict(arm=r["arm"], seed=r["seed"], si=r["si"], scale=r["scale"],
                         n_fail=nfail, n_fail_near=nfn, n_pass=npass, n_pass_near=npn,
                         n_elem=int(R.numel()), argmax_near=am_near,
                         max_ratio32=r["max_budget_ratio"], tstar32=t32,
                         max_ratio64=ratio64.max().item(), label64=label64, tstar64=t64))
    agg["n_fail"] += nfail; agg["n_fail_near"] += nfn
    agg["n_pass"] += npass; agg["n_pass_near"] += npn
    agg["n_elem"] += int(R.numel()); agg["n_argmax_near"] += int(am_near)

phi_cond = agg["n_fail_near"] / agg["n_fail"] if agg["n_fail"] else None
phi_pass = agg["n_pass_near"] / agg["n_pass"] if agg["n_pass"] else None
phi_abs = (agg["n_fail_near"] + agg["n_pass_near"]) / agg["n_elem"]
phi_argmax = agg["n_argmax_near"] / len(kept)
med_f = torch.cat(nearR_f).median().item() if nearR_f else None
med_p = torch.cat(nearR_p).median().item() if nearR_p else None
branch = "A_phi_heavy" if (phi_cond is not None and phi_cond >= FZ["phi_branch"]) else "B_flat"
frac64 = sum(1 for r in out_rows if r["label64"]) / len(out_rows)
remedy = "green" if frac64 >= FZ["remedy_green"] else ("partial" if frac64 >= FZ["remedy_partial"] else "fail")

# T* quantiles per arm x cand
def q(vs):
    if not vs: return None
    vs = sorted(vs); n = len(vs)
    return dict(min=vs[0], q25=vs[max(0, int(.25 * n) - 1)], q50=vs[max(0, int(.5 * n) - 1)],
                q75=vs[max(0, int(.75 * n) - 1)], q90=vs[max(0, int(.90 * n) - 1)], max=vs[-1])

tq = {}
for arm in ("ordmag", "cancel"):
    for k in ("tstar32", "tstar64"):
        vs = [r[k] for r in out_rows if r["arm"] == arm]
        tq[f"{arm}.{k}"] = q(vs)

# ---- declared re-evaluation: d2 verdict machinery verbatim, label64 ----
for r, o in zip(kept, out_rows):  # same order; attach frozen-arm label
    r["label64"] = o["label64"]
for r in kept:
    r["score"] = D.deployed_z(r, record=r)  # geometry z, unchanged from d2
units = {}
for a in sorted({r["arm"] for r in kept}):
    rs = [r for r in kept if r["arm"] == a]
    t, J = D.youden_tstar([r["score"] for r in rs], [r["label64"] for r in rs])
    mean_signed = st.fmean(sum((r["z_raw"][f] if D.ORIENT[f] else -r["z_raw"][f]) for f in D.KEPT) / len(D.KEPT) for r in rs)
    units[a] = dict(n=len(rs), n_pos64=sum(1 for r in rs if r["label64"]),
                    youden_J64=J, t_star64=t, d_raw=mean_signed)
conv_units = {a: u for a, u in units.items() if u["t_star64"] is not None}
d_mean = st.fmean(u["d_raw"] for u in conv_units.values()) if conv_units else 0.0
for a, u in units.items():
    u["d_cen"] = u["d_raw"] - d_mean if a in conv_units else None
    if u["t_star64"] is not None:
        u["T_law64"] = D.A_DAG + D.GAMMA * u["d_cen"] + D.B_PRED

prec_fr64 = acc_fr64 = prec_law64 = acc_law64 = None
acc = [r for r in kept if r["score"] >= D.T_FROZEN]
if acc:
    prec_fr64 = sum(1 for r in acc if r["label64"]) / len(acc); acc_fr64 = len(acc) / len(kept)
law_pairs = [(r, units[r["arm"]].get("T_law64")) for r in kept]
if all(t is not None for _, t in law_pairs):
    acc = [r for r, t in law_pairs if r["score"] >= t]
    prec_law64 = (sum(1 for r in acc if r["label64"]) / len(acc)) if acc else 0.0
    acc_law64 = len(acc) / len(kept)

R2_pool64 = None; beta64 = None
if conv_units:
    d1 = json.load(open(os.path.join(RES, "d1_cellaffine.json")))
    tab = list(d1["table"])
    X, y = [], []
    allu = [(u["src"], u["d_raw"], u["t_star"]) for u in tab]
    fammean = {}
    for src in ("b4", "b5", "d2"):
        du = [d for s, d, _ in allu if s == src] + \
             ([u["d_raw"] for u in conv_units.values()] if src == "d2" else [])
        fammean[src] = st.fmean(du) if du else 0.0
    for s, d, t in allu:
        X.append([1.0, 1.0 if s == "b5" else 0.0, 1.0 if s == "d2" else 0.0, d - fammean[s]]); y.append(t)
    for u in conv_units.values():
        X.append([1.0, 0.0, 1.0, u["d_raw"] - fammean["d2"]]); y.append(u["t_star64"])
    p = 4
    A = [[sum(X[r][i] * X[r][j] for r in range(len(X))) for j in range(p)] for i in range(p)]
    bb = [sum(X[r][i] * y[r] for r in range(len(X))) for i in range(p)]
    M = [A[i][:] + [bb[i]] for i in range(p)]
    for c in range(p):
        piv = max(range(c, p), key=lambda rr: abs(M[rr][c]))
        M[c], M[piv] = M[piv], M[c]; pv = M[c][c]
        if abs(pv) < 1e-12: continue
        M[c] = [v / pv for v in M[c]]
        for rr in range(p):
            if rr != c and abs(M[rr][c]) > 0:
                f = M[rr][c]; M[rr] = [a1 - f * b1 for a1, b1 in zip(M[rr], M[c])]
    beta64 = [M[i][p] for i in range(p)]
    ybar = st.fmean(y)
    sstot = sum((v - ybar) ** 2 for v in y)
    ssres = sum((y[i] - sum(beta64[j] * X[i][j] for j in range(p))) ** 2 for i in range(len(y)))
    R2_pool64 = 1 - ssres / sstot if sstot > 0 else float("nan")

b_meas = (st.fmean(u["t_star64"] for u in conv_units.values()) - D.A_DAG) if conv_units else None
rider = None
if b_meas is not None:
    rider = dict(b_meas=b_meas, b_pred=D.B_PRED, absdiff=abs(b_meas - D.B_PRED),
                 bound=D.BOUND, pass_=bool(abs(b_meas - D.B_PRED) <= D.BOUND))

V = {}
V["reeval_evaluable"] = bool(conv_units)
if conv_units:
    V["K-a_frozen_t_sufficient_64"] = bool(prec_fr64 is not None and prec_fr64 >= 0.90 and acc_fr64 is not None and acc_fr64 >= 0.20)
    V["K-b_repaired_below_0.80_64"] = bool(prec_law64 is not None and prec_law64 < 0.80)
    V["K-c_cellfit_R2_below_0.5_64"] = bool(R2_pool64 is not None and R2_pool64 < 0.5)
    V["G_construction_signal_64"] = bool(prec_law64 is not None and prec_law64 >= 0.90 and acc_law64 is not None and acc_law64 >= 0.20)
    if rider: V["offset_law_rider_PASS_64"] = rider["pass_"]

out = dict(when=__import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
           freeze_sha=hashlib.sha256(open(os.path.join(RES, "d3_freeze.json"), "rb").read()).hexdigest()[:16],
           n_rows=len(kept), agg=agg,
           phi_cond=phi_cond, phi_pass=phi_pass, phi_abs=phi_abs, phi_argmax=phi_argmax,
           median_absR_fail=med_f, median_absR_pass=med_p,
           branch=branch, frac_label64=frac64, remedy64=remedy, tstar_q=tq,
           units64=units, prec_frozen64=prec_fr64, acc_frozen64=acc_fr64,
           prec_law64=prec_law64, acc_law64=acc_law64,
           pooled_R2_3family64=R2_pool64, pooled_beta64=beta64,
           rider64=rider, verdict64=V,
           env=dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0)))

with open(os.path.join(RES, "d3_margin.jsonl"), "w") as f:
    for r in out_rows:
        f.write(json.dumps(r) + "\n")
json.dump(out, open(os.path.join(RES, "d3_margin.json"), "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k != "tstar_q"}, indent=1, default=str))
print("T*quantiles:", json.dumps(out["tstar_q"], indent=1))
