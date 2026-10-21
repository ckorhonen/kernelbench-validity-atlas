"""CD runner -- conv-discriminator (spark GPU). Executes cd_freeze verbatim.

PRE-REGISTRATION: lab/cd_freeze.py + results/cd_freeze.json (committed BEFORE
this run). Adds no freedom: universe, t* convention, thresholds, verdicts
frozen there. Tie-out vs committed results/d2_conv.jsonl asserted BEFORE any
cd number is recorded (d3 pattern verbatim).

Run: cd ~/kb-lab && ~/kernelbench/.venv/bin/python cd_discrim.py
"""
import json, os, sys, hashlib
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F
import d2_conv as D  # module level: loads guard_v1/b2 corpus/d2_freeze, asserts shas

FZ = json.load(open(os.path.join(RES, "cd_freeze.json")))
TH = FZ["thresholds"]
for key, fn in (("d2_freeze", "d2_freeze.json"), ("d2_conv", "d2_conv.jsonl"),
                ("guard_v1", "guard_v1.json")):
    s = hashlib.sha256(open(os.path.join(RES, fn), "rb").read()).hexdigest()
    assert s == FZ["shas"][key], (key, s, FZ["shas"][key])

# ---- verbatim row universe via recorder-wrapped builders (d3 lines 33-44) ----
_rec = []
_of, _oc = D.ref_fwd, D.cand_fwd
def _ref_rec(x, W, b):
    R = _of(x, W, b); _rec.append(["R", x, W, b, R]); return R
def _cand_rec(x, W, b):
    T = _oc(x, W, b); _rec.append(["T", None, None, None, T]); return T
D.ref_fwd, D.cand_fwd = _ref_rec, _cand_rec

rows_all1 = D.build_rows([10 ** (6 * si / 23) for si in range(24)])
rows_c = D.build_rows_contingency()
D.ref_fwd, D.cand_fwd = _of, _oc
assert len(_rec) == 2 * (len(rows_all1) + len(rows_c))

all_rows = rows_all1 + rows_c
for i, r in enumerate(all_rows):
    r["_rec"] = 2 * i
kept = [r for r in all_rows if r["arm"] == "cancel"] + rows_c
assert len(kept) == 116 and {r["arm"] for r in kept} == {"ordmag", "cancel"}

# ---- tie-out vs committed d2_conv.jsonl (d3 lines 60-67 verbatim) ----
comm = {}
for line in open(os.path.join(RES, "d2_conv.jsonl")):
    r = json.loads(line)
    comm[(r["arm"], r["seed"], r["si"], round(float(r["scale"]), 12))] = r["max_budget_ratio"]
for r in kept:
    k = (r["arm"], r["seed"], r["si"], round(float(r["scale"]), 12))
    assert abs(comm[k] - r["max_budget_ratio"]) <= 1e-9 * abs(comm[k]), (k, comm[k], r["max_budget_ratio"])

# ---- frozen arms ----
def gemm64_raw(x, W, b):
    # d2 cand path in full fp64, NO cast (exact-arith pair for tA2)
    P = x.double().unfold(2, D.K, 1).unfold(3, D.K, 1)
    P = P.permute(0, 2, 3, 1, 4, 5).reshape(D.N * D.OH * D.OW, D.CIN * D.K * D.K)
    Wf = W.double().reshape(D.COUT, -1)
    y = P @ Wf.t() + b.double()
    return y.reshape(D.N, D.OH, D.OW, D.COUT).permute(0, 3, 1, 2).contiguous()

def tstar(cand, ref):
    d = (cand.double() - ref.double()).abs().max().item()
    return d / (1.0 + ref.double().abs().max().item())

out_rows = []
for r in kept:
    _, x, W, b, C32 = _rec[r["_rec"]]          # ref path fp32 = conv32
    assert x.is_cuda and C32.is_cuda, "env fidelity: expected cuda tensors"
    G32 = _rec[r["_rec"] + 1][4]               # cand path fp32 = gemm32
    C64 = F.conv2d(x.double(), W.double(), b.double())   # exact reference
    G64 = gemm64_raw(x, W, b)
    tA1 = tstar(C32, C64); tA2 = tstar(G64, C64); tB1 = tstar(G32, C64)
    # path arm (frozen trio); flags restored after
    b0 = torch.backends.cudnn.benchmark
    ts = {}
    ts["default"] = tstar(C32, C64)
    try:
        torch.backends.cudnn.benchmark = True
        ts["bench"] = tstar(F.conv2d(x, W, b), C64)
    finally:
        torch.backends.cudnn.benchmark = b0
    xc = x.to(memory_format=torch.channels_last); wc = W.to(memory_format=torch.channels_last)
    ts["cl"] = tstar(F.conv2d(xc, wc, b).contiguous(), C64)
    vals = list(ts.values())
    d0 = F.conv2d(x, W, b)
    dbench = F.conv2d(x, W, b) if False else None  # bench recompute below
    torch.backends.cudnn.benchmark = True
    dbench = F.conv2d(x, W, b)
    torch.backends.cudnn.benchmark = b0
    degenerate = bool(torch.equal(d0, dbench) and torch.equal(d0, F.conv2d(xc, wc, b).contiguous()))
    spread = ((max(vals) - min(vals)) / st.median(vals)) if st.median(vals) else 0.0
    out_rows.append(dict(arm=r["arm"], seed=r["seed"], si=r["si"], scale=r["scale"],
                         tA1=tA1, tA2=tA2, tB1=tB1, paths=ts,
                         path_spread=spread, paths_degenerate=bool(degenerate)))

med = lambda k: st.median([o[k] for o in out_rows])
med_spread = st.median([o["path_spread"] for o in out_rows])
n_degen = sum(1 for o in out_rows if o["paths_degenerate"])

V = {}
m_tA1 = med("tA1"); m_tA2 = med("tA2")
V["F1_exact_disagreement"] = bool(m_tA2 > TH["exact_agree"])
if m_tA1 >= TH["confirm_tA1"]:
    V["verdict"] = "V1_conv_side_CONFIRMED"
elif m_tA1 <= TH["dead_tA1"]:
    V["verdict"] = "V2_conv_side_DEAD"
else:
    V["verdict"] = "V3_partial_recorded"
V["reconsider_if"] = "transform-share lane reopen note" if V["verdict"].startswith("V1") else None
V["R1_path"] = "degenerate_void" if n_degen == len(out_rows) else \
               ("invariant" if med_spread <= TH["path_spread"] else "path_sensitive")

def q(vs):
    vs = sorted(vs); n = len(vs)
    return dict(min=vs[0], q25=vs[max(0, int(.25 * n) - 1)], q50=vs[max(0, int(.5 * n) - 1)],
                q75=vs[max(0, int(.75 * n) - 1)], q90=vs[max(0, int(.90 * n) - 1)], max=vs[-1])

tq = {k: q([o[k] for o in out_rows]) for k in ("tA1", "tA2", "tB1", "path_spread")}

out = dict(when=__import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
           freeze_sha=hashlib.sha256(open(os.path.join(RES, "cd_freeze.json"), "rb").read()).hexdigest()[:16],
           n_rows=len(out_rows), med_tA1=m_tA1, med_tA2=m_tA2, med_tB1=med("tB1"),
           med_path_spread=med_spread, n_paths_degenerate=n_degen,
           verdict=V, tstar_q=tq,
           env=dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                    cudnn=torch.backends.cudnn.version()))

with open(os.path.join(RES, "cd_discrim.jsonl"), "w") as f:
    for o in out_rows:
        f.write(json.dumps(o) + "\n")
json.dump(out, open(os.path.join(RES, "cd_discrim.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "tstar_q"}, indent=1))
print("t* quantiles:", json.dumps(out["tstar_q"]["tA1"], indent=1))
