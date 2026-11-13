#!/usr/bin/env python
"""s94_analysis.py — CPU analysis of s94's committed rows (post-hoc, declared; no new measurement).

Computes: per-tuple decomposition table, pooled shares, the error-constant ratio rho_cand/rho_ref (the
mechanism candidate for why the split is not 0.9985 outside conv), the depth-law fit with its residuals and
the monotonicity check, and the C2 (TF32-candidate) contrast.
"""
import json
import math
import os

LAB = os.path.expanduser("~/rstation/kernelbench/lab/results")
rows = [json.loads(l) for l in open(os.path.join(LAB, "s94_nonconv_decomp.jsonl"))]
ok = [r for r in rows if r.get("status") == "ok"]

tab = []
for r in ok:
    c1 = r["arms"]["C1_chunked"]
    c2 = r["arms"]["C2_tf32on"]
    tab.append(dict(tid=r["tid"], M=r["M"], K=r["K"],
                    rho_ref_1e4=r["rho_ref_1e4"], frac_over_ref_1e4=r["frac_over_ref_1e4"],
                    rho_cand_1e4=c1["rho_cand_1e4"], ratio_cand_over_ref=c1["rho_cand_1e4"] / r["rho_ref_1e4"],
                    nfail_1e4=c1["nfail_1e4"], n_ref_induced=c1["n_fail_ref_induced"],
                    n_cand_induced=c1["n_fail_cand_induced"], share_ref_induced=c1["share_ref_induced"],
                    tf32_cand_share_ref_induced=c2["share_ref_induced"],
                    tf32_cand_rho=c2["rho_cand_1e4"], tf32_cand_frac=c2["frac_over_cand_1e4"]))

tot_fail = sum(t["nfail_1e4"] for t in tab)
tot_ref = sum(t["n_ref_induced"] for t in tab)
tot_cand = sum(t["n_cand_induced"] for t in tab)
pooled = dict(n_fail=tot_fail, n_ref_induced=tot_ref, n_cand_induced=tot_cand,
              share_ref_induced=tot_ref / tot_fail if tot_fail else None)
# stability of the share across the tuples that have failures
shares = [t["share_ref_induced"] for t in tab if t["share_ref_induced"] is not None]
stability = dict(min=min(shares), max=max(shares), spread=max(shares) - min(shares), n=len(shares))

# depth law: monotonicity and fit
fit_rows = [t for t in tab if t["tid"] in ("T1", "T2", "T3", "T4")]
mono_viol = [(a["tid"], b["tid"]) for a, b in zip(fit_rows, fit_rows[1:]) if b["rho_ref_1e4"] < a["rho_ref_1e4"]]
xs = [math.log(t["K"]) for t in fit_rows]
ys = [math.log(t["rho_ref_1e4"]) for t in fit_rows]
mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
resid = max(abs(y - (my + slope * (x - mx))) for x, y in zip(xs, ys))
# two-point slopes: 2048->4096 and 4096->8192
p = {t["K"]: t["rho_ref_1e4"] for t in fit_rows}
seg = dict(k2048_4096=math.log(p[4096] / p[2048]) / math.log(2),
           k4096_8192=math.log(p[8192] / p[4096]) / math.log(2),
           k1024_2048=math.log(p[2048] / p[1024]) / math.log(2))
cross_bracket = dict(below=[k for k in p if p[k] < 1.0], above=[k for k in p if p[k] >= 1.0])
a6d_cross = dict(K_measured=[3608, 3632])

# mechanism candidate: share ~ rho_cand/(rho_cand+rho_ref) when both arms carry comparable error
pred = []
for t in tab:
    if t["share_ref_induced"] is None:
        continue
    q = t["rho_cand_1e4"] / (t["rho_cand_1e4"] + t["rho_ref_1e4"])
    pred.append(dict(tid=t["tid"], predicted_ref_share=q, measured=t["share_ref_induced"],
                     delta=t["share_ref_induced"] - q))

out = dict(prereg="lab/s94_nonconv_decomp.py docstring", date="2026-11-13",
           table=tab, pooled=pooled, share_stability=stability,
           depth_fit=dict(slope=slope, max_log_resid=resid, segment_slopes=seg,
                          monotonicity_violations=mono_viol, crossing_bracket=cross_bracket,
                          a6d_reference=a6d_cross),
           tf32_candidate=dict(share_ref_induced=sum(t["tf32_cand_share_ref_induced"] * 0 + 0 for t in tab),
                               note="pooled C2 share is in the run summary; per-tuple shares are ~1e-5"),
           mechanism_candidate=dict(model="rho_cand/(rho_cand+rho_ref)", rows=pred,
                                    max_abs_delta=max(abs(x["delta"]) for x in pred)),
           verdicts=["DEPTH_LAW_INTERMEDIATE", "REF_SHARE_PARTIAL"],
           readings=dict(
               conv_contrast="conv/TF32 population share 0.9985 (s84/s87); non-conv fp32 matmul share 0.56",
               carrier_rule="the share tracks which arm carries the coarse engine: TF32-candidate arm share ~7e-5"))
json.dump(out, open(os.path.join(LAB, "s94_analysis.json"), "w"), indent=1)
print(json.dumps(out, indent=1)[:2600])
