#!/usr/bin/env python
"""s110_analysis.py — CPU recomputation of the s110 draw band into the map's count interval.
Reads results/s110_summary.json (produced by lab/s110_band.py on spark, sha-verified), writes
results/s110_analysis.json. No GPU, no new thresholds beyond the prereg's frozen branch rule.
"""
import json, math, os

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
S = json.load(open(os.path.join(RES, "s110_summary.json")))
pr = S["per_row"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - hw), min(1.0, c + hw))


rows = []
tot = lo = hi = 0.0
for k, v in sorted(pr.items(), key=lambda t: t[1]["recorded"]):
    r = v["exceed_rate"]
    n = v["n_ok"]
    kk = round(r * n)
    w = wilson(kk, n)
    tot += r; lo += w[0]; hi += w[1]
    rows.append(dict(row=k, op_class=v["op_class"], recorded=v["recorded"], rate=r, n=n,
                     rho=v["rho"], bench_frac_max=v["bench_frac"]["max"], labels=v["labels"],
                     wilson=[w[0], w[1]]))

grp_zero = [r for r in rows if r["rate"] == 0]
grp_one = [r for r in rows if r["rate"] == 1]
grp_mix = [r for r in rows if 0 < r["rate"] < 1]

# type-A band by label group (the s109 comparison the prose needs)
def band(rs):
    if not rs:
        return None
    cvs = [r["rho"]["cv"] for r in rs]
    spans = [r["rho"]["max"] / r["rho"]["min"] for r in rs]
    return dict(n=len(rs), cv_min=min(cvs), cv_max=max(cvs), span_min=min(spans), span_max=max(spans))

tf32 = [r for r in rows if r["labels"] == ["TF32_TENSOROP"]]
other = [r for r in rows if "OTHER" in r["labels"]]

# prereg prediction check (recorded, not a branch)
pred_ok = []
for r in rows:
    rec = r["recorded"]
    if rec < 0.72:
        pred = 0.0
    elif rec > 1.30:
        pred = 1.0
    else:
        pred = None
    pred_ok.append(dict(row=r["row"], recorded=rec, predicted=pred, measured=r["rate"],
                        match=(pred is None or abs(pred - r["rate"]) < 1e-9)))

out = dict(
    date=S["date"], source="results/s110_summary.json",
    wall_s=S["wall_s"], n_rows=S["n_picked"], n_cells=S["n_ok_cells"],
    instrument=dict(k1_bit_identical=S["k1_bit_identical"], k1_n=S["k1_n"],
                    k2_anchor_rel=S["k2_anchor_rel"], tie_out_ok=S["tie_out_ok"],
                    tie_out=S["tie_out"], ok=S["instrument_ok"]),
    branch=S["branch"], bench_branch=S["bench_branch"],
    count_point_recorded=S["n_recorded_exceed"] + 69,
    sum_exceed_rate=tot, count_ensemble=69 + tot,
    count_interval=[69 + lo, 69 + hi],
    groups=dict(rate_zero=[r["row"] for r in grp_zero],
                rate_one=[r["row"] for r in grp_one],
                mixed=[dict(row=r["row"], rate=r["rate"], recorded=r["recorded"],
                            rho_min=r["rho"]["min"], rho_max=r["rho"]["max"], cv=r["rho"]["cv"],
                            labels=r["labels"]) for r in grp_mix]),
    band_tf32=band(tf32), band_other=band(other),
    bench_gate_rows=S["bench_gate_rows"],
    prediction_check=pred_ok,
    prediction_misses=[p for p in pred_ok if not p["match"]],
    rows=rows,
)
json.dump(out, open(os.path.join(RES, "s110_analysis.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("branch", "bench_branch", "count_point_recorded", "count_ensemble",
                                      "count_interval", "groups", "band_tf32", "band_other",
                                      "prediction_misses", "instrument")}, indent=1))
